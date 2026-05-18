import os
import pickle
import threading
from collections import deque

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from app.core.algorithm_bridge import AlgorithmFusionBridge
from config.config import (
    ATTENTION_THRESHOLDS,
    DATASET_PATH,
    MODEL_PATH,
    ORIGINAL_DATASET_PATH,
    SCORING_MODE,
    UHMF_PRIMARY_WEIGHT,
    YOLO_MODEL_PATH,
)


class AttentionDetector:
    def __init__(self):
        self.model = None
        self.yolo_model = None
        self.yolo_checked = False
        self.rf_feature_columns = [
            "no_of_face",
            "face_x",
            "face_y",
            "face_w",
            "face_h",
            "face_con",
            "no_of_hand",
            "pose_x",
            "pose_y",
            "phone",
            "phone_con",
            "pose_encoded",
        ]
        self.frame_index = 0
        self.phone_detection_interval = 6
        self.phone_detection_imgsz = 320
        self.phone_detection_conf = 0.18
        self.phone_detection_crop_conf = 0.14
        self.phone_hold_frames = 8
        self.last_phone_frame = -1000
        self.face_detection_scale = 0.45
        self.cached_phone_data = self._empty_phone_data()
        self.facemark_interval = 3
        self.facemark_checked = False
        self.facemark_available = False
        self.facemark_error = ""
        self.facemark_model = None
        self.facemark_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models",
            "facemark",
            "lbfmodel.yaml",
        )
        self.cached_facemark_face = self._empty_facemark_face()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_alt_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        self.profile_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

        self.prev_face_data = None
        self.prev_mouth_roi = None
        self.eye_state_history = deque(maxlen=8)
        self.eye_closure_history = deque(maxlen=90)
        self.ear_history = deque(maxlen=60)
        self.mar_history = deque(maxlen=60)
        self.head_pose_history = deque(maxlen=45)
        self.mouth_open_history = deque(maxlen=10)
        self.mouth_closed_baseline = None
        self.eye_open_baseline = None
        self.prev_eyes_closed = False
        self.blink_count = 0
        self.consecutive_face_misses = 0
        self.calibration_lock = threading.Lock()
        self.calibration_active = False
        self.calibration_required_samples = 12
        self.calibration_samples = []
        self.calibration_profile = None
        self.face_movement_history = deque(maxlen=30)
        self.expression_history = deque(maxlen=20)
        self.score_history = deque(maxlen=12)
        self.face_detection_history = deque(maxlen=4)
        self.last_smoothed_score = None
        self.multimodal_provider = None

        self.backend_bridge = AlgorithmFusionBridge()
        self.backend_bridge.warmup_async()

        self.load_or_train_model()

    def load_or_train_model(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as file_obj:
                    self.model = pickle.load(file_obj)
                print("Attention model loaded successfully")
                return
            except Exception as exc:
                print(f"Model load failed, retraining fallback enabled: {exc}")

        self.train_model()

    def train_model(self):
        print("Training attention detection model...")

        dataset_path = DATASET_PATH if os.path.exists(DATASET_PATH) else ORIGINAL_DATASET_PATH
        df = pd.read_csv(dataset_path)

        pose_mapping = {"forward": 0, "left": 1, "right": 2, "down": 3}
        df["pose_encoded"] = df["pose"].map(pose_mapping).fillna(0)

        x_data = df[self.rf_feature_columns].fillna(0)
        y_data = df["label"]

        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )

        self.model = RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Attention model accuracy: {accuracy:.2%}")

        with open(MODEL_PATH, "wb") as file_obj:
            pickle.dump(self.model, file_obj)
        print(f"Attention model saved to {MODEL_PATH}")

    def detect_frame_with_movement(self, frame):
        if frame is None or frame.size == 0:
            return 10.0, self._build_runtime_info({}, {}, None), {"horizontal": 0.0, "vertical": 0.0, "total": 0.0}

        self.frame_index += 1

        face_data = self._detect_face_with_movement(frame)
        facemark_data = self._detect_facemark(frame, face_data)
        pose_data = self._detect_pose(frame, face_data, facemark_data)
        phone_data = self._detect_phone(frame)
        hand_count = self._estimate_hand_count(face_data)
        expression_data = self._detect_expression(frame, face_data, facemark_data)

        features = self._extract_features(frame, face_data, pose_data, phone_data, hand_count, expression_data)
        self._collect_calibration_sample(features)
        base_score = self._calculate_attention_score(features, expression_data)

        context = self._build_backend_context(base_score)
        backend_result = self.backend_bridge.analyze(features, context)
        final_score = self._combine_with_backend(base_score, backend_result, features)
        final_score = self._apply_personal_calibration(final_score, features)
        smoothed_score = self._smooth_score(final_score, face_data["detected"], features)
        movement_info = self._calculate_movement_info()

        runtime_info = self._build_runtime_info(
            features,
            expression_data,
            backend_result,
            context.get("multimodal"),
        )
        runtime_info["base_score"] = base_score
        runtime_info["final_score"] = final_score
        runtime_info["smoothed_score"] = smoothed_score
        runtime_info["score_source"] = "UHMF" if backend_result.get("available") else "local_fallback"
        runtime_info["calibration"] = self.get_calibration_status()
        runtime_info["score_explanation"] = self._build_score_explanation(
            features,
            expression_data,
            backend_result,
            base_score,
            final_score,
            smoothed_score,
            context.get("multimodal"),
        )

        return smoothed_score, runtime_info, movement_info

    def reset_runtime_state(self):
        self.frame_index = 0
        self.cached_phone_data = self._empty_phone_data()
        self.cached_facemark_face = self._empty_facemark_face()
        self.last_phone_frame = -1000
        self.prev_face_data = None
        self.prev_mouth_roi = None
        self.eye_state_history.clear()
        self.eye_closure_history.clear()
        self.ear_history.clear()
        self.mar_history.clear()
        self.head_pose_history.clear()
        self.mouth_open_history.clear()
        self.mouth_closed_baseline = None
        self.eye_open_baseline = None
        self.prev_eyes_closed = False
        self.blink_count = 0
        self.consecutive_face_misses = 0
        self.face_movement_history.clear()
        self.expression_history.clear()
        self.score_history.clear()
        self.face_detection_history.clear()
        self.last_smoothed_score = None

    def set_multimodal_provider(self, provider):
        self.multimodal_provider = provider

    def start_focus_calibration(self):
        with self.calibration_lock:
            self.calibration_active = True
            self.calibration_samples = []
        return "开始校准：请保持正常专注姿态 2-3 秒"

    def clear_focus_calibration(self):
        with self.calibration_lock:
            self.calibration_active = False
            self.calibration_samples = []
            self.calibration_profile = None
        return "个人校准已清除"

    def get_calibration_status(self):
        with self.calibration_lock:
            if self.calibration_profile:
                return {
                    "active": False,
                    "ready": True,
                    "progress": 1.0,
                    "sample_count": len(self.calibration_samples),
                    "required_samples": self.calibration_required_samples,
                    "message": "个人校准已启用",
                }
            if self.calibration_active:
                sample_count = len(self.calibration_samples)
                return {
                    "active": True,
                    "ready": False,
                    "progress": sample_count / self.calibration_required_samples,
                    "sample_count": sample_count,
                    "required_samples": self.calibration_required_samples,
                    "message": f"校准中 {sample_count}/{self.calibration_required_samples}",
                }
            return {
                "active": False,
                "ready": False,
                "progress": 0.0,
                "sample_count": 0,
                "required_samples": self.calibration_required_samples,
                "message": "尚未校准",
            }

    def _collect_calibration_sample(self, features):
        with self.calibration_lock:
            if not self.calibration_active:
                return
            if features.get("no_of_face", 0) != 1 or features.get("phone", 0) == 1:
                return

            sample = {
                "face_center_x_norm": features.get("face_center_x_norm", 0.5),
                "face_center_y_norm": features.get("face_center_y_norm", 0.5),
                "face_area_ratio": features.get("face_area_ratio", 0.0),
                "pose_x": features.get("pose_x", 0.0),
                "pose_y": features.get("pose_y", 0.0),
                "movement_total_norm": features.get("movement_total_norm", 0.0),
                "mouth_movement_norm": features.get("mouth_movement_norm", 0.0),
                "mouth_open_ratio": features.get("mouth_open_ratio", 0.0),
                "mouth_open_strength": features.get("mouth_open_strength", 0.0),
                "eye_open_score": features.get("eye_open_score", 0.0),
                "body_slouch": features.get("body_slouch", 0.0),
                "body_lean": features.get("body_lean", 0.0),
                "perclos": features.get("perclos", 0.0),
                "head_pose_pressure": features.get("head_pose_pressure", 0.0),
            }
            self.calibration_samples.append(sample)

            if len(self.calibration_samples) < self.calibration_required_samples:
                return

            self.calibration_profile = {
                key: float(np.median([item[key] for item in self.calibration_samples]))
                for key in self.calibration_samples[0]
            }
            self.calibration_active = False

    def _apply_personal_calibration(self, score, features):
        with self.calibration_lock:
            profile = dict(self.calibration_profile) if self.calibration_profile else None

        if not profile or features.get("no_of_face", 0) != 1:
            return score

        center_delta = abs(features.get("face_center_x_norm", 0.5) - profile["face_center_x_norm"])
        center_delta += abs(features.get("face_center_y_norm", 0.5) - profile["face_center_y_norm"])
        pose_delta = abs(features.get("pose_x", 0.0) - profile["pose_x"])
        pose_delta += abs(features.get("pose_y", 0.0) - profile["pose_y"])
        area_delta = abs(features.get("face_area_ratio", 0.0) - profile["face_area_ratio"])
        movement = features.get("movement_total_norm", 0.0)

        deviation = center_delta * 2.6 + pose_delta / 18.0 + area_delta * 3.0 + movement * 1.4
        similarity = float(np.clip(1.0 - deviation, 0.0, 1.0))

        calibrated_score = score
        if features.get("phone", 0) == 1:
            calibrated_score -= 10.0
        elif similarity >= 0.72:
            calibrated_score += 8.0
            if score < 70:
                calibrated_score = max(calibrated_score, 70.0 + similarity * 8.0)
        elif similarity < 0.38:
            calibrated_score -= 10.0

        return float(np.clip(calibrated_score, 10, 100))

    def _build_backend_context(self, base_score):
        history = list(self.score_history)
        movement_history = list(self.face_movement_history)
        multimodal = self._safe_multimodal_snapshot()

        focused_threshold = ATTENTION_THRESHOLDS["focused"]
        distracted_threshold = ATTENTION_THRESHOLDS["moderate"]

        if history:
            avg_score = float(np.mean(history))
            focused_ratio = sum(score >= focused_threshold for score in history) / len(history)
            distraction_ratio = sum(score < distracted_threshold for score in history) / len(history)
            score_delta = float((history[-1] - history[-2]) / 100.0) if len(history) > 1 else 0.0
            stability = float(max(0.0, 1.0 - (np.std(history) / 35.0)))
        else:
            avg_score = base_score
            focused_ratio = 0.0
            distraction_ratio = 0.0
            score_delta = 0.0
            stability = 0.0

        face_detection_rate = (
            float(sum(self.face_detection_history) / len(self.face_detection_history))
            if self.face_detection_history
            else 0.0
        )

        status_weights = {
            "focused": focused_ratio,
            "moderate": max(0.0, 1.0 - focused_ratio - distraction_ratio),
            "distracted": distraction_ratio,
        }

        return {
            "base_score": base_score,
            "avg_score": avg_score,
            "score_history": history[-20:],
            "recent_movements": [
                item["total_norm"] for item in movement_history[-20:] if "total_norm" in item
            ],
            "face_detection_rate": face_detection_rate,
            "focused_ratio": focused_ratio,
            "distraction_ratio": distraction_ratio,
            "score_delta": score_delta,
            "stability": stability,
            "status_weights": status_weights,
            "session_minutes": len(history) / 6.0,
            "multimodal": multimodal,
        }

    def _combine_with_backend(self, base_score, backend_result, features):
        if not backend_result.get("available") or backend_result.get("score") is None:
            return base_score

        backend_score = backend_result["score"]
        confidence = backend_result.get("confidence", 0.0)

        calibrated_backend = backend_score
        if features["no_of_face"] == 1 and features["phone"] == 0:
            calibrated_backend += 4.0
        if features["pose_encoded"] == 0:
            calibrated_backend += 6.0
        elif features["pose_encoded"] == 3:
            calibrated_backend -= 10.0
        else:
            calibrated_backend -= 4.0

        if features.get("movement_total_norm", 0.0) < 0.05:
            calibrated_backend += 3.0
        if features.get("mouth_movement_norm", 0.0) > 0.55:
            calibrated_backend -= 8.0
        if features.get("eyes_closed", 0) == 1:
            calibrated_backend -= 15.0
        if features.get("yawn_detected", 0) == 1:
            calibrated_backend -= 18.0
        elif features.get("mouth_open", 0) == 1:
            calibrated_backend -= 9.0
        if features.get("perclos", 0.0) > 0.32:
            calibrated_backend -= 12.0
        if features.get("head_pose_pressure", 0.0) > 0.58:
            calibrated_backend -= 8.0
        if features.get("body_slouch", 0.0) > 0.55:
            calibrated_backend -= 12.0
        elif features.get("body_lean", 0.0) > 0.6:
            calibrated_backend -= 7.0
        if features["phone"] == 1:
            calibrated_backend -= 18.0

        calibrated_backend = float(np.clip(calibrated_backend, 10, 100))

        if SCORING_MODE == "uhmf_primary":
            backend_weight = UHMF_PRIMARY_WEIGHT if confidence >= 0.65 else 0.62
        else:
            backend_weight = min(0.45, 0.12 + confidence * 0.25)

        combined_score = (1 - backend_weight) * base_score + backend_weight * calibrated_backend
        return float(np.clip(combined_score, 10, 100))

    def _build_runtime_info(self, features, expression_data, backend_result, multimodal=None):
        return {
            "features": features,
            "expression": expression_data,
            "backend": backend_result or {"available": False, "status": self.backend_bridge.get_status()},
            "multimodal": multimodal or self._safe_multimodal_snapshot(),
        }

    def _safe_multimodal_snapshot(self):
        if self.multimodal_provider is None:
            return {
                "available": False,
                "running": False,
                "status": "真实多模态未接入",
                "error": "",
                "audio": {"available": False},
                "text": {"available": False, "last_transcript": ""},
            }

        try:
            snapshot = self.multimodal_provider.snapshot()
            if isinstance(snapshot, dict):
                return snapshot
        except Exception as exc:
            return {
                "available": False,
                "running": False,
                "status": f"真实多模态读取失败（{exc}）",
                "error": str(exc),
                "audio": {"available": False},
                "text": {"available": False, "last_transcript": ""},
            }

        return {
            "available": False,
            "running": False,
            "status": "真实多模态不可用",
            "error": "",
            "audio": {"available": False},
            "text": {"available": False, "last_transcript": ""},
        }

    def _init_facemark(self):
        if self.facemark_checked:
            return self.facemark_available

        self.facemark_checked = True
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "createFacemarkLBF"):
            self.facemark_error = "当前 OpenCV 不包含 facemark 模块"
            return False
        if not os.path.exists(self.facemark_model_path):
            self.facemark_error = "缺少 LBF 人脸关键点模型"
            return False

        try:
            self.facemark_model = cv2.face.createFacemarkLBF()
            self.facemark_model.loadModel(self.facemark_model_path)
            self.facemark_available = True
            self.facemark_error = ""
            print("OpenCV LBF facemark enhancement enabled")
            return True
        except Exception as exc:
            self.facemark_available = False
            self.facemark_error = f"LBF 人脸关键点初始化失败: {exc}"
            print(self.facemark_error)
            return False

    def _detect_facemark(self, frame, face_data):
        if not face_data.get("detected"):
            self.cached_facemark_face = self._empty_facemark_face()
            return {"available": False, "status": "未检测到人脸", "face": self.cached_facemark_face}

        if not self._init_facemark():
            return {
                "available": False,
                "status": self.facemark_error or "LBF 人脸关键点不可用",
                "face": self._empty_facemark_face(error=self.facemark_error),
            }

        if self.frame_index % self.facemark_interval != 0 and self.cached_facemark_face.get("detected"):
            return {"available": True, "status": "LBF 关键点缓存", "face": self.cached_facemark_face}

        frame_h, frame_w = frame.shape[:2]
        x = int(max(0, min(frame_w - 1, face_data.get("x", 0))))
        y = int(max(0, min(frame_h - 1, face_data.get("y", 0))))
        w = int(max(1, min(frame_w - x, face_data.get("w", 1))))
        h = int(max(1, min(frame_h - y, face_data.get("h", 1))))
        faces = np.array([[x, y, w, h]], dtype=np.int32)

        try:
            ok, landmarks = self.facemark_model.fit(frame, faces)
            if not ok or landmarks is None or len(landmarks) == 0:
                self.cached_facemark_face = self._empty_facemark_face()
            else:
                points = np.asarray(landmarks[0], dtype=np.float32).reshape(-1, 2)
                self.cached_facemark_face = self._extract_facemark_face(points, frame.shape)
        except Exception as exc:
            self.cached_facemark_face = self._empty_facemark_face(error=str(exc))

        return {"available": self.cached_facemark_face.get("detected", False), "status": "OpenCV LBF 已启用", "face": self.cached_facemark_face}

    def _empty_facemark_face(self, error=""):
        return {
            "detected": False,
            "eyes_detected": 0,
            "ear": 0.0,
            "eye_closed_candidate": False,
            "mouth_open_ratio": 0.0,
            "head_yaw": 0.0,
            "head_pitch": 0.0,
            "head_roll": 0.0,
            "head_yaw_deg": 0.0,
            "head_pitch_deg": 0.0,
            "head_roll_deg": 0.0,
            "head_pose_reliable": False,
            "head_pose_source": "none",
            "source": "none",
            "error": error,
        }

    def _extract_facemark_face(self, points, frame_shape):
        if points is None or len(points) < 68:
            return self._empty_facemark_face()

        left_ear = self._eye_aspect_ratio(
            points[36],
            points[39],
            points[37],
            points[41],
            points[38],
            points[40],
        )
        right_ear = self._eye_aspect_ratio(
            points[42],
            points[45],
            points[43],
            points[47],
            points[44],
            points[46],
        )
        ear = float(np.mean([left_ear, right_ear]))
        if self.eye_open_baseline is None and ear > 0.12:
            self.eye_open_baseline = ear
        elif self.eye_open_baseline is not None and ear > self.eye_open_baseline * 0.72:
            self.eye_open_baseline = max(
                self.eye_open_baseline * 0.96 + ear * 0.04,
                self.eye_open_baseline,
            )
        eye_threshold = max(0.13, (self.eye_open_baseline or 0.22) * 0.58)
        eye_closed_candidate = ear < eye_threshold

        mouth_width = max(self._point_distance(points[60], points[64]), 1.0)
        mouth_open_ratio = float(
            np.clip(
                (self._point_distance(points[62], points[66]) + self._point_distance(points[63], points[65]))
                / (2.0 * mouth_width),
                0.0,
                1.0,
            )
        )

        jaw_mid_x = (points[0][0] + points[16][0]) / 2.0
        face_height = max(self._point_distance(points[27], points[8]), 1.0)
        fallback_yaw = float(np.clip((points[30][0] - jaw_mid_x) / max(self._point_distance(points[0], points[16]) * 0.26, 1.0), -1.0, 1.0))
        fallback_pitch = float(np.clip((points[30][1] - ((points[27][1] + points[8][1]) / 2.0)) / max(face_height * 0.35, 1.0), -1.0, 1.0))
        head_pose = self._estimate_head_pose_solvepnp(points, frame_shape)
        if head_pose["head_pose_reliable"]:
            head_yaw = float(np.clip(head_pose["head_yaw"] * 0.75 + fallback_yaw * 0.25, -1.0, 1.0))
            if abs(fallback_pitch) >= 0.18 and np.sign(head_pose["head_pitch"]) != np.sign(fallback_pitch):
                head_pitch = fallback_pitch
            else:
                head_pitch = float(np.clip(head_pose["head_pitch"] * 0.7 + fallback_pitch * 0.3, -1.0, 1.0))
            head_roll = float(head_pose["head_roll"])
            head_pose_source = "opencv_solvepnp"
            head_pose_reliable = True
            head_yaw_deg = float(head_pose["head_yaw_deg"])
            head_pitch_deg = float(head_pose["head_pitch_deg"])
            head_roll_deg = float(head_pose["head_roll_deg"])
        else:
            head_yaw = fallback_yaw
            head_pitch = fallback_pitch
            head_roll = 0.0
            head_pose_source = "lbf_geometry"
            head_pose_reliable = False
            head_yaw_deg = float(fallback_yaw * 30.0)
            head_pitch_deg = float(fallback_pitch * 25.0)
            head_roll_deg = 0.0

        return {
            "detected": True,
            "eyes_detected": 0 if eye_closed_candidate else 2,
            "ear": ear,
            "eye_closed_candidate": eye_closed_candidate,
            "mouth_open_ratio": mouth_open_ratio,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "head_roll": head_roll,
            "head_yaw_deg": head_yaw_deg,
            "head_pitch_deg": head_pitch_deg,
            "head_roll_deg": head_roll_deg,
            "head_pose_reliable": head_pose_reliable,
            "head_pose_source": head_pose_source,
            "source": "opencv_lbf",
            "error": "",
        }

    def _estimate_head_pose_solvepnp(self, points, frame_shape):
        try:
            frame_h, frame_w = frame_shape[:2]
            image_points = np.array(
                [
                    points[30],  # nose tip
                    points[8],   # chin
                    points[36],  # left eye corner
                    points[45],  # right eye corner
                    points[48],  # left mouth corner
                    points[54],  # right mouth corner
                ],
                dtype=np.float64,
            )
            model_points = np.array(
                [
                    (0.0, 0.0, 0.0),
                    (0.0, -330.0, -65.0),
                    (-225.0, 170.0, -135.0),
                    (225.0, 170.0, -135.0),
                    (-150.0, -150.0, -125.0),
                    (150.0, -150.0, -125.0),
                ],
                dtype=np.float64,
            )
            focal_length = float(frame_w)
            camera_matrix = np.array(
                [
                    [focal_length, 0.0, frame_w / 2.0],
                    [0.0, focal_length, frame_h / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            distortion = np.zeros((4, 1), dtype=np.float64)
            ok, rotation_vec, _translation_vec = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                raise RuntimeError("solvePnP failed")

            rotation_matrix, _jacobian = cv2.Rodrigues(rotation_vec)
            angles = cv2.RQDecomp3x3(rotation_matrix)[0]
            pitch_deg = self._compact_head_angle(float(angles[0]))
            yaw_deg = self._compact_head_angle(float(angles[1]))
            roll_deg = self._compact_head_angle(float(angles[2]))
            reliable = all(np.isfinite([yaw_deg, pitch_deg, roll_deg])) and max(
                abs(yaw_deg),
                abs(pitch_deg),
                abs(roll_deg),
            ) <= 75.0
            return {
                "head_yaw": float(np.clip(yaw_deg / 30.0, -1.0, 1.0)),
                "head_pitch": float(np.clip(pitch_deg / 25.0, -1.0, 1.0)),
                "head_roll": float(np.clip(roll_deg / 30.0, -1.0, 1.0)),
                "head_yaw_deg": yaw_deg,
                "head_pitch_deg": pitch_deg,
                "head_roll_deg": roll_deg,
                "head_pose_reliable": reliable,
            }
        except Exception:
            return {
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "head_roll": 0.0,
                "head_yaw_deg": 0.0,
                "head_pitch_deg": 0.0,
                "head_roll_deg": 0.0,
                "head_pose_reliable": False,
            }

    def _compact_head_angle(self, angle):
        if angle > 90.0:
            return angle - 180.0
        if angle < -90.0:
            return angle + 180.0
        return angle

    def _eye_aspect_ratio(self, outer, inner, upper_1, lower_1, upper_2, lower_2):
        horizontal = self._point_distance(outer, inner)
        if horizontal <= 1e-6:
            return 0.0
        vertical = self._point_distance(upper_1, lower_1) + self._point_distance(upper_2, lower_2)
        return float(vertical / (2.0 * horizontal))

    def _point_distance(self, point_a, point_b):
        return float(np.linalg.norm(point_a - point_b))

    def _detect_face_with_movement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        scaled_gray = cv2.resize(
            gray,
            None,
            fx=self.face_detection_scale,
            fy=self.face_detection_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        faces = []
        detector_candidates = [
            (self.face_cascade, scaled_gray, 1.08, 4, (30, 30), False),
            (self.face_alt_cascade, scaled_gray, 1.05, 3, (26, 26), False),
            (self.profile_face_cascade, scaled_gray, 1.08, 4, (30, 30), False),
            (self.profile_face_cascade, cv2.flip(scaled_gray, 1), 1.08, 4, (30, 30), True),
        ]

        for cascade, image, scale_factor, min_neighbors, min_size, mirrored in detector_candidates:
            if cascade.empty():
                continue
            detected = cascade.detectMultiScale(
                image,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
            )
            if len(detected) == 0:
                continue

            frame_w = frame.shape[1]
            normalized = []
            for item in detected:
                x, y, w, h = item
                if mirrored:
                    x = image.shape[1] - x - w
                x = int(x / self.face_detection_scale)
                y = int(y / self.face_detection_scale)
                w = int(w / self.face_detection_scale)
                h = int(h / self.face_detection_scale)
                x = max(0, min(frame_w - 1, x))
                y = max(0, min(frame.shape[0] - 1, y))
                normalized.append((x, y, w, h))
            faces = normalized
            break

        if len(faces) == 0:
            self.consecutive_face_misses += 1
            if self.consecutive_face_misses > 3:
                self.prev_face_data = None
            self.face_movement_history.append(
                {"horizontal": 0.0, "vertical": 0.0, "total": 0.0, "total_norm": 0.0}
            )
            return {
                "detected": False,
                "x": 0.0,
                "y": 0.0,
                "w": 0.0,
                "h": 0.0,
                "confidence": 0.0,
                "movement": 0.0,
                "horizontal_movement": 0.0,
                "vertical_movement": 0.0,
                "eyes_detected": 0,
                "source": "none",
            }

        self.consecutive_face_misses = 0
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        roi_gray = gray[y : y + h, x : x + w]
        upper_roi_gray = roi_gray[: max(1, int(h * 0.58)), :]
        eyes = self.eye_cascade.detectMultiScale(
            upper_roi_gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(max(10, int(w * 0.08)), max(8, int(h * 0.05))),
        )
        eye_count = min(len(eyes), 2)

        frame_h, frame_w = frame.shape[:2]
        area_ratio = (w * h) / float(frame_w * frame_h)
        confidence = 60.0 + min(25.0, area_ratio * 220.0)
        confidence += 5.0 * eye_count
        confidence = float(np.clip(confidence, 55.0, 95.0))

        return self._build_face_movement_data(frame, x, y, w, h, confidence, eye_count, "opencv")

    def _build_face_movement_data(self, frame, x, y, w, h, confidence, eye_count, source):
        self.consecutive_face_misses = 0
        current_face = {
            "x": float(x + w / 2.0),
            "y": float(y + h / 2.0),
            "w": float(w),
            "h": float(h),
        }

        horizontal_movement = 0.0
        vertical_movement = 0.0
        total_movement = 0.0
        if self.prev_face_data is not None:
            horizontal_movement = abs(current_face["x"] - self.prev_face_data["x"])
            vertical_movement = abs(current_face["y"] - self.prev_face_data["y"])
            total_movement = horizontal_movement + vertical_movement

        self.prev_face_data = current_face

        frame_h, frame_w = frame.shape[:2]
        movement_norm = float(np.clip(total_movement / max(frame_w, frame_h), 0.0, 1.0))
        self.face_movement_history.append(
            {
                "horizontal": horizontal_movement,
                "vertical": vertical_movement,
                "total": total_movement,
                "total_norm": movement_norm,
            }
        )

        return {
            "detected": True,
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "confidence": confidence,
            "movement": total_movement,
            "horizontal_movement": horizontal_movement,
            "vertical_movement": vertical_movement,
            "eyes_detected": eye_count,
            "source": source,
        }

    def _detect_pose(self, frame, face_data, facemark_data=None):
        if not face_data["detected"]:
            return {
                "x": 0.0,
                "y": 0.0,
                "pose": "down",
                "body_visible": False,
                "body_slouch": 0.0,
                "body_lean": 0.0,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "head_roll": 0.0,
                "head_yaw_deg": 0.0,
                "head_pitch_deg": 0.0,
                "head_roll_deg": 0.0,
                "head_pose_reliable": False,
                "head_pose_source": "none",
                "posture_status": "未检测",
            }

        frame_h, frame_w = frame.shape[:2]
        face_center_x = face_data["x"] + face_data["w"] / 2.0
        face_center_y = face_data["y"] + face_data["h"] / 2.0

        pose_x_norm = (face_center_x - (frame_w / 2.0)) / max(frame_w / 2.0, 1.0)
        pose_y_norm = (face_center_y - (frame_h / 2.0)) / max(frame_h / 2.0, 1.0)

        # Map to the same approximate numeric range as the training dataset.
        pose_x_value = float(np.clip(pose_x_norm * 18.0, -20.0, 20.0))
        pose_y_value = float(np.clip(pose_y_norm * 18.0, -20.0, 20.0))

        if pose_y_value > 8.5:
            pose_name = "down"
        elif pose_x_value < -5.5:
            pose_name = "left"
        elif pose_x_value > 5.5:
            pose_name = "right"
        else:
            pose_name = "forward"

        facemark_face = (facemark_data or {}).get("face", {})
        landmark_face = facemark_face
        head_yaw = 0.0
        head_pitch = 0.0
        head_roll = 0.0
        head_yaw_deg = 0.0
        head_pitch_deg = 0.0
        head_roll_deg = 0.0
        head_pose_reliable = False
        head_pose_source = "none"
        if landmark_face.get("detected"):
            head_yaw = float(landmark_face.get("head_yaw", 0.0))
            head_pitch = float(landmark_face.get("head_pitch", 0.0))
            head_roll = float(landmark_face.get("head_roll", 0.0))
            head_yaw_deg = float(landmark_face.get("head_yaw_deg", head_yaw * 30.0))
            head_pitch_deg = float(landmark_face.get("head_pitch_deg", head_pitch * 25.0))
            head_roll_deg = float(landmark_face.get("head_roll_deg", head_roll * 30.0))
            head_pose_reliable = bool(landmark_face.get("head_pose_reliable", False))
            head_pose_source = landmark_face.get("head_pose_source", "lbf_geometry")
            self.head_pose_history.append(max(abs(head_yaw), abs(head_pitch), abs(head_roll) * 0.85))
            if head_pitch > 0.38:
                pose_name = "down"
                pose_y_value = max(pose_y_value, 10.0)
            elif head_yaw < -0.38:
                pose_name = "left"
                pose_x_value = min(pose_x_value, -8.0)
            elif head_yaw > 0.38:
                pose_name = "right"
                pose_x_value = max(pose_x_value, 8.0)
            elif abs(head_roll) > 0.52:
                pose_name = "left" if head_roll < 0 else "right"
                pose_x_value = -7.0 if head_roll < 0 else 7.0

        body_slouch = 0.0
        if body_slouch > 0.58:
            pose_name = "down"
            pose_y_value = max(pose_y_value, 10.5)

        head_pose_pressure = max(abs(head_yaw), abs(head_pitch), abs(head_roll) * 0.85)
        body_lean = float(np.clip(max(0.0, abs(head_roll) - 0.28) / 0.72, 0.0, 1.0))
        if pose_name == "down":
            posture_status = "低头/视线下移"
        elif pose_name in {"left", "right"}:
            posture_status = "头部偏转"
        elif abs(head_roll) > 0.42:
            posture_status = "头部倾斜"
        elif head_pose_pressure < 0.24:
            posture_status = "头部姿态稳定"
        else:
            posture_status = "轻微偏离"

        return {
            "x": pose_x_value,
            "y": pose_y_value,
            "pose": pose_name,
            "body_visible": False,
            "body_slouch": body_slouch,
            "body_lean": body_lean,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "head_roll": head_roll,
            "head_yaw_deg": head_yaw_deg,
            "head_pitch_deg": head_pitch_deg,
            "head_roll_deg": head_roll_deg,
            "head_pose_pressure": head_pose_pressure,
            "head_pose_reliable": head_pose_reliable,
            "head_pose_source": head_pose_source,
            "posture_status": posture_status,
        }

    def _build_score_explanation(
        self,
        features,
        expression_data,
        backend_result,
        base_score,
        final_score,
        smoothed_score,
        multimodal=None,
    ):
        reasons = []
        multimodal = multimodal or self._safe_multimodal_snapshot()
        audio_info = multimodal.get("audio", {})
        text_info = multimodal.get("text", {})

        if features.get("no_of_face", 0) == 0:
            reasons.append("未稳定检测到人脸，分数被压低")
        else:
            reasons.append(
                f"已检测到人脸，置信度 {features.get('face_con', 0):.0f}%，眼睛 {features.get('eyes_detected', 0)} 个"
            )

        pose_map = {0: "正视屏幕", 1: "偏左", 2: "偏右", 3: "低头"}
        pose_text = pose_map.get(features.get("pose_encoded", 0), "未知")
        reasons.append(f"当前姿态: {pose_text}")

        if features.get("phone", 0) == 1:
            phone_conf = features.get("phone_con", 0.0)
            phone_source = features.get("phone_source", "YOLO")
            reasons.append(f"检测到手机，分数会明显下降（{phone_source}，置信度 {phone_conf:.2f}）")
        elif features.get("movement_total_norm", 0.0) > 0.12:
            reasons.append("头部移动较大，系统判为不够稳定")
        elif features.get("movement_total_norm", 0.0) < 0.04:
            reasons.append("头部较稳定，这会加分")

        if expression_data.get("mouth_movement", 0.0) > 18:
            reasons.append("嘴部动作较大，系统可能认为你在讲话或分心")
        if expression_data.get("eyes_closed"):
            reasons.append("连续多帧眼睛不可见，疑似闭眼或视线偏离")
        elif expression_data.get("perclos", 0.0) > 0.28:
            reasons.append(f"PERCLOS 闭眼比例偏高（{expression_data.get('perclos', 0.0) * 100:.0f}%），疑似疲劳")
        if expression_data.get("yawn_detected"):
            reasons.append("嘴巴持续张开，疑似打哈欠或疲劳")
        elif expression_data.get("mouth_open"):
            reasons.append("嘴巴张开度偏高，系统会轻微降低专注判断")

        if features.get("head_pose_pressure", 0.0) > 0.55:
            reasons.append("solvePnP 头部姿态显示偏转较大，可能没有持续看向屏幕")
        if features.get("body_slouch", 0.0) > 0.55:
            reasons.append("身体姿态偏低，疑似低头或趴桌")
        elif features.get("body_lean", 0.0) > 0.6:
            reasons.append("身体倾斜明显，坐姿稳定性下降")

        if backend_result.get("available"):
            reasons.append(
                f"UHMF 后端分数 {backend_result.get('score', 0):.1f}，最终分数按后端主导融合"
            )
        else:
            reasons.append("UHMF 后端暂不可用，当前使用本地兜底评分")

        if multimodal.get("available"):
            transcript = (text_info.get("last_transcript") or "").strip()
            if transcript:
                reasons.append(f"已接入真实语音文本，最近转写: {transcript[:10]}")
            else:
                reasons.append("已接入真实音频模态，语音活动将参与后端融合")

        calibration_status = self.get_calibration_status()
        if calibration_status["ready"]:
            reasons.append("已启用个人专注姿态校准")
        elif calibration_status["active"]:
            reasons.append(calibration_status["message"])

        face_data_source = features.get("face_source", "-")
        metrics = {
            "base_score": round(float(base_score), 1),
            "final_score": round(float(final_score), 1),
            "smoothed_score": round(float(smoothed_score), 1),
            "face_detected": bool(features.get("no_of_face", 0)),
            "pose": pose_text,
            "phone_detected": bool(features.get("phone", 0)),
            "phone_confidence": round(float(features.get("phone_con", 0.0)), 2),
            "phone_source": features.get("phone_source", "-"),
            "movement_norm": round(float(features.get("movement_total_norm", 0.0)), 3),
            "mouth_movement": round(float(expression_data.get("mouth_movement", 0.0)), 1),
            "mouth_open_ratio": round(float(expression_data.get("mouth_open_ratio", 0.0)), 3),
            "mouth_open_strength": round(float(expression_data.get("mouth_open_strength", 0.0)), 3),
            "mouth_baseline": round(float(expression_data.get("mouth_baseline", 0.0)), 3),
            "mar": round(float(expression_data.get("mar", 0.0)), 3),
            "mar_avg": round(float(expression_data.get("mar_avg", 0.0)), 3),
            "mouth_open": bool(expression_data.get("mouth_open", False)),
            "yawn_detected": bool(expression_data.get("yawn_detected", False)),
            "eyes_detected": int(expression_data.get("eyes_detected", 0)),
            "eyes_closed": bool(expression_data.get("eyes_closed", False)),
            "eye_closed_strength": round(float(expression_data.get("eye_closed_strength", 0.0)), 3),
            "eye_open_score": round(float(expression_data.get("eye_open_score", 0.0)), 3),
            "ear": round(float(expression_data.get("ear", 0.0)), 3),
            "ear_avg": round(float(expression_data.get("ear_avg", 0.0)), 3),
            "perclos": round(float(expression_data.get("perclos", 0.0)), 3),
            "fatigue_score": round(float(expression_data.get("fatigue_score", 0.0)), 3),
            "eye_status": expression_data.get("eye_status", "-"),
            "blink_count": int(expression_data.get("blink_count", 0)),
            "vision_source": expression_data.get("face_analysis_source", face_data_source),
            "face_source": face_data_source,
            "body_visible": bool(features.get("body_visible", False)),
            "body_slouch": round(float(features.get("body_slouch", 0.0)), 3),
            "body_lean": round(float(features.get("body_lean", 0.0)), 3),
            "head_yaw": round(float(features.get("head_yaw", 0.0)), 3),
            "head_pitch": round(float(features.get("head_pitch", 0.0)), 3),
            "head_roll": round(float(features.get("head_roll", 0.0)), 3),
            "head_yaw_deg": round(float(features.get("head_yaw_deg", 0.0)), 1),
            "head_pitch_deg": round(float(features.get("head_pitch_deg", 0.0)), 1),
            "head_roll_deg": round(float(features.get("head_roll_deg", 0.0)), 1),
            "head_pose_pressure": round(float(features.get("head_pose_pressure", 0.0)), 3),
            "head_pose_source": features.get("head_pose_source", "-"),
            "head_pose_reliable": bool(features.get("head_pose_reliable", False)),
            "posture_status": features.get("posture_status", "-"),
            "calibration": calibration_status["message"],
            "multimodal_enabled": bool(multimodal.get("available")),
            "multimodal_status": multimodal.get("status", "-"),
            "audio_energy": round(float(audio_info.get("energy", 0.0)), 4),
            "speech_ratio": round(float(audio_info.get("speech_ratio", 0.0)), 3),
            "speaking_now": bool(audio_info.get("speaking_now", False)),
            "last_transcript": (text_info.get("last_transcript") or "")[:24],
        }

        return {"reasons": reasons[:5], "metrics": metrics}

    def _detect_expression(self, frame, face_data, facemark_data=None):
        if not face_data["detected"]:
            self.prev_mouth_roi = None
            self.eye_state_history.clear()
            self.eye_closure_history.clear()
            self.ear_history.clear()
            self.mar_history.clear()
            self.mouth_open_history.clear()
            self.prev_eyes_closed = False
            return {
                "expression_score": 0.2,
                "mouth_movement": 0.0,
                "mouth_open_ratio": 0.0,
                "mouth_open_strength": 0.0,
                "mouth_baseline": 0.0,
                "mouth_open": False,
                "yawn_detected": False,
                "eyes_detected": 0,
                "eyes_closed": False,
                "eye_closed_strength": 0.0,
                "eye_open_score": 0.0,
                "ear": 0.0,
                "ear_avg": 0.0,
                "mar": 0.0,
                "mar_avg": 0.0,
                "perclos": 0.0,
                "fatigue_score": 0.0,
                "eye_status": "未检测",
                "face_analysis_source": "none",
                "blink_count": self.blink_count,
            }

        x = int(face_data["x"])
        y = int(face_data["y"])
        w = int(face_data["w"])
        h = int(face_data["h"])
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(frame_w - 1, x))
        y = max(0, min(frame_h - 1, y))
        w = max(1, min(frame_w - x, w))
        h = max(1, min(frame_h - y, h))

        mouth_y_start = y + int(h * 0.55)
        mouth_roi = frame[mouth_y_start : y + h, x : x + w]
        if mouth_roi.size == 0:
            return {
                "expression_score": 0.5,
                "mouth_movement": 0.0,
                "mouth_open_ratio": 0.0,
                "mouth_open_strength": 0.0,
                "mouth_baseline": float(self.mouth_closed_baseline or 0.0),
                "mouth_open": False,
                "yawn_detected": False,
                "eyes_detected": int(face_data.get("eyes_detected", 0)),
                "eyes_closed": False,
                "eye_closed_strength": 0.0,
                "eye_open_score": min(int(face_data.get("eyes_detected", 0)) / 2.0, 1.0),
                "ear": 0.0,
                "ear_avg": float(np.mean(self.ear_history)) if self.ear_history else 0.0,
                "mar": 0.0,
                "mar_avg": float(np.mean(self.mar_history)) if self.mar_history else 0.0,
                "perclos": float(np.mean(self.eye_closure_history)) if self.eye_closure_history else 0.0,
                "fatigue_score": 0.0,
                "eye_status": "眼睛状态不稳定",
                "face_analysis_source": face_data.get("source", "opencv"),
                "blink_count": self.blink_count,
            }

        mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
        mouth_gray = cv2.resize(mouth_gray, (48, 24))

        movement = 0.0
        if self.prev_mouth_roi is not None and self.prev_mouth_roi.shape == mouth_gray.shape:
            diff = cv2.absdiff(mouth_gray, self.prev_mouth_roi)
            movement = float(diff.mean())

        self.prev_mouth_roi = mouth_gray
        self.expression_history.append(movement)

        facemark_face = (facemark_data or {}).get("face", {})
        has_facemark = bool(facemark_face.get("detected"))
        landmark_face = facemark_face
        if has_facemark:
            face_analysis_source = "OpenCV LBF landmarks"
        else:
            face_analysis_source = "OpenCV"

        has_landmarks = has_facemark

        if has_landmarks:
            mouth_open_ratio = float(landmark_face.get("mouth_open_ratio", 0.0))
        else:
            mouth_open_ratio = self._estimate_mouth_open_ratio(mouth_gray)

        if self.mouth_closed_baseline is None:
            self.mouth_closed_baseline = mouth_open_ratio
        elif movement < 10 and mouth_open_ratio < max(0.18 if has_landmarks else 0.34, self.mouth_closed_baseline + 0.08):
            # Track each student's normal closed-mouth appearance to reduce shadow/skin-tone false positives.
            update_rate = 0.16 if mouth_open_ratio < self.mouth_closed_baseline else 0.05
            self.mouth_closed_baseline = (
                self.mouth_closed_baseline * (1.0 - update_rate) + mouth_open_ratio * update_rate
            )

        mouth_baseline = float(self.mouth_closed_baseline or 0.0)
        mouth_relative_range = 0.11 if has_landmarks else 0.22
        mouth_absolute_floor = 0.08 if has_landmarks else 0.34
        mouth_absolute_range = 0.18 if has_landmarks else 0.28
        relative_open = float(np.clip((mouth_open_ratio - mouth_baseline) / mouth_relative_range, 0.0, 1.0))
        absolute_open = float(np.clip((mouth_open_ratio - mouth_absolute_floor) / mouth_absolute_range, 0.0, 1.0))
        mouth_open_strength = max(relative_open, absolute_open * 0.9)
        self.mouth_open_history.append(mouth_open_strength)
        recent_mouth_open = list(self.mouth_open_history)[-5:]
        avg_mouth_open_strength = (
            float(np.mean(recent_mouth_open)) if recent_mouth_open else mouth_open_strength
        )
        mouth_open = avg_mouth_open_strength > 0.48
        yawn_detected = (
            len(recent_mouth_open) >= 4
            and sum(item > 0.74 for item in recent_mouth_open) >= 3
            and mouth_open_ratio > (0.1 if has_landmarks else 0.38)
        )

        eyes_detected = int(landmark_face.get("eyes_detected", face_data.get("eyes_detected", 0)))
        eye_closed_candidate = bool(landmark_face.get("eye_closed_candidate", False)) if has_landmarks else eyes_detected == 0
        ear = float(landmark_face.get("ear", 0.0)) if has_landmarks else 0.0
        mar = float(mouth_open_ratio)
        if has_landmarks:
            self.ear_history.append(ear)
            self.mar_history.append(mar)
        eyes_for_history = 0 if eye_closed_candidate else max(eyes_detected, 2 if has_landmarks else eyes_detected)
        self.eye_state_history.append(eyes_for_history)
        recent_eye_counts = list(self.eye_state_history)[-5:]
        eye_presence = (
            float(np.mean([min(item / 2.0, 1.0) for item in recent_eye_counts]))
            if recent_eye_counts
            else 0.0
        )
        closed_ratio = (
            sum(1 for item in recent_eye_counts if item == 0) / len(recent_eye_counts)
            if recent_eye_counts
            else 0.0
        )
        eye_signal_reliable = (
            has_landmarks
            or face_data.get("confidence", 0.0) >= 68.0
            and face_data.get("w", 0.0) >= 70.0
            and face_data.get("h", 0.0) >= 70.0
        )
        if eye_signal_reliable:
            self.eye_closure_history.append(1.0 if eye_closed_candidate else 0.0)
        eye_closed_strength = closed_ratio if eye_signal_reliable else 0.0
        eyes_closed = len(recent_eye_counts) >= 5 and eye_closed_strength >= 0.8
        if eyes_closed and not self.prev_eyes_closed:
            self.blink_count += 1
        self.prev_eyes_closed = eyes_closed

        if eyes_closed:
            eye_status = "疑似闭眼"
            eye_open_score = 0.12
        elif eye_presence >= 0.75:
            eye_status = "双眼稳定可见"
            eye_open_score = 1.0
        elif eye_presence >= 0.45:
            eye_status = "眼睛部分可见"
            eye_open_score = 0.68
        else:
            eye_status = "眼睛识别不稳定"
            eye_open_score = 0.52 if eye_signal_reliable else 0.62

        recent_mean = float(np.mean(self.expression_history)) if self.expression_history else movement
        if recent_mean < 6:
            expression_score = 0.82
        elif recent_mean < 12:
            expression_score = 0.65
        elif recent_mean < 20:
            expression_score = 0.48
        else:
            expression_score = 0.3

        if yawn_detected:
            expression_score = min(expression_score, 0.18)
        elif mouth_open:
            expression_score = min(expression_score, 0.34)
        if eyes_closed:
            expression_score = min(expression_score, 0.26)

        perclos = float(np.mean(self.eye_closure_history)) if self.eye_closure_history else 0.0
        ear_avg = float(np.mean(self.ear_history)) if self.ear_history else ear
        mar_avg = float(np.mean(self.mar_history)) if self.mar_history else mar
        fatigue_score = float(
            np.clip(
                perclos * 0.62
                + avg_mouth_open_strength * 0.24
                + (1.0 if yawn_detected else 0.0) * 0.14,
                0.0,
                1.0,
            )
        )

        return {
            "expression_score": expression_score,
            "mouth_movement": movement,
            "mouth_open_ratio": mouth_open_ratio,
            "mouth_open_strength": avg_mouth_open_strength,
            "mouth_baseline": mouth_baseline,
            "mouth_open": mouth_open,
            "yawn_detected": yawn_detected,
            "eyes_detected": eyes_detected,
            "eyes_closed": eyes_closed,
            "eye_closed_strength": eye_closed_strength,
            "eye_open_score": eye_open_score,
            "ear": ear,
            "ear_avg": ear_avg,
            "mar": mar,
            "mar_avg": mar_avg,
            "perclos": perclos,
            "fatigue_score": fatigue_score,
            "eye_status": eye_status,
            "face_analysis_source": face_analysis_source,
            "blink_count": self.blink_count,
        }

    def _estimate_mouth_open_ratio(self, mouth_gray):
        if mouth_gray is None or mouth_gray.size == 0:
            return 0.0

        center = mouth_gray[:, int(mouth_gray.shape[1] * 0.18) : int(mouth_gray.shape[1] * 0.82)]
        if center.size == 0:
            return 0.0

        blurred = cv2.GaussianBlur(center, (3, 3), 0)
        try:
            _threshold, dark_mask = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
        except Exception:
            dark_mask = (blurred < np.percentile(blurred, 35)).astype(np.uint8) * 255

        dark_ratio = float(np.mean(dark_mask > 0))
        contours, _hierarchy = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return float(np.clip(dark_ratio * 0.65, 0.0, 1.0))

        largest = max(contours, key=cv2.contourArea)
        _x, _y, contour_w, contour_h = cv2.boundingRect(largest)
        area_ratio = float(cv2.contourArea(largest) / max(center.shape[0] * center.shape[1], 1))
        vertical_ratio = contour_h / max(center.shape[0], 1)
        horizontal_ratio = contour_w / max(center.shape[1], 1)
        open_ratio = vertical_ratio * 0.48 + area_ratio * 0.34 + dark_ratio * 0.14 + horizontal_ratio * 0.04
        return float(np.clip(open_ratio, 0.0, 1.0))

    def _detect_phone(self, frame):
        if self.frame_index % self.phone_detection_interval != 0:
            return self.cached_phone_data

        if self.yolo_model is None and not self.yolo_checked:
            self._load_yolo()

        if self.yolo_model is None:
            self.cached_phone_data = self._empty_phone_data("disabled")
            return self.cached_phone_data

        try:
            results = self.yolo_model(
                frame,
                verbose=False,
                imgsz=self.phone_detection_imgsz,
                conf=self.phone_detection_conf,
            )
            best_phone = self._extract_best_phone(results, frame.shape, source="yolo_full")

            if best_phone is None:
                frame_h = frame.shape[0]
                crop_y = int(frame_h * 0.35)
                lower_frame = frame[crop_y:frame_h, :]
                if lower_frame.size > 0:
                    crop_results = self.yolo_model(
                        lower_frame,
                        verbose=False,
                        imgsz=self.phone_detection_imgsz,
                        conf=self.phone_detection_crop_conf,
                    )
                    best_phone = self._extract_best_phone(
                        crop_results,
                        frame.shape,
                        y_offset=crop_y,
                        source="yolo_lower_crop",
                    )

            if best_phone:
                self.last_phone_frame = self.frame_index
                self.cached_phone_data = best_phone
                return self.cached_phone_data

            frames_since_phone = self.frame_index - self.last_phone_frame
            if self.cached_phone_data.get("detected") and frames_since_phone <= self.phone_hold_frames:
                held_phone = dict(self.cached_phone_data)
                held_phone["source"] = "yolo_hold"
                held_phone["confidence"] = max(0.2, held_phone.get("confidence", 0.0) * 0.86)
                self.cached_phone_data = held_phone
            else:
                self.cached_phone_data = self._empty_phone_data()
        except Exception as exc:
            print(f"Phone detection fallback: {exc}")
            self.cached_phone_data = self._empty_phone_data("error")
        return self.cached_phone_data

    def _empty_phone_data(self, source="none"):
        return {
            "detected": False,
            "x": 0,
            "y": 0,
            "w": 0,
            "h": 0,
            "confidence": 0.0,
            "source": source,
        }

    def _extract_best_phone(self, results, frame_shape, x_offset=0, y_offset=0, source="yolo"):
        best_phone = None
        best_conf = 0.0
        frame_h, frame_w = frame_shape[:2]
        frame_area = max(float(frame_w * frame_h), 1.0)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls)
                if not self._is_phone_class(cls):
                    continue

                conf = float(box.conf)
                if conf <= best_conf:
                    continue

                x, y, w, h = box.xywh[0].cpu().numpy()
                area_ratio = float((w * h) / frame_area)
                if area_ratio < 0.0008 or area_ratio > 0.55:
                    continue

                best_conf = conf
                best_phone = {
                    "detected": True,
                    "x": int(x + x_offset),
                    "y": int(y + y_offset),
                    "w": int(w),
                    "h": int(h),
                    "confidence": conf,
                    "source": source,
                }

        return best_phone

    def _is_phone_class(self, cls):
        if cls == 67:
            return True

        names = getattr(self.yolo_model, "names", {}) or {}
        name = names.get(cls, "") if isinstance(names, dict) else names[cls] if cls < len(names) else ""
        normalized = str(name).strip().lower()
        return normalized in {"cell phone", "mobile phone", "phone", "smartphone"}

    def _load_yolo(self):
        self.yolo_checked = True
        if not os.path.exists(YOLO_MODEL_PATH):
            print("YOLO model file not found, phone detection disabled.")
            return False

        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            print("YOLO model loaded successfully")
            return True
        except Exception as exc:
            print(f"Error loading YOLO model: {exc}")
            return False

    def _estimate_hand_count(self, face_data):
        if not face_data["detected"]:
            return 0
        if face_data["movement"] > 18:
            return 1
        return 0

    def _extract_features(self, frame, face_data, pose_data, phone_data, hand_count, expression_data):
        pose_mapping = {"forward": 0, "left": 1, "right": 2, "down": 3}
        frame_h, frame_w = frame.shape[:2]

        face_center_x = face_data["x"] + face_data["w"] / 2.0 if face_data["detected"] else 0.0
        face_center_y = face_data["y"] + face_data["h"] / 2.0 if face_data["detected"] else 0.0
        face_area_ratio = (
            (face_data["w"] * face_data["h"]) / float(frame_w * frame_h)
            if face_data["detected"]
            else 0.0
        )

        movement_horizontal_norm = float(np.clip(face_data["horizontal_movement"] / frame_w, 0.0, 1.0))
        movement_vertical_norm = float(np.clip(face_data["vertical_movement"] / frame_h, 0.0, 1.0))
        movement_total_norm = float(
            np.clip(face_data["movement"] / max(frame_w, frame_h), 0.0, 1.0)
        )

        return {
            "no_of_face": 1 if face_data["detected"] else 0,
            "face_x": face_data["x"],
            "face_y": face_data["y"],
            "face_w": face_data["w"],
            "face_h": face_data["h"],
            "face_con": face_data["confidence"],
            "face_source": face_data.get("source", "opencv"),
            "no_of_hand": hand_count,
            "pose_x": pose_data["x"],
            "pose_y": pose_data["y"],
            "body_visible": bool(pose_data.get("body_visible", False)),
            "body_slouch": float(pose_data.get("body_slouch", 0.0)),
            "body_lean": float(pose_data.get("body_lean", 0.0)),
            "head_yaw": float(pose_data.get("head_yaw", 0.0)),
            "head_pitch": float(pose_data.get("head_pitch", 0.0)),
            "head_roll": float(pose_data.get("head_roll", 0.0)),
            "head_yaw_deg": float(pose_data.get("head_yaw_deg", 0.0)),
            "head_pitch_deg": float(pose_data.get("head_pitch_deg", 0.0)),
            "head_roll_deg": float(pose_data.get("head_roll_deg", 0.0)),
            "head_pose_pressure": float(pose_data.get("head_pose_pressure", 0.0)),
            "head_pose_reliable": bool(pose_data.get("head_pose_reliable", False)),
            "head_pose_source": pose_data.get("head_pose_source", "none"),
            "posture_status": pose_data.get("posture_status", "未检测"),
            "phone": 1 if phone_data.get("detected") else 0,
            "phone_con": float(phone_data.get("confidence", 0.0)),
            "phone_source": phone_data.get("source", "none"),
            "pose_encoded": pose_mapping.get(pose_data["pose"], 0),
            "expression_score": expression_data["expression_score"],
            "mouth_movement": expression_data["mouth_movement"],
            "mouth_open_ratio": float(expression_data.get("mouth_open_ratio", 0.0)),
            "mouth_open_strength": float(expression_data.get("mouth_open_strength", 0.0)),
            "mouth_baseline": float(expression_data.get("mouth_baseline", 0.0)),
            "mouth_open": 1 if expression_data.get("mouth_open") else 0,
            "yawn_detected": 1 if expression_data.get("yawn_detected") else 0,
            "eyes_detected": int(expression_data.get("eyes_detected", 0)),
            "eyes_closed": 1 if expression_data.get("eyes_closed") else 0,
            "eye_closed_strength": float(expression_data.get("eye_closed_strength", 0.0)),
            "eye_open_score": float(expression_data.get("eye_open_score", 0.0)),
            "ear": float(expression_data.get("ear", 0.0)),
            "ear_avg": float(expression_data.get("ear_avg", 0.0)),
            "mar": float(expression_data.get("mar", 0.0)),
            "mar_avg": float(expression_data.get("mar_avg", 0.0)),
            "perclos": float(expression_data.get("perclos", 0.0)),
            "fatigue_score": float(expression_data.get("fatigue_score", 0.0)),
            "face_center_x_norm": float(face_center_x / frame_w) if frame_w else 0.0,
            "face_center_y_norm": float(face_center_y / frame_h) if frame_h else 0.0,
            "face_area_ratio": float(face_area_ratio),
            "pose_x_norm": float(pose_data["x"]),
            "pose_y_norm": float(pose_data["y"]),
            "movement_horizontal_norm": movement_horizontal_norm,
            "movement_vertical_norm": movement_vertical_norm,
            "movement_total_norm": movement_total_norm,
            "mouth_movement_norm": float(np.clip(expression_data["mouth_movement"] / 25.0, 0.0, 1.0)),
            "mouth_open_norm": float(np.clip(expression_data.get("mouth_open_strength", 0.0), 0.0, 1.0)),
            "eye_closed_norm": float(np.clip(expression_data.get("eye_closed_strength", 0.0), 0.0, 1.0)),
            "perclos_norm": float(np.clip(expression_data.get("perclos", 0.0), 0.0, 1.0)),
            "fatigue_norm": float(np.clip(expression_data.get("fatigue_score", 0.0), 0.0, 1.0)),
        }

    def _calculate_attention_score(self, features, expression_data):
        if features["no_of_face"] == 0:
            if self.consecutive_face_misses <= 2 and self.score_history:
                recent_score = float(np.mean(list(self.score_history)[-3:]))
                return max(35.0, recent_score - 12.0)
            return 10.0

        focused_prob = 0.5
        if self.model is not None:
            feature_list = [
                features["no_of_face"],
                features["face_x"],
                features["face_y"],
                features["face_w"],
                features["face_h"],
                features["face_con"],
                features["no_of_hand"],
                features["pose_x"],
                features["pose_y"],
                features["phone"],
                features["phone_con"],
                features["pose_encoded"],
            ]

            try:
                feature_frame = pd.DataFrame([feature_list], columns=self.rf_feature_columns)
                probabilities = self.model.predict_proba(feature_frame)[0]
                class_probabilities = {
                    int(label): float(prob)
                    for label, prob in zip(self.model.classes_, probabilities)
                }
                # The provided dataset maps label 0 to focused-looking samples.
                focused_prob = class_probabilities.get(0, 0.5)
            except Exception as exc:
                print(f"Attention score fallback: {exc}")

        score = 30.0 + focused_prob * 40.0

        if features["face_con"] > 88:
            score += 8
        elif features["face_con"] > 75:
            score += 4

        if features["pose_encoded"] == 0:
            score += 14
        elif features["pose_encoded"] == 3:
            score -= 18
        else:
            score -= 10

        head_pose_pressure = features.get("head_pose_pressure", 0.0)
        if head_pose_pressure > 0.62:
            score -= 12.0 + head_pose_pressure * 8.0
        elif head_pose_pressure > 0.42:
            score -= head_pose_pressure * 9.0

        movement_penalty = min(18.0, features["movement_total_norm"] * 30.0)
        score -= movement_penalty

        if features["phone"] == 1:
            score -= 28

        mouth_movement = expression_data["mouth_movement"]
        if mouth_movement < 5:
            score += 6
        elif mouth_movement > 18:
            score -= 12

        if features.get("eyes_closed", 0) == 1:
            score -= 24 + features.get("eye_closed_strength", 0.0) * 6.0
        elif features.get("eyes_detected", 0) >= 2:
            score += 4
        perclos = features.get("perclos", 0.0)
        if perclos > 0.42:
            score -= 18.0
        elif perclos > 0.25:
            score -= 7.0 + perclos * 18.0

        mouth_open_strength = features.get("mouth_open_strength", 0.0)
        if features.get("yawn_detected", 0) == 1:
            score -= 26
        elif features.get("mouth_open", 0) == 1:
            score -= 12 + mouth_open_strength * 8.0
        elif mouth_open_strength > 0.28:
            score -= mouth_open_strength * 7.0
        fatigue_score = features.get("fatigue_score", 0.0)
        if fatigue_score > 0.55:
            score -= fatigue_score * 10.0

        body_slouch = features.get("body_slouch", 0.0)
        body_lean = features.get("body_lean", 0.0)
        if body_slouch > 0.55:
            score -= 10.0 + body_slouch * 12.0
        elif body_lean > 0.6:
            score -= 6.0 + body_lean * 8.0

        score = score * 0.82 + expression_data["expression_score"] * 22.0
        return float(np.clip(score, 10, 100))

    def _calculate_movement_info(self):
        if len(self.face_movement_history) < 3:
            return {"horizontal": 0.0, "vertical": 0.0, "total": 0.0}

        recent_movements = list(self.face_movement_history)[-5:]
        avg_horizontal = float(np.mean([item["horizontal"] for item in recent_movements]))
        avg_vertical = float(np.mean([item["vertical"] for item in recent_movements]))
        avg_total = float(np.mean([item["total"] for item in recent_movements]))

        return {"horizontal": avg_horizontal, "vertical": avg_vertical, "total": avg_total}

    def _smooth_score(self, current_score, face_detected, features=None):
        features = features or {}
        self.face_detection_history.append(1 if face_detected else 0)

        face_detection_rate = (
            sum(self.face_detection_history) / len(self.face_detection_history)
            if self.face_detection_history
            else 0.0
        )
        if face_detection_rate < 0.45:
            if self.consecutive_face_misses >= 3:
                current_score = min(current_score, 22.0)

        self.score_history.append(current_score)
        if self.last_smoothed_score is None:
            self.last_smoothed_score = current_score
            return float(np.clip(current_score, 10, 100))

        previous_score = self.last_smoothed_score
        score_delta = current_score - previous_score
        phone_detected = features.get("phone", 0) == 1
        pose_encoded = features.get("pose_encoded", 0)

        if phone_detected or not face_detected:
            alpha = 0.96 if score_delta < 0 else 0.68
            max_drop = 52.0
            max_rise = 24.0
        elif pose_encoded != 0:
            alpha = 0.86 if score_delta < 0 else 0.66
            max_drop = 36.0
            max_rise = 22.0
        else:
            alpha = 0.74 if score_delta < 0 else 0.66
            max_drop = 28.0
            max_rise = 24.0

        if abs(score_delta) > 18:
            alpha = max(alpha, 0.82)

        smoothed_score = previous_score + score_delta * alpha
        if score_delta < 0:
            smoothed_score = max(smoothed_score, previous_score - max_drop)
        else:
            smoothed_score = min(smoothed_score, previous_score + max_rise)

        self.last_smoothed_score = float(np.clip(smoothed_score, 10, 100))
        return self.last_smoothed_score

    def get_status(self, score):
        if score >= ATTENTION_THRESHOLDS["focused"]:
            return "focused", "专注"
        if score >= ATTENTION_THRESHOLDS["moderate"]:
            return "moderate", "一般"
        return "distracted", "不专注"
