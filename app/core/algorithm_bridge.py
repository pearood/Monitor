import importlib.util
import os
import sys
import threading

import numpy as np


class AlgorithmFusionBridge:
    """
    Bridge the UHMF backend model into the desktop app.

    The original research project expects 512/256/768 dimensional multimodal
    features. For the desktop runtime we derive stable pseudo-modal embeddings
    from live visual/temporal signals, then let UHMF provide a lightweight
    auxiliary engagement estimate.
    """

    CLASS_LABELS = ["HD", "DE", "EG", "HE"]
    CLASS_SCORES = np.array([18.0, 42.0, 74.0, 92.0], dtype=np.float32)
    SCENE_LABELS = ["lecture", "discussion", "practice"]

    def __init__(self):
        self._lock = threading.Lock()
        self._loading_started = False
        self._ready = False
        self._error = ""
        self._model = None
        self._torch = None

    @property
    def ready(self):
        return self._ready

    @property
    def error(self):
        return self._error

    def warmup_async(self):
        with self._lock:
            if self._loading_started:
                return
            self._loading_started = True

        worker = threading.Thread(target=self._load_model, daemon=True)
        worker.start()

    def get_status(self):
        if self._ready:
            return {"state": "ready", "message": "算法增强已就绪"}
        if self._error:
            return {"state": "fallback", "message": f"算法增强已降级: {self._error}"}
        if self._loading_started:
            return {"state": "loading", "message": "算法增强正在预热"}
        return {"state": "idle", "message": "算法增强未启动"}

    def analyze(self, features, context):
        if not self._loading_started:
            self.warmup_async()

        multimodal = context.get("multimodal") or {}

        if not self._ready or self._model is None or self._torch is None:
            return {
                "available": False,
                "status": self.get_status(),
                "score": None,
                "label": None,
                "scene": None,
                "confidence": 0.0,
                "fusion_weights": {},
                "scene_probs": {},
                "real_modalities": self._real_modality_flags(multimodal),
            }

        torch = self._torch
        video_vec = self._build_video_features(features, context)
        audio_vec = self._build_audio_features(features, context)
        text_vec = self._build_text_features(features, context)

        with torch.no_grad():
            logits, weights, scene_probs, _, _ = self._model(
                torch.tensor(video_vec[None, :], dtype=torch.float32),
                torch.tensor(audio_vec[None, :], dtype=torch.float32),
                torch.tensor(text_vec[None, :], dtype=torch.float32),
                return_weights=True,
            )

            class_probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            weight_values = weights.cpu().numpy()[0]
            scene_values = scene_probs.cpu().numpy()[0]

        engagement_score = float(np.dot(class_probs, self.CLASS_SCORES))
        label_index = int(np.argmax(class_probs))
        scene_index = int(np.argmax(scene_values))

        return {
            "available": True,
            "status": self.get_status(),
            "model_name": "UHMF",
            "model_source": "archive/algorithm-research/results/exp3_uhmf/main/best_model.pt",
            "score": engagement_score,
            "label": self.CLASS_LABELS[label_index],
            "scene": self.SCENE_LABELS[scene_index],
            "confidence": float(np.max(class_probs)),
            "fusion_weights": {
                "video": float(weight_values[0]),
                "audio": float(weight_values[1]),
                "text": float(weight_values[2]),
            },
            "scene_probs": {
                name: float(scene_values[idx]) for idx, name in enumerate(self.SCENE_LABELS)
            },
            "real_modalities": self._real_modality_flags(multimodal),
        }

    def _load_model(self):
        try:
            self._ensure_typing_extensions()
            import torch

            model_file = self._resolve_resource(
                ("archive", "algorithm-research", "src", "models", "uhmf", "model.py"),
                ("multimodal_edu", "src", "models", "uhmf", "model.py"),
                ("algorithms", "uhmf", "model.py"),
            )
            checkpoint_file = self._resolve_resource(
                (
                    "archive",
                    "algorithm-research",
                    "results",
                    "exp3_uhmf",
                    "main",
                    "best_model.pt",
                ),
                ("multimodal_edu", "results", "exp3_uhmf", "main", "best_model.pt"),
                ("algorithms", "uhmf", "best_model.pt"),
            )

            if not model_file or not checkpoint_file:
                raise FileNotFoundError("未找到 UHMF 模型文件")

            spec = importlib.util.spec_from_file_location("focus_uhmf_model", model_file)
            if spec is None or spec.loader is None:
                raise RuntimeError("UHMF 模型定义加载失败")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            model = module.UHMF(
                video_dim=512,
                audio_dim=256,
                text_dim=768,
                hidden_dim=128,
                n_scenes=3,
                n_classes=4,
            )

            state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()

            self._torch = torch
            self._model = model
            self._ready = True
            self._error = ""
        except Exception as exc:
            self._ready = False
            self._error = str(exc)

    def _ensure_typing_extensions(self):
        existing_module = sys.modules.get("typing_extensions")
        if existing_module is not None and hasattr(existing_module, "deprecated"):
            return

        candidate = self._resolve_resource(
            (
                "archive",
                "algorithm-research",
                "venv",
                "lib",
                "python3.14",
                "site-packages",
                "typing_extensions.py",
            ),
            ("multimodal_edu", "venv", "lib", "python3.14", "site-packages", "typing_extensions.py"),
            ("vendor", "typing_extensions.py"),
        )
        if candidate:
            spec = importlib.util.spec_from_file_location("typing_extensions", candidate)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules["typing_extensions"] = module
                return

        if existing_module is not None and not hasattr(existing_module, "deprecated"):
            def _noop_deprecated(*_args, **_kwargs):
                def decorator(obj):
                    return obj
                return decorator

            existing_module.deprecated = _noop_deprecated

    def _resolve_resource(self, *relative_options):
        base_candidates = []
        current_file = os.path.abspath(__file__)
        app_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        workspace_root = os.path.dirname(app_root)
        base_candidates.extend(
            [
                app_root,
                workspace_root,
                getattr(sys, "_MEIPASS", ""),
                os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else "",
            ]
        )

        for base in base_candidates:
            if not base:
                continue
            for parts in relative_options:
                candidate = os.path.join(base, *parts)
                if os.path.exists(candidate):
                    return candidate
        return None

    def _expand_signal(self, values, target_dim):
        base = np.asarray(values, dtype=np.float32)
        if base.size == 0:
            return np.zeros(target_dim, dtype=np.float32)

        stats = np.array(
            [
                base.mean(),
                base.std(),
                base.min(),
                base.max(),
                np.median(base),
            ],
            dtype=np.float32,
        )
        expanded = np.concatenate([base, stats, np.sin(base), np.cos(base), np.square(base)])
        repeats = int(np.ceil(target_dim / expanded.size))
        tiled = np.tile(expanded, repeats)[:target_dim]
        return np.clip(tiled, -3.0, 3.0).astype(np.float32)

    def _safe_series_stats(self, series):
        if not series:
            return [0.0, 0.0, 0.0, 0.0]

        arr = np.asarray(series, dtype=np.float32)
        return [float(arr[-1]), float(arr.mean()), float(arr.std()), float(arr.max())]

    def _real_modality_flags(self, multimodal):
        audio = multimodal.get("audio", {}) if isinstance(multimodal, dict) else {}
        text = multimodal.get("text", {}) if isinstance(multimodal, dict) else {}
        return {
            "audio": bool(audio.get("available") and audio.get("sample_count", 0)),
            "text": bool(text.get("available") and text.get("last_transcript")),
        }

    def _hash_text_features(self, text, bins=128):
        vector = np.zeros(bins, dtype=np.float32)
        if not text:
            return vector

        compact = "".join(str(text).split())[:240]
        for index, char in enumerate(compact):
            bucket = (ord(char) + index * 31) % bins
            vector[bucket] += 1.0

        total = float(vector.sum())
        if total > 0:
            vector /= total
        return vector

    def _build_video_features(self, features, context):
        values = [
            features.get("face_center_x_norm", 0.0),
            features.get("face_center_y_norm", 0.0),
            features.get("face_area_ratio", 0.0),
            features.get("face_con", 0.0) / 100.0,
            features.get("pose_x_norm", 0.0),
            features.get("pose_y_norm", 0.0),
            features.get("movement_horizontal_norm", 0.0),
            features.get("movement_vertical_norm", 0.0),
            features.get("movement_total_norm", 0.0),
            float(features.get("phone", 0.0)),
            features.get("phone_con", 0.0),
            features.get("expression_score", 0.0),
            features.get("mouth_movement_norm", 0.0),
            features.get("mouth_open_norm", 0.0),
            features.get("eye_open_score", 0.0),
            features.get("eye_closed_norm", 0.0),
            features.get("perclos_norm", 0.0),
            features.get("fatigue_norm", 0.0),
            features.get("ear_avg", 0.0),
            features.get("mar_avg", 0.0),
            features.get("head_yaw", 0.0),
            features.get("head_pitch", 0.0),
            features.get("head_roll", 0.0),
            features.get("head_pose_pressure", 0.0),
            float(features.get("yawn_detected", 0.0)),
            1.0 if features.get("body_visible", False) else 0.0,
            features.get("body_slouch", 0.0),
            features.get("body_lean", 0.0),
            context.get("base_score", 0.0) / 100.0,
            context.get("face_detection_rate", 0.0),
        ]
        values.extend(self._safe_series_stats(context.get("recent_movements", [])))
        return self._expand_signal(values, 512)

    def _build_audio_features(self, features, context):
        multimodal = context.get("multimodal") or {}
        audio = multimodal.get("audio", {})
        has_real_audio = bool(audio.get("available") and audio.get("sample_count", 0))

        values = [
            1.0 if has_real_audio else 0.0,
            float(audio.get("energy", 0.0)),
            float(audio.get("energy_mean", 0.0)),
            float(audio.get("energy_std", 0.0)),
            float(audio.get("zcr_mean", 0.0)),
            float(audio.get("speech_ratio", 0.0)),
            1.0 if audio.get("speaking_now") else 0.0,
            float(np.clip(audio.get("last_activity_age", 999.0) / 10.0, 0.0, 1.0)),
            features.get("mouth_movement_norm", 0.0),
            features.get("mouth_open_norm", 0.0),
            features.get("expression_score", 0.0),
            features.get("movement_total_norm", 0.0),
            context.get("score_delta", 0.0),
            context.get("stability", 0.0),
            context.get("focused_ratio", 0.0),
            context.get("distraction_ratio", 0.0),
        ]
        if has_real_audio:
            values.extend(self._safe_series_stats(audio.get("energy_history", [])))
        else:
            values.extend(self._safe_series_stats(context.get("score_history", [])))
        return self._expand_signal(values, 256)

    def _build_text_features(self, features, context):
        weights = context.get("status_weights", {})
        multimodal = context.get("multimodal") or {}
        text_info = multimodal.get("text", {})
        transcript = (text_info.get("last_transcript") or "").strip()
        hashed = self._hash_text_features(transcript, bins=128)
        values = [
            1.0 if transcript else 0.0,
            float(len(transcript) / 36.0),
            float(text_info.get("study_keyword_hits", 0.0) / 4.0),
            float(text_info.get("study_keyword_ratio", 0.0) * 10.0),
            float(text_info.get("question_ratio", 0.0) * 10.0),
            float(np.clip(text_info.get("transcript_count", 0.0) / 6.0, 0.0, 1.0)),
            float(np.clip(text_info.get("last_text_age", 999.0) / 20.0, 0.0, 1.0)),
            context.get("session_minutes", 0.0),
            context.get("base_score", 0.0) / 100.0,
            context.get("avg_score", 0.0) / 100.0,
            context.get("focused_ratio", 0.0),
            context.get("distraction_ratio", 0.0),
            context.get("score_delta", 0.0),
            context.get("stability", 0.0),
            float(weights.get("focused", 0.0)),
            float(weights.get("moderate", 0.0)),
            float(weights.get("distracted", 0.0)),
            features.get("face_con", 0.0) / 100.0,
            float(features.get("phone", 0.0)),
        ]
        values.extend(self._safe_series_stats(context.get("score_history", [])))
        values.extend(hashed.tolist())
        return self._expand_signal(values, 768)
