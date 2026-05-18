import json
import os
import threading
import time
from collections import deque

import numpy as np

from config.config import VOSK_MODEL_PATH


class RealMultimodalStream:
    """Collect real audio and text signals for runtime multimodal fusion."""

    STUDY_KEYWORDS = (
        "学习",
        "课程",
        "老师",
        "同学",
        "题目",
        "解题",
        "练习",
        "作业",
        "课堂",
        "知识",
        "讲解",
        "笔记",
        "阅读",
        "单词",
        "计算",
    )
    QUESTION_KEYWORDS = ("为什么", "怎么", "多少", "是否", "吗", "呢", "?", "？")

    def __init__(self, sample_rate=16000, frames_per_buffer=2048):
        self.sample_rate = sample_rate
        self.frames_per_buffer = frames_per_buffer
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._status = "未启动"
        self._error = ""
        self._noise_floor = 0.010
        self._last_activity_at = 0.0
        self._last_text_at = 0.0
        self._latest_partial = ""
        self._energy_history = deque(maxlen=90)
        self._zcr_history = deque(maxlen=90)
        self._speech_history = deque(maxlen=90)
        self._transcripts = deque(maxlen=6)

    @property
    def is_running(self):
        return bool(self._running)

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._running = True
            self._status = "正在连接真实多模态输入"
            self._error = ""
            self._latest_partial = ""
            self._energy_history.clear()
            self._zcr_history.clear()
            self._speech_history.clear()
            self._transcripts.clear()
            self._noise_floor = 0.010
            self._last_activity_at = 0.0
            self._last_text_at = 0.0
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        thread = None
        with self._lock:
            self._running = False
            thread = self._thread
            self._thread = None
            if not self._error:
                self._status = "已停止"
        if thread and thread.is_alive():
            thread.join(timeout=1.8)

    def snapshot(self):
        with self._lock:
            energy_values = list(self._energy_history)
            zcr_values = list(self._zcr_history)
            speech_values = list(self._speech_history)
            transcripts = list(self._transcripts)
            latest_text = transcripts[-1]["text"] if transcripts else self._latest_partial
            now = time.time()
            speaking_now = bool(speech_values[-1]) if speech_values else False

            study_hits = sum(word in latest_text for word in self.STUDY_KEYWORDS)
            question_hits = sum(word in latest_text for word in self.QUESTION_KEYWORDS)
            transcript_chars = len(latest_text)
            transcript_count = len(transcripts)
            keyword_ratio = (
                float(study_hits) / max(1.0, float(transcript_chars))
                if transcript_chars
                else 0.0
            )
            question_ratio = (
                float(question_hits) / max(1.0, float(transcript_chars))
                if transcript_chars
                else 0.0
            )

            text_available = transcript_count > 0 or bool(self._latest_partial)
            audio_available = bool(energy_values)
            return {
                "available": audio_available,
                "running": self._running,
                "status": self._status,
                "error": self._error,
                "audio": {
                    "available": audio_available,
                    "energy": float(energy_values[-1]) if energy_values else 0.0,
                    "energy_mean": float(np.mean(energy_values)) if energy_values else 0.0,
                    "energy_std": float(np.std(energy_values)) if energy_values else 0.0,
                    "zcr_mean": float(np.mean(zcr_values)) if zcr_values else 0.0,
                    "speech_ratio": float(np.mean(speech_values)) if speech_values else 0.0,
                    "speaking_now": speaking_now,
                    "last_activity_age": float(now - self._last_activity_at) if self._last_activity_at else 999.0,
                    "sample_count": len(energy_values),
                    "energy_history": energy_values[-12:],
                },
                "text": {
                    "available": text_available,
                    "last_transcript": latest_text,
                    "transcript_count": transcript_count,
                    "study_keyword_hits": study_hits,
                    "study_keyword_ratio": keyword_ratio,
                    "question_ratio": question_ratio,
                    "last_text_age": float(now - self._last_text_at) if self._last_text_at else 999.0,
                },
            }

    def _run(self):
        py_audio = None
        audio_stream = None
        vosk_recognizer = None
        recognizer_model = None
        try:
            import pyaudio
        except Exception as exc:
            self._set_error(f"真实多模态未启用：缺少音频依赖（{exc}）")
            return

        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel

            if VOSK_MODEL_PATH and os.path.exists(VOSK_MODEL_PATH):
                SetLogLevel(-1)
                recognizer_model = Model(VOSK_MODEL_PATH)
                vosk_recognizer = KaldiRecognizer(recognizer_model, self.sample_rate)
        except Exception:
            vosk_recognizer = None

        try:
            py_audio = pyaudio.PyAudio()
            audio_stream = py_audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frames_per_buffer,
            )
            self._set_status(
                "真实多模态已启用（音频+文本）"
                if vosk_recognizer is not None
                else "真实多模态已启用（音频）"
            )
        except Exception as exc:
            self._set_error(f"真实多模态未启用：无法打开麦克风（{exc}）")
            if audio_stream is not None:
                try:
                    audio_stream.close()
                except Exception:
                    pass
            if py_audio is not None:
                try:
                    py_audio.terminate()
                except Exception:
                    pass
            return

        try:
            while self._running:
                try:
                    payload = audio_stream.read(self.frames_per_buffer, exception_on_overflow=False)
                except Exception as exc:
                    self._set_error(f"真实多模态采集异常（{exc}）")
                    break

                samples = np.frombuffer(payload, dtype=np.int16).astype(np.float32)
                if samples.size == 0:
                    continue

                normalized = samples / 32768.0
                rms = float(np.sqrt(np.mean(np.square(normalized))) + 1e-8)
                sign_changes = np.count_nonzero(np.diff(np.signbit(normalized)))
                zcr = float(sign_changes / max(1, normalized.size - 1))

                if rms < self._noise_floor * 1.4:
                    self._noise_floor = self._noise_floor * 0.94 + rms * 0.06

                speech_gate = max(0.015, self._noise_floor * 2.2)
                speaking = rms >= speech_gate

                with self._lock:
                    self._energy_history.append(rms)
                    self._zcr_history.append(zcr)
                    self._speech_history.append(1.0 if speaking else 0.0)
                    if speaking:
                        self._last_activity_at = time.time()

                if vosk_recognizer is None:
                    continue

                text = ""
                partial = ""
                try:
                    accepted = vosk_recognizer.AcceptWaveform(payload)
                    if accepted:
                        result = json.loads(vosk_recognizer.Result() or "{}")
                        text = (result.get("text") or "").strip()
                    else:
                        partial_result = json.loads(vosk_recognizer.PartialResult() or "{}")
                        partial = (partial_result.get("partial") or "").strip()
                except Exception:
                    text = ""
                    partial = ""

                if partial:
                    with self._lock:
                        self._latest_partial = partial

                if text:
                    with self._lock:
                        self._latest_partial = ""
                        self._last_text_at = time.time()
                        self._transcripts.append({"text": text, "timestamp": self._last_text_at})
        finally:
            if audio_stream is not None:
                try:
                    audio_stream.stop_stream()
                except Exception:
                    pass
                try:
                    audio_stream.close()
                except Exception:
                    pass
            if py_audio is not None:
                try:
                    py_audio.terminate()
                except Exception:
                    pass

    def _set_status(self, message):
        with self._lock:
            self._status = message
            self._error = ""

    def _set_error(self, message):
        with self._lock:
            self._status = message
            self._error = message
            self._running = False
