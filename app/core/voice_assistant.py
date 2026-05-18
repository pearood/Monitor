import base64
import json
import os
import platform
import re
import audioop
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

from PyQt5.QtCore import QThread, pyqtSignal
from config.config import (
    DOUBAO_ASR_API_KEY,
    DOUBAO_ASR_RESOURCE_ID,
    DOUBAO_SPEECH_APPID,
    DOUBAO_SPEECH_TOKEN,
    DOUBAO_TTS_ACCESS_KEY,
    DOUBAO_TTS_API_KEY,
    DOUBAO_TTS_APP_KEY,
    DOUBAO_TTS_CLUSTER,
    DOUBAO_TTS_ENCODING,
    DOUBAO_TTS_RATE,
    DOUBAO_TTS_RESOURCE_ID,
    DOUBAO_TTS_VERSION,
    DOUBAO_TTS_VOICE_TYPE,
    VOSK_MODEL_PATH,
)


class VoiceSpeaker:
    DOUBAO_TTS_V1_URL = "https://openspeech.bytedance.com/api/v1/tts"
    DOUBAO_TTS_V2_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"

    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._engine = None
        self._generation = 0
        self._temp_audio_path = None
        self._voice_type = DOUBAO_TTS_VOICE_TYPE

    def set_voice_type(self, voice_type):
        voice_type = (voice_type or "").strip()
        if not voice_type:
            return
        with self._lock:
            self._voice_type = voice_type

    def voice_type(self):
        with self._lock:
            return self._voice_type or DOUBAO_TTS_VOICE_TYPE

    def speak(self, text):
        if not text:
            return
        with self._lock:
            self._generation += 1
            generation = self._generation
        self.stop(advance_generation=False)
        worker = threading.Thread(target=self._speak_blocking, args=(text, generation), daemon=True)
        worker.start()

    def stop(self, advance_generation=True):
        with self._lock:
            if advance_generation:
                self._generation += 1
            process = self._process
            engine = self._engine
            temp_audio_path = self._temp_audio_path
            self._process = None
            self._engine = None
            self._temp_audio_path = None

        try:
            if process and process.poll() is None:
                process.terminate()
        except Exception:
            pass

        try:
            if engine:
                engine.stop()
        except Exception:
            pass

        self._cleanup_temp_audio(temp_audio_path)

    def is_speaking(self):
        with self._lock:
            process = self._process
            engine = self._engine
        try:
            if process and process.poll() is None:
                return True
        except Exception:
            pass
        return engine is not None

    def _speak_blocking(self, text, generation):
        try:
            if platform.system() == "Darwin" and self._doubao_tts_ready():
                if self._speak_with_doubao(text, generation):
                    return

            if platform.system() == "Darwin":
                with self._lock:
                    if generation != self._generation:
                        return
                    process = subprocess.Popen(["say", "-v", "Ting-Ting", text])
                    self._process = process
                try:
                    process.wait()
                finally:
                    with self._lock:
                        if self._process is process:
                            self._process = None
                return

            import pyttsx3

            engine = pyttsx3.init()
            with self._lock:
                if generation != self._generation:
                    return
                self._engine = engine
            try:
                engine.say(text)
                engine.runAndWait()
            finally:
                with self._lock:
                    if self._engine is engine:
                        self._engine = None
        except Exception:
            print(f"Voice reply: {text}")

    def _doubao_tts_ready(self):
        if self._use_doubao_tts_v2():
            return bool(
                DOUBAO_SPEECH_APPID
                and self.voice_type()
                and (DOUBAO_TTS_API_KEY or DOUBAO_TTS_ACCESS_KEY)
            )
        return bool(DOUBAO_SPEECH_APPID and DOUBAO_SPEECH_TOKEN and self.voice_type())

    def _use_doubao_tts_v2(self):
        return str(DOUBAO_TTS_VERSION).strip().lower() in {"2", "2.0", "v2", "v3"}

    def _speak_with_doubao(self, text, generation):
        try:
            audio_bytes = self._request_doubao_tts_audio(text)
            if not audio_bytes:
                return False

            audio_path = self._write_temp_audio(audio_bytes)
            with self._lock:
                if generation != self._generation:
                    self._cleanup_temp_audio(audio_path)
                    return True
                process = subprocess.Popen(["afplay", audio_path])
                self._process = process
                self._temp_audio_path = audio_path
            try:
                process.wait()
            finally:
                with self._lock:
                    if self._process is process:
                        self._process = None
                    if self._temp_audio_path == audio_path:
                        self._temp_audio_path = None
                self._cleanup_temp_audio(audio_path)
            return True
        except Exception:
            return False

    def _request_doubao_tts_audio(self, text):
        if self._use_doubao_tts_v2():
            return self._request_doubao_tts_v2_audio(text)
        return self._request_doubao_tts_v1_audio(text)

    def _request_doubao_tts_v1_audio(self, text):
        voice_type = self.voice_type()
        payload = {
            "app": {
                "appid": DOUBAO_SPEECH_APPID,
                "token": DOUBAO_SPEECH_TOKEN,
                "cluster": DOUBAO_TTS_CLUSTER,
            },
            "user": {"uid": "xiaorui-study"},
            "audio": {
                "voice_type": voice_type,
                "encoding": DOUBAO_TTS_ENCODING,
                "rate": DOUBAO_TTS_RATE,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "query",
            },
        }
        request = urllib.request.Request(
            self.DOUBAO_TTS_V1_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer;{DOUBAO_SPEECH_TOKEN}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=18) as response:
            raw = response.read().decode("utf-8")
        result = json.loads(raw)
        audio_b64 = result.get("data") or ""
        if not audio_b64:
            raise RuntimeError(result.get("message") or result.get("Message") or "Doubao TTS returned no audio")
        if audio_b64.startswith("data:"):
            audio_b64 = audio_b64.split(",", 1)[1]
        return base64.b64decode(audio_b64)

    def _request_doubao_tts_v2_audio(self, text):
        voice_type = self.voice_type()
        payload = {
            "user": {"uid": "xiaorui-study"},
            "req_params": {
                "text": text,
                "speaker": voice_type,
                "audio_params": {
                    "format": (DOUBAO_TTS_ENCODING or "mp3").lower(),
                    "sample_rate": DOUBAO_TTS_RATE,
                },
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Connection": "keep-alive",
            "X-Api-App-Id": DOUBAO_SPEECH_APPID,
            "X-Api-Resource-Id": DOUBAO_TTS_RESOURCE_ID or "seed-tts-2.0",
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        if DOUBAO_TTS_ACCESS_KEY:
            headers["X-Api-Access-Key"] = DOUBAO_TTS_ACCESS_KEY
            if DOUBAO_TTS_APP_KEY:
                headers["X-Api-App-Key"] = DOUBAO_TTS_APP_KEY
        elif DOUBAO_TTS_API_KEY:
            headers["X-Api-Key"] = DOUBAO_TTS_API_KEY

        request = urllib.request.Request(
            self.DOUBAO_TTS_V2_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        audio = bytearray()
        raw_lines = []
        with urllib.request.urlopen(request, timeout=45) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                raw_lines.append(line)
                if line.startswith("data:"):
                    line = line.split("data:", 1)[1].strip()
                if line == "[DONE]":
                    break
                self._append_doubao_tts_v2_audio_chunk(line, audio)

        if audio:
            return bytes(audio)

        # Some gateways may buffer the whole JSON body rather than line-delimit chunks.
        joined = "".join(raw_lines).strip()
        if joined:
            if joined.startswith("data:"):
                joined = joined.split("data:", 1)[1].strip()
            self._append_doubao_tts_v2_audio_chunk(joined, audio)
        if audio:
            return bytes(audio)
        raise RuntimeError("Doubao TTS 2.0 returned no audio")

    def _append_doubao_tts_v2_audio_chunk(self, line, audio):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            return

        code = result.get("code")
        message = result.get("message") or result.get("msg") or result.get("Message") or ""
        audio_b64 = result.get("data") or ""
        if audio_b64:
            if audio_b64.startswith("data:"):
                audio_b64 = audio_b64.split(",", 1)[1]
            audio.extend(base64.b64decode(audio_b64))
            return

        try:
            numeric_code = int(code)
        except (TypeError, ValueError):
            numeric_code = 0
        if numeric_code and numeric_code != 20000000:
            raise RuntimeError(message or f"Doubao TTS 2.0 error: {numeric_code}")

    def _write_temp_audio(self, audio_bytes):
        suffix = f".{(DOUBAO_TTS_ENCODING or 'mp3').lower()}"
        with tempfile.NamedTemporaryFile(prefix="xiaorui_tts_", suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            return temp_file.name

    def _cleanup_temp_audio(self, path):
        if not path:
            return
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


class VoiceAssistantWorker(QThread):
    DOUBAO_ASR_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"

    status_changed = pyqtSignal(str)
    activated = pyqtSignal()
    command_recognized = pyqtSignal(str)
    dependency_error = pyqtSignal(str)

    WAKE_WORDS = (
        "你好小睿",
        "你好 小睿",
        "你好小瑞",
        "你好小锐",
        "你好小蕊",
    )
    ACTIVE_WINDOW_SECONDS = 18.0
    AUTO_RECALIBRATE_SECONDS = 18.0
    FAILURE_RECALIBRATE_THRESHOLD = 3
    MIN_COMMAND_CHARS = 2
    IDLE_LISTEN_TIMEOUT = 1.6
    ACTIVE_LISTEN_TIMEOUT = 3.2
    WAKE_PHRASE_TIME_LIMIT = 3.8
    COMMAND_PHRASE_TIME_LIMIT = 10.5
    MIN_WAKE_AUDIO_SECONDS = 0.25
    MIN_COMMAND_AUDIO_SECONDS = 0.42
    MIN_INTERRUPT_AUDIO_SECONDS = 0.18
    MIN_AUDIO_RMS = 120
    MIN_AUDIO_PEAK = 480
    MAX_NOISE_ZCR = 0.33
    SPEAKER_COOLDOWN_SECONDS = 0.25
    SPEAKER_LISTEN_TIMEOUT = 1.0
    SPEAKER_PHRASE_TIME_LIMIT = 2.8
    WAKE_PATTERNS = (
        r"你好小[睿瑞锐蕊芮]",
        r"你好晓[睿瑞锐蕊芮]",
        r"你好[晓小]瑞",
        r"你好[晓小]锐",
        r"小睿你好",
        r"小睿在吗",
        r"哈喽小睿",
        r"hello小睿",
    )
    INTERRUPT_PHRASES = (
        "停",
        "停一下",
        "停一停",
        "先停",
        "暂停一下",
        "停止播报",
        "停止说话",
        "停止",
        "先别说",
        "别说了",
        "别讲了",
        "别读了",
        "不要说了",
        "别播了",
        "可以了",
        "好了",
        "打住",
        "打断",
        "暂停",
    )
    COMMAND_HINTS = (
        "打开",
        "查看",
        "显示",
        "开始",
        "停止",
        "检测",
        "专注",
        "分数",
        "建议",
        "原因",
        "为什么",
        "介绍",
        "自己",
        "返回",
        "恢复",
        "最小化",
        "什么",
        "怎么",
        "谁",
        "能不能",
        "可以",
        "会不会",
        "解释",
        "总结",
        "整理",
        "生成",
        "功能",
        "系统",
        "项目",
        "班级",
        "同步",
        "学习",
    )

    def __init__(self, parent=None, speaker_active_callback=None):
        super().__init__(parent)
        self._running = False
        self._active_until = 0.0
        self._offline_ready = False
        self._vosk_model = None
        self._last_calibration_at = 0.0
        self._consecutive_failures = 0
        self._noise_rms_floor = 90.0
        self._speaker_active_callback = speaker_active_callback
        self._speaker_quiet_until = 0.0

    def stop(self):
        self._running = False
        self.requestInterruption()

    def run(self):
        try:
            import speech_recognition as sr
        except Exception:
            self.dependency_error.emit(
                "缺少语音识别依赖：请安装 SpeechRecognition 和 PyAudio 后再启用小睿。"
            )
            return

        self._running = True
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.energy_threshold = 120
        recognizer.dynamic_energy_adjustment_damping = 0.18
        recognizer.dynamic_energy_ratio = 1.15
        recognizer.operation_timeout = 8
        recognizer.pause_threshold = 0.9
        recognizer.non_speaking_duration = 0.35
        recognizer.phrase_threshold = 0.18

        try:
            microphone = sr.Microphone(sample_rate=16000, chunk_size=1024)
        except Exception as exc:
            self.dependency_error.emit(f"无法打开麦克风：{exc}")
            return

        self._prepare_offline_backend(show_status=not self._doubao_asr_ready())

        with microphone as source:
            self.status_changed.emit("小睿正在适应环境噪声")
            self._calibrate_ambient_noise(recognizer, source, duration=0.4)
            if self._doubao_asr_ready():
                self.status_changed.emit("小睿待命中：豆包语音识别已启用，请说“你好，小睿”")
            elif self._offline_ready:
                self.status_changed.emit("小睿待命中：本地语音兜底已启用，请说“你好，小睿”")
            else:
                self.status_changed.emit("小睿待命中：请说“你好，小睿”")

            while self._running and not self.isInterruptionRequested():
                speaker_active = self._speaker_is_active()
                if speaker_active:
                    self._speaker_quiet_until = time.time() + self.SPEAKER_COOLDOWN_SECONDS
                if not speaker_active and time.time() < self._speaker_quiet_until:
                    time.sleep(0.05)
                    continue

                is_active = time.time() < self._active_until
                if not is_active and (time.time() - self._last_calibration_at) >= self.AUTO_RECALIBRATE_SECONDS:
                    self._calibrate_ambient_noise(recognizer, source, duration=0.18)
                try:
                    audio = recognizer.listen(
                        source,
                        timeout=(
                            self.SPEAKER_LISTEN_TIMEOUT
                            if speaker_active
                            else self.IDLE_LISTEN_TIMEOUT
                            if not is_active
                            else self.ACTIVE_LISTEN_TIMEOUT
                        ),
                        phrase_time_limit=(
                            self.SPEAKER_PHRASE_TIME_LIMIT
                            if speaker_active
                            else self.WAKE_PHRASE_TIME_LIMIT
                            if not is_active
                            else self.COMMAND_PHRASE_TIME_LIMIT
                        ),
                    )
                except sr.WaitTimeoutError:
                    continue
                except Exception as exc:
                    self.status_changed.emit(f"语音监听异常：{exc}")
                    self._record_failure(recognizer, source, is_active)
                    continue

                if not self._looks_like_speech_audio(audio, is_active, speaker_active=speaker_active):
                    if not speaker_active:
                        self._record_failure(recognizer, source, is_active)
                    continue

                try:
                    candidates = self._recognize_candidates(recognizer, audio)
                except sr.UnknownValueError:
                    if not speaker_active:
                        self._record_failure(recognizer, source, is_active)
                    continue
                except sr.RequestError as exc:
                    self.status_changed.emit(f"语音识别网络异常：{exc}")
                    if not speaker_active:
                        self._record_failure(recognizer, source, is_active)
                    continue
                except Exception as exc:
                    self.status_changed.emit(f"语音识别失败：{exc}")
                    if not speaker_active:
                        self._record_failure(recognizer, source, is_active)
                    continue

                if not candidates:
                    if not speaker_active:
                        self._record_failure(recognizer, source, is_active)
                    continue

                matched = False
                for text in candidates:
                    normalized = self._normalize(text)
                    if not normalized or len(normalized) < self.MIN_COMMAND_CHARS:
                        continue

                    command = self._extract_command(normalized, speaker_active=speaker_active)
                    if command is None:
                        continue

                    self._consecutive_failures = 0
                    self.command_recognized.emit(command)
                    matched = True
                    break

                if not matched and not speaker_active:
                    self._record_failure(recognizer, source, is_active)

    def _speaker_is_active(self):
        if not self._speaker_active_callback:
            return False
        try:
            return bool(self._speaker_active_callback())
        except Exception:
            return False

    def _looks_like_speech_audio(self, audio, is_active, speaker_active=False):
        try:
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
        except Exception:
            return True

        if not raw:
            return False

        sample_count = max(1, len(raw) // 2)
        duration = sample_count / 16000.0
        if speaker_active:
            min_duration = self.MIN_INTERRUPT_AUDIO_SECONDS
        else:
            min_duration = self.MIN_COMMAND_AUDIO_SECONDS if is_active else self.MIN_WAKE_AUDIO_SECONDS
        if duration < min_duration:
            return False

        try:
            rms = float(audioop.rms(raw, 2))
            peak = float(audioop.max(raw, 2))
        except Exception:
            return True

        if speaker_active:
            adaptive_gate = max(self.MIN_AUDIO_RMS * 0.75, self._noise_rms_floor * 1.55)
        else:
            adaptive_gate = max(self.MIN_AUDIO_RMS, self._noise_rms_floor * (1.9 if is_active else 2.1))
        if rms < adaptive_gate and peak < self.MIN_AUDIO_PEAK:
            self._update_noise_floor(rms)
            return False

        zcr = self._zero_crossing_rate(raw)
        zcr_limit = self.MAX_NOISE_ZCR + (0.08 if speaker_active else 0.0)
        if zcr > zcr_limit and rms < adaptive_gate * (1.7 if speaker_active else 1.45):
            self._update_noise_floor(rms)
            return False

        return True

    def _update_noise_floor(self, rms):
        try:
            rms = float(rms)
        except Exception:
            return
        self._noise_rms_floor = self._noise_rms_floor * 0.9 + min(max(rms, 20.0), 500.0) * 0.1

    def _zero_crossing_rate(self, raw):
        try:
            samples = memoryview(raw).cast("h")
            if len(samples) < 2:
                return 0.0
            step = max(1, len(samples) // 2400)
            previous = samples[0]
            crossings = 0
            total = 0
            for index in range(step, len(samples), step):
                current = samples[index]
                if (previous < 0 <= current) or (previous >= 0 > current):
                    crossings += 1
                previous = current
                total += 1
            return crossings / max(total, 1)
        except Exception:
            return 0.0

    def _normalize(self, text):
        compact = re.sub(r"[\s，。！？,.!?、]", "", text or "")
        return self._normalize_assistant_name(compact)

    def _normalize_assistant_name(self, text):
        if not text:
            return ""
        replacements = {
            "小瑞": "小睿",
            "小锐": "小睿",
            "小蕊": "小睿",
            "小芮": "小睿",
            "小慧": "小睿",
            "小惠": "小睿",
            "小辉": "小睿",
            "小晖": "小睿",
            "小微": "小睿",
            "小薇": "小睿",
            "小维": "小睿",
            "小威": "小睿",
            "小为": "小睿",
            "小魏": "小睿",
            "晓睿": "小睿",
            "晓瑞": "小睿",
            "晓锐": "小睿",
            "晓蕊": "小睿",
            "晓芮": "小睿",
            "晓慧": "小睿",
            "晓惠": "小睿",
            "晓辉": "小睿",
            "晓微": "小睿",
            "晓薇": "小睿",
            "晓维": "小睿",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text

    def _recognize_candidates(self, recognizer, audio):
        if self._doubao_asr_ready():
            try:
                doubao_results = self._recognize_candidates_doubao(audio)
                if doubao_results:
                    return doubao_results
            except Exception:
                if not self._offline_ready:
                    raise

        if self._offline_ready:
            return self._recognize_candidates_vosk(audio)

        results = []
        try:
            raw = recognizer.recognize_google(audio, language="zh-CN", show_all=True)
        except Exception:
            raw = None

        alternatives = []
        if isinstance(raw, dict):
            alternatives = raw.get("alternative", []) or []
        elif isinstance(raw, list):
            alternatives = raw

        for item in alternatives:
            if isinstance(item, dict):
                transcript = item.get("transcript")
            else:
                transcript = str(item)
            if transcript and transcript not in results:
                results.append(transcript)

        if results:
            return results

        fallback = recognizer.recognize_google(audio, language="zh-CN")
        if fallback:
            results.append(fallback)
        return results

    def _doubao_asr_ready(self):
        return bool(DOUBAO_ASR_API_KEY)

    def _recognize_candidates_doubao(self, audio):
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
        payload = {
            "user": {"uid": "xiaorui-study"},
            "audio": {"data": base64.b64encode(wav_bytes).decode("utf-8")},
            "request": {"model_name": "bigmodel"},
        }
        request = urllib.request.Request(
            self.DOUBAO_ASR_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": DOUBAO_ASR_API_KEY,
                "X-Api-Resource-Id": DOUBAO_ASR_RESOURCE_ID,
                "X-Api-Request-Id": str(uuid.uuid4()),
                "X-Api-Sequence": "-1",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=22) as response:
            raw = response.read().decode("utf-8")
        result = json.loads(raw)

        candidates = []
        text = (((result.get("result") or {}).get("text")) or "").strip()
        if text:
            candidates.append(text)

        for utterance in (result.get("result") or {}).get("utterances", []) or []:
            utterance_text = str(utterance.get("text") or "").strip()
            if utterance_text and utterance_text not in candidates:
                candidates.append(utterance_text)

        return candidates

    def _prepare_offline_backend(self, show_status=True):
        try:
            from vosk import Model, SetLogLevel

            if not VOSK_MODEL_PATH:
                return
            SetLogLevel(-1)
            if show_status:
                self.status_changed.emit("小睿正在加载本地语音兜底模型")
            self._vosk_model = Model(VOSK_MODEL_PATH)
            self._offline_ready = True
        except Exception:
            self._offline_ready = False
            self._vosk_model = None

    def _calibrate_ambient_noise(self, recognizer, source, duration):
        try:
            recognizer.adjust_for_ambient_noise(source, duration=duration)
            self._last_calibration_at = time.time()
            self._consecutive_failures = 0
        except Exception:
            pass

    def _record_failure(self, recognizer, source, is_active):
        if is_active:
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.FAILURE_RECALIBRATE_THRESHOLD:
            self.status_changed.emit("环境有些嘈杂，小睿正在重新适应噪声")
            self._calibrate_ambient_noise(recognizer, source, duration=0.22)

    def _recognize_candidates_vosk(self, audio):
        if not self._vosk_model:
            return []

        from vosk import KaldiRecognizer

        grammar = json.dumps(
            [
                "你好 小睿",
                "你好 小瑞",
                "你好 小锐",
                "你好 小蕊",
                "你好 晓睿",
                "你好 小慧",
                "你好 小微",
                "小睿 你好",
                "小睿 在 吗",
                "哈喽 小睿",
                "停",
                "停 一下",
                "停 一停",
                "先 停",
                "暂停 一下",
                "停止 播报",
                "停止 说话",
                "别 说 了",
                "别 讲 了",
                "不要 说 了",
                "可以 了",
                "好了",
                "打住",
                "打断",
                "打开 详细 数据 页面",
                "打开 详细 分析 页面",
                "查看 详细 数据",
                "查看 详细 分析",
                "打开 详细 数据",
                "打开 详细 分析",
                "打开 数据 页面",
                "打开 分析 页面",
                "当前 我 的 专注度 是 多少",
                "当前 专注度 是 多少",
                "当前 我的 专注度 是 多少",
                "我 现在 的 专注度 是 多少",
                "现在 专注度 是 多少",
                "开始 检测",
                "开启 检测",
                "启动 检测",
                "停止 检测",
                "结束 检测",
                "关闭 检测",
                "程序 最小化",
                "窗口 最小化",
                "最小化",
                "隐藏 窗口",
                "返回 检测 界面",
                "回到 检测 界面",
                "返回 主 界面",
                "回到 主 界面",
                "恢复 窗口",
                "介绍 一下 自己",
                "介绍 下 自己",
                "介绍 你 自己",
                "介绍 一下 你 自己",
                "你 是 谁",
                "你 能 做 什么",
                "你好 小睿 介绍 一下 自己",
                "你好 小睿 介绍 你 自己",
                "你好 小睿 你 是 谁",
                "你好 小睿 你 能 做 什么",
                "[unk]",
            ],
            ensure_ascii=False,
        )

        recognizer = KaldiRecognizer(self._vosk_model, 16000, grammar)
        raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        accepted = recognizer.AcceptWaveform(raw_data)

        results = []
        try:
            final = json.loads(recognizer.FinalResult())
        except Exception:
            final = {}
        text = final.get("text", "")
        if text:
            results.append(text)
        if not accepted:
            try:
                partial = json.loads(recognizer.PartialResult())
            except Exception:
                partial = {}
            partial_text = partial.get("partial", "")
            if partial_text and partial_text not in results:
                results.append(partial_text)
        return results

    def _extract_command(self, normalized, speaker_active=False):
        for wake_word in self.WAKE_WORDS:
            wake = self._normalize(wake_word)
            if wake and wake in normalized:
                self._active_until = time.time() + self.ACTIVE_WINDOW_SECONDS
                self.activated.emit()
                command = normalized.split(wake, 1)[-1]
                if command:
                    return command
                return "__wake__"

        for pattern in self.WAKE_PATTERNS:
            match = re.search(pattern, normalized)
            if not match:
                continue
            self._active_until = time.time() + self.ACTIVE_WINDOW_SECONDS
            self.activated.emit()
            command = normalized[match.end():]
            if command:
                return command
            return "__wake__"

        if speaker_active and any(phrase in normalized for phrase in self.INTERRUPT_PHRASES):
            self._active_until = time.time() + self.ACTIVE_WINDOW_SECONDS
            return "__interrupt__"

        if speaker_active:
            return None

        if time.time() < self._active_until:
            if not self._looks_like_command_text(normalized):
                return None
            return normalized
        return None

    def _looks_like_command_text(self, normalized):
        if not normalized:
            return False
        if normalized == "__wake__":
            return True
        if len(normalized) < 3:
            return False
        if any(keyword in normalized for keyword in self.COMMAND_HINTS):
            return True
        if any(ch.isdigit() for ch in normalized):
            return True
        chinese_chars = sum(1 for ch in normalized if "\u4e00" <= ch <= "\u9fff")
        return chinese_chars >= 6
