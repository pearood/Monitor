import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCE_DIR = getattr(sys, "_MEIPASS", BASE_DIR)
RUNTIME_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else BASE_DIR

APP_NAME = "小睿伴学"
APP_BUNDLE_NAME = "小睿伴学.app"
APP_DOWNLOAD_NAME = "小睿伴学-macos-arm64.zip"
APP_USER_AGENT = "XiaoXiaoRuiStudy/1.0"
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com"
DOUBAO_API_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
SUPPORTED_REMOTE_LLM_MODELS = [
    ("DeepSeek Chat", "deepseek:deepseek-chat"),
    ("DeepSeek Reasoner", "deepseek:deepseek-reasoner"),
    ("Doubao Vision (Images)", "doubao:doubao-seed-1-6-vision-250815"),
    ("Doubao Image Generation", "doubao:doubao-seedream-5-0-lite-260128"),
]


def resource_path(*parts):
    """Return a readable asset path that also works after PyInstaller packaging."""
    bundled = os.path.join(RESOURCE_DIR, *parts)
    if os.path.exists(bundled):
        return bundled
    return os.path.join(BASE_DIR, *parts)


def runtime_path(*parts):
    """Return a writable runtime path beside the executable/source tree."""
    return os.path.join(RUNTIME_DIR, *parts)


def _load_simple_env_file(path):
    values = {}
    if not os.path.exists(path):
        return values

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return {}

    return values


DATA_DIR = resource_path("data")
DATASET_PATH = resource_path("data", "attention_detection_dataset_augmented.csv")
ORIGINAL_DATASET_PATH = resource_path("data", "attention_detection_dataset_v1.csv")

DB_PATH = runtime_path("runtime", "users.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

API_BASE_URL = os.environ.get("FOCUS_API_BASE_URL", "http://47.238.152.124").rstrip("/")
REMOTE_API_ENABLED = os.environ.get("FOCUS_REMOTE_API", "1").lower() not in {"0", "false", "no", "local"}

_DEEPSEEK_ENV_VALUES = _load_simple_env_file(os.path.join(BASE_DIR, "config", "deepseek.env"))
_DOUBAO_ENV_VALUES = _load_simple_env_file(os.path.join(BASE_DIR, "config", "doubao.env"))
_LLM_ENV_VALUES = {**_DEEPSEEK_ENV_VALUES, **_DOUBAO_ENV_VALUES}
DEEPSEEK_LLM_API_KEY = (
    os.environ.get("DEEPSEEK_API_KEY")
    or _DEEPSEEK_ENV_VALUES.get("DEEPSEEK_API_KEY", "")
).strip()
DOUBAO_LLM_API_KEY = (
    os.environ.get("DOUBAO_API_KEY")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_API_KEY", "")
).strip()
DOUBAO_SPEECH_APPID = (
    os.environ.get("DOUBAO_SPEECH_APPID")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_SPEECH_APPID", "")
).strip()
DOUBAO_SPEECH_TOKEN = (
    os.environ.get("DOUBAO_SPEECH_TOKEN")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_SPEECH_TOKEN", "")
).strip()
DOUBAO_TTS_VERSION = (
    os.environ.get("DOUBAO_TTS_VERSION")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_TTS_VERSION", "1")
).strip()
DOUBAO_TTS_API_KEY = (
    os.environ.get("DOUBAO_TTS_API_KEY")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_TTS_API_KEY", "")
).strip()
DOUBAO_TTS_ACCESS_KEY = (
    os.environ.get("DOUBAO_TTS_ACCESS_KEY")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_TTS_ACCESS_KEY", "")
    or DOUBAO_SPEECH_TOKEN
).strip()
DOUBAO_TTS_APP_KEY = (
    os.environ.get("DOUBAO_TTS_APP_KEY")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_TTS_APP_KEY", "")
    or DOUBAO_SPEECH_APPID
).strip()
DOUBAO_TTS_RESOURCE_ID = (
    os.environ.get("DOUBAO_TTS_RESOURCE_ID")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_TTS_RESOURCE_ID", "seed-tts-2.0")
).strip()
DOUBAO_TTS_CLUSTER = (
    os.environ.get("DOUBAO_TTS_CLUSTER")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_TTS_CLUSTER", "volcano_tts")
).strip()
DOUBAO_TTS_VOICE_TYPE = (
    os.environ.get("DOUBAO_TTS_VOICE_TYPE")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_TTS_VOICE_TYPE", "BV700_streaming")
).strip()
SUPPORTED_DOUBAO_TTS_VOICES = [
    ("小何 2.0", "zh_female_xiaohe_uranus_bigtts"),
    ("Vivi 2.0", "zh_female_vv_uranus_bigtts"),
    ("云舟 2.0", "zh_male_m191_uranus_bigtts"),
    ("小天 2.0", "zh_male_taocheng_uranus_bigtts"),
]
DOUBAO_TTS_ENCODING = (
    os.environ.get("DOUBAO_TTS_ENCODING")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_TTS_ENCODING", "mp3")
).strip()
DOUBAO_TTS_RATE = int(
    os.environ.get("DOUBAO_TTS_RATE")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_TTS_RATE", "24000")
)
DOUBAO_ASR_API_KEY = (
    os.environ.get("DOUBAO_ASR_API_KEY")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_ASR_API_KEY", "")
).strip()
DOUBAO_ASR_RESOURCE_ID = (
    os.environ.get("DOUBAO_ASR_RESOURCE_ID")
    or _DOUBAO_ENV_VALUES.get("DOUBAO_ASR_RESOURCE_ID", "volc.bigasr.auc_turbo")
).strip()
LOCAL_LLM_ENABLED = os.environ.get("LOCAL_LLM_ENABLED", "1").lower() not in {"0", "false", "no"}
LOCAL_LLM_PROVIDER = os.environ.get(
    "LOCAL_LLM_PROVIDER",
    _LLM_ENV_VALUES.get("LOCAL_LLM_PROVIDER", "deepseek"),
).strip().lower() or "deepseek"
_DEFAULT_API_BASE_URL = {
    "deepseek": DEEPSEEK_API_BASE_URL,
    "doubao": DOUBAO_API_BASE_URL,
}.get(LOCAL_LLM_PROVIDER, DEEPSEEK_API_BASE_URL)
LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", _DEFAULT_API_BASE_URL).rstrip("/")
LOCAL_LLM_MODEL = os.environ.get(
    "LOCAL_LLM_MODEL",
    (
        _DEEPSEEK_ENV_VALUES.get("DEEPSEEK_MODEL", "deepseek-chat")
        if LOCAL_LLM_PROVIDER == "deepseek"
        else _DOUBAO_ENV_VALUES.get("DOUBAO_MODEL", "doubao-seed-1-6-vision-250815")
        if LOCAL_LLM_PROVIDER == "doubao"
        else "deepseek-chat"
    ),
).strip() or (
    "deepseek-chat"
    if LOCAL_LLM_PROVIDER == "deepseek"
    else "doubao-seed-1-6-vision-250815"
    if LOCAL_LLM_PROVIDER == "doubao"
    else "deepseek-chat"
)
LOCAL_LLM_API_KEY = (
    os.environ.get("LOCAL_LLM_API_KEY")
    or (
        DEEPSEEK_LLM_API_KEY
        if LOCAL_LLM_PROVIDER == "deepseek"
        else DOUBAO_LLM_API_KEY
        if LOCAL_LLM_PROVIDER == "doubao"
        else DEEPSEEK_LLM_API_KEY
    )
).strip()
LOCAL_LLM_TIMEOUT = max(
    15.0,
    float(
        os.environ.get(
            "LOCAL_LLM_TIMEOUT",
            (
                _DEEPSEEK_ENV_VALUES.get("DEEPSEEK_TIMEOUT", "15")
                if LOCAL_LLM_PROVIDER == "deepseek"
                else _DOUBAO_ENV_VALUES.get("DOUBAO_TIMEOUT", "15")
                if LOCAL_LLM_PROVIDER == "doubao"
                else _DEEPSEEK_ENV_VALUES.get("DEEPSEEK_TIMEOUT", "15")
            ),
        )
    ),
)

LOGO_PATH = resource_path("assets", "branding", "logo.jpg")
ICON_PATH = resource_path("assets", "branding", "app_icon.icns")

MODEL_PATH = runtime_path("models", "attention_model.pkl")
YOLO_MODEL_PATH = resource_path("models", "yolo11n.pt")
VOSK_MODEL_PATH = resource_path("models", "vosk-model-small-cn-0.22")

WINDOW_WIDTH = 1360
WINDOW_HEIGHT = 860

ROBOT_SIZE = 120

SCORING_MODE = "uhmf_primary"
UHMF_PRIMARY_WEIGHT = 0.78

DEFAULT_CLASSES = ["计算机1班", "计算机2班"]

ATTENTION_THRESHOLDS = {
    "focused": 70,
    "moderate": 40,
    "distracted": 0,
}

COLORS = {
    "focused": "#2ECC71",
    "moderate": "#F39C12",
    "distracted": "#E74C3C",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "primary": "#3498DB",
    "secondary": "#9B59B6",
    "background": "#ECF0F1",
    "surface": "#FFFFFF",
    "muted": "#95A5A6",
    "text": "#2C3E50",
}
