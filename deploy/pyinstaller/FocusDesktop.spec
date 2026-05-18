# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path.cwd()
focus_root = project_root.parent
algorithm_root = project_root / "archive" / "algorithm-research"
app_display_name = "小睿伴学"
app_bundle_name = f"{app_display_name}.app"
app_icon_path = project_root / "assets" / "branding" / "app_icon.icns"

datas = [
    (str(project_root / "assets"), "assets"),
    (str(project_root / "data"), "data"),
    (str(project_root / "models"), "models"),
]

if app_icon_path.exists():
    datas.append((str(app_icon_path), "."))

if (algorithm_root / "src/models/uhmf/model.py").exists():
    datas.append(
        (
            str(algorithm_root / "src/models/uhmf/model.py"),
            "archive/algorithm-research/src/models/uhmf",
        )
    )

if (algorithm_root / "results/exp3_uhmf/main/best_model.pt").exists():
    datas.append(
        (
            str(algorithm_root / "results/exp3_uhmf/main/best_model.pt"),
            "archive/algorithm-research/results/exp3_uhmf/main",
        )
    )

typing_extensions_file = algorithm_root / "venv/lib/python3.14/site-packages/typing_extensions.py"
if typing_extensions_file.exists():
    datas.append(
        (
            str(typing_extensions_file),
            "archive/algorithm-research/venv/lib/python3.14/site-packages",
        )
    )

hiddenimports = ["ultralytics"]

excludes = [
    "tensorflow",
    "tensorflow-plugins",
    "keras",
    "tensorboard",
    "torch.utils.tensorboard",
    "torchaudio",
    "librosa",
    "transformers",
    "openpyxl",
    "sqlalchemy",
    "grpc",
    "grpcio",
    "h5py",
    "numba",
    "llvmlite",
    "polars",
    "tkinter",
    "pytest",
]

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    [],
    name="FocusDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="FocusDesktop",
)

app = BUNDLE(
    coll,
    name=app_bundle_name,
    icon=str(app_icon_path) if app_icon_path.exists() else None,
    bundle_identifier="com.xiaoxiaorui.study",
    info_plist={
        "CFBundleDisplayName": app_display_name,
        "CFBundleName": app_display_name,
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1",
        "NSCameraUsageDescription": f"{app_display_name}需要访问摄像头，用于实时检测学习状态和专注度。",
        "NSMicrophoneUsageDescription": f"{app_display_name}需要访问麦克风，用于语音唤醒和本地语音助手交互。",
        "LSApplicationCategoryType": "public.app-category.education",
    },
)
