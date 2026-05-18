from config.config import COLORS


APP_BG = "#F4F7FB"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F7FBFF"
SUBTLE = "#EDF5FF"
BORDER = "#D7E4F3"
BORDER_STRONG = "#BFD3EC"
TEXT = "#2C3E50"
MUTED = "#7F8C8D"
SOFT_TEXT = "#A8B5C2"
SHADOW = "rgba(52, 152, 219, 0.08)"


def panel_style(radius=26, background=SURFACE, border=BORDER):
    return f"""
        QFrame {{
            background-color: {background};
            border: 1px solid {border};
            border-radius: {radius}px;
        }}
    """


def button_style(kind="default", radius=16, padding_y=10, padding_x=16):
    presets = {
        "default": {
            "bg": COLORS["primary"],
            "border": COLORS["primary"],
            "hover_bg": "#2980B9",
            "hover_border": "#2980B9",
            "text": "#FFFFFF",
        },
        "subtle": {
            "bg": "#EBF5FF",
            "border": "#AED6F1",
            "hover_bg": "#D6EAF8",
            "hover_border": "#7FB3D5",
            "text": COLORS["primary"],
        },
        "selected": {
            "bg": COLORS["primary"],
            "border": COLORS["primary"],
            "hover_bg": "#2980B9",
            "hover_border": "#2980B9",
            "text": "#FFFFFF",
        },
        "accent": {
            "bg": COLORS["secondary"],
            "border": COLORS["secondary"],
            "hover_bg": "#8E44AD",
            "hover_border": "#8E44AD",
            "text": "#FFFFFF",
        },
        "danger": {
            "bg": COLORS["danger"],
            "border": COLORS["danger"],
            "hover_bg": "#C0392B",
            "hover_border": "#C0392B",
            "text": "#FFFFFF",
        },
        "success": {
            "bg": COLORS["focused"],
            "border": COLORS["focused"],
            "hover_bg": "#27AE60",
            "hover_border": "#27AE60",
            "text": "#FFFFFF",
        },
    }
    preset = presets.get(kind, presets["default"])
    return f"""
        QPushButton {{
            background-color: {preset['bg']};
            color: {preset['text']};
            border: 1px solid {preset['border']};
            border-radius: {radius}px;
            padding: {padding_y}px {padding_x}px;
            font: 10.5pt "Microsoft YaHei";
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: {preset['hover_bg']};
            border-color: {preset['hover_border']};
        }}
        QPushButton:pressed {{
            background-color: {preset['hover_bg']};
        }}
        QPushButton:disabled {{
            background-color: #D6DBDF;
            color: #FFFFFF;
            border-color: #D6DBDF;
        }}
    """


def combo_style(radius=16, min_width=140):
    return f"""
        QComboBox {{
            background-color: {SURFACE};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: {radius}px;
            padding: 8px 14px;
            min-width: {min_width}px;
        }}
        QComboBox:hover {{
            border-color: {COLORS["primary"]};
            background-color: {SURFACE_ALT};
        }}
        QComboBox:on {{
            border-color: {COLORS["primary"]};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 28px;
        }}
        QComboBox QAbstractItemView {{
            background: {SURFACE};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 6px;
            outline: none;
            selection-background-color: {SUBTLE};
            selection-color: {TEXT};
        }}
        QComboBox QAbstractItemView::item {{
            min-height: 28px;
            padding: 6px 12px;
            margin: 2px 0;
            border-radius: 10px;
            color: {TEXT};
        }}
        QComboBox QAbstractItemView::item:hover {{
            background: #EBF5FF;
            border: 1px solid #AED6F1;
            color: {TEXT};
        }}
        QComboBox QAbstractItemView::item:selected {{
            background: #D6EAF8;
            border: 1px solid {COLORS["primary"]};
            color: {TEXT};
        }}
    """


def input_style(multiline=False, radius=18):
    padding = "12px 14px" if multiline else "0 14px"
    return f"""
        background-color: {SURFACE};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: {radius}px;
        padding: {padding};
    """


def text_edit_style():
    return f"""
        QTextEdit {{
            background-color: {SURFACE};
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 18px;
            padding: 12px 14px;
        }}
    """


def badge_style(kind="default"):
    palette = {
        "default": ("#EBF5FF", "#AED6F1", COLORS["primary"]),
        "accent": ("#F4ECF7", "#D2B4DE", COLORS["secondary"]),
        "success": ("#EAFAF1", "#A9DFBF", COLORS["focused"]),
        "danger": ("#FDEDEC", "#F5B7B1", COLORS["danger"]),
    }
    bg, border, color = palette.get(kind, palette["default"])
    return f"""
        QLabel {{
            background-color: {bg};
            color: {color};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 8px 14px;
        }}
    """


def table_style():
    return f"""
        QTableWidget {{
            background-color: {SURFACE};
            alternate-background-color: #FAFCFF;
            border: 1px solid {BORDER};
            border-radius: 18px;
            color: {TEXT};
            gridline-color: {BORDER};
            selection-background-color: #D6EAF8;
            selection-color: {TEXT};
        }}
        QHeaderView::section {{
            background-color: #EDF5FF;
            color: {TEXT};
            border: none;
            border-bottom: 1px solid {BORDER};
            padding: 10px;
            font-weight: 600;
        }}
        QTableWidget::item {{
            padding: 10px;
            border-bottom: 1px solid {BORDER};
        }}
    """


SEMANTIC_TEXT = {
    "focused": COLORS["focused"],
    "moderate": COLORS["moderate"],
    "distracted": COLORS["distracted"],
}
