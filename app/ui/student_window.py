import base64
import ctypes
import html
import os
import re
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
from PyQt5.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QEvent, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QDesktopServices, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPen, QPixmap, QTextOption
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QToolButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMenu,
)

from app.core.multimodal_stream import RealMultimodalStream
from app.core.study_helper import (
    StudyAssistantError,
    build_default_study_filename,
    build_curated_learning_sources,
    detect_web_search_command,
    desktop_dir,
    extract_site_preferences,
    extract_topic,
    extract_urls,
    fetch_page_content,
    get_active_browser_tabs,
    infer_learning_outputs,
    infer_output_format,
    infer_save_mode,
    is_current_tabs_command,
    is_current_page_command,
    is_material_collection_command,
    save_document,
    save_text,
    search_learning_sources,
)
from app.core.voice_assistant import VoiceAssistantWorker, VoiceSpeaker
from app.core.voice_llm_controller import VoiceLLMController, VoiceLLMControllerError
from app.ui.analytics_windows import StudentAnalyticsWindow
from app.ui.robot_widget import MiniRobotWidget
from config.config import (
    APP_NAME,
    COLORS,
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_LLM_API_KEY,
    DOUBAO_API_BASE_URL,
    DOUBAO_LLM_API_KEY,
    DOUBAO_TTS_VOICE_TYPE,
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL,
    LOCAL_LLM_PROVIDER,
    SUPPORTED_DOUBAO_TTS_VOICES,
    SUPPORTED_REMOTE_LLM_MODELS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)


class VoiceHintBubble(QFrame):
    def __init__(self):
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setObjectName("voiceHintBubble")
        self.setFixedWidth(250)
        self._current_level = "moderate"

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self.opacity_effect)

        self.fade_in = QPropertyAnimation(self.opacity_effect, b"opacity", self)
        self.fade_in.setDuration(180)
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)
        self.fade_in.setEasingCurve(QEasingCurve.OutCubic)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        self.title_label = QLabel("小睿播报")
        self.title_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        layout.addWidget(self.title_label)

        self.label = QLabel("")
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Microsoft YaHei", 10, QFont.Medium))
        layout.addWidget(self.label)

        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide)
        self.set_theme("moderate")

    def set_theme(self, level):
        level = level if level in {"focused", "moderate", "distracted"} else "moderate"
        self._current_level = level
        palette = {
            "focused": ("rgba(46, 204, 113, 0.16)", "rgba(46, 204, 113, 0.45)", COLORS["focused"]),
            "moderate": ("rgba(243, 156, 18, 0.16)", "rgba(243, 156, 18, 0.45)", COLORS["moderate"]),
            "distracted": ("rgba(231, 76, 60, 0.16)", "rgba(231, 76, 60, 0.45)", COLORS["distracted"]),
        }
        background, border, accent = palette[level]
        self.setStyleSheet(
            f"""
            QFrame#voiceHintBubble {{
                background: {background};
                border: 1px solid {border};
                border-radius: 16px;
            }}
            QLabel {{
                color: {COLORS['text']};
                background: transparent;
            }}
            """
        )
        self.title_label.setStyleSheet(f"color: {accent};")

    def show_near(self, anchor_widget, text, duration_ms=3200, level="moderate", title="小睿播报"):
        if not anchor_widget or not text:
            return

        self.set_theme(level)
        self.title_label.setText(title)
        self.label.setText(text)
        self.adjustSize()

        anchor_top_left = anchor_widget.mapToGlobal(anchor_widget.rect().topLeft())
        anchor_top_right = anchor_widget.mapToGlobal(anchor_widget.rect().topRight())
        anchor_bottom_left = anchor_widget.mapToGlobal(anchor_widget.rect().bottomLeft())
        bubble_size = self.sizeHint()
        screen_geometry = anchor_widget.screen().availableGeometry() if anchor_widget.screen() else None

        x = anchor_top_left.x() - bubble_size.width() - 14
        y = anchor_top_left.y() + max(0, (anchor_widget.height() - bubble_size.height()) // 2)

        if screen_geometry and x < screen_geometry.left() + 8:
            x = anchor_top_right.x() + 14

        if screen_geometry:
            max_x = screen_geometry.right() - bubble_size.width() - 8
            max_y = screen_geometry.bottom() - bubble_size.height() - 8
            x = max(screen_geometry.left() + 8, min(x, max_x))
            y = max(screen_geometry.top() + 8, min(y, max_y))

        if screen_geometry and y + bubble_size.height() > screen_geometry.bottom():
            y = anchor_bottom_left.y() - bubble_size.height() - 12

        self.move(QPoint(x, y))
        self.opacity_effect.setOpacity(0.0)
        self.show()
        self.raise_()
        self.fade_in.stop()
        self.fade_in.start()
        self.hide_timer.start(duration_ms)


class AttachmentDropTextEdit(QTextEdit):
    fileDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._base_stylesheet = ""
        self._drag_active = False

    def set_base_stylesheet(self, stylesheet):
        self._base_stylesheet = stylesheet or ""
        self._apply_drop_state()

    def dragEnterEvent(self, event):
        if self._extract_local_file(event.mimeData()):
            self._drag_active = True
            self._apply_drop_state()
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event):
        if self._extract_local_file(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._drag_active = False
        self._apply_drop_state()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        path = self._extract_local_file(event.mimeData())
        self._drag_active = False
        self._apply_drop_state()
        if path:
            self.fileDropped.emit(path)
            event.acceptProposedAction()
            return
        event.ignore()

    def _apply_drop_state(self):
        extra = ""
        if self._drag_active:
            extra = (
                "\nQTextEdit {"
                " border: 1.5px dashed rgba(52, 152, 219, 0.88);"
                " background: rgba(236, 245, 255, 0.95);"
                " }"
            )
        self.setStyleSheet(f"{self._base_stylesheet}{extra}")

    @staticmethod
    def _extract_local_file(mime_data):
        if not mime_data or not mime_data.hasUrls():
            return None
        for url in mime_data.urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if path and os.path.isfile(path):
                    return path
        return None


class ConversationListItemWidget(QFrame):
    clicked = pyqtSignal()
    delete_requested = pyqtSignal()

    def __init__(self, title, tooltip=""):
        super().__init__()
        self.setObjectName("assistantConversationRow")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            """
            QFrame#assistantConversationRow {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 10px;
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 8, 7)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Microsoft YaHei", 9))
        self.title_label.setStyleSheet(f"color: {COLORS['text']};")
        self.title_label.setWordWrap(False)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.title_label, 1)

        self.delete_btn = QPushButton("×")
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.setFixedSize(22, 22)
        self.delete_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: white;
                color: {COLORS['danger']};
                border: 1px solid rgba(231, 76, 60, 0.22);
                border-radius: 11px;
                font-size: 14px;
                font-weight: 700;
                padding: 0;
            }}
            QPushButton:hover {{
                background: rgba(231, 76, 60, 0.08);
                border-color: rgba(231, 76, 60, 0.38);
            }}
            """
        )
        self.delete_btn.hide()
        self.delete_btn.clicked.connect(self.delete_requested.emit)
        layout.addWidget(self.delete_btn, 0)

        if tooltip:
            self.setToolTip(tooltip)
            self.title_label.setToolTip(tooltip)

        self.set_selected(False)

    def set_selected(self, selected):
        if selected:
            self.setStyleSheet(
                """
                QFrame#assistantConversationRow {
                    background: #EDF3FA;
                    border: 1px solid #2C3E50;
                    border-radius: 10px;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame#assistantConversationRow {
                    background: transparent;
                    border: 1px solid transparent;
                    border-radius: 10px;
                }
                """
            )

    def enterEvent(self, event):
        self.delete_btn.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.delete_btn.hide()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class MiniAssistantPanel(QFrame):
    submitted = pyqtSignal(str)
    restore_requested = pyqtSignal()
    new_chat_requested = pyqtSignal()
    voice_toggle_requested = pyqtSignal()
    upload_image_requested = pyqtSignal()
    upload_file_requested = pyqtSignal()
    clear_attachment_requested = pyqtSignal()
    COMPACT_SIZE = (380, 430)
    EXPANDED_SIZE = (640, 700)

    def __init__(self):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setObjectName("miniAssistantPanel")
        self._is_expanded = False
        self._drag_active = False
        self._drag_offset = QPoint()
        self._last_anchor_widget = None
        self.setFixedSize(*self.COMPACT_SIZE)

        self.setStyleSheet(
            f"""
            QFrame#miniAssistantPanel {{
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(52, 152, 219, 0.24);
                border-radius: 18px;
            }}
            QLabel {{
                background: transparent;
                color: {COLORS['text']};
            }}
            QTextEdit {{
                background: rgba(236, 240, 241, 0.78);
                border: 1px solid rgba(44, 62, 80, 0.08);
                border-radius: 12px;
                padding: 8px;
                color: {COLORS['text']};
            }}
            QLineEdit {{
                background: white;
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 10px;
                padding: 8px 10px;
                color: {COLORS['text']};
            }}
            QComboBox {{
                background: white;
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 10px;
                padding: 4px 10px;
                color: {COLORS['text']};
            }}
            QComboBox:hover {{
                border-color: #2C3E50;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 18px;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.header_frame = QFrame()
        self.header_frame.setObjectName("miniAssistantHeader")
        self.header_frame.setStyleSheet("QFrame#miniAssistantHeader { background: transparent; border: none; }")
        self.header_frame.installEventFilter(self)
        header = QHBoxLayout(self.header_frame)
        header.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel("小睿文本助手")
        self.title_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.title_label.installEventFilter(self)
        header.addWidget(self.title_label)
        header.addStretch()

        self.expand_btn = QPushButton("⤢")
        self.expand_btn.setCursor(Qt.PointingHandCursor)
        self.expand_btn.setToolTip("放大")
        self.expand_btn.setFixedSize(28, 28)
        self.expand_btn.setFixedHeight(28)
        self.expand_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: white;
                color: {COLORS['text']};
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 8px;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                border-color: #2C3E50;
            }}
            """
        )
        self.expand_btn.clicked.connect(self.toggle_expanded)
        header.addWidget(self.expand_btn)

        self.restore_btn = QPushButton("回到检测")
        self.restore_btn.setCursor(Qt.PointingHandCursor)
        self.restore_btn.setFixedHeight(28)
        self.restore_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                background: #2980B9;
            }}
            """
        )
        self.restore_btn.clicked.connect(self.restore_requested.emit)
        header.addWidget(self.restore_btn)
        layout.addWidget(self.header_frame)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(8)

        self.new_chat_btn = QPushButton("新对话")
        self.new_chat_btn.setCursor(Qt.PointingHandCursor)
        self.new_chat_btn.setFixedHeight(28)
        self.new_chat_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: white;
                color: {COLORS['text']};
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 8px;
                padding: 0 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                border-color: #2C3E50;
                background: #F7FAFD;
            }}
            """
        )
        self.new_chat_btn.clicked.connect(self.new_chat_requested.emit)
        controls_row.addWidget(self.new_chat_btn)

        self.voice_toggle_btn = QPushButton("启动语音")
        self.voice_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.voice_toggle_btn.setFixedHeight(28)
        self.voice_toggle_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: {COLORS['secondary']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: #8E44AD;
            }}
            """
        )
        self.voice_toggle_btn.clicked.connect(self.voice_toggle_requested.emit)
        controls_row.addWidget(self.voice_toggle_btn, 1)
        layout.addLayout(controls_row)

        self.attachment_preview = QFrame()
        self.attachment_preview.setStyleSheet(
            f"""
            QFrame {{
                background: rgba(236, 245, 255, 0.78);
                border: 1px solid rgba(52, 152, 219, 0.18);
                border-radius: 10px;
            }}
            """
        )
        preview_layout = QHBoxLayout(self.attachment_preview)
        preview_layout.setContentsMargins(10, 8, 10, 8)
        preview_layout.setSpacing(8)
        self.attachment_thumb = QLabel()
        self.attachment_thumb.setFixedSize(40, 40)
        self.attachment_thumb.setAlignment(Qt.AlignCenter)
        self.attachment_thumb.setStyleSheet(
            "background: rgba(255,255,255,0.9); border-radius: 8px; color: #7F8C8D; font-weight: 700;"
        )
        preview_layout.addWidget(self.attachment_thumb, 0)
        self.attachment_label = QLabel("未附图片/文件")
        self.attachment_label.setWordWrap(True)
        self.attachment_label.setFont(QFont("Microsoft YaHei", 8))
        self.attachment_label.setStyleSheet(f"color: {COLORS['text']};")
        preview_layout.addWidget(self.attachment_label, 1)
        self.clear_attachment_btn = QPushButton("删除")
        self.clear_attachment_btn.setCursor(Qt.PointingHandCursor)
        self.clear_attachment_btn.setFixedHeight(26)
        self.clear_attachment_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: white;
                color: {COLORS['danger']};
                border: 1px solid rgba(231, 76, 60, 0.18);
                border-radius: 8px;
                padding: 0 10px;
            }}
            """
        )
        self.clear_attachment_btn.clicked.connect(self.clear_attachment_requested.emit)
        preview_layout.addWidget(self.clear_attachment_btn, 0)
        layout.addWidget(self.attachment_preview)

        self.chat_view = AttachmentDropTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setFont(QFont("Microsoft YaHei", 9))
        self.chat_view.setLineWrapMode(QTextEdit.WidgetWidth)
        self.chat_view.setWordWrapMode(QTextOption.WrapAnywhere)
        self.chat_view.setPlaceholderText("把图片或文件直接拖到这里，也可以继续文字对话。")
        self.chat_view.set_base_stylesheet(
            f"""
            QTextEdit {{
                background: rgba(236, 240, 241, 0.78);
                border: 1px solid rgba(44, 62, 80, 0.08);
                border-radius: 12px;
                padding: 8px;
                color: {COLORS['text']};
            }}
            """
        )
        layout.addWidget(self.chat_view, 1)

        self.status_label = QLabel("输入文字或上传图片/文件后，小睿会在这里回复你。")
        self.status_label.setWordWrap(True)
        self.status_label.setFont(QFont("Microsoft YaHei", 9))
        self.status_label.setStyleSheet(f"color: {COLORS['muted']};")
        layout.addWidget(self.status_label)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.plus_btn = QToolButton()
        self.plus_btn.setText("+")
        self.plus_btn.setCursor(Qt.PointingHandCursor)
        self.plus_btn.setFixedSize(30, 30)
        self.plus_btn.setPopupMode(QToolButton.InstantPopup)
        self.plus_btn.setStyleSheet(
            f"""
            QToolButton {{
                background: white;
                color: {COLORS['text']};
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 15px;
                font-size: 18px;
                font-weight: 700;
            }}
            QToolButton:hover {{
                border-color: #2C3E50;
            }}
            QToolButton::menu-indicator {{
                image: none;
                width: 0px;
            }}
            """
        )
        plus_menu = QMenu(self.plus_btn)
        plus_menu.addAction("上传图片", self.upload_image_requested.emit)
        plus_menu.addAction("上传文件", self.upload_file_requested.emit)
        self.plus_btn.setMenu(plus_menu)
        input_row.addWidget(self.plus_btn, 0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("例如：整理成学习笔记，或先上传图片/文件再提问")
        self.input_edit.returnPressed.connect(self._emit_submit)
        input_row.addWidget(self.input_edit, 1)

        self.send_btn = QPushButton("发送")
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setFixedHeight(36)
        self.send_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: {COLORS['secondary']};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0 14px;
            }}
            QPushButton:hover {{
                background: #8E44AD;
            }}
            """
        )
        self.send_btn.clicked.connect(self._emit_submit)
        input_row.addWidget(self.send_btn)
        layout.addLayout(input_row)

    def _emit_submit(self):
        text = self.input_edit.text().strip()
        self.input_edit.clear()
        self.submitted.emit(text)

    def toggle_expanded(self):
        self._is_expanded = not self._is_expanded
        if self._is_expanded:
            self.setFixedSize(*self.EXPANDED_SIZE)
            self.expand_btn.setText("⤡")
            self.expand_btn.setToolTip("缩小")
        else:
            self.setFixedSize(*self.COMPACT_SIZE)
            self.expand_btn.setText("⤢")
            self.expand_btn.setToolTip("放大")
        self._reposition(self._last_anchor_widget)

    def _reposition(self, anchor_widget=None):
        if anchor_widget is not None:
            self._last_anchor_widget = anchor_widget
        if self._is_expanded:
            screen_geometry = None
            if anchor_widget and anchor_widget.screen():
                screen_geometry = anchor_widget.screen().availableGeometry()
            elif self.screen():
                screen_geometry = self.screen().availableGeometry()
            if screen_geometry:
                x = screen_geometry.left() + (screen_geometry.width() - self.width()) // 2
                y = screen_geometry.top() + max(20, (screen_geometry.height() - self.height()) // 2)
                self.move(QPoint(x, y))
            return

        if not anchor_widget:
            return
        self.adjustSize()
        bubble_size = self.size()
        anchor_top_left = anchor_widget.mapToGlobal(anchor_widget.rect().topLeft())
        anchor_top_right = anchor_widget.mapToGlobal(anchor_widget.rect().topRight())
        screen_geometry = anchor_widget.screen().availableGeometry() if anchor_widget.screen() else None

        x = anchor_top_left.x() - bubble_size.width() - 16
        y = anchor_top_left.y() - 4

        if screen_geometry and x < screen_geometry.left() + 10:
            x = anchor_top_right.x() + 16

        if screen_geometry:
            max_x = screen_geometry.right() - bubble_size.width() - 10
            max_y = screen_geometry.bottom() - bubble_size.height() - 10
            x = max(screen_geometry.left() + 10, min(x, max_x))
            y = max(screen_geometry.top() + 10, min(y, max_y))

        self.move(QPoint(x, y))

    def eventFilter(self, watched, event):
        watched_targets = {
            target
            for target in (
                getattr(self, "header_frame", None),
                getattr(self, "title_label", None),
            )
            if target is not None
        }
        if watched in watched_targets:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._drag_active = True
                self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
                return True
            if event.type() == QEvent.MouseMove and self._drag_active and event.buttons() & Qt.LeftButton:
                self.move(event.globalPos() - self._drag_offset)
                return True
            if event.type() == QEvent.MouseButtonRelease:
                self._drag_active = False
                return True
        return super().eventFilter(watched, event)

    def set_chat_html(self, content):
        self.chat_view.setHtml(content)
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def set_status(self, text):
        self.status_label.setText(text)

    def set_voice_enabled(self, enabled):
        if enabled:
            self.voice_toggle_btn.setText("关闭语音")
            self.voice_toggle_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background: {COLORS['danger']};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 0 12px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: #C0392B;
                }}
                """
            )
        else:
            self.voice_toggle_btn.setText("启动语音")
            self.voice_toggle_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background: {COLORS['secondary']};
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 0 12px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: #8E44AD;
                }}
                """
            )

    def set_attachment_text(self, text, has_attachment=False, picker_visible=False, pixmap=None, badge_text=""):
        self.attachment_label.setText(text)
        self.clear_attachment_btn.setEnabled(bool(has_attachment))
        self.attachment_preview.setVisible(bool(has_attachment))
        self.plus_btn.setVisible(bool(picker_visible))
        if pixmap and not pixmap.isNull():
            self.attachment_thumb.setPixmap(pixmap)
            self.attachment_thumb.setText("")
        else:
            self.attachment_thumb.setPixmap(QPixmap())
            self.attachment_thumb.setText(badge_text or "")

    def show_near(self, anchor_widget):
        self._reposition(anchor_widget)
        self.show()
        self.raise_()


class StudentMainWindow(QWidget):
    logout_signal = pyqtSignal()
    voice_ai_result = pyqtSignal(int, dict, str)
    voice_ai_error = pyqtSignal(int, str, str)
    text_ai_result = pyqtSignal(int, dict, str)
    text_ai_error = pyqtSignal(int, str, str)
    study_task_result = pyqtSignal(int, dict, str)
    study_task_error = pyqtSignal(int, str, str)
    detector_loaded = pyqtSignal(object, str)
    assistant_conversations_loaded = pyqtSignal(list, object)
    assistant_messages_loaded = pyqtSignal(object, list)
    VOICE_INTERRUPT_ARM_SECONDS = 5.0
    DEFAULT_REMOTE_LLM_MODEL = "deepseek:deepseek-chat"
    PROVIDER_DEFAULT_MODELS = {
        "deepseek": "deepseek:deepseek-chat",
        "doubao": "doubao:doubao-seed-1-6-vision-250815",
    }
    PROVIDER_REASONING_MODELS = {
        "deepseek": "deepseek:deepseek-reasoner",
    }
    VIDEO_ASPECT_RATIO = 4 / 3
    IMAGE_ATTACHMENT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    DOCUMENT_ATTACHMENT_EXTENSIONS = VoiceLLMController.TEXT_FILE_EXTENSIONS | {
        ".docx",
        ".pdf",
        ".xlsx",
        ".pptx",
    }
    DIAGRAM_IMAGE_KEYWORDS = (
        ("思维导图", "mindmap", "思维导图"),
        ("导图", "mindmap", "思维导图"),
        ("脑图", "mindmap", "思维导图"),
        ("流程图", "flowchart", "流程图"),
        ("架构图", "architecture", "架构图"),
        ("结构图", "structure", "结构图"),
        ("框架图", "framework", "框架图"),
    )

    def __init__(self, user_info, db):
        super().__init__()
        self.user_info = user_info
        self.db = db
        self.detector = None
        self.detector_loading = False
        self.pending_start_after_detector_load = False
        self.multimodal_stream = RealMultimodalStream()

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.detector_executor = ThreadPoolExecutor(max_workers=1)
        self.detector_load_executor = ThreadPoolExecutor(max_workers=1)
        self.record_executor = ThreadPoolExecutor(max_workers=1)
        self.assistant_io_executor = ThreadPoolExecutor(max_workers=1)
        self.voice_command_executor = ThreadPoolExecutor(max_workers=2)
        self.presence_executor = ThreadPoolExecutor(max_workers=1)
        self.pending_detection = None
        self.pending_detection_time = None
        self.frame_counter = 0
        self.detection_stride = 3
        self.record_save_inflight = False

        self.analytics_window = None
        self.mini_robot = None
        self.mini_assistant_panel = None
        self.voice_hint_bubble = VoiceHintBubble()
        self.voice_worker = None
        self.voice_speaker = VoiceSpeaker()
        self.selected_tts_voice_type = DOUBAO_TTS_VOICE_TYPE
        self.voice_speaker.set_voice_type(self.selected_tts_voice_type)
        self.multimodal_paused_for_voice = False
        self.selected_llm_model = self._normalized_remote_llm_model(f"{LOCAL_LLM_PROVIDER}:{LOCAL_LLM_MODEL}")
        initial_provider, initial_model = self._split_llm_spec(self.selected_llm_model)
        self.voice_llm_controller = VoiceLLMController(
            local_enabled=True,
            local_provider=initial_provider,
            local_base_url=self._base_url_for_provider(initial_provider),
            local_model=initial_model,
            local_api_key=self._api_key_for_provider(initial_provider),
        )
        self.assistant_conversations = []
        self.current_assistant_conversation_id = None
        self.current_assistant_conversation_title = "新对话"
        self.assistant_messages = []
        self.assistant_pending_text = ""
        self.selected_attachment = None
        self.pending_assistant_log = None

        self.attention_scores = []
        self.live_records = []
        self.current_score = 0.0
        self.is_detecting = False
        self.is_minimized = False
        self.latest_runtime_info = {
            "backend": {"available": False},
            "multimodal": {"available": False, "status": "真实多模态未启动"},
        }
        self.voice_enabled = False
        self.voice_request_id = 0
        self.voice_runtime_active = True
        self.voice_interrupt_armed_until = 0.0
        self.llm_request_inflight = False
        self.last_llm_request_at = 0.0
        self.llm_request_cooldown_seconds = 2.2

        self.distraction_count = 0
        self.focused_duration_seconds = 0.0
        self.total_session_duration_seconds = 0.0
        self.session_start_time = None
        self.last_frame_ts = None
        self.display_last_ts = None
        self.preview_fps = 0.0
        self.inference_latency_ms = 0.0
        self.last_saved_at = 0.0
        self.last_saved_status = ""

        self.voice_ai_result.connect(self._apply_voice_ai_result)
        self.voice_ai_error.connect(self._handle_voice_ai_error)
        self.text_ai_result.connect(self._apply_text_ai_result)
        self.text_ai_error.connect(self._handle_text_ai_error)
        self.study_task_result.connect(self._apply_study_task_result)
        self.study_task_error.connect(self._handle_study_task_error)
        self.detector_loaded.connect(self._handle_detector_loaded)
        self.assistant_conversations_loaded.connect(self._handle_assistant_conversations_loaded)
        self.assistant_messages_loaded.connect(self._handle_assistant_messages_loaded)
        self.init_ui()
        self._load_assistant_conversations_async()
        self._refresh_assistant_views()
        self._refresh_attachment_views()
        self.presence_inflight = False
        self.presence_timer = QTimer(self)
        self.presence_timer.timeout.connect(self._heartbeat_presence)
        self.presence_timer.start(12000)
        self._set_presence(True)

    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} - 学生端 - {self.user_info.get('name', self.user_info['username'])}")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowSystemMenuHint
        )
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"QWidget {{ background-color: {COLORS['background']}; }}")

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        content_column = QVBoxLayout()
        content_column.setSpacing(20)
        content_column.addWidget(self.create_left_panel(), stretch=6)
        content_column.addWidget(self.create_right_panel(), stretch=9)

        main_layout.addLayout(content_column, stretch=12)
        main_layout.addWidget(self.create_assistant_panel(), stretch=7)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_video_viewport()

    def create_left_panel(self):
        panel = QFrame()
        panel.setStyleSheet("QFrame { background-color: white; border-radius: 15px; }")
        panel.setMinimumWidth(500)
        outer_layout = QHBoxLayout(panel)
        outer_layout.setContentsMargins(20, 20, 20, 20)
        outer_layout.setSpacing(18)

        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(220)
        sidebar_frame.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['background']}; border-radius: 14px; }}"
        )
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(14, 16, 14, 16)
        sidebar_layout.setSpacing(10)

        content_layout = QVBoxLayout()
        content_layout.setSpacing(15)
        content_layout.setAlignment(Qt.AlignTop)

        header_layout = QHBoxLayout()
        title_layout = QVBoxLayout()
        title_layout.setSpacing(3)

        title = QLabel("实时检测")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['text']};")
        title_layout.addWidget(title)

        self.class_info_label = QLabel("")
        self.class_info_label.setFont(QFont("Microsoft YaHei", 10))
        self.class_info_label.setStyleSheet(f"color: {COLORS['muted']};")
        title_layout.addWidget(self.class_info_label)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        content_layout.addLayout(header_layout)

        self.video_label = QLabel('点击"开始检测"启动摄像头')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumWidth(360)
        self.video_label.setMaximumWidth(16777215)
        self.video_label.setMinimumHeight(220)
        self.video_label.setMaximumHeight(300)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.video_label.setStyleSheet(
            """
            QLabel {
                background-color: #2C3E50;
                border-radius: 12px;
                color: white;
                font-size: 16px;
            }
            """
        )
        content_layout.addWidget(self.video_label, 0, Qt.AlignTop)
        content_layout.addSpacing(10)
        content_layout.addStretch(1)

        detect_label = QLabel("检测操作")
        detect_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        detect_label.setStyleSheet(f"color: {COLORS['muted']};")
        sidebar_layout.addWidget(detect_label)

        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self.toggle_detection)
        self.start_btn.setMinimumHeight(44)
        self.start_btn.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.start_btn.setStyleSheet(self._button_style(COLORS["focused"], "#27AE60"))
        sidebar_layout.addWidget(self.start_btn)

        action_label = QLabel("页面操作")
        action_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        action_label.setStyleSheet(f"color: {COLORS['muted']};")
        sidebar_layout.addWidget(action_label)

        self.analytics_btn = QPushButton("查看详细分析")
        self.analytics_btn.clicked.connect(self.open_analytics_window)
        self.analytics_btn.setMinimumHeight(40)
        self.analytics_btn.setMinimumWidth(190)
        self.analytics_btn.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.analytics_btn.setStyleSheet(self._button_style("#27AE60", "#219653"))
        sidebar_layout.addWidget(self.analytics_btn)

        self.join_class_btn = QPushButton("加入班级")
        self.join_class_btn.clicked.connect(self.join_class)
        self.join_class_btn.setMinimumHeight(40)
        self.join_class_btn.setMinimumWidth(190)
        self.join_class_btn.setStyleSheet(self._button_style(COLORS["secondary"], "#8E44AD"))
        sidebar_layout.addWidget(self.join_class_btn)

        self.minimize_btn = QPushButton("最小化")
        self.minimize_btn.clicked.connect(self.toggle_minimize)
        self.minimize_btn.setMinimumHeight(40)
        self.minimize_btn.setMinimumWidth(190)
        self.minimize_btn.setStyleSheet(self._button_style(COLORS["primary"], "#2980B9"))
        sidebar_layout.addWidget(self.minimize_btn)

        self.logout_btn = QPushButton("退出登录")
        self.logout_btn.clicked.connect(self.handle_logout)
        self.logout_btn.setMinimumHeight(40)
        self.logout_btn.setMinimumWidth(190)
        self.logout_btn.setStyleSheet(self._button_style(COLORS["danger"], "#C0392B"))
        sidebar_layout.addWidget(self.logout_btn)

        sidebar_layout.addStretch(1)

        outer_layout.addWidget(sidebar_frame, 0)
        outer_layout.addLayout(content_layout, 1)

        self._refresh_class_info()
        self._sync_video_viewport()

        return panel

    def create_assistant_panel(self):
        panel = QFrame()
        panel.setStyleSheet("QFrame { background-color: white; border-radius: 15px; }")
        panel.setMinimumWidth(600)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(18, 20, 18, 20)
        layout.setSpacing(14)

        main_column = QVBoxLayout()
        main_column.setSpacing(12)
        main_column.setContentsMargins(0, 0, 0, 0)

        title_row = QHBoxLayout()
        title = QLabel("小睿助手")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['text']};")
        title_row.addWidget(title)
        title_row.addStretch()
        main_column.addLayout(title_row)

        subtitle = QLabel("支持文本提问、语音控制、学习建议、资料采集，以及图片/文件问答")
        subtitle.setFont(QFont("Microsoft YaHei", 10))
        subtitle.setStyleSheet(f"color: {COLORS['muted']};")
        main_column.addWidget(subtitle)

        model_row = QHBoxLayout()
        model_row.setSpacing(8)

        model_title = QLabel("回答模型")
        model_title.setFont(QFont("Microsoft YaHei", 9, QFont.Medium))
        model_title.setStyleSheet(f"color: {COLORS['muted']};")
        model_row.addWidget(model_title)
        model_row.addStretch()

        self.llm_model_combo = QComboBox()
        self.llm_model_combo.setMinimumHeight(32)
        self.llm_model_combo.setStyleSheet(
            """
            QComboBox {
                background: white;
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 16px;
                padding: 6px 14px;
                color: #2C3E50;
                min-width: 164px;
            }
            QComboBox:hover {
                border: 1px solid #2C3E50;
                color: #2C3E50;
            }
            QComboBox:on {
                border: 1px solid #2C3E50;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 12px;
                padding: 6px;
                background: white;
                selection-background-color: #EDF3FA;
                selection-color: #2C3E50;
            }
            QComboBox QAbstractItemView::item {
                min-height: 28px;
                padding: 6px 12px;
                margin: 2px 0;
                border-radius: 10px;
                color: #2C3E50;
            }
            QComboBox QAbstractItemView::item:hover {
                background: #F7FAFD;
                border: 1px solid #2C3E50;
                color: #2C3E50;
            }
            QComboBox QAbstractItemView::item:selected {
                background: #EDF3FA;
                border: 1px solid #2C3E50;
                color: #2C3E50;
            }
            """
        )
        for label, model_id in SUPPORTED_REMOTE_LLM_MODELS:
            self.llm_model_combo.addItem(label, model_id)
        self._sync_llm_combo_selection()
        self.llm_model_combo.currentIndexChanged.connect(self._handle_llm_model_changed)
        model_row.addWidget(self.llm_model_combo)
        main_column.addLayout(model_row)

        self.model_hint_label = QLabel("")
        self.model_hint_label.setWordWrap(True)
        self.model_hint_label.setFont(QFont("Microsoft YaHei", 8))
        self.model_hint_label.setStyleSheet(f"color: {COLORS['muted']};")
        main_column.addWidget(self.model_hint_label)

        self.voice_status_label = QLabel(self._assistant_idle_text())
        self.voice_status_label.setWordWrap(True)
        self.voice_status_label.setFont(QFont("Microsoft YaHei", 9))
        self.voice_status_label.setStyleSheet(f"color: {COLORS['muted']};")
        main_column.addWidget(self.voice_status_label)

        voice_action_row = QHBoxLayout()
        voice_action_row.setSpacing(8)
        self.voice_btn = QPushButton("启动小睿语音助手")
        self.voice_btn.clicked.connect(self.toggle_voice_assistant)
        self.voice_btn.setMinimumHeight(38)
        self.voice_btn.setStyleSheet(self._button_style(COLORS["secondary"], "#8E44AD"))
        voice_action_row.addWidget(self.voice_btn, 1)

        self.tts_voice_combo = QComboBox()
        self.tts_voice_combo.setToolTip("切换小睿播报声音")
        self.tts_voice_combo.setFixedWidth(112)
        self.tts_voice_combo.setMinimumHeight(38)
        self.tts_voice_combo.setStyleSheet(
            """
            QComboBox {
                background: white;
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 14px;
                padding: 6px 10px;
                color: #2C3E50;
            }
            QComboBox:hover {
                border: 1px solid #2C3E50;
                color: #2C3E50;
            }
            QComboBox:on {
                border: 1px solid #2C3E50;
            }
            QComboBox::drop-down {
                border: none;
                width: 18px;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 12px;
                padding: 6px;
                background: white;
                selection-background-color: #EDF3FA;
                selection-color: #2C3E50;
            }
            """
        )
        for label, voice_type in self._available_tts_voice_options():
            self.tts_voice_combo.addItem(label, voice_type)
        self._sync_tts_voice_combo_selection()
        self.tts_voice_combo.currentIndexChanged.connect(self._handle_tts_voice_changed)
        voice_action_row.addWidget(self.tts_voice_combo, 0)
        main_column.addLayout(voice_action_row)

        self.assistant_chat_view = AttachmentDropTextEdit()
        self.assistant_chat_view.setReadOnly(True)
        self.assistant_chat_view.setMinimumHeight(360)
        self.assistant_chat_view.setMinimumWidth(360)
        self.assistant_chat_view.setFont(QFont("Microsoft YaHei", 10))
        self.assistant_chat_view.setLineWrapMode(QTextEdit.WidgetWidth)
        self.assistant_chat_view.setWordWrapMode(QTextOption.WrapAnywhere)
        self.assistant_chat_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.assistant_chat_view.setPlaceholderText("把图片或文件直接拖到这里，或继续和小睿对话。")
        self.assistant_chat_view.set_base_stylesheet(
            """
            QTextEdit {
                background: white;
                border: 1px solid rgba(44, 62, 80, 0.08);
                border-radius: 12px;
                padding: 10px 18px 10px 10px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 8px 4px 8px 0;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: rgba(44, 62, 80, 0.18);
                border-radius: 3px;
                min-height: 28px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(44, 62, 80, 0.36);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                border: none;
                background: transparent;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }
            """
        )
        self.assistant_chat_view.fileDropped.connect(self.handle_assistant_drop)
        main_column.addWidget(self.assistant_chat_view, 1)

        self.attachment_preview = QFrame()
        self.attachment_preview.setStyleSheet(
            """
            QFrame {
                background: rgba(236, 245, 255, 0.78);
                border: 1px solid rgba(52, 152, 219, 0.18);
                border-radius: 12px;
            }
            """
        )
        preview_layout = QHBoxLayout(self.attachment_preview)
        preview_layout.setContentsMargins(12, 10, 12, 10)
        preview_layout.setSpacing(10)
        self.attachment_thumb = QLabel()
        self.attachment_thumb.setFixedSize(46, 46)
        self.attachment_thumb.setAlignment(Qt.AlignCenter)
        self.attachment_thumb.setStyleSheet(
            "background: rgba(255,255,255,0.94); border-radius: 10px; color: #7F8C8D; font-weight: 700;"
        )
        preview_layout.addWidget(self.attachment_thumb, 0)
        self.attachment_label = QLabel("未附图片/文件")
        self.attachment_label.setWordWrap(True)
        self.attachment_label.setFont(QFont("Microsoft YaHei", 9))
        self.attachment_label.setStyleSheet(f"color: {COLORS['text']};")
        preview_layout.addWidget(self.attachment_label, 1)
        self.clear_attachment_btn = QPushButton("删除")
        self.clear_attachment_btn.setMinimumHeight(30)
        self.clear_attachment_btn.setStyleSheet(
            """
            QPushButton {
                background: white;
                color: #E74C3C;
                border: 1px solid rgba(231, 76, 60, 0.18);
                border-radius: 9px;
                padding: 0 12px;
                font-weight: 600;
            }
            """
        )
        self.clear_attachment_btn.clicked.connect(self.clear_assistant_attachment)
        preview_layout.addWidget(self.clear_attachment_btn, 0)
        main_column.addWidget(self.attachment_preview)

        text_input_row = QHBoxLayout()
        text_input_row.setSpacing(8)

        self.plus_btn = QToolButton()
        self.plus_btn.setText("+")
        self.plus_btn.setCursor(Qt.PointingHandCursor)
        self.plus_btn.setFixedSize(38, 38)
        self.plus_btn.setPopupMode(QToolButton.InstantPopup)
        self.plus_btn.setStyleSheet(
            """
            QToolButton {
                background: white;
                color: #2C3E50;
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 19px;
                font-size: 20px;
                font-weight: 700;
            }
            QToolButton:hover {
                border-color: #2C3E50;
                background: #F7FAFD;
            }
            QToolButton::menu-indicator {
                image: none;
                width: 0px;
            }
            """
        )
        plus_menu = QMenu(self.plus_btn)
        plus_menu.addAction("上传图片", self.pick_assistant_image)
        plus_menu.addAction("上传文件", self.pick_assistant_file)
        self.plus_btn.setMenu(plus_menu)
        text_input_row.addWidget(self.plus_btn, 0)

        self.assistant_input = QLineEdit()
        self.assistant_input.setPlaceholderText("例如：帮我收集高中数学函数的学习资料，或把这个文档整理成学习笔记")
        self.assistant_input.setMinimumHeight(38)
        self.assistant_input.returnPressed.connect(self.submit_text_query)
        self.assistant_input.setStyleSheet(
            """
            QLineEdit {
                background: white;
                border: 1px solid rgba(44, 62, 80, 0.12);
                border-radius: 10px;
                padding: 0 12px;
            }
            """
        )
        text_input_row.addWidget(self.assistant_input, 1)

        self.assistant_send_btn = QPushButton("发送")
        self.assistant_send_btn.setMinimumHeight(38)
        self.assistant_send_btn.clicked.connect(self.submit_text_query)
        self.assistant_send_btn.setStyleSheet(self._button_style(COLORS["primary"], "#2980B9"))
        text_input_row.addWidget(self.assistant_send_btn)
        main_column.addLayout(text_input_row)

        history_frame = QFrame()
        history_frame.setFixedWidth(186)
        history_frame.setStyleSheet(f"QFrame {{ background-color: {COLORS['background']}; border-radius: 14px; }}")
        history_layout = QVBoxLayout(history_frame)
        history_layout.setContentsMargins(12, 14, 12, 14)
        history_layout.setSpacing(10)

        history_title = QLabel("历史对话")
        history_title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        history_title.setStyleSheet(f"color: {COLORS['muted']};")
        history_layout.addWidget(history_title)

        self.new_chat_btn = QPushButton("新对话")
        self.new_chat_btn.setMinimumHeight(34)
        self.new_chat_btn.clicked.connect(self._start_new_assistant_conversation)
        self.new_chat_btn.setStyleSheet(self._button_style(COLORS["primary"], "#2980B9"))
        history_layout.addWidget(self.new_chat_btn)

        self.assistant_conversation_list = QListWidget()
        self.assistant_conversation_list.setMouseTracking(True)
        self.assistant_conversation_list.setStyleSheet(
            """
            QListWidget {
                background: white;
                border: 1px solid rgba(44, 62, 80, 0.08);
                border-radius: 12px;
                padding: 6px;
            }
            QListWidget::item {
                padding: 0;
                margin: 2px 0;
                border: none;
                background: transparent;
            }
            """
        )
        self.assistant_conversation_list.itemSelectionChanged.connect(self._handle_assistant_conversation_selected)
        history_layout.addWidget(self.assistant_conversation_list, 1)

        layout.addLayout(main_column, 1)
        layout.addWidget(history_frame, 0)

        return panel

    def create_right_panel(self):
        panel = QFrame()
        panel.setStyleSheet("QFrame { background-color: white; border-radius: 15px; }")
        panel.setMinimumWidth(420)
        panel.setMinimumHeight(420)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(18)

        title = QLabel("专注度状态")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['text']};")
        layout.addWidget(title)

        score_frame = QFrame()
        score_frame.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['background']}; border-radius: 15px; }}"
        )
        score_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        score_frame.setMinimumHeight(140)
        score_layout = QVBoxLayout(score_frame)
        score_layout.setContentsMargins(20, 20, 20, 20)
        score_layout.setSpacing(10)

        self.score_label = QLabel("--")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(QFont("Arial", 30, QFont.Bold))
        self.score_label.setStyleSheet(f"color: {COLORS['text']};")
        score_layout.addWidget(self.score_label)

        self.status_label = QLabel("未检测")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Microsoft YaHei", 15, QFont.Medium))
        self.status_label.setStyleSheet(f"color: {COLORS['text']};")
        score_layout.addWidget(self.status_label)

        self.backend_status_label = QLabel("算法增强正在预热")
        self.backend_status_label.setAlignment(Qt.AlignCenter)
        self.backend_status_label.setWordWrap(True)
        self.backend_status_label.setFont(QFont("Microsoft YaHei", 11))
        self.backend_status_label.setStyleSheet(f"color: {COLORS['muted']};")
        score_layout.addWidget(self.backend_status_label)

        layout.addWidget(score_frame)

        progress_title = QLabel("专注度进度")
        progress_title.setFont(QFont("Microsoft YaHei", 12))
        progress_title.setStyleSheet(f"color: {COLORS['text']};")
        layout.addWidget(progress_title)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setStyleSheet(self._progress_style(COLORS["moderate"]))
        layout.addWidget(self.progress_bar)

        status_body_layout = QHBoxLayout()
        status_body_layout.setSpacing(14)

        action_frame = QFrame()
        action_frame.setFixedWidth(220)
        action_frame.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['background']}; border-radius: 14px; }}"
        )
        action_layout = QVBoxLayout(action_frame)
        action_layout.setContentsMargins(14, 16, 14, 16)
        action_layout.setSpacing(10)

        action_label = QLabel("状态操作")
        action_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        action_label.setStyleSheet(f"color: {COLORS['muted']};")
        action_layout.addWidget(action_label)

        self.calibration_btn = QPushButton("校准专注姿态")
        self.calibration_btn.clicked.connect(self.start_focus_calibration)
        self.calibration_btn.setMinimumHeight(40)
        self.calibration_btn.setMinimumWidth(190)
        self.calibration_btn.setStyleSheet(self._button_style(COLORS["primary"], "#2980B9"))
        action_layout.addWidget(self.calibration_btn)

        self.clear_calibration_btn = QPushButton("清除校准")
        self.clear_calibration_btn.clicked.connect(self.clear_focus_calibration)
        self.clear_calibration_btn.setMinimumHeight(40)
        self.clear_calibration_btn.setMinimumWidth(190)
        self.clear_calibration_btn.setStyleSheet(self._button_style(COLORS["muted"], "#7F8C8D"))
        action_layout.addWidget(self.clear_calibration_btn)
        action_layout.addStretch(1)

        self.debug_frame = QFrame()
        self.debug_frame.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['background']}; border-radius: 12px; }}"
        )
        self.debug_frame.setMinimumHeight(180)
        self.debug_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        debug_layout = QVBoxLayout(self.debug_frame)
        debug_layout.setContentsMargins(16, 14, 16, 14)
        debug_layout.setSpacing(6)

        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(18)

        summary_items = [
            ("平均专注度", "avg_score_label", "0%"),
            ("累计专注时长", "focus_duration_label", "0s"),
            ("分心次数", "distraction_label", "0"),
        ]
        for index, (label_text, attr_name, initial_value) in enumerate(summary_items):
            item_layout = QVBoxLayout()
            item_layout.setSpacing(4)

            stat_label = self._create_stat_label(label_text)
            stat_label.setStyleSheet(f"color: {COLORS['muted']};")
            item_layout.addWidget(stat_label)

            value_label = QLabel(initial_value)
            value_label.setAlignment(Qt.AlignLeft)
            value_label.setFont(QFont("Arial", 17, QFont.Bold))
            value_label.setStyleSheet(f"color: {COLORS['primary']};")
            item_layout.addWidget(value_label)
            item_layout.addStretch()

            setattr(self, attr_name, value_label)
            summary_layout.addLayout(item_layout, 1)

            if index < len(summary_items) - 1:
                divider = QFrame()
                divider.setFixedWidth(1)
                divider.setStyleSheet("background-color: rgba(44, 62, 80, 0.10); border: none;")
                summary_layout.addWidget(divider)

        debug_layout.addLayout(summary_layout)

        summary_divider = QFrame()
        summary_divider.setFixedHeight(1)
        summary_divider.setStyleSheet("background-color: rgba(44, 62, 80, 0.10); border: none;")
        debug_layout.addWidget(summary_divider)

        debug_title = QLabel("检测调试")
        debug_title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        debug_title.setStyleSheet(f"color: {COLORS['text']};")
        debug_layout.addWidget(debug_title)

        self.debug_metrics_label = QLabel("等待检测启动")
        self.debug_metrics_label.setWordWrap(True)
        self.debug_metrics_label.setMinimumHeight(34)
        self.debug_metrics_label.setFont(QFont("Microsoft YaHei", 9))
        self.debug_metrics_label.setStyleSheet(f"color: {COLORS['text']};")
        debug_layout.addWidget(self.debug_metrics_label)

        self.debug_reason_label = QLabel("这里会显示主要判分原因")
        self.debug_reason_label.setWordWrap(True)
        self.debug_reason_label.setMinimumHeight(24)
        self.debug_reason_label.setFont(QFont("Microsoft YaHei", 9))
        self.debug_reason_label.setStyleSheet(f"color: {COLORS['muted']};")
        debug_layout.addWidget(self.debug_reason_label)

        status_body_layout.addWidget(action_frame, 0)
        status_body_layout.addWidget(self.debug_frame, 1)
        layout.addLayout(status_body_layout, 1)

        return panel

    def _ensure_minimized_assistant_panel(self):
        if self.mini_assistant_panel is None:
            self.mini_assistant_panel = MiniAssistantPanel()
            self.mini_assistant_panel.submitted.connect(self.submit_text_query)
            self.mini_assistant_panel.restore_requested.connect(self.show_normal)
            self.mini_assistant_panel.new_chat_requested.connect(self._start_new_assistant_conversation)
            self.mini_assistant_panel.voice_toggle_requested.connect(self.toggle_voice_assistant)
            self.mini_assistant_panel.upload_image_requested.connect(self.pick_assistant_image)
            self.mini_assistant_panel.upload_file_requested.connect(self.pick_assistant_file)
            self.mini_assistant_panel.clear_attachment_requested.connect(self.clear_assistant_attachment)
            self.mini_assistant_panel.chat_view.fileDropped.connect(self.handle_assistant_drop)
            self.mini_assistant_panel.set_attachment_text(
                self._attachment_summary_text(),
                bool(self.selected_attachment),
                self._supports_attachment_input(),
            )
            self.mini_assistant_panel.set_voice_enabled(self.voice_enabled)
        return self.mini_assistant_panel

    def _toggle_minimized_assistant_panel(self):
        if not self.is_minimized or not self.mini_robot:
            return
        panel = self._ensure_minimized_assistant_panel()
        if panel.isVisible():
            panel.hide()
            return
        if self.voice_hint_bubble:
            self.voice_hint_bubble.hide()
        panel.set_status(self.voice_status_label.text())
        self._refresh_assistant_views()
        panel.show_near(self.mini_robot)

    def _reposition_minimized_overlays(self):
        if not self.is_minimized or not self.mini_robot:
            return
        if self.mini_assistant_panel and self.mini_assistant_panel.isVisible():
            self.mini_assistant_panel.show_near(self.mini_robot)
        if self.voice_hint_bubble and self.voice_hint_bubble.isVisible():
            self.voice_hint_bubble.show_near(
                self.mini_robot,
                self.voice_hint_bubble.label.text(),
                duration_ms=max(1200, self.voice_hint_bubble.hide_timer.remainingTime()),
                level=getattr(self.voice_hint_bubble, "_current_level", "moderate"),
                title=self.voice_hint_bubble.title_label.text() or "小睿播报",
            )

    def _load_assistant_conversations(self, select_conversation_id=None):
        conversations = []
        if hasattr(self.db, "get_assistant_conversations"):
            try:
                conversations = self.db.get_assistant_conversations(self.user_info["id"], limit=40) or []
            except Exception:
                conversations = []
        if not conversations and hasattr(self.db, "create_assistant_conversation"):
            try:
                created = self.db.create_assistant_conversation(self.user_info["id"], "新对话")
                if created:
                    conversations = [created]
            except Exception:
                conversations = []
        self.assistant_conversations = conversations
        target_id = select_conversation_id or self.current_assistant_conversation_id
        if not target_id and conversations:
            target_id = conversations[0].get("id")
        self._refresh_assistant_conversation_list(target_id)
        if target_id:
            self._switch_assistant_conversation(target_id, refresh_list=False)

    def _load_assistant_conversations_async(self, select_conversation_id=None):
        if not hasattr(self.db, "get_assistant_conversations"):
            return
        self.assistant_io_executor.submit(self._load_assistant_conversations_worker, select_conversation_id)

    def _load_assistant_conversations_worker(self, select_conversation_id=None):
        conversations = []
        if hasattr(self.db, "get_assistant_conversations"):
            try:
                conversations = self.db.get_assistant_conversations(self.user_info["id"], limit=40) or []
            except Exception:
                conversations = []
        if not conversations and hasattr(self.db, "create_assistant_conversation"):
            try:
                created = self.db.create_assistant_conversation(self.user_info["id"], "新对话")
                if created:
                    conversations = [created]
            except Exception:
                conversations = []
        self.assistant_conversations_loaded.emit(conversations, select_conversation_id)

    def _handle_assistant_conversations_loaded(self, conversations, select_conversation_id=None):
        self.assistant_conversations = conversations or []
        target_id = select_conversation_id or self.current_assistant_conversation_id
        if not target_id and self.assistant_conversations:
            target_id = self.assistant_conversations[0].get("id")
        self._refresh_assistant_conversation_list(target_id)
        if target_id and not self.current_assistant_conversation_id:
            conversation = next((item for item in self.assistant_conversations if item.get("id") == target_id), None)
            self.current_assistant_conversation_id = target_id
            self.current_assistant_conversation_title = (conversation or {}).get("title") or "新对话"
            self.assistant_messages = []
            self.assistant_pending_text = "正在加载历史对话..."
            self._refresh_assistant_views()
            self._load_assistant_messages_async(target_id)

    def _refresh_assistant_conversation_list(self, selected_id=None):
        if not hasattr(self, "assistant_conversation_list"):
            return
        self.assistant_conversation_list.blockSignals(True)
        self.assistant_conversation_list.clear()
        for conversation in self.assistant_conversations:
            title = (conversation.get("title") or "新对话").strip() or "新对话"
            conversation_id = conversation.get("id")
            item = QListWidgetItem()
            item.setData(Qt.UserRole, conversation_id)
            item.setSizeHint(QSize(0, 38))
            preview = (conversation.get("preview") or "").strip()
            self.assistant_conversation_list.addItem(item)
            row_widget = ConversationListItemWidget(title[:18], preview or title)
            row_widget.clicked.connect(
                lambda _checked=False, it=item, cid=conversation_id: self._select_assistant_conversation_item(it, cid)
            )
            row_widget.delete_requested.connect(
                lambda _checked=False, cid=conversation_id, ctitle=title: self._delete_current_assistant_conversation(cid, ctitle)
            )
            self.assistant_conversation_list.setItemWidget(item, row_widget)
            if selected_id and conversation_id == selected_id:
                self.assistant_conversation_list.setCurrentItem(item)
        self.assistant_conversation_list.blockSignals(False)
        self._sync_assistant_conversation_row_states()

    def _sync_assistant_conversation_row_states(self):
        if not hasattr(self, "assistant_conversation_list"):
            return
        current_item = self.assistant_conversation_list.currentItem()
        for index in range(self.assistant_conversation_list.count()):
            item = self.assistant_conversation_list.item(index)
            widget = self.assistant_conversation_list.itemWidget(item)
            if widget:
                widget.set_selected(item == current_item)

    def _select_assistant_conversation_item(self, item, conversation_id):
        if hasattr(self, "assistant_conversation_list") and item:
            self.assistant_conversation_list.setCurrentItem(item)
        self._switch_assistant_conversation_async(conversation_id, refresh_list=False)

    def _switch_assistant_conversation_async(self, conversation_id, refresh_list=True):
        if not conversation_id:
            return
        self.current_assistant_conversation_id = conversation_id
        conversation = next((item for item in self.assistant_conversations if item.get("id") == conversation_id), None)
        self.current_assistant_conversation_title = (conversation or {}).get("title") or "新对话"
        self.assistant_messages = []
        self.assistant_pending_text = "正在加载历史对话..."
        if refresh_list:
            self._refresh_assistant_conversation_list(conversation_id)
        else:
            self._sync_assistant_conversation_row_states()
        self._refresh_assistant_views()
        self._load_assistant_messages_async(conversation_id)

    def _load_assistant_messages_async(self, conversation_id):
        self.assistant_io_executor.submit(self._load_assistant_messages_worker, conversation_id)

    def _load_assistant_messages_worker(self, conversation_id):
        messages = []
        if hasattr(self.db, "get_assistant_conversation_messages"):
            try:
                messages = self.db.get_assistant_conversation_messages(self.user_info["id"], conversation_id, limit=120) or []
            except Exception:
                messages = []
        self.assistant_messages_loaded.emit(conversation_id, messages)

    def _handle_assistant_messages_loaded(self, conversation_id, messages):
        if conversation_id != self.current_assistant_conversation_id:
            return
        rebuilt = []
        for row in messages or []:
            query_text = (row.get("query_text") or "").strip()
            reply_text = (row.get("reply_text") or "").strip()
            if query_text:
                rebuilt.append(("user", query_text))
            if reply_text:
                rebuilt.append(("assistant", reply_text))
        self.assistant_messages = rebuilt[-80:]
        self.assistant_pending_text = ""
        self._refresh_assistant_views()

    def _assistant_preview_from_messages(self, messages=None):
        rows = messages if messages is not None else self.assistant_messages
        for role, text in reversed(rows or []):
            normalized = (text or "").strip()
            if normalized:
                return normalized[:24]
        return ""

    def _assistant_title_from_messages(self, messages=None):
        rows = messages if messages is not None else self.assistant_messages
        for role, text in rows or []:
            if role == "user" and (text or "").strip():
                return (text or "").strip()[:24]
        preview = self._assistant_preview_from_messages(rows)
        return preview or "新对话"

    def _remember_current_conversation_in_sidebar(self):
        conversation_id = self.current_assistant_conversation_id
        if not conversation_id or not self.assistant_messages:
            return
        title = (self.current_assistant_conversation_title or "").strip()
        if not title or title == "新对话":
            title = self._assistant_title_from_messages()
            self.current_assistant_conversation_title = title
        preview = self._assistant_preview_from_messages()
        existing = next((item for item in self.assistant_conversations if item.get("id") == conversation_id), None)
        if existing:
            existing["title"] = title
            existing["preview"] = preview
        else:
            self.assistant_conversations.insert(
                0,
                {
                    "id": conversation_id,
                    "title": title,
                    "preview": preview,
                },
            )

    def _switch_assistant_conversation(self, conversation_id, refresh_list=True):
        if not conversation_id:
            return
        messages = []
        if hasattr(self.db, "get_assistant_conversation_messages"):
            try:
                messages = self.db.get_assistant_conversation_messages(self.user_info["id"], conversation_id, limit=120) or []
            except Exception:
                messages = []
        self.current_assistant_conversation_id = conversation_id
        conversation = next((item for item in self.assistant_conversations if item.get("id") == conversation_id), None)
        self.current_assistant_conversation_title = (conversation or {}).get("title") or "新对话"
        rebuilt = []
        for row in messages:
            query_text = (row.get("query_text") or "").strip()
            reply_text = (row.get("reply_text") or "").strip()
            if query_text:
                rebuilt.append(("user", query_text))
            if reply_text:
                rebuilt.append(("assistant", reply_text))
        self.assistant_messages = rebuilt[-80:]
        self.assistant_pending_text = ""
        if refresh_list:
            self._refresh_assistant_conversation_list(conversation_id)
        else:
            self._sync_assistant_conversation_row_states()
        self._refresh_assistant_views()

    def _start_new_assistant_conversation(self):
        self._remember_current_conversation_in_sidebar()
        if not hasattr(self.db, "create_assistant_conversation"):
            self.assistant_messages = []
            self.assistant_pending_text = ""
            self.pending_assistant_log = None
            self._refresh_assistant_views()
            return
        try:
            conversation = self.db.create_assistant_conversation(self.user_info["id"], "新对话")
        except Exception as exc:
            self._set_assistant_status(f"新建对话失败：{exc}")
            return
        self._load_assistant_conversations(select_conversation_id=(conversation or {}).get("id"))
        self.assistant_messages = []
        self.assistant_pending_text = ""
        self.pending_assistant_log = None
        self._refresh_assistant_views()
        self._set_assistant_status("已开启新对话。")

    def _delete_current_assistant_conversation(self, conversation_id=None, title=None):
        conversation_id = conversation_id or self.current_assistant_conversation_id
        if not conversation_id or not hasattr(self.db, "delete_assistant_conversation"):
            return
        title = title or self.current_assistant_conversation_title or "当前对话"
        confirmed = QMessageBox.question(
            self._assistant_dialog_parent(),
            "删除对话",
            f"要彻底删除“{title}”吗？删除后无法恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmed != QMessageBox.Yes:
            return
        try:
            deleted = self.db.delete_assistant_conversation(self.user_info["id"], conversation_id)
        except Exception as exc:
            self._set_assistant_status(f"删除对话失败：{exc}")
            return
        if not deleted:
            self._set_assistant_status("这段对话删除失败，请稍后再试。")
            return
        self.current_assistant_conversation_id = None
        self.current_assistant_conversation_title = "新对话"
        self.assistant_messages = []
        self.assistant_pending_text = ""
        self.pending_assistant_log = None
        self._load_assistant_conversations()
        self._set_assistant_status("已彻底删除这段对话。")

    def _handle_assistant_conversation_selected(self):
        if not hasattr(self, "assistant_conversation_list"):
            return
        item = self.assistant_conversation_list.currentItem()
        self._sync_assistant_conversation_row_states()
        if not item:
            return
        conversation_id = item.data(Qt.UserRole)
        if conversation_id and conversation_id != self.current_assistant_conversation_id:
            self._switch_assistant_conversation_async(conversation_id)

    def _assistant_history_html(self):
        fragments = []
        if not self.assistant_messages and not self.assistant_pending_text:
            return (
                "<div style='color:#95A5A6;font-size:12px;line-height:1.7;'>"
                "你可以直接输入：介绍一下自己、当前专注度是多少、打开详细数据页面、开始检测。"
                "</div>"
            )

        for role, text in self.assistant_messages[-24:]:
            bg = "#ECF5FF" if role == "user" else "#F8F9FB"
            title = "你" if role == "user" else "小睿"
            title_color = COLORS["primary"] if role == "user" else COLORS["text"]
            safe_text = html.escape(text).replace("\n", "<br>")
            fragments.append(
                "<div style='margin-bottom:8px;'>"
                f"<div style='font-size:11px;color:{title_color};font-weight:600;margin-bottom:3px;'>{title}</div>"
                f"<div style='background:{bg};border-radius:10px;padding:9px 11px;font-size:12px;line-height:1.75;color:{COLORS['text']};"
                f"white-space:normal;word-break:break-word;'>{safe_text}</div>"
                "</div>"
            )
        if self.assistant_pending_text:
            fragments.append(
                "<div style='margin-bottom:8px;'>"
                f"<div style='font-size:11px;color:{COLORS['text']};font-weight:600;margin-bottom:3px;'>小睿</div>"
                "<div style='background:#F8F9FB;border-radius:10px;padding:9px 11px;font-size:12px;line-height:1.75;"
                f"color:{COLORS['muted']};font-style:italic;'>{html.escape(self.assistant_pending_text)}</div>"
                "</div>"
            )
        return "".join(fragments)

    def _refresh_assistant_views(self):
        content = self._assistant_history_html()
        if hasattr(self, "assistant_chat_view"):
            self.assistant_chat_view.setHtml(content)
            self.assistant_chat_view.verticalScrollBar().setValue(self.assistant_chat_view.verticalScrollBar().maximum())
        if self.mini_assistant_panel:
            self.mini_assistant_panel.set_chat_html(content)

    def _append_assistant_message(self, role, text):
        text = self._normalize_formula_text((text or "").strip())
        if not text:
            return
        if role == "assistant":
            self.assistant_pending_text = ""
        self.assistant_messages.append((role, text))
        if len(self.assistant_messages) > 40:
            self.assistant_messages = self.assistant_messages[-40:]
        self._refresh_assistant_views()

    def _normalize_formula_text(self, text):
        text = (text or "").strip()
        if not text:
            return ""

        replacements = {
            "\\times": "×",
            "\\cdot": "·",
            "\\leq": "≤",
            "\\geq": "≥",
            "\\neq": "≠",
            "\\approx": "≈",
            "\\pm": "±",
            "\\infty": "∞",
            "\\left": "",
            "\\right": "",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)

        text = re.sub(r"\$\$(.*?)\$\$", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"\\\[(.*?)\\\]", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"\\\((.*?)\\\)", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"\$(.*?)\$", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
        text = re.sub(r"([A-Za-z0-9])\^\{([^{}]+)\}", r"\1^\2", text)
        text = re.sub(r"([A-Za-z0-9])_\{([^{}]+)\}", r"\1_\2", text)
        text = text.replace("{", "").replace("}", "")
        text = text.replace("\\\\", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _stage_assistant_log(self, query_text, source):
        query_text = (query_text or "").strip()
        if not query_text:
            self.pending_assistant_log = None
            return
        if not self.current_assistant_conversation_id and hasattr(self.db, "create_assistant_conversation"):
            try:
                conversation = self.db.create_assistant_conversation(self.user_info["id"], "新对话")
                if conversation:
                    self.current_assistant_conversation_id = conversation.get("id")
                    self.current_assistant_conversation_title = conversation.get("title") or "新对话"
                    self._load_assistant_conversations(select_conversation_id=self.current_assistant_conversation_id)
            except Exception:
                pass
        self.pending_assistant_log = {
            "query_text": query_text,
            "source": source or "text",
            "model_name": self._current_llm_label() if self.voice_llm_controller.available else "内置指令",
            "conversation_id": self.current_assistant_conversation_id,
        }

    def _commit_assistant_log(self, reply_text):
        pending = self.pending_assistant_log or {}
        query_text = (pending.get("query_text") or "").strip()
        if not query_text:
            self.pending_assistant_log = None
            return
        if hasattr(self.db, "save_assistant_query_log"):
            try:
                self.db.save_assistant_query_log(
                    self.user_info["id"],
                    query_text,
                    (reply_text or "").strip(),
                    source=pending.get("source") or "text",
                    model_name=pending.get("model_name") or "",
                    conversation_id=pending.get("conversation_id"),
                )
            except Exception:
                pass
        self._load_assistant_conversations(select_conversation_id=pending.get("conversation_id") or self.current_assistant_conversation_id)
        self.pending_assistant_log = None

    def _clear_pending_assistant_log(self):
        self.pending_assistant_log = None

    def _set_assistant_status(self, text):
        if hasattr(self, "voice_status_label"):
            self.voice_status_label.setText(text)
        if self.mini_assistant_panel:
            self.mini_assistant_panel.set_status(text)

    def _create_stat_label(self, text):
        label = QLabel(text)
        label.setFont(QFont("Microsoft YaHei", 10))
        label.setStyleSheet(f"color: {COLORS['text']};")
        return label

    def _set_presence(self, online):
        if not hasattr(self.db, "update_presence"):
            return
        if self.presence_inflight and online:
            return
        self.presence_inflight = True
        self.presence_executor.submit(self._run_presence_update, bool(online))

    def _run_presence_update(self, online):
        try:
            self.db.update_presence(self.user_info["id"], online=online)
        finally:
            self.presence_inflight = False

    def _heartbeat_presence(self):
        self._set_presence(True)

    def _refresh_class_info(self):
        class_name = self.user_info.get("class_name") or "暂未加入班级"
        self.class_info_label.setText(f"当前班级：{class_name}")

    def _assistant_dialog_parent(self):
        if self.is_minimized and self.mini_assistant_panel and self.mini_assistant_panel.isVisible():
            return self.mini_assistant_panel
        return self

    def _sync_video_viewport(self):
        if not hasattr(self, "video_label"):
            return
        width = max(self.video_label.width(), self.video_label.minimumWidth())
        target_height = int(width / self.VIDEO_ASPECT_RATIO)
        target_height = max(220, min(290, target_height))
        if self.video_label.height() != target_height:
            self.video_label.setFixedHeight(target_height)

    def _available_llm_specs(self):
        return [model_id for _, model_id in SUPPORTED_REMOTE_LLM_MODELS]

    def _split_llm_spec(self, spec):
        text = (spec or "").strip().lower()
        if ":" in text:
            provider, model_name = text.split(":", 1)
            return (provider or "deepseek", model_name or "deepseek-chat")
        if text.startswith("doubao-"):
            return ("doubao", text)
        return ("deepseek", text or "deepseek-chat")

    def _provider_label_for_spec(self, spec):
        provider, _model_name = self._split_llm_spec(spec)
        if provider == "doubao":
            return "Doubao"
        return "DeepSeek"

    def _base_url_for_provider(self, provider):
        provider = (provider or "").strip().lower()
        if provider == LOCAL_LLM_PROVIDER:
            return LOCAL_LLM_BASE_URL
        if provider == "doubao":
            return DOUBAO_API_BASE_URL
        return DEEPSEEK_API_BASE_URL

    def _api_key_for_provider(self, provider):
        provider = (provider or "").strip().lower()
        if provider == LOCAL_LLM_PROVIDER and LOCAL_LLM_API_KEY:
            return LOCAL_LLM_API_KEY
        if provider == "doubao":
            return DOUBAO_LLM_API_KEY
        return DEEPSEEK_LLM_API_KEY

    def _normalized_remote_llm_model(self, model_name):
        normalized = (model_name or "").strip().lower()
        valid_models = set(self._available_llm_specs())
        if normalized in valid_models:
            return normalized
        provider, bare_model = self._split_llm_spec(normalized)
        fallback = f"{provider}:{bare_model}"
        return fallback if fallback in valid_models else self.DEFAULT_REMOTE_LLM_MODEL

    def _is_reasoning_model(self, model_name=None):
        current = self._normalized_remote_llm_model(model_name or self.selected_llm_model)
        return current in set(self.PROVIDER_REASONING_MODELS.values())

    def _current_llm_label(self):
        current = self._normalized_remote_llm_model(self.selected_llm_model)
        for label, model_id in SUPPORTED_REMOTE_LLM_MODELS:
            if model_id == current:
                return label
        return "Doubao Vision"

    def _current_model_hint(self):
        current = self._normalized_remote_llm_model(self.selected_llm_model)
        if current == "doubao:doubao-seedream-5-0-lite-260128":
            return "Doubao Image Generation 5.0 Lite：更适合生成思维导图、流程图、架构图等图片。"
        if current == "doubao:doubao-seed-1-6-vision-250815":
            return "Doubao Vision：更适合图片理解；文件/代码仍会优先走稳的文本模型。"
        if current == "deepseek:deepseek-reasoner":
            return "DeepSeek Reasoner：适合更深入的推理问题。"
        return "DeepSeek Chat：适合日常文本问答；上传附件时会自动选择更合适的模型。"

    def _supports_attachment_input(self):
        return True

    def _assistant_idle_text(self):
        label = self._current_llm_label()
        provider_label = self._provider_label_for_spec(self.selected_llm_model)
        if self.voice_llm_controller.available:
            suffix = "当前为推理模型，复杂问题会回答得更充分。" if self._is_reasoning_model() else "可语音唤醒，也可以直接输入文字和小睿对话。"
            return f"当前问答模型：{label}（{provider_label}）。{suffix}"
        return f"当前问答模型：{label}（{provider_label}）。复杂问答需配置 API Key，固定指令仍可直接使用。"

    def _sync_llm_combo_selection(self):
        if not hasattr(self, "llm_model_combo"):
            return
        selected = self._normalized_remote_llm_model(self.selected_llm_model)
        index = self.llm_model_combo.findData(selected)
        if index >= 0:
            self.llm_model_combo.blockSignals(True)
            self.llm_model_combo.setCurrentIndex(index)
            self.llm_model_combo.blockSignals(False)
        self._sync_reasoning_button()

    def _sync_reasoning_button(self):
        if self.mini_assistant_panel:
            self.mini_assistant_panel.set_status(self.voice_status_label.text())

    def _available_tts_voice_options(self):
        options = list(SUPPORTED_DOUBAO_TTS_VOICES)
        known_voice_types = {voice_type for _label, voice_type in options}
        current_voice = (self.selected_tts_voice_type or "").strip()
        if current_voice and current_voice not in known_voice_types:
            options.insert(0, ("当前配置音色", current_voice))
        return options

    def _current_tts_voice_label(self):
        for label, voice_type in self._available_tts_voice_options():
            if voice_type == self.selected_tts_voice_type:
                return label
        return "当前配置音色"

    def _sync_tts_voice_combo_selection(self):
        if hasattr(self, "tts_voice_combo"):
            index = self.tts_voice_combo.findData(self.selected_tts_voice_type)
            if index >= 0:
                self.tts_voice_combo.blockSignals(True)
                self.tts_voice_combo.setCurrentIndex(index)
                self.tts_voice_combo.blockSignals(False)

    def _handle_tts_voice_changed(self, _index=None):
        if not hasattr(self, "tts_voice_combo"):
            return
        voice_type = self.tts_voice_combo.currentData()
        if voice_type:
            self._set_tts_voice_type(voice_type, announce=True)

    def _set_tts_voice_type(self, voice_type, announce=False):
        voice_type = (voice_type or "").strip()
        if not voice_type:
            return
        self.selected_tts_voice_type = voice_type
        self.voice_speaker.set_voice_type(voice_type)
        self._sync_tts_voice_combo_selection()
        if announce:
            label = self._current_tts_voice_label()
            self._set_assistant_status(f"已切换小睿播报声音为：{label}。下一句回复会使用新声音。")

    def _attachment_summary_text(self):
        attachment = self.selected_attachment or {}
        if not attachment:
            return "未附图片/文件"
        kind_label = "图片" if attachment.get("kind") == "image" else "文件"
        return f"已附{kind_label}：{attachment.get('name') or '未命名内容'}"

    def _attachment_preview_payload(self):
        attachment = self.selected_attachment or {}
        if not attachment:
            return None, ""
        if attachment.get("kind") == "image":
            pixmap = QPixmap(attachment.get("path", ""))
            if not pixmap.isNull():
                return pixmap.scaled(46, 46, Qt.KeepAspectRatio, Qt.SmoothTransformation), ""
            return None, "IMG"
        return None, "FILE"

    def _refresh_attachment_views(self):
        text = self._attachment_summary_text()
        has_attachment = bool(self.selected_attachment)
        picker_visible = True
        pixmap, badge_text = self._attachment_preview_payload()
        if hasattr(self, "attachment_label"):
            self.attachment_label.setText(text)
        if hasattr(self, "attachment_preview"):
            self.attachment_preview.setVisible(has_attachment)
        if hasattr(self, "attachment_thumb"):
            if pixmap and not pixmap.isNull():
                self.attachment_thumb.setPixmap(pixmap)
                self.attachment_thumb.setText("")
            else:
                self.attachment_thumb.setPixmap(QPixmap())
                self.attachment_thumb.setText(badge_text)
        if hasattr(self, "clear_attachment_btn"):
            self.clear_attachment_btn.setEnabled(has_attachment)
        if hasattr(self, "plus_btn"):
            self.plus_btn.setVisible(picker_visible)
        if self.mini_assistant_panel:
            self.mini_assistant_panel.set_attachment_text(text, has_attachment, picker_visible, pixmap, badge_text)
        if hasattr(self, "model_hint_label"):
            self.model_hint_label.setText(self._current_model_hint())

    def pick_assistant_image(self):
        dialog_parent = None
        if self.is_minimized and self.mini_assistant_panel and self.mini_assistant_panel.isVisible():
            dialog_parent = self.mini_assistant_panel
        elif self.isVisible():
            dialog_parent = self
        path, _ = QFileDialog.getOpenFileName(
            dialog_parent,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)",
        )
        if not path:
            return
        self._set_assistant_attachment(path, "image")

    def pick_assistant_file(self):
        dialog_parent = None
        if self.is_minimized and self.mini_assistant_panel and self.mini_assistant_panel.isVisible():
            dialog_parent = self.mini_assistant_panel
        elif self.isVisible():
            dialog_parent = self
        path, _ = QFileDialog.getOpenFileName(
            dialog_parent,
            "选择文件",
            "",
            "支持的文件 (*.txt *.md *.markdown *.csv *.json *.log *.py *.js *.jsx *.ts *.tsx *.java *.c *.cc *.cpp *.cxx *.h *.hpp *.go *.rs *.php *.rb *.swift *.kt *.kts *.scala *.sql *.sh *.bash *.zsh *.html *.htm *.css *.scss *.xml *.toml *.yml *.yaml *.ini *.cfg *.docx *.pdf *.xlsx *.pptx);;所有文件 (*)",
        )
        if not path:
            return
        self._set_assistant_attachment(path, "file")

    def _set_assistant_attachment(self, path, kind):
        path = (path or "").strip()
        if not path or not os.path.exists(path):
            return
        self.selected_attachment = {
            "path": path,
            "kind": "image" if kind == "image" else "file",
            "name": os.path.basename(path),
        }
        self._refresh_attachment_views()
        label = "图片" if self.selected_attachment["kind"] == "image" else "文件"
        if self.selected_attachment["kind"] == "image" and DOUBAO_LLM_API_KEY:
            self._set_assistant_status(f"已附{label}《{self.selected_attachment['name']}》，发送时会自动优先用 Doubao Vision 分析。")
        elif self.selected_attachment["kind"] == "file" and DEEPSEEK_LLM_API_KEY:
            self._set_assistant_status(f"已附{label}《{self.selected_attachment['name']}》，发送时会自动优先用 DeepSeek Chat 解读。")
        else:
            self._set_assistant_status(f"已附{label}《{self.selected_attachment['name']}》，现在可以直接提问。")

    def handle_assistant_drop(self, path):
        path = (path or "").strip()
        if not path or not os.path.isfile(path):
            self._set_assistant_status("拖入的内容不存在，请重新试一次。")
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in self.IMAGE_ATTACHMENT_EXTENSIONS:
            self._set_assistant_attachment(path, "image")
            return
        if ext in self.DOCUMENT_ATTACHMENT_EXTENSIONS:
            self._set_assistant_attachment(path, "file")
            return
        self._set_assistant_status("当前支持拖入常见图片、代码文件，以及 docx/pdf/xlsx/pptx。")

    def clear_assistant_attachment(self, silent=False):
        self.selected_attachment = None
        self._refresh_attachment_views()
        if not silent:
            self._set_assistant_status("已清除当前上传内容。")

    def _rebuild_voice_llm_controller(self, model_name=None, announce=False):
        selected = self._normalized_remote_llm_model(model_name or self.selected_llm_model)
        provider, bare_model = self._split_llm_spec(selected)
        old_controller = getattr(self, "voice_llm_controller", None)
        self.selected_llm_model = selected
        self.voice_llm_controller = VoiceLLMController(
            local_enabled=True,
            local_provider=provider,
            local_base_url=self._base_url_for_provider(provider),
            local_model=bare_model,
            local_api_key=self._api_key_for_provider(provider),
        )
        if old_controller and old_controller is not self.voice_llm_controller:
            try:
                old_controller.shutdown()
            except Exception:
                pass
        self._sync_llm_combo_selection()
        self._refresh_attachment_views()
        if announce:
            if self._is_reasoning_model():
                self._set_assistant_status(f"已切换到推理模型 {self._current_llm_label()}，更适合回答复杂问题。")
            else:
                self._set_assistant_status(f"已切换回答模型到{self._current_llm_label()}。你可以继续语音或文字提问。")
        elif hasattr(self, "voice_status_label") and not self.voice_enabled:
            self._set_assistant_status(self._assistant_idle_text())

    def _handle_llm_model_changed(self, _index):
        if not hasattr(self, "llm_model_combo"):
            return
        model_name = self.llm_model_combo.currentData()
        if not model_name:
            return
        self._rebuild_voice_llm_controller(model_name=model_name, announce=True)

    def _detect_diagram_image_task(self, command, attachment=None, urls=None, save_mode="dialog"):
        command = (command or "").strip()
        if not command:
            return None

        trigger_tokens = ("生成", "绘制", "画", "出图", "做一个", "做一张", "整理成", "转成")
        if not any(token in command for token in trigger_tokens):
            return None

        diagram_kind = ""
        diagram_label = ""
        for keyword, kind, label in self.DIAGRAM_IMAGE_KEYWORDS:
            if keyword in command:
                diagram_kind = kind
                diagram_label = label
                break
        if not diagram_kind:
            return None

        if attachment:
            source_kind = "attachment"
            source_value = dict(attachment)
        elif urls:
            source_kind = "url"
            source_value = urls[0]
        elif is_current_tabs_command(command):
            source_kind = "current_tabs"
            source_value = None
        elif is_current_page_command(command) or any(token in command for token in ("当前内容", "当前图片", "当前文档", "当前文件")):
            source_kind = "current_page"
            source_value = None
        else:
            source_kind = "prompt_only"
            source_value = None

        return {
            "kind": "generate_diagram_image",
            "diagram_kind": diagram_kind,
            "diagram_label": diagram_label,
            "save_mode": save_mode,
            "source_kind": source_kind,
            "source_value": source_value,
        }

    def _detect_study_task(self, command, attachment=None):
        command = (command or "").strip()
        if not command:
            return None

        save_mode = infer_save_mode(command)
        urls = extract_urls(command)
        diagram_task = self._detect_diagram_image_task(command, attachment=attachment, urls=urls, save_mode=save_mode)
        if diagram_task:
            return diagram_task

        if detect_web_search_command(command) and not self._is_study_generation_request(command):
            return None

        if is_material_collection_command(command):
            topic = extract_topic(command)
            if not topic:
                return {
                    "kind": "collect_materials",
                    "error": "我知道你想采集学习资料，但还缺少明确主题。你可以说“帮我收集高中数学函数的学习资料并保存到桌面”。",
                }
            return {
                "kind": "collect_materials",
                "topic": topic,
                "save_mode": save_mode,
                "output_format": infer_output_format(command),
                "source_urls": urls,
            }

        outputs = infer_learning_outputs(command)
        if outputs:
            source_kind = None
            source_value = None
            if attachment:
                source_kind = "attachment"
                source_value = dict(attachment)
            elif urls:
                source_kind = "url"
                source_value = urls[0]
            elif is_current_tabs_command(command):
                source_kind = "current_tabs"
                source_value = None
            elif is_current_page_command(command):
                source_kind = "current_page"
                source_value = None
            else:
                source_kind = "missing"

            return {
                "kind": "process_learning_content",
                "outputs": outputs,
                "save_mode": save_mode,
                "output_format": infer_output_format(command),
                "source_kind": source_kind,
                "source_value": source_value,
            }

        return None

    def _is_study_generation_request(self, command):
        command = command or ""
        return any(
            token in command
            for token in (
                "保存",
                "桌面",
                "word",
                "docx",
                "文档",
                "资料包",
                "整理成",
                "整理为",
                "生成",
                "收集",
                "搜集",
                "采集",
                "抓取",
            )
        )

    def _prepare_study_save_plan(self, task):
        kind = task.get("kind")
        save_mode = task.get("save_mode", "dialog")
        title_hint = task.get("topic") or task.get("title_hint") or (task.get("diagram_label") if kind == "generate_diagram_image" else "学习资料")
        output_format = task.get("output_format") or "md"
        outputs = task.get("outputs", [])
        if kind == "generate_diagram_image":
            diagram_label = task.get("diagram_label") or "示意图"
            default_name = build_default_study_filename(diagram_label, title_hint, "png")
            if save_mode == "desktop":
                file_path = os.path.join(desktop_dir(), default_name)
            else:
                file_path, _ = QFileDialog.getSaveFileName(
                    self._assistant_dialog_parent(),
                    f"保存{diagram_label}",
                    os.path.join(desktop_dir(), default_name),
                    "PNG 图片 (*.png)",
                )
            if not file_path:
                return None
            return {"mode": "file", "main_path": file_path}
        mindmap_only = outputs == ["mindmap"]
        effective_output_format = "png" if mindmap_only else output_format
        if effective_output_format == "png":
            file_filter = "PNG 图片 (*.png)"
        elif effective_output_format == "xlsx":
            file_filter = "Excel 表格 (*.xlsx)"
        elif effective_output_format == "docx":
            file_filter = "Word 文档 (*.docx)"
        else:
            file_filter = "Markdown 文档 (*.md)"

        if kind == "collect_materials":
            output_prefix = "学习资料表" if effective_output_format == "xlsx" else "学习资料包"
            default_name = build_default_study_filename(output_prefix, title_hint, effective_output_format)
            if save_mode == "desktop":
                file_path = os.path.join(desktop_dir(), default_name)
            else:
                file_path, _ = QFileDialog.getSaveFileName(
                    self._assistant_dialog_parent(),
                    "保存学习资料表" if effective_output_format == "xlsx" else "保存学习资料包",
                    os.path.join(desktop_dir(), default_name),
                    file_filter,
                )
            if not file_path:
                return None
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            return {
                "mode": "file",
                "main_path": file_path,
                "sources_path": os.path.join(
                    os.path.dirname(file_path),
                    f"{base_name}_来源.txt",
                ),
                "mindmap_path": os.path.join(
                    os.path.dirname(file_path),
                    f"{base_name}_思维导图.png",
                ),
            }

        output_label = "学习笔记"
        if outputs == ["table"]:
            output_label = "学习表格"
        elif "mindmap" in outputs:
            output_label = "思维导图"
        elif "outline" in outputs and "notes" not in outputs:
            output_label = "复习提纲"
        elif len(outputs) > 1:
            output_label = "学习处理结果"

        default_name = build_default_study_filename(output_label, title_hint, effective_output_format)
        if save_mode == "desktop":
            file_path = os.path.join(desktop_dir(), default_name)
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self._assistant_dialog_parent(),
                "保存学习文档",
                os.path.join(desktop_dir(), default_name),
                file_filter,
            )
        if not file_path:
            return None
        save_plan = {"mode": "file", "main_path": file_path, "main_is_mindmap": mindmap_only}
        if "mindmap" in outputs:
            if mindmap_only:
                save_plan["mindmap_path"] = file_path
            else:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                save_plan["mindmap_path"] = os.path.join(
                    os.path.dirname(file_path),
                    f"{base_name}_思维导图.png",
                )
        return save_plan

    def _study_task_pending_text(self, task):
        if task.get("kind") == "generate_diagram_image":
            return f"正在生成{task.get('diagram_label') or '示意图'}..."
        if task.get("kind") == "collect_materials":
            return "正在采集并整理学习资料..."
        outputs = task.get("outputs", [])
        if outputs == ["table"]:
            return "正在整理 Excel 表格..."
        if outputs == ["notes"]:
            return "正在整理学习笔记..."
        if outputs == ["outline"]:
            return "正在生成复习提纲..."
        if outputs == ["mindmap"]:
            return "正在生成思维导图..."
        return "正在处理学习内容..."

    def _is_current_content_query(self, command):
        command = (command or "").strip()
        if not command:
            return False
        targets = (
            "当前图片",
            "这张图片",
            "当前文档",
            "当前文件",
            "这个文档",
            "这个文件",
            "当前页面",
            "当前网页",
            "这个网页",
            "本页面",
            "当前内容",
        )
        actions = (
            "解释",
            "讲解",
            "说明",
            "总结",
            "概括",
            "分析",
            "看看",
            "看一下",
            "识别",
            "讲讲",
            "介绍",
            "内容",
            "是什么",
            "有什么",
        )
        return any(token in command for token in targets) and any(token in command for token in actions)

    def _describe_current_source(self, current_source):
        current_source = dict(current_source or {})
        source_kind = str(current_source.get("kind") or "").strip().lower()
        if source_kind == "attachment":
            attachment = dict(current_source.get("value") or {})
            if self._is_screen_capture_attachment(attachment):
                return "当前屏幕内容"
            kind_label = "图片" if attachment.get("kind") == "image" else "文件"
            name = attachment.get("name") or current_source.get("title") or "当前内容"
            return f"当前{kind_label}《{name}》"
        return current_source.get("title") or "当前页面"

    def _current_source_pending_text(self, current_source):
        current_source = dict(current_source or {})
        source_kind = str(current_source.get("kind") or "").strip().lower()
        if source_kind == "attachment":
            attachment = dict(current_source.get("value") or {})
            if self._is_screen_capture_attachment(attachment):
                return "正在分析当前屏幕内容..."
            if attachment.get("kind") == "image":
                return "正在分析当前图片..."
            return "正在解读当前文档..."
        return "正在总结当前页面..."

    def _current_source_status_text(self, current_source):
        current_source = dict(current_source or {})
        source_kind = str(current_source.get("kind") or "").strip().lower()
        source_label = self._describe_current_source(current_source)
        if source_kind == "attachment":
            attachment = dict(current_source.get("value") or {})
            if self._is_screen_capture_attachment(attachment):
                return "已获取当前屏幕内容，小睿正在用视觉模型理解..."
            if attachment.get("kind") == "image":
                return f"已读取{source_label}，小睿正在识别图片内容..."
            return f"已读取{source_label}，小睿正在整理文档内容..."
        return f"已读取{source_label}，小睿正在提取网页要点..."

    def _confirm_online_web_access(self, purpose, detail_lines=None, will_open=False, will_read=True, will_search=True):
        actions = []
        if will_open:
            actions.append("打开浏览器页面")
        if will_search:
            actions.append("联网搜索相关资料")
        if will_read:
            actions.append("读取网页正文内容")
        if not actions:
            actions.append("访问互联网")
        action_text = "、".join(dict.fromkeys(actions))
        usage_text = (
            "，并把获取到的网页内容用于回答或整理学习资料。"
            if will_read
            else "。小睿只会打开对应网页，不会后台读取网页正文。"
        )
        prompt = (
            f"是否允许小睿{purpose}？\n\n"
            f"本次可能会{action_text}{usage_text}"
        )
        clean_details = [str(line).strip() for line in (detail_lines or []) if str(line).strip()]
        if clean_details:
            prompt += "\n\n本次操作：\n" + "\n".join(f"• {line}" for line in clean_details[:4])
        prompt += "\n\n小睿只会按这一次指令使用网络，不会保存你的浏览器账号或密码。"

        decision = QMessageBox.question(
            self._assistant_dialog_parent(),
            "确认联网访问",
            prompt,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return decision == QMessageBox.Yes

    def _study_task_requires_online_access(self, task):
        task = dict(task or {})
        kind = task.get("kind")
        source_kind = str(task.get("source_kind") or "").strip().lower()
        if kind == "collect_materials":
            return True
        return kind in {"process_learning_content", "generate_diagram_image"} and source_kind in {"url", "urls"}

    def _confirm_study_task_online_access(self, task, command):
        task = dict(task or {})
        kind = task.get("kind")
        source_kind = str(task.get("source_kind") or "").strip().lower()
        if kind == "collect_materials":
            topic = task.get("topic") or "学习资料"
            return self._confirm_online_web_access(
                "联网搜索并读取相关网页内容",
                detail_lines=[
                    f"资料主题：{topic}",
                    "用途：采集学习资料并整理成文档",
                ],
                will_open=self._should_offer_browser_open(task, command),
                will_read=True,
                will_search=True,
            )
        if source_kind == "url":
            return self._confirm_online_web_access(
                "联网读取这个网页内容",
                detail_lines=["用途：读取网页正文并生成学习内容"],
                will_open=False,
                will_read=True,
                will_search=False,
            )
        if source_kind == "urls":
            source_value = task.get("source_value") or []
            count = len(source_value) if isinstance(source_value, list) else 0
            return self._confirm_online_web_access(
                "联网读取当前标签页网页内容",
                detail_lines=[
                    f"网页数量：{count or '多个'}",
                    "用途：汇总网页正文并生成学习内容",
                ],
                will_open=False,
                will_read=True,
                will_search=False,
            )
        return True

    def _submit_current_content_request(self, command, request_id, source="text"):
        try:
            current_source = self._confirm_and_capture_current_page()
        except StudyAssistantError as exc:
            self._deliver_assistant_reply(
                str(exc),
                source=source,
                bubble_title="读取失败",
                bubble_level="moderate",
            )
            return True
        if not current_source:
            self._set_assistant_status("已取消读取当前内容。")
            return True

        allowed, blocked_message = self._can_send_llm_request()
        if not allowed:
            self._cleanup_temporary_screen_capture_source(current_source)
            self._set_assistant_status(blocked_message)
            self._deliver_assistant_reply(
                blocked_message,
                source=source,
                bubble_title="请求稍快",
                bubble_level="moderate",
            )
            return True

        self.llm_request_inflight = True
        self.last_llm_request_at = time.time()
        self.assistant_pending_text = self._current_source_pending_text(current_source)
        self._refresh_assistant_views()
        self._sync_reasoning_button()
        self._set_assistant_status(self._current_source_status_text(current_source))
        self.voice_command_executor.submit(
            self._run_current_content_ai,
            request_id,
            command,
            dict(current_source),
            source,
        )
        return True

    def _should_offer_browser_open(self, task, command):
        if (task or {}).get("kind") != "collect_materials":
            return False
        command = command or ""
        return any(token in command for token in ("打开", "浏览器", "网页", "网站"))

    def _candidate_browser_sources(self, task, command):
        explicit_urls = list((task or {}).get("source_urls") or [])
        if explicit_urls:
            return [{"title": "指定网页", "url": url} for url in explicit_urls]
        topic = (task or {}).get("topic") or ""
        if not topic:
            return []
        candidates = build_curated_learning_sources(topic, command=command, limit=3)
        site_preferences = extract_site_preferences(command)
        if site_preferences and candidates:
            return candidates[:1]
        return candidates

    def _confirm_and_open_browser_sources(self, task, command, already_confirmed=False):
        sources = self._candidate_browser_sources(task, command)
        if not sources:
            return []

        if not already_confirmed:
            preview_lines = []
            for item in sources[:2]:
                preview_lines.append(f"• {item.get('title', '未命名页面')}")
            prompt = "我准备为你打开相关网页并继续整理资料：\n\n" + "\n".join(preview_lines)
            if len(sources) > 2:
                prompt += f"\n• 另外 {len(sources) - 2} 个页面"
            prompt += "\n\n是否继续打开浏览器页面？"

            decision = QMessageBox.question(
                self._assistant_dialog_parent(),
                "确认打开网页",
                prompt,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if decision != QMessageBox.Yes:
                return []

        opened = []
        for item in sources:
            url = item.get("url")
            if not url:
                continue
            if QDesktopServices.openUrl(QUrl(url)):
                opened.append(item)
        return opened

    def _confirm_and_capture_current_page(self):
        decision = QMessageBox.question(
            self._assistant_dialog_parent(),
            "确认获取当前屏幕内容",
            "是否能获取我当前屏幕内容？\n\n小睿会截取当前屏幕画面，并使用视觉模型理解页面、图片或文档内容。截图可能包含屏幕上的无关信息，请确认后继续。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if decision != QMessageBox.Yes:
            return None

        try:
            return self._capture_current_screen_source()
        except Exception as exc:
            raise StudyAssistantError(f"获取当前屏幕内容失败：{exc}") from exc

    def _capture_current_screen_source(self):
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            raise StudyAssistantError("暂时无法获取当前屏幕，请确认系统已允许应用录屏/截屏权限。")

        # Let the confirmation dialog close before taking the screenshot.
        QGuiApplication.processEvents()
        pixmap = screen.grabWindow(0)
        if pixmap.isNull():
            raise StudyAssistantError("当前屏幕截图为空，请确认系统已允许应用录屏/截屏权限。")

        temp_dir = os.path.join(tempfile.gettempdir(), "xiaorui_current_screen")
        os.makedirs(temp_dir, exist_ok=True)
        screenshot_path = os.path.join(temp_dir, f"current_screen_{int(time.time() * 1000)}.png")
        if not pixmap.save(screenshot_path, "PNG"):
            raise StudyAssistantError("当前屏幕截图保存失败，请稍后再试。")

        attachment = {
            "kind": "image",
            "path": screenshot_path,
            "name": "当前屏幕内容.png",
            "_xiaorui_temp_screen_capture": True,
        }
        return {
            "kind": "attachment",
            "value": attachment,
            "title": "当前屏幕内容",
        }

    def _is_screen_capture_attachment(self, attachment):
        attachment = dict(attachment or {})
        name = str(attachment.get("name") or "")
        path = str(attachment.get("path") or "")
        return "当前屏幕内容" in name or "current_screen_" in os.path.basename(path)

    def _temporary_screen_capture_dir(self):
        return os.path.abspath(os.path.join(tempfile.gettempdir(), "xiaorui_current_screen"))

    def _is_temporary_screen_capture_attachment(self, attachment):
        attachment = dict(attachment or {})
        path = str(attachment.get("path") or "").strip()
        if not path:
            return False
        abs_path = os.path.abspath(path)
        temp_dir = self._temporary_screen_capture_dir()
        in_temp_dir = abs_path.startswith(temp_dir + os.sep)
        generated_name = os.path.basename(abs_path).startswith("current_screen_")
        generated_marker = bool(attachment.get("_xiaorui_temp_screen_capture"))
        return in_temp_dir and (generated_marker or generated_name)

    def _cleanup_temporary_screen_capture_attachment(self, attachment):
        if not self._is_temporary_screen_capture_attachment(attachment):
            return False
        path = os.path.abspath(str((attachment or {}).get("path") or ""))
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            return False
        try:
            os.rmdir(self._temporary_screen_capture_dir())
        except OSError:
            pass
        return True

    def _cleanup_temporary_screen_capture_source(self, current_source):
        current_source = dict(current_source or {})
        if str(current_source.get("kind") or "").strip().lower() != "attachment":
            return False
        return self._cleanup_temporary_screen_capture_attachment(current_source.get("value") or {})

    def _cleanup_temporary_screen_capture_task(self, task):
        task = dict(task or {})
        cleaned = False
        if str(task.get("source_kind") or "").strip().lower() == "attachment":
            cleaned = self._cleanup_temporary_screen_capture_attachment(task.get("source_value") or {}) or cleaned
        cleaned = self._cleanup_temporary_screen_capture_attachment(task.get("attachment") or {}) or cleaned
        return cleaned

    def _confirm_and_capture_current_tabs(self):
        decision = QMessageBox.question(
            self._assistant_dialog_parent(),
            "确认读取当前标签页",
            "我需要读取你当前浏览器窗口里的多个标签页网址，再继续整理成学习笔记/提纲/导图。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if decision != QMessageBox.Yes:
            return None

        try:
            return get_active_browser_tabs()
        except StudyAssistantError:
            raise
        except Exception as exc:
            raise StudyAssistantError(f"读取当前标签页失败：{exc}") from exc

    def _submit_study_task(self, task, request_id, command, source="text", attachment=None):
        if task.get("error"):
            self._deliver_assistant_reply(
                task["error"],
                source=source,
                bubble_title="还差一点信息",
                bubble_level="moderate",
            )
            return False

        if task.get("kind") == "process_learning_content" and task.get("source_kind") == "missing":
            self._deliver_assistant_reply(
                "要处理当前页面或文档，请先上传文件/图片，或者在问题里直接贴网页链接，也可以说“整理当前打开的这些页面”。",
                source=source,
                bubble_title="缺少内容来源",
                bubble_level="moderate",
            )
            return False

        task = dict(task)
        if task.get("kind") in {"process_learning_content", "generate_diagram_image"} and task.get("source_kind") == "current_page":
            try:
                current_source = self._confirm_and_capture_current_page()
            except StudyAssistantError as exc:
                self._deliver_assistant_reply(
                    str(exc),
                    source=source,
                    bubble_title="读取失败",
                    bubble_level="moderate",
                )
                return False
            if not current_source:
                self._set_assistant_status("已取消读取当前内容。")
                return False
            task["source_kind"] = current_source.get("kind") or "missing"
            task["source_value"] = current_source.get("value")
            task["title_hint"] = current_source.get("title") or "当前内容"

        if task.get("kind") in {"process_learning_content", "generate_diagram_image"} and task.get("source_kind") == "current_tabs":
            try:
                current_tabs = self._confirm_and_capture_current_tabs()
            except StudyAssistantError as exc:
                self._deliver_assistant_reply(
                    str(exc),
                    source=source,
                    bubble_title="读取失败",
                    bubble_level="moderate",
                )
                return False
            if not current_tabs:
                self._set_assistant_status("已取消读取当前标签页。")
                return False
            task["source_kind"] = "urls"
            task["source_value"] = current_tabs
            task["title_hint"] = "当前标签页"

        online_access_confirmed = False
        if self._study_task_requires_online_access(task):
            online_access_confirmed = self._confirm_study_task_online_access(task, command)
            if not online_access_confirmed:
                self._cleanup_temporary_screen_capture_task(task)
                self._set_assistant_status("已取消联网搜索和网页读取。")
                self._deliver_assistant_reply(
                    "已取消联网搜索和网页读取，本次不会访问网页。",
                    source=source,
                    bubble_title="已取消联网",
                    bubble_level="moderate",
                )
                return True

        opened_sources = []
        if self._should_offer_browser_open(task, command):
            opened_sources = self._confirm_and_open_browser_sources(
                task,
                command,
                already_confirmed=online_access_confirmed,
            )

        save_plan = self._prepare_study_save_plan(task)
        if not save_plan:
            self._cleanup_temporary_screen_capture_task(task)
            self._set_assistant_status("已取消本次学习资料整理。")
            return False

        allowed, blocked_message = self._can_send_llm_request()
        if not allowed:
            self._cleanup_temporary_screen_capture_task(task)
            self._set_assistant_status(blocked_message)
            self._deliver_assistant_reply(
                blocked_message,
                source=source,
                bubble_title="请求稍快",
                bubble_level="moderate",
            )
            return False

        task_payload = dict(task)
        task_payload["save_plan"] = save_plan
        task_payload["opened_sources"] = opened_sources
        if attachment:
            task_payload["attachment"] = dict(attachment)

        self.llm_request_inflight = True
        self.last_llm_request_at = time.time()
        self.assistant_pending_text = self._study_task_pending_text(task_payload)
        self._refresh_assistant_views()
        self._set_assistant_status(self.assistant_pending_text)
        self.voice_command_executor.submit(
            self._run_study_task,
            request_id,
            task_payload,
            command,
            source,
        )
        return True

    def join_class(self):
        class_code, accepted = QInputDialog.getText(self, "加入班级", "请输入教师提供的班级码：")
        if not accepted:
            return

        success, message, class_info = self.db.join_class(self.user_info["id"], class_code)
        if success:
            self.user_info["class_name"] = (class_info or {}).get("class_name") or self.user_info.get("class_name")
            self._refresh_class_info()
            QMessageBox.information(self, "加入成功", message)
            if self.analytics_window:
                self.analytics_window.refresh_view()
            return

        QMessageBox.warning(self, "加入失败", message)

    def _create_value_label(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignRight)
        label.setFont(QFont("Arial", 14, QFont.Bold))
        label.setStyleSheet(f"color: {COLORS['primary']};")
        return label

    def _button_style(self, background, hover):
        return f"""
            QPushButton {{
                background-color: {background};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """

    def _progress_style(self, color):
        return f"""
            QProgressBar {{
                background-color: {COLORS['background']};
                border-radius: 10px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 10px;
            }}
        """

    def _ensure_detector_ready(self, start_after_load=False):
        if self.detector is not None:
            return True

        if start_after_load:
            self.pending_start_after_detector_load = True

        if self.detector_loading:
            self._set_assistant_status("检测模型正在加载，请稍等几秒。")
            return False

        self.detector_loading = True
        if hasattr(self, "start_btn"):
            self.start_btn.setEnabled(False)
            self.start_btn.setText("模型加载中...")
        if hasattr(self, "backend_status_label"):
            self.backend_status_label.setText("算法状态：正在加载检测模型")
        self._set_assistant_status("正在加载检测模型，界面不会卡住。")
        self.detector_load_executor.submit(self._load_detector_worker)
        return False

    def _load_detector_worker(self):
        try:
            from app.core.attention_detector import AttentionDetector

            detector = AttentionDetector()
            detector.set_multimodal_provider(self.multimodal_stream)
            self.detector_loaded.emit(detector, "")
        except Exception as exc:
            self.detector_loaded.emit(None, str(exc))

    def _handle_detector_loaded(self, detector, error):
        self.detector_loading = False
        if detector is None:
            self.pending_start_after_detector_load = False
            if hasattr(self, "start_btn"):
                self.start_btn.setEnabled(True)
                self.start_btn.setText("开始检测")
            if hasattr(self, "backend_status_label"):
                self.backend_status_label.setText(f"算法状态：加载失败 {error}")
            QMessageBox.warning(self, "检测模型加载失败", f"检测模型暂时无法加载：{error}")
            return

        self.detector = detector
        if hasattr(self, "backend_status_label"):
            self.backend_status_label.setText("算法状态：检测模型已就绪")
        if hasattr(self, "start_btn"):
            self.start_btn.setEnabled(True)
            self.start_btn.setText("开始检测")
        self._set_assistant_status("检测模型已就绪。")

        if self.pending_start_after_detector_load and not self.is_detecting:
            self.pending_start_after_detector_load = False
            self.start_detection()

    def _get_detector_status(self, score):
        if self.detector is not None:
            return self.detector.get_status(score)
        if score >= 70:
            return "focused", "专注"
        if score >= 40:
            return "moderate", "一般"
        return "distracted", "分心"

    def toggle_detection(self):
        if self.is_detecting:
            self.stop_detection()
        else:
            self.start_detection()

    def start_detection(self):
        if not self._ensure_detector_ready(start_after_load=True):
            return

        cap, camera_desc = self._open_available_camera()
        if cap is None:
            QMessageBox.warning(self, "摄像头不可用", self._camera_unavailable_message())
            self.cap = None
            return

        self.cap = cap

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.detector.reset_runtime_state()

        self.is_detecting = True
        self.start_btn.setText("停止检测")
        self.start_btn.setStyleSheet(self._button_style(COLORS["danger"], "#C0392B"))

        self.attention_scores.clear()
        self.live_records.clear()
        self.current_score = 0.0
        self.distraction_count = 0
        self.focused_duration_seconds = 0.0
        self.total_session_duration_seconds = 0.0
        self.session_start_time = time.time()
        self.last_frame_ts = self.session_start_time
        self.last_saved_at = 0.0
        self.last_saved_status = ""
        self.pending_detection = None
        self.pending_detection_time = None
        self.video_label.setText(f"摄像头已连接：{camera_desc}")
        self.frame_counter = 0
        self.display_last_ts = None
        self.preview_fps = 0.0
        self.inference_latency_ms = 0.0

        self.multimodal_paused_for_voice = False
        self._start_multimodal_stream_if_needed()
        self.timer.start(50)
        if self.analytics_window:
            self.analytics_window.refresh_view()

    def _open_available_camera(self):
        for index, backend, label in self._camera_candidates():
            cap = self._try_open_camera(index, backend)
            if cap is not None:
                return cap, label
        return None, ""

    def _camera_candidates(self):
        candidates = []
        seen = set()

        backend_options = [("默认后端", cv2.CAP_ANY)]
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backend_options.insert(0, ("AVFoundation", cv2.CAP_AVFOUNDATION))
        elif sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            backend_options.insert(0, ("DirectShow", cv2.CAP_DSHOW))

        for index in (0, 1, 2):
            for backend_name, backend in backend_options:
                key = (index, backend)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((index, backend, f"{backend_name} / 摄像头 {index}"))
        return candidates

    def _try_open_camera(self, index, backend):
        try:
            cap = cv2.VideoCapture(index, backend)
        except Exception:
            return None

        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return None

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _frame = cap.read()
        except Exception:
            ret = False

        if not ret:
            cap.release()
            return None
        return cap

    def _camera_unavailable_message(self):
        if sys.platform == "darwin":
            return (
                "无法打开摄像头。\n\n"
                "请优先检查：\n"
                f"1. 系统设置 -> 隐私与安全性 -> 摄像头，给 Terminal 或打包后的 {APP_NAME} 开启权限；\n"
                "2. 关闭正在占用摄像头的软件，例如微信、腾讯会议、FaceTime、浏览器；\n"
                "3. 如果接了外接摄像头，请重新插拔后再试。\n\n"
                "程序已经自动尝试了多个摄像头索引和 macOS 后端，但仍未成功连接。"
            )
        return (
            "无法打开摄像头。\n\n"
            "请检查：\n"
            "1. 摄像头权限是否已授予当前程序；\n"
            "2. 是否有其他软件正在占用摄像头；\n"
            "3. 摄像头设备是否连接正常。\n\n"
            "程序已经自动尝试了多个摄像头索引，但仍未成功连接。"
        )

    def stop_detection(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

        if self.pending_detection is not None:
            self.pending_detection.cancel()
            self.pending_detection = None
            self.pending_detection_time = None

        self._stop_multimodal_stream()

        if self.is_detecting:
            self.persist_current_record(force=True)

        self.is_detecting = False
        self.start_btn.setText("开始检测")
        self.start_btn.setStyleSheet(self._button_style(COLORS["focused"], "#27AE60"))

        self.video_label.clear()
        self.video_label.setText('点击"开始检测"启动摄像头')
        self.reset_ui_to_idle()
        if self.analytics_window:
            self.analytics_window.refresh_view()

    def reset_ui_to_idle(self):
        self.score_label.setText("--")
        self.status_label.setText("未检测")
        self.progress_bar.setValue(0)
        self.avg_score_label.setText("0%")
        self.focus_duration_label.setText("0s")
        self.distraction_label.setText("0")
        self.backend_status_label.setText("算法状态：预热中")
        self.debug_metrics_label.setText("等待检测启动")
        self.debug_reason_label.setText("这里会显示主要原因")

    def _start_multimodal_stream_if_needed(self):
        if self.multimodal_paused_for_voice:
            return
        if self.voice_worker and self.voice_worker.isRunning():
            self.multimodal_paused_for_voice = True
            return
        self.multimodal_stream.start()

    def _stop_multimodal_stream(self):
        self.multimodal_stream.stop()
        self.multimodal_paused_for_voice = False

    def _resume_multimodal_after_voice(self):
        if self.is_detecting and self.multimodal_paused_for_voice:
            self.multimodal_paused_for_voice = False
            self.multimodal_stream.start()

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        self.frame_counter += 1
        now = time.time()
        if self.display_last_ts is not None:
            delta = max(now - self.display_last_ts, 1e-3)
            instant_fps = 1.0 / delta
            self.preview_fps = instant_fps if self.preview_fps == 0 else self.preview_fps * 0.8 + instant_fps * 0.2
        self.display_last_ts = now

        self.render_frame(frame)

        if self.pending_detection is not None and self.pending_detection.done():
            detection_finished_at = time.time()
            try:
                score, runtime_info, _movement = self.pending_detection.result()
            except Exception as exc:
                print(f"Detection error: {exc}")
                self.stop_detection()
                return

            self.pending_detection = None
            reference_time = self.pending_detection_time or detection_finished_at
            self.pending_detection_time = None
            self.apply_detection_result(score, runtime_info, reference_time, detection_finished_at)

        if self.pending_detection is None and self.frame_counter % self.detection_stride == 0:
            frame_for_detection = frame.copy()
            self.pending_detection = self.detector_executor.submit(
                self.detector.detect_frame_with_movement, frame_for_detection
            )
            self.pending_detection_time = time.time()

        if self.mini_robot:
            self.mini_robot.set_attention(self.current_score)

    def render_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, channel = rgb_frame.shape
        bytes_per_line = channel * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )

        if not self.is_minimized:
            self.video_label.setPixmap(scaled_pixmap)

    def apply_detection_result(self, score, runtime_info, started_at, finished_at):
        current_time = finished_at
        baseline_time = self.last_frame_ts or started_at
        delta = max(0.0, current_time - baseline_time)
        self.last_frame_ts = current_time
        self.inference_latency_ms = max(0.0, (finished_at - started_at) * 1000.0)

        self.current_score = score
        self.latest_runtime_info = runtime_info

        if len(self.attention_scores) >= 500:
            self.attention_scores.pop(0)
        self.attention_scores.append(score)

        self.total_session_duration_seconds += delta
        if score >= 70:
            self.focused_duration_seconds += delta

        if score < 40:
            if len(self.attention_scores) == 1 or self.attention_scores[-2] >= 40:
                self.distraction_count += 1

        status_key, status_text = self._get_detector_status(score)
        self.live_records.append(
            {
                "timestamp": time.strftime("%H:%M:%S"),
                "attention_score": float(score),
                "status": status_text,
                "metrics": self.build_current_metrics(),
            }
        )
        if len(self.live_records) > 500:
            self.live_records.pop(0)
        self.update_ui(score, status_key, status_text)
        self.persist_current_record()

    def update_ui(self, score, status_key, status_text):
        color = COLORS[status_key]
        self.score_label.setText(f"{int(score)}%")
        self.score_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {color};")
        self.progress_bar.setValue(int(score))
        self.progress_bar.setStyleSheet(self._progress_style(color))

        avg_score = sum(self.attention_scores) / len(self.attention_scores) if self.attention_scores else 0.0
        self.avg_score_label.setText(f"{avg_score:.1f}%")
        self.focus_duration_label.setText(self.format_duration(self.focused_duration_seconds))
        self.distraction_label.setText(str(self.distraction_count))

        backend = self.latest_runtime_info.get("backend", {})
        if backend.get("available"):
            self.backend_status_label.setText(
                f"算法状态：正常 · 场景 {backend.get('scene', '-')} · 模型 {backend.get('score', 0):.1f}"
            )
        else:
            message = backend.get("status", {}).get("message", "算法增强未启用")
            self.backend_status_label.setText(f"算法状态：{message}")

        self.update_debug_panel()

    def start_focus_calibration(self):
        if not self.is_detecting:
            QMessageBox.information(self, "先开始检测", "请先点击“开始检测”，让摄像头画面正常显示后再校准。")
            return
        if self.detector is None:
            QMessageBox.information(self, "模型未就绪", "检测模型还在加载，请稍等。")
            return

        message = self.detector.start_focus_calibration()
        self.debug_reason_label.setText(message)
        self.calibration_btn.setText("校准中...")

    def clear_focus_calibration(self):
        if self.detector is None:
            self.debug_reason_label.setText("检测模型尚未启动，暂无个人校准数据。")
            return
        message = self.detector.clear_focus_calibration()
        self.debug_reason_label.setText(message)
        self.calibration_btn.setText("校准专注姿态")

    def update_debug_panel(self):
        explanation = self.latest_runtime_info.get("score_explanation", {})
        metrics = explanation.get("metrics", {})
        backend = self.latest_runtime_info.get("backend", {})
        multimodal = self.latest_runtime_info.get("multimodal", {})
        audio_info = multimodal.get("audio", {})
        text_info = multimodal.get("text", {})
        calibration = self.latest_runtime_info.get("calibration")
        if not calibration and self.detector is not None:
            calibration = self.detector.get_calibration_status()
        if not calibration:
            calibration = {"message": "尚未启动"}
        face_text = "是" if metrics.get("face_detected") else "否"
        if metrics.get("phone_detected"):
            source_map = {"yolo_full": "全图", "yolo_lower_crop": "局部"}
            phone_source = source_map.get(metrics.get("phone_source"), metrics.get("phone_source", "-"))
            phone_text = f"是 {metrics.get('phone_confidence', 0):.2f} {phone_source}"
        else:
            phone_text = "否"
        if metrics.get("yawn_detected"):
            mouth_text = "疑似打哈欠"
        elif metrics.get("mouth_open"):
            mouth_text = f"张开偏高 {metrics.get('mouth_open_strength', 0) * 100:.0f}%"
        else:
            mouth_text = "稳定"
        eye_text = metrics.get("eye_status", "-")
        posture_text = metrics.get("posture_status", "-")
        vision_text = metrics.get("vision_source", metrics.get("face_source", "-"))
        perclos_text = f"{metrics.get('perclos', 0) * 100:.0f}%"
        head_pose_text = (
            f"Y{metrics.get('head_yaw_deg', 0):.0f}/"
            f"P{metrics.get('head_pitch_deg', 0):.0f}/"
            f"R{metrics.get('head_roll_deg', 0):.0f}"
        )
        backend_score = backend.get("score")
        backend_text = f"{backend_score:.1f}" if isinstance(backend_score, (int, float)) else "-"
        if self.multimodal_paused_for_voice:
            multimodal_text = "因小睿占用麦克风暂时暂停"
        elif multimodal.get("available"):
            transcript = (text_info.get("last_transcript") or "-").replace("\n", " ")
            transcript = transcript[:10] + ("..." if len(transcript) > 10 else "")
            multimodal_text = (
                f"已启用 · 语音活跃 {audio_info.get('speech_ratio', 0.0) * 100:.0f}% · 转写：{transcript}"
            )
        else:
            multimodal_text = multimodal.get("status", "未启用")
        if calibration.get("active"):
            self.calibration_btn.setText(f"校准中 {calibration.get('sample_count', 0)}/{calibration.get('required_samples', 0)}")
        elif calibration.get("ready"):
            self.calibration_btn.setText("重新校准")
        else:
            self.calibration_btn.setText("校准专注姿态")

        self.debug_metrics_label.setText(
            f"FPS {self.preview_fps:.1f} · 延时 {self.inference_latency_ms:.0f}ms · 人脸 {face_text} · 姿态 {metrics.get('pose', '-')}\n"
            f"手机 {phone_text} · 眼睛 {eye_text} · PERCLOS {perclos_text} · 嘴部 {mouth_text}\n"
            f"头姿 {head_pose_text} · 坐姿 {posture_text} · 当前 {metrics.get('final_score', '-')} · 模型 {backend_text}\n"
            f"视觉 {vision_text} · 校准 {calibration.get('message', '未知')}"
        )

        reasons = explanation.get("reasons", [])
        if reasons:
            display_reasons = "；".join(reasons[:2])
            if len(reasons) > 2:
                display_reasons += "……"
            self.debug_reason_label.setText(f"主要原因：{display_reasons}")
        else:
            self.debug_reason_label.setText("主要原因：这里会显示主要原因")

    def persist_current_record(self, force=False):
        if not self.is_detecting and not force:
            return

        status_key, status_text = self._get_detector_status(self.current_score)
        now = time.time()
        should_save = force or not self.last_saved_at or (now - self.last_saved_at) >= 5.0
        if not should_save and status_text != self.last_saved_status:
            should_save = True

        if not should_save:
            return

        if self.record_save_inflight and not force:
            return

        metrics = self.build_current_metrics()
        score = float(self.current_score)
        self.last_saved_at = now
        self.last_saved_status = status_text
        self.record_save_inflight = True
        self.record_executor.submit(self._save_attention_record_worker, score, status_text, metrics)

    def _save_attention_record_worker(self, score, status_text, metrics):
        try:
            self.db.save_attention_record(self.user_info["id"], score, status_text, metrics=metrics)
        except Exception as exc:
            print(f"Save attention record failed: {exc}")
        finally:
            self.record_save_inflight = False

    def build_current_metrics(self):
        avg_score = sum(self.attention_scores) / len(self.attention_scores) if self.attention_scores else 0.0
        explanation = self.latest_runtime_info.get("score_explanation", {})
        runtime_metrics = explanation.get("metrics", {})
        return {
            "avg_score": avg_score,
            "focused_duration_seconds": self.focused_duration_seconds,
            "distraction_count": self.distraction_count,
            "backend": self.latest_runtime_info.get("backend", {}),
            "multimodal": self.latest_runtime_info.get("multimodal", {}),
            "base_score": self.latest_runtime_info.get("base_score"),
            "final_score": self.latest_runtime_info.get("final_score"),
            "smoothed_score": self.latest_runtime_info.get("smoothed_score"),
            "score_source": self.latest_runtime_info.get("score_source"),
            "runtime": runtime_metrics,
            "calibration": self.latest_runtime_info.get("calibration"),
            "reasons": explanation.get("reasons", []),
        }

    def get_snapshot(self):
        avg_score = sum(self.attention_scores) / len(self.attention_scores) if self.attention_scores else 0.0
        return {
            "is_detecting": self.is_detecting,
            "current_score": self.current_score,
            "avg_score": avg_score,
            "attention_scores": list(self.attention_scores),
            "live_records": list(self.live_records),
            "distraction_count": self.distraction_count,
            "focused_duration_seconds": self.focused_duration_seconds,
            "total_session_duration_seconds": self.total_session_duration_seconds,
            "latest_runtime_info": self.latest_runtime_info,
        }

    def open_analytics_window(self):
        if self.is_minimized or not self.isVisible():
            self.show_normal()
            QTimer.singleShot(120, self._present_analytics_window)
            return

        self._present_analytics_window()

    def _present_analytics_window(self):
        if self.analytics_window is None:
            self.analytics_window = StudentAnalyticsWindow(
                self.user_info,
                self.db,
                snapshot_provider=self.get_snapshot,
            )
        self.raise_()
        self.activateWindow()
        self.analytics_window.show()
        self.analytics_window.raise_()
        self.analytics_window.activateWindow()
        self.analytics_window.refresh_view()

    def toggle_voice_assistant(self):
        if self.voice_worker and self.voice_worker.isRunning():
            self.stop_voice_assistant()
        else:
            self.start_voice_assistant()

    def start_voice_assistant(self):
        if self.is_detecting and self.multimodal_stream.is_running:
            self.multimodal_stream.stop()
            self.multimodal_paused_for_voice = True
        self.voice_worker = VoiceAssistantWorker(self, speaker_active_callback=self.voice_speaker.is_speaking)
        self.voice_worker.status_changed.connect(self.update_voice_status)
        self.voice_worker.dependency_error.connect(self.handle_voice_error)
        self.voice_worker.activated.connect(self.handle_voice_wake)
        self.voice_worker.command_recognized.connect(self.handle_voice_command)
        self.voice_worker.finished.connect(self.handle_voice_finished)
        self.voice_worker.start()
        self.voice_enabled = True
        self.voice_btn.setText("关闭小睿语音助手")
        self.voice_btn.setStyleSheet(self._button_style(COLORS["danger"], "#C0392B"))
        if self.mini_assistant_panel:
            self.mini_assistant_panel.set_voice_enabled(True)
        self._set_assistant_status("小睿启动中，请稍等...")

    def stop_voice_assistant(self):
        if self.voice_worker:
            self.voice_worker.stop()
            self.voice_worker.wait(6500)
            self.voice_worker = None
        self.voice_speaker.stop()
        self.voice_interrupt_armed_until = 0.0
        self.voice_enabled = False
        self.voice_btn.setText("启动小睿语音助手")
        self.voice_btn.setStyleSheet(self._button_style(COLORS["secondary"], "#8E44AD"))
        if self.mini_assistant_panel:
            self.mini_assistant_panel.set_voice_enabled(False)
        self._set_assistant_status(self._assistant_idle_text())
        self._resume_multimodal_after_voice()

    def update_voice_status(self, message):
        self._set_assistant_status(message)

    def handle_voice_error(self, message):
        self._set_assistant_status(message)
        self.voice_speaker.speak("小睿语音依赖还没有安装，暂时不能识别语音。")
        self.stop_voice_assistant()

    def handle_voice_finished(self):
        if self.voice_enabled:
            self.stop_voice_assistant()

    def submit_text_query(self, text=None):
        try:
            command = ""
            if isinstance(text, str):
                command = text.strip()
            elif hasattr(self, "assistant_input"):
                command = self.assistant_input.text().strip()
                self.assistant_input.clear()

            attachment = dict(self.selected_attachment) if self.selected_attachment else None
            if not command and not attachment:
                return
            if not command and attachment:
                command = "请帮我看看这个上传内容。"

            if not self.voice_runtime_active:
                return

            display_command = command
            if attachment:
                label = "图片" if attachment.get("kind") == "image" else "文件"
                display_command = f"{command}\n[{label}] {attachment.get('name', '未命名内容')}"

            self._stage_assistant_log(command, "text")
            self._append_assistant_message("user", display_command)
            request_id = self._advance_voice_request()
            self.voice_interrupt_armed_until = 0.0
            if self.voice_speaker.is_speaking():
                self.voice_speaker.stop()

            study_task = self._detect_study_task(command, attachment=attachment)
            if study_task:
                if self._submit_study_task(study_task, request_id, command, source="text", attachment=attachment):
                    if attachment:
                        self.clear_assistant_attachment(silent=True)
                return

            if attachment:
                if self.voice_llm_controller.available:
                    if self._submit_attachment_llm_request(command, attachment, request_id):
                        self.clear_assistant_attachment(silent=True)
                    return
                self._deliver_assistant_reply(
                    f"当前上传图片/文件问答仅通过{self._current_llm_label()}回答，请先配置对应的 API Key 后再试。",
                    source="text",
                    bubble_title="模型未连接",
                    bubble_level="moderate",
                )
                return

            self._dispatch_assistant_command(command, request_id, source="text")
        except Exception as exc:
            self._set_assistant_status(f"刚刚这次发送没有成功：{exc}")
            self._deliver_assistant_reply(
                f"刚刚这次发送没有成功，我已经拦住程序崩溃了。错误信息：{exc}",
                source="text",
                bubble_title="发送失败",
                bubble_level="moderate",
            )

    def _submit_attachment_llm_request(self, command, attachment, request_id):
        allowed, blocked_message = self._can_send_llm_request()
        if not allowed:
            self._set_assistant_status(blocked_message)
            self._deliver_assistant_reply(
                blocked_message,
                source="text",
                bubble_title="请求稍快",
                bubble_level="moderate",
            )
            return False

        label = "图片" if attachment.get("kind") == "image" else "文件"
        self.llm_request_inflight = True
        self.last_llm_request_at = time.time()
        self.assistant_pending_text = "思考中..." if self._is_reasoning_model() else f"正在分析{label}..."
        self._refresh_assistant_views()
        self._sync_reasoning_button()
        self._set_assistant_status(f"已收到{label}《{attachment.get('name', '未命名内容')}》，小睿正在分析...")
        self.voice_command_executor.submit(
            self._run_attachment_ai,
            request_id,
            command,
            self._build_text_ai_context(command),
            dict(attachment),
        )
        return True

    def _dispatch_assistant_command(self, command, request_id, source="voice"):
        thinking_message = f"{'识别到' if source == 'voice' else '收到'}：{command}。小睿正在思考..."

        study_task = self._detect_study_task(command)
        if study_task and source == "voice":
            if self._submit_study_task(study_task, request_id, command, source=source):
                return

        web_search_task = detect_web_search_command(command)
        if web_search_task:
            self._handle_web_search_task(web_search_task, source=source)
            return

        if self._is_current_content_query(command):
            if self._submit_current_content_request(command, request_id, source=source):
                return

        if source == "text":
            if self.voice_llm_controller.available:
                if self._submit_llm_request(command, request_id, source, thinking_message):
                    return
                return
            self._deliver_assistant_reply(
                f"当前文本问答仅通过{self._current_llm_label()}回答，请先配置对应的 API Key 后再试。",
                source="text",
                bubble_title="模型未连接",
                bubble_level="moderate",
            )
            return

        if self._is_start_detection_command(command):
            self._handle_voice_command_locally(command, request_id=request_id, source=source)
            return

        if self._is_stop_detection_command(command):
            self._handle_voice_command_locally(command, request_id=request_id, source=source)
            return

        if self._is_minimize_command(command):
            self._handle_voice_command_locally(command, request_id=request_id, source=source)
            return

        if self._is_show_detection_command(command):
            self._handle_voice_command_locally(command, request_id=request_id, source=source)
            return

        if self._is_open_analytics_command(command):
            self._handle_voice_command_locally(command, request_id=request_id, source=source)
            return

        if self._match_project_info_topic(command):
            self._handle_voice_command_locally(command, request_id=request_id, source=source)
            return

        if self._is_intro_query(command):
            if self.voice_llm_controller.available:
                if self._submit_llm_request(command, request_id, source, thinking_message):
                    return
                return
            self._handle_voice_command_locally(command, request_id=request_id, source=source)
            return

        if self._is_advice_query(command):
            if self.voice_llm_controller.available:
                if self._submit_llm_request(command, request_id, source, thinking_message):
                    return
                return
            self._handle_voice_command_locally(command, request_id=request_id, source=source)
            return

        if self._is_focus_query(command):
            self._handle_voice_command_locally(command, request_id=request_id, source=source)
            return

        if self.voice_llm_controller.available:
            if self._submit_llm_request(command, request_id, source, thinking_message):
                return
            return

        self._handle_voice_command_locally(command, request_id=request_id, source=source)

    def _handle_web_search_task(self, task, source="text"):
        target_url = (task or {}).get("url", "")
        site_label = (task or {}).get("site_label") or "网页"
        query = (task or {}).get("query") or ""
        if not target_url:
            self._deliver_assistant_reply(
                "我还没有识别出要打开的网站或搜索内容，你可以说“打开哔哩哔哩搜索高中数学函数”。",
                source=source,
                bubble_title="缺少搜索信息",
                bubble_level="moderate",
            )
            return True

        detail_lines = [f"网站：{site_label}"]
        if query:
            detail_lines.append(f"搜索内容：{query}")
        else:
            detail_lines.append("操作：打开网站首页")
        if not self._confirm_online_web_access(
            "联网搜索并打开相关网页",
            detail_lines=detail_lines,
            will_open=True,
            will_read=False,
            will_search=bool(query),
        ):
            self._set_assistant_status("已取消打开网页。")
            self._deliver_assistant_reply(
                "已取消联网搜索和打开网页。",
                source=source,
                bubble_title="已取消联网",
                bubble_level="moderate",
            )
            return True

        opened = QDesktopServices.openUrl(QUrl(target_url))
        if not opened:
            self._deliver_assistant_reply(
                f"我没能打开浏览器，你可以手动访问：\n{target_url}",
                source=source,
                bubble_title="打开失败",
                bubble_level="moderate",
            )
            return True

        if query:
            reply = f"已为你打开{site_label}，并搜索：{query}"
        else:
            reply = f"已为你打开{site_label}。"
        self._deliver_assistant_reply(
            reply,
            source=source,
            bubble_title="已打开网页",
            bubble_level="focused",
        )
        return True

    def _submit_llm_request(self, command, request_id, source, thinking_message):
        allowed, blocked_message = self._can_send_llm_request()
        if not allowed:
            self._set_assistant_status(blocked_message)
            if source == "text":
                self._deliver_assistant_reply(
                    blocked_message,
                    source="text",
                    bubble_title="请求稍快",
                    bubble_level="moderate",
                )
            return False

        self.llm_request_inflight = True
        self.last_llm_request_at = time.time()
        self.assistant_pending_text = "思考中..." if self._is_reasoning_model() else "正在思考..."
        self._refresh_assistant_views()
        self._sync_reasoning_button()
        self._set_assistant_status(thinking_message)
        self.voice_command_executor.submit(
            self._run_text_ai if source == "text" else self._run_voice_ai,
            request_id,
            command,
            self._build_text_ai_context(command) if source == "text" else self._build_voice_ai_context(command),
        )
        return True

    def _can_send_llm_request(self):
        if self.llm_request_inflight:
            return False, "我还在处理上一条问题，请等这条回答完再继续问我。"

        now = time.time()
        remaining = self.llm_request_cooldown_seconds - (now - self.last_llm_request_at)
        if remaining > 0:
            seconds = max(1, int(round(remaining)))
            return False, f"你问得有点快，等 {seconds} 秒再问我会更稳定。"
        return True, ""

    def _finish_llm_request(self, request_id):
        if request_id == self.voice_request_id or request_id >= self.voice_request_id - 1:
            self.llm_request_inflight = False
            self.assistant_pending_text = ""
            self._refresh_assistant_views()
            self._sync_reasoning_button()

    def handle_voice_wake(self):
        if not self.voice_runtime_active:
            return
        self.voice_interrupt_armed_until = time.time() + self.VOICE_INTERRUPT_ARM_SECONDS
        if self.voice_speaker.is_speaking():
            self._advance_voice_request()
            self.voice_speaker.stop()
            self._set_assistant_status("已打断小睿播报。请继续说新指令。")
            return
        self._advance_voice_request()
        self.voice_speaker.stop()
        self._set_assistant_status("小睿已唤醒：请在 8 秒内说出指令，例如“当前专注度是多少”“打开详细数据”或“介绍一下自己”。")
        self.voice_speaker.speak("我在，请讲。")

    def handle_voice_command(self, command):
        if not self.voice_runtime_active:
            return
        command = command or ""
        if command == "__wake__":
            return
        if command == "__interrupt__":
            self._advance_voice_request()
            self.voice_interrupt_armed_until = time.time() + self.VOICE_INTERRUPT_ARM_SECONDS
            self.voice_speaker.stop()
            self._set_assistant_status("已停止播报。你可以继续说新指令。")
            return

        if not self._looks_like_meaningful_voice_command(command):
            self._set_assistant_status("刚才更像是环境噪声，我先忽略这次输入。")
            return

        if self.voice_speaker.is_speaking() and time.time() > self.voice_interrupt_armed_until:
            self._set_assistant_status("小睿正在播报中。若要打断，请先说“你好小睿”再下达新命令。")
            return

        request_id = self._advance_voice_request()
        self.voice_interrupt_armed_until = 0.0
        self.voice_speaker.stop()
        self._stage_assistant_log(command, "voice")

        self._dispatch_assistant_command(command, request_id, source="voice")

    def _advance_voice_request(self):
        self.voice_request_id += 1
        return self.voice_request_id

    def _run_voice_ai(self, request_id, command, context):
        try:
            result = self.voice_llm_controller.interpret(command, context)
        except VoiceLLMControllerError as exc:
            self.voice_ai_error.emit(request_id, command, str(exc))
            return
        except Exception as exc:
            self.voice_ai_error.emit(request_id, command, f"大模型处理异常：{exc}")
            return
        self.voice_ai_result.emit(request_id, result, command)

    def _run_text_ai(self, request_id, command, context):
        try:
            if self._should_use_text_assist_mode(command):
                result = self.voice_llm_controller.assist_text(command, context)
            else:
                reply = self.voice_llm_controller.chat(command, context)
                result = {
                    "intent": "general_chat",
                    "reply": reply,
                    "confidence": 0.92,
                }
        except VoiceLLMControllerError as exc:
            self.text_ai_error.emit(request_id, command, str(exc))
            return
        except Exception as exc:
            self.text_ai_error.emit(request_id, command, f"大模型处理异常：{exc}")
            return
        self.text_ai_result.emit(request_id, result, command)

    def _run_attachment_ai(self, request_id, command, context, attachment):
        try:
            reply = self.voice_llm_controller.chat_with_attachment(command, context, attachment)
            result = {
                "intent": "general_chat",
                "reply": reply,
                "confidence": 0.94,
            }
        except VoiceLLMControllerError as exc:
            self.text_ai_error.emit(request_id, command, str(exc))
            return
        except Exception as exc:
            self.text_ai_error.emit(request_id, command, f"大模型处理异常：{exc}")
            return
        self.text_ai_result.emit(request_id, result, command)

    def _run_current_content_ai(self, request_id, command, current_source, source="text"):
        try:
            reply = self._answer_current_content_query(command, current_source)
            result = {
                "intent": "general_chat",
                "reply": reply,
                "confidence": 0.95,
                "source": source,
            }
        except (StudyAssistantError, VoiceLLMControllerError) as exc:
            result = {
                "intent": "general_chat",
                "reply": f"当前内容分析没有完成：{exc}",
                "confidence": 0.2,
                "source": source,
            }
            self.text_ai_result.emit(request_id, result, command)
            return
        except Exception as exc:
            result = {
                "intent": "general_chat",
                "reply": f"当前内容分析异常：{exc}",
                "confidence": 0.2,
                "source": source,
            }
            self.text_ai_result.emit(request_id, result, command)
            return
        finally:
            self._cleanup_temporary_screen_capture_source(current_source)
        self.text_ai_result.emit(request_id, result, command)

    def _run_study_task(self, request_id, task, command, source):
        try:
            result = self._execute_study_task(task, command)
        except StudyAssistantError as exc:
            self.study_task_error.emit(request_id, command, str(exc))
            return
        except VoiceLLMControllerError as exc:
            self.study_task_error.emit(request_id, command, str(exc))
            return
        except Exception as exc:
            self.study_task_error.emit(request_id, command, f"学习整理异常：{exc}")
            return
        finally:
            self._cleanup_temporary_screen_capture_task(task)
        result["source"] = source
        self.study_task_result.emit(request_id, result, command)

    def _execute_study_task(self, task, command):
        kind = task.get("kind")
        if kind == "generate_diagram_image":
            return self._execute_diagram_image_generation(task, command)
        if kind == "collect_materials":
            return self._execute_material_collection(task, command)
        if kind == "process_learning_content":
            return self._execute_learning_processing(task, command)
        raise StudyAssistantError("暂时不支持这种学习助手任务。")

    def _execute_diagram_image_generation(self, task, command):
        diagram_label = task.get("diagram_label") or "示意图"
        source_text, title_hint = self._resolve_diagram_generation_source(task, command)
        prompt = self._build_diagram_image_prompt(
            task.get("diagram_kind") or "structure",
            diagram_label,
            command,
            source_text=source_text,
            title_hint=title_hint,
        )
        controller = self._build_learning_task_controller("doubao:doubao-seedream-5-0-lite-260128")
        try:
            image_result = controller.generate_image(prompt, size="2K")
        except VoiceLLMControllerError as exc:
            raise StudyAssistantError(str(exc)) from exc

        save_plan = task.get("save_plan") or {}
        output_path = self._persist_generated_image_result(image_result, save_plan.get("main_path"))
        summary = self._shorten_reply(source_text or command, 120)
        return {
            "reply": f"已为你生成{diagram_label}图片，并保存到：\n{output_path}\n\n依据内容：{summary}",
            "saved_path": output_path,
            "title_hint": title_hint or diagram_label,
        }

    def _resolve_diagram_generation_source(self, task, command):
        source_kind = task.get("source_kind")
        source_value = task.get("source_value")
        diagram_label = task.get("diagram_label") or "示意图"
        title_hint = task.get("title_hint") or diagram_label

        if source_kind == "attachment":
            attachment = dict(task.get("attachment") or source_value or {})
            if not attachment.get("path"):
                raise StudyAssistantError("当前没有可用于生成图片的附件内容。")
            title_hint = attachment.get("name") or title_hint
            if attachment.get("kind") == "image":
                summary_prompt = (
                    f"请阅读这张图片，并提炼出适合生成{diagram_label}的关键要点。"
                    "请用中文输出 8 到 12 条要点，每条一句话，不要输出多余解释。"
                )
                summary = self._chat_for_learning_task(
                    summary_prompt,
                    self._build_learning_context(title_hint, "图片结构整理"),
                    attachment=attachment,
                    max_tokens_override=320,
                    retry_tokens_override=220,
                )
                return summary, title_hint

            summary_prompt = (
                f"请阅读这份资料，并提炼出适合生成{diagram_label}的关键信息。"
                "请用中文输出 8 到 12 条要点，优先保留模块关系、流程顺序和核心结构。"
            )
            summary = self._chat_for_learning_task(
                summary_prompt,
                self._build_learning_context(title_hint, "文档结构整理"),
                attachment=attachment,
                max_tokens_override=700,
                retry_tokens_override=520,
            )
            return summary, title_hint

        if source_kind == "url":
            page = fetch_page_content(source_value, timeout=8, max_chars=4200)
            return page["content"], page["title"] or title_hint

        if source_kind == "urls":
            pages = []
            for item in list(source_value or [])[:3]:
                url = (item or {}).get("url", "")
                if not url:
                    continue
                try:
                    page = fetch_page_content(url, timeout=8, max_chars=2200)
                except Exception:
                    continue
                pages.append(page)
            if not pages:
                raise StudyAssistantError("当前标签页里还没有可读取的正文内容，暂时无法生成图片。")
            joined = []
            for index, page in enumerate(pages, start=1):
                joined.append(f"[页面{index}] {page['title']}\n{page['content']}")
            return "\n\n".join(joined), f"当前标签页汇总（{len(pages)}页）"

        if source_kind == "prompt_only":
            return "", title_hint

        raise StudyAssistantError("这次还没有找到可用于生成图片的内容来源。")

    def _build_diagram_image_prompt(self, diagram_kind, diagram_label, command, source_text="", title_hint=""):
        style_map = {
            "mindmap": "中文思维导图，中心主题清晰，分支层级明确",
            "flowchart": "中文流程图，步骤顺序清楚，箭头清晰",
            "architecture": "中文系统架构图，模块分层明确，连接关系清晰",
            "structure": "中文结构图，层级和组成关系清楚",
            "framework": "中文框架图，模块结构清晰，适合论文或答辩展示",
        }
        style_text = style_map.get(diagram_kind, f"中文{diagram_label}")
        source_block = ""
        if source_text:
            source_block = (
                f"\n请严格依据以下资料生成，不要凭空添加无关模块：\n"
                f"标题：{title_hint or '未命名资料'}\n"
                f"资料内容：\n{source_text[:5200]}"
            )
        else:
            source_block = f"\n请直接依据用户要求生成：{command}"
        return (
            f"请生成一张{style_text}图片。"
            "整体要求：白色背景，横版信息图风格，蓝灰色主色，中文文字清晰可读，排版简洁专业，"
            "适合毕业设计、课程汇报或答辩展示。"
            "只生成图表本身，不要照片风格，不要人物，不要水印，不要多余装饰，不要界面截图效果。"
            "请保证节点名称简洁，连线/箭头明确，层次分明。"
            f"{source_block}"
        )

    def _persist_generated_image_result(self, image_result, output_path):
        target = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        result_type = str((image_result or {}).get("type") or "").strip()
        result_data = (image_result or {}).get("data")
        if result_type == "b64_json":
            try:
                image_bytes = base64.b64decode(result_data)
            except Exception as exc:
                raise StudyAssistantError(f"图片生成结果解码失败：{exc}") from exc
            with open(target, "wb") as handle:
                handle.write(image_bytes)
            return target
        if result_type == "url":
            try:
                request = urllib.request.Request(str(result_data), headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=25) as response, open(target, "wb") as handle:
                    handle.write(response.read())
            except Exception as exc:
                raise StudyAssistantError(f"下载生成图片失败：{exc}") from exc
            return target
        raise StudyAssistantError("图片生成没有返回可保存的结果。")

    def _execute_material_collection(self, task, command):
        topic = task.get("topic", "").strip()
        if not topic:
            raise StudyAssistantError("学习资料采集需要明确主题。")

        prefer_brief = any(token in (command or "") for token in ("简短", "精简", "简要", "简单一些", "不用太详细"))
        source_limit = 2 if prefer_brief else 3
        page_chars = 1800 if prefer_brief else 2800
        bundle_limit = 6000 if prefer_brief else 9000

        sources = search_learning_sources(topic, command=command, limit=source_limit)
        if not sources:
            sources = build_curated_learning_sources(topic, command=command, limit=source_limit)
        if not sources:
            raise StudyAssistantError("暂时没有搜索到合适的学习资料来源，你可以换一个更具体的主题再试。")

        fetched = []
        for source in sources:
            try:
                page = fetch_page_content(source["url"], timeout=8, max_chars=page_chars)
            except Exception:
                continue
            fetched.append(
                {
                    "title": page["title"],
                    "url": page["url"],
                    "content": page["content"],
                }
            )
            if len(fetched) >= source_limit:
                break

        combined_text = []
        if fetched:
            for index, item in enumerate(fetched, start=1):
                combined_text.append(f"[资料{index}] {item['title']}\n链接：{item['url']}\n内容：\n{item['content']}")
        else:
            for index, item in enumerate(sources, start=1):
                combined_text.append(
                    f"[推荐来源{index}] {item.get('title', '未命名来源')}\n"
                    f"链接：{item.get('url', '')}\n"
                    f"线索：{item.get('snippet', '该来源适合继续检索相关学习内容。')}"
                )
        source_bundle = "\n\n".join(combined_text)[:bundle_limit]
        source_mode_text = "已提取网页正文" if fetched else "网页正文提取不足，以下内容以推荐学习来源和检索入口为主"
        output_format = (task.get("output_format") or "").lower()
        wants_xlsx = output_format == "xlsx" or str((task.get("save_plan") or {}).get("main_path") or "").lower().endswith(".xlsx")

        def build_material_prompt(concise=False):
            if wants_xlsx:
                return (
                    f"请围绕主题《{topic}》整理一份适合导出为 Excel 的学习资料表，并使用 Markdown 表格输出。\n"
                    "请优先输出 1 个主表格，表头固定为：模块 | 知识点/任务 | 核心内容 | 公式/例题 | 易错点 | 复习建议。\n"
                    "要求：\n"
                    "1. 每一行只写一个知识点或学习任务，便于写入 Excel。\n"
                    "2. 内容中等详细，但每个单元格不要太长。\n"
                    "3. 如果资料不足，可以基于主题和来源线索整理学习路径，但不要编造具体来源没有的信息。\n"
                    "4. 公式使用纯文本，例如 x^2、sqrt(x)、(a+b)/c，避免 LaTeX 乱码。\n\n"
                    f"用户原始要求：{command}\n\n"
                    f"资料情况说明：{source_mode_text}\n\n"
                    f"资料内容如下：\n{source_bundle[:5200]}"
                )
            if concise:
                return (
                    f"请围绕主题《{topic}》整理一份简洁版学习资料包，并使用 Markdown 输出。\n"
                    "请包含以下一级标题：\n"
                    "# 学习资料总览\n"
                    "# 学习笔记\n"
                    "# 复习提纲\n"
                    "# 参考来源\n"
                    "内容控制在适中长度，优先保证能快速生成可用结果。\n"
                    "重点写：学习目标、核心知识点、推荐学习顺序、练习建议。\n\n"
                    f"用户原始要求：{command}\n\n"
                    f"资料情况说明：{source_mode_text}\n\n"
                    f"资料内容如下：\n{source_bundle[:4200]}"
                )
            return (
                f"请围绕主题《{topic}》整理一份学习资料包，并使用 Markdown 输出。\n"
                "请严格包含以下几个一级标题：\n"
                "# 学习资料总览\n"
                "# 学习笔记\n"
                "# 复习提纲\n"
                "# 思维导图\n"
                "# 参考来源\n"
                "其中“思维导图”部分请输出 Mermaid mindmap 代码块。\n"
                "内容要适合学生自学，结构清晰，语言简洁，尽量做到中等详细度，不要太短，也不要写得过长。\n"
                "请重点补充：学习目标、核心知识点、推荐学习顺序、典型题型/方法、易错点、练习建议。\n"
                "学习笔记部分尽量分成 3 到 4 个小节，复习提纲尽量分层清楚。\n"
                "如果资料主要来自推荐来源而不是正文，请基于主题和来源线索整理一份可直接上手的学习方案。\n\n"
                f"用户原始要求：{command}\n\n"
                f"资料情况说明：{source_mode_text}\n\n"
                f"资料内容如下：\n{source_bundle}"
            )

        try:
            content = self._chat_for_learning_task(
                build_material_prompt(concise=prefer_brief),
                self._build_learning_context(topic, "学习资料采集"),
                max_tokens_override=1400 if prefer_brief else 1800,
                retry_tokens_override=1000 if prefer_brief else 1200,
            )
        except VoiceLLMControllerError as exc:
            if "超时" not in str(exc) and "timed out" not in str(exc):
                raise
            content = self._chat_for_learning_task(
                build_material_prompt(concise=True),
                self._build_learning_context(topic, "学习资料采集"),
                max_tokens_override=1000,
                retry_tokens_override=800,
            )
        save_plan = task.get("save_plan") or {}
        main_path, mindmap_path, persisted_content = self._persist_learning_outputs(
            content,
            save_plan,
            task.get("topic") or "学习资料包",
        )
        sources_text = "\n\n".join(
            [
                f"{index}. {item.get('title', '未命名来源')}\n{item.get('url', '')}\n{item.get('snippet', '')}"
                for index, item in enumerate(fetched or sources, start=1)
            ]
        )
        sources_path = save_text(save_plan.get("sources_path"), sources_text)
        preview = self._shorten_reply(persisted_content, 180)
        mode_line = f"共整理 {len(fetched)} 个可读来源" if fetched else f"已整理 {len(sources)} 个推荐来源"
        opened_sources = task.get("opened_sources") or []
        opened_line = ""
        if opened_sources:
            opened_titles = "、".join(item.get("title", "相关页面") for item in opened_sources[:2])
            if len(opened_sources) > 2:
                opened_titles += f" 等 {len(opened_sources)} 个页面"
            opened_line = f"我也已经为你打开：{opened_titles}\n"
        mindmap_line = f"\n思维导图图片：{mindmap_path}" if mindmap_path else ""
        return {
            "reply": f"{opened_line}已为你整理《{topic}》的学习资料，{mode_line}，并保存到：\n{main_path}{mindmap_line}\n来源清单：{sources_path}\n\n预览：{preview}",
            "saved_path": main_path,
            "extra_path": sources_path,
            "mindmap_path": mindmap_path,
        }

    def _execute_learning_processing(self, task, command):
        outputs = task.get("outputs") or ["notes"]
        source_kind = task.get("source_kind")
        source_value = task.get("source_value")

        if source_kind == "attachment":
            content = self._generate_learning_doc_from_attachment(task.get("attachment") or source_value, outputs, command)
            title_hint = (task.get("attachment") or source_value or {}).get("name") or "学习资料"
        elif source_kind == "url":
            page = fetch_page_content(source_value, timeout=8, max_chars=7000)
            content = self._generate_learning_doc_from_text(page["content"], page["title"], outputs, command, page["url"])
            title_hint = page["title"]
        elif source_kind == "urls":
            pages = []
            for item in list(source_value or [])[:4]:
                url = (item or {}).get("url", "")
                if not url:
                    continue
                try:
                    page = fetch_page_content(url, timeout=8, max_chars=4500)
                except Exception:
                    continue
                pages.append(page)
            if not pages:
                raise StudyAssistantError("当前标签页还没有可读取的正文内容，你可以先打开具体文章/课程页再试。")
            joined = []
            for index, page in enumerate(pages, start=1):
                joined.append(f"[页面{index}] {page['title']}\n链接：{page['url']}\n内容：\n{page['content']}")
            title_hint = f"当前标签页汇总（{len(pages)}页）"
            content = self._generate_learning_doc_from_text("\n\n".join(joined), title_hint, outputs, command)
        else:
            raise StudyAssistantError("请先上传文件/图片，或在问题里附上网页链接，也可以让我整理当前打开的页面。")

        save_plan = task.get("save_plan") or {}
        main_path, mindmap_path, persisted_content = self._persist_learning_outputs(content, save_plan, title_hint)
        preview = self._shorten_reply(persisted_content, 180)
        mindmap_line = f"\n思维导图图片：{mindmap_path}" if mindmap_path else ""
        return {
            "reply": f"已完成学习化处理，并保存到：\n{main_path}{mindmap_line}\n\n预览：{preview}",
            "saved_path": main_path,
            "mindmap_path": mindmap_path,
            "title_hint": title_hint,
        }

    def _generate_learning_doc_from_attachment(self, attachment, outputs, command):
        attachment = dict(attachment or {})
        if not attachment.get("path"):
            raise StudyAssistantError("当前没有可处理的附件。")
        attachment_kind = str(attachment.get("kind") or "").strip().lower()
        if attachment_kind == "image":
            return self._generate_learning_doc_from_image_attachment(attachment, outputs, command)
        if attachment_kind == "file":
            return self._generate_learning_doc_from_file_attachment(attachment, outputs, command)
        prompt = self._learning_prompt_from_outputs(outputs, command)
        return self._chat_for_learning_task(
            prompt,
            self._build_learning_context(attachment.get("name") or "附件内容", "学习化处理"),
            attachment=attachment,
        )

    def _generate_learning_doc_from_file_attachment(self, attachment, outputs, command):
        attachment = dict(attachment or {})
        path = str(attachment.get("path") or "").strip()
        title_hint = attachment.get("name") or "附件内容"
        if not path or not os.path.exists(path):
            raise StudyAssistantError("当前没有可处理的文件附件。")

        try:
            extracted_text = self.voice_llm_controller._extract_file_text(path)
        except Exception as exc:
            raise StudyAssistantError(f"读取文件内容失败：{exc}") from exc

        draft_doc = self._build_basic_learning_doc_from_text_extract(extracted_text, outputs, title_hint)
        prompt = (
            f"{self._build_file_learning_enhancement_prompt(outputs, command)}\n\n"
            f"资料标题：{title_hint}\n"
            "下面是从文件中提取出的原始内容节选：\n"
            f"{(extracted_text or '')[:5200]}\n\n"
            "下面是一份已经生成的草稿，请在不编造事实的前提下，把它整理得更自然、更清楚一些：\n"
            f"{draft_doc}"
        )
        try:
            return self._chat_for_learning_task(
                prompt,
                self._build_learning_context(title_hint, "学习化处理"),
                max_tokens_override=760,
                retry_tokens_override=520,
            )
        except VoiceLLMControllerError as exc:
            error_text = str(exc)
            if "超时" not in error_text and "timed out" not in error_text and "连接失败" not in error_text:
                raise
            return draft_doc

    def _generate_learning_doc_from_image_attachment(self, attachment, outputs, command):
        attachment = dict(attachment or {})
        title_hint = attachment.get("name") or "图片内容"
        is_screen_capture = self._is_screen_capture_attachment(attachment)
        screen_note = (
            "这张图片是用户授权截取的当前屏幕内容。请优先关注屏幕中央或主要窗口中的学习内容，"
            "忽略桌面背景、应用边框、通知、无关图标和小睿自身窗口。\n"
            if is_screen_capture
            else ""
        )
        extraction_prompt = screen_note + (
            "请先读取这张图片，并把它整理成后续可用于生成学习笔记的原始素材。\n"
            "要求：\n"
            "1. 尽量准确提取图片中的标题、要点、层级结构、模块关系、关键术语和公式。\n"
            "2. 如果图片里有图示、框架图、结构图或流程，请把结构关系按条目写清楚。\n"
            "3. 不要直接写成学习笔记，也不要长篇发挥，只做内容提取。\n"
            "4. 输出格式尽量使用下面的结构：\n"
            "# 图片主题\n"
            "# 关键内容\n"
            "# 结构关系\n"
            "# 可识别文字与公式\n"
        )
        try:
            extracted_source = self._chat_for_learning_task(
                extraction_prompt,
                self._build_learning_context(title_hint, "图片内容提取"),
                attachment=attachment,
                max_tokens_override=180,
                retry_tokens_override=120,
            )
        except VoiceLLMControllerError as exc:
            error_text = str(exc)
            if "超时" not in error_text and "timed out" not in error_text and "连接失败" not in error_text:
                raise
            fallback_prompt = (
                "请快速阅读这张图片，只提取最关键的信息，输出尽量短。\n"
                "只保留这四部分：\n"
                "# 图片主题\n"
                "# 关键内容\n"
                "# 结构关系\n"
                "# 可识别文字与公式\n"
            )
            extracted_source = self._chat_for_learning_task(
                fallback_prompt,
                self._build_learning_context(title_hint, "图片内容提取"),
                attachment=attachment,
                max_tokens_override=120,
                retry_tokens_override=90,
            )
        prompt = (
            f"{self._build_image_learning_enhancement_prompt(outputs, command)}\n\n"
            f"资料标题：{title_hint}\n"
            "下面是从图片中提取出的原始学习素材：\n"
            f"{(extracted_source or '')[:2600]}\n\n"
            "下面是一份已经生成的学习笔记草稿，请在不编造事实的前提下，把它整理得更自然、更完整一些：\n"
            f"{self._build_basic_learning_doc_from_image_extract(extracted_source, outputs, title_hint)}"
        )
        draft_doc = self._build_basic_learning_doc_from_image_extract(extracted_source, outputs, title_hint)
        try:
            return self._chat_for_learning_task(
                prompt,
                self._build_learning_context(title_hint, "学习化处理"),
                max_tokens_override=700,
                retry_tokens_override=480,
            )
        except VoiceLLMControllerError as exc:
            error_text = str(exc)
            if "超时" not in error_text and "timed out" not in error_text and "连接失败" not in error_text:
                raise
            return draft_doc

    def _build_basic_learning_doc_from_image_extract(self, extracted_source, outputs, title_hint):
        outputs = outputs or ["notes"]
        parsed = self._parse_extracted_image_sections(extracted_source)
        topic = parsed.get("topic") or (title_hint or "当前图片内容")
        key_points = parsed.get("key_points") or ["请结合原图进一步核对关键细节。"]
        structure_points = parsed.get("structure_points") or ["原图主要体现了若干模块或知识点之间的关系。"]
        formula_points = parsed.get("formula_points") or ["本次图片中未稳定提取出更多公式，建议结合原图查看。"]
        summary_sentence = self._compose_image_summary_sentence(topic, key_points, structure_points)

        parts = []
        if "table" in outputs:
            parts.extend(
                [
                    "# Excel 表格",
                    "| 模块 | 知识点/任务 | 核心内容 | 公式/例题 | 易错点 | 复习建议 |",
                    "| --- | --- | --- | --- | --- | --- |",
                    *[
                        f"| 图片内容 | {item} | {summary_sentence} | {formula_points[idx] if idx < len(formula_points) else ''} | 结合原图核对细节 | 先理解该项，再回到原图复述 |"
                        for idx, item in enumerate(key_points[:8])
                    ],
                    "",
                ]
            )
        if "notes" in outputs:
            parts.extend(
                [
                    "# 学习笔记",
                    f"## 学习主题\n{topic}",
                    f"## 内容概览\n{summary_sentence}",
                    "## 学习目标",
                    "- 先理解图片想表达的整体主题和核心结构。",
                    "- 再抓住关键概念、模块关系和容易混淆的部分。",
                    "## 关键内容",
                    *[f"- {item}" for item in key_points[:8]],
                    "## 结构关系",
                    *[f"- {item}" for item in structure_points[:6]],
                    "## 可识别文字与公式",
                    *[f"- {item}" for item in formula_points[:6]],
                    "## 建议理解顺序",
                    "- 先看标题或主题，明确这张图在讲什么。",
                    "- 再按结构关系理解模块之间的连接方式。",
                    "- 最后结合关键术语与公式回到原图核对细节。",
                    "## 复习建议",
                    "- 先通读图片中的整体框架，再逐个理解各模块含义。",
                    "- 遇到术语或公式时，建议结合教材原文进一步核对。",
                    "- 如果这是知识框架图，建议把各节点转成自己的口头复述。",
                    "",
                ]
            )
        if "outline" in outputs:
            parts.extend(
                [
                    "# 复习提纲",
                    f"1. {topic}",
                    *[f"{idx + 2}. {item}" for idx, item in enumerate(key_points[:6])],
                    "",
                ]
            )
        if "mindmap" in outputs:
            parts.extend(
                [
                    "# 思维导图",
                    self._build_basic_image_mindmap_mermaid(topic, key_points, structure_points, formula_points),
                    "",
                ]
            )
        return "\n".join(parts).strip()

    def _build_basic_learning_doc_from_text_extract(self, extracted_text, outputs, title_hint):
        outputs = outputs or ["notes"]
        topic, headings, key_points = self._parse_text_source_for_learning(extracted_text, title_hint)
        parts = []
        if "table" in outputs:
            table_items = key_points[:10] or headings[:10] or [topic]
            parts.extend(
                [
                    "# Excel 表格",
                    "| 模块 | 知识点/任务 | 核心内容 | 公式/例题 | 易错点 | 复习建议 |",
                    "| --- | --- | --- | --- | --- | --- |",
                    *[
                        f"| {headings[idx % len(headings)] if headings else topic} | {item} | 围绕“{topic}”展开的关键内容 |  | 结合原文核对定义和条件 | 做成卡片并配合例题复习 |"
                        for idx, item in enumerate(table_items)
                    ],
                    "",
                ]
            )
        if "notes" in outputs:
            parts.extend(
                [
                    "# 学习笔记",
                    f"## 学习主题\n{topic}",
                    "## 内容概览",
                    f"这份资料主要围绕“{topic}”展开，内容重点集中在若干核心知识点和层级关系上。",
                    "## 关键内容",
                    *[f"- {item}" for item in key_points[:10]],
                    "## 建议理解顺序",
                    *[f"- {item}" for item in headings[:6]],
                    "## 复习建议",
                    "- 先按章节或模块理解整体结构，再逐条回看关键内容。",
                    "- 对公式、定义和结论，建议结合原文再次核对。",
                    "",
                ]
            )
        if "outline" in outputs:
            parts.extend(
                [
                    "# 复习提纲",
                    f"1. {topic}",
                    *[f"{idx + 2}. {item}" for idx, item in enumerate(headings[:8] or key_points[:8])],
                    "",
                ]
            )
        if "mindmap" in outputs:
            parts.extend(
                [
                    "# 思维导图",
                    self._build_basic_mindmap_mermaid(topic, headings, key_points),
                    "",
                ]
            )
        return "\n".join(parts).strip()

    def _parse_extracted_image_sections(self, extracted_source):
        text = (extracted_source or "").strip()
        result = {
            "topic": "",
            "key_points": [],
            "structure_points": [],
            "formula_points": [],
        }
        if not text:
            return result
        current = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized = line.lstrip("#").strip()
            if normalized.startswith("图片主题"):
                current = "topic"
                continue
            if normalized.startswith("关键内容"):
                current = "key_points"
                continue
            if normalized.startswith("结构关系"):
                current = "structure_points"
                continue
            if normalized.startswith("可识别文字与公式"):
                current = "formula_points"
                continue
            cleaned = re.sub(r"^[\-\*\d\.\)\(、\s]+", "", line).strip()
            if not cleaned:
                continue
            if current == "topic":
                if not result["topic"]:
                    result["topic"] = cleaned
            elif current in {"key_points", "structure_points", "formula_points"}:
                result[current].append(cleaned)
            elif not result["topic"]:
                result["topic"] = cleaned
            else:
                result["key_points"].append(cleaned)
        return result

    def _parse_text_source_for_learning(self, extracted_text, title_hint):
        text = (extracted_text or "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_lines = [re.sub(r"^[#>\-\*\d\.\)\(、\s]+", "", line).strip() for line in lines]
        cleaned_lines = [line for line in cleaned_lines if line]
        topic = cleaned_lines[0] if cleaned_lines else (title_hint or "学习资料")
        headings = []
        key_points = []
        for line in cleaned_lines[1:]:
            if len(headings) < 10 and (len(line) <= 30 or any(token in line for token in ("章", "节", "模块", "部分", "定义", "定理", "步骤", "方法"))):
                if line not in headings:
                    headings.append(line)
            if len(key_points) < 12 and line not in key_points:
                key_points.append(line)
        if not headings:
            headings = key_points[:6]
        if not key_points:
            key_points = [topic]
        return topic, headings, key_points

    def _build_basic_mindmap_mermaid(self, topic, headings, key_points):
        root = self._mermaid_safe_label(topic or "学习资料")
        primary_nodes = headings[:6] or key_points[:6]
        detail_nodes = key_points[:10]
        lines = [
            "```mermaid",
            "mindmap",
            f"  root(({root}))",
        ]
        for index, node in enumerate(primary_nodes, start=1):
            primary = self._mermaid_safe_label(node)
            lines.append(f"    {primary}")
            for detail in detail_nodes[(index - 1) * 2:index * 2]:
                detail_label = self._mermaid_safe_label(detail)
                if detail_label != primary:
                    lines.append(f"      {detail_label}")
        lines.append("```")
        return "\n".join(lines)

    def _mermaid_safe_label(self, text):
        cleaned = re.sub(r"[`\"]+", "", (text or "").strip())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:40] or "节点"

    def _compose_image_summary_sentence(self, topic, key_points, structure_points):
        lead = key_points[0] if key_points else "当前图片主要围绕若干知识点展开"
        relation = structure_points[0] if structure_points else "并展示了这些内容之间的结构关系"
        return f"这张图片主要围绕“{topic}”展开，重点内容包括“{lead}”，同时{relation}。"

    def _build_basic_image_mindmap_mermaid(self, topic, key_points, structure_points, formula_points):
        root = self._mermaid_safe_label(topic or "图片内容")
        primary_nodes = (structure_points[:4] or key_points[:4] or [topic])[:4]
        detail_nodes = (key_points[:8] + formula_points[:4])[:12]
        lines = [
            "```mermaid",
            "mindmap",
            f"  root(({root}))",
        ]
        for index, node in enumerate(primary_nodes, start=1):
            primary = self._mermaid_safe_label(node)
            lines.append(f"    {primary}")
            for detail in detail_nodes[(index - 1) * 2:index * 2]:
                detail_label = self._mermaid_safe_label(detail)
                if detail_label != primary:
                    lines.append(f"      {detail_label}")
        lines.append("```")
        return "\n".join(lines)

    def _build_image_learning_enhancement_prompt(self, outputs, command):
        outputs = outputs or ["notes"]
        sections = []
        if "notes" in outputs:
            sections.append("学习笔记")
        if "outline" in outputs:
            sections.append("复习提纲")
        if "mindmap" in outputs:
            sections.append("思维导图（如果无法稳定生成导图，可保留说明文字）")
        if "table" in outputs:
            sections.append("Excel 表格（使用 Markdown 表格，表头清晰）")
        return (
            "你是一名中文学习助手，请基于用户上传图片中提取出的素材和现有草稿，生成一份更自然、更完整的 Markdown 学习文档。\n"
            "要求：\n"
            "1. 以现有草稿为基础润色和补充，不要推翻结构。\n"
            "2. 内容要比草稿稍微丰富，但不要冗长，优先保证能快速生成。\n"
            "3. 不要编造图片中没有出现的知识点，可以适度补成更清楚的学习表达。\n"
            "4. 如果涉及公式，请使用纯文本可读写法，例如 x^2、sqrt(x)、(a+b)/c。\n"
            "5. 如果包含 Excel 表格，请输出 Markdown 表格，推荐表头：模块 | 知识点/任务 | 核心内容 | 公式/例题 | 易错点 | 复习建议。\n"
            f"6. 用户当前需要的输出包括：{', '.join(sections)}。\n"
            f"用户原始要求：{command}\n"
        )

    def _build_file_learning_enhancement_prompt(self, outputs, command):
        outputs = outputs or ["notes"]
        sections = []
        if "notes" in outputs:
            sections.append("学习笔记")
        if "outline" in outputs:
            sections.append("复习提纲")
        if "mindmap" in outputs:
            sections.append("思维导图")
        if "table" in outputs:
            sections.append("Excel 表格（使用 Markdown 表格，表头清晰）")
        return (
            "你是一名中文学习助手，请根据用户提供的文件内容节选和草稿，输出一份更清楚、更适合复习的 Markdown 文档。\n"
            "要求：\n"
            "1. 以现有草稿为基础润色，不要完全推翻结构。\n"
            "2. 内容要比草稿更自然，但不要过长，优先保证稳定生成。\n"
            "3. 不要编造原文中没有的知识点。\n"
            "4. 如果包含思维导图，请输出 Mermaid mindmap 代码块。\n"
            "5. 如果包含 Excel 表格，请输出 Markdown 表格，推荐表头：模块 | 知识点/任务 | 核心内容 | 公式/例题 | 易错点 | 复习建议。\n"
            f"6. 用户当前需要的输出包括：{', '.join(sections)}。\n"
            f"用户原始要求：{command}\n"
        )

    def _generate_learning_doc_from_text(self, source_text, title_hint, outputs, command, source_url=""):
        url_line = f"资料链接：{source_url}\n" if source_url else ""
        prompt = (
            f"{self._learning_prompt_from_outputs(outputs, command)}\n\n"
            f"资料标题：{title_hint}\n"
            f"{url_line}"
            "下面是可供整理的原始内容，请基于这些内容输出：\n"
            f"{source_text[:14000]}"
        )
        return self._chat_for_learning_task(
            prompt,
            self._build_learning_context(title_hint or "网页内容", "学习化处理"),
        )

    def _learning_prompt_from_outputs(self, outputs, command):
        outputs = outputs or ["notes"]
        sections = []
        if "table" in outputs:
            sections.append("# Excel 表格（使用 Markdown 表格）")
        if "notes" in outputs:
            sections.append("# 学习笔记")
        if "outline" in outputs:
            sections.append("# 复习提纲")
        if "mindmap" in outputs:
            sections.append("# 思维导图（使用 Mermaid mindmap 代码块）")
        section_text = "\n".join(sections)
        return (
            "你是一名中文学习助手，请把用户提供的学习资料整理成适合复习的 Markdown 文档。\n"
            "要求：\n"
            "1. 语言清晰、简洁、像真正的学习笔记，但内容不能过于单薄。\n"
            "2. 只输出用户要求的部分。\n"
            "3. 如果包含“思维导图”，请输出 Mermaid mindmap 代码块，结构清晰、层级不要过深。\n"
            "4. 如果包含“Excel 表格”，请输出 Markdown 表格，推荐表头：模块 | 知识点/任务 | 核心内容 | 公式/例题 | 易错点 | 复习建议。\n"
            "5. 表格内容要适合写入 Excel：一行一个知识点或任务，单元格不要过长，公式用纯文本避免乱码。\n"
            "6. 不要编造资料中没有的知识点，但可以把已有信息整理成更完整的学习结构。\n"
            "7. 尽量补充学习目标、关键知识点、方法总结、常见误区、复习建议。\n"
            "8. 如果内容允许，请给出分层小标题，不要只写一段概述。\n"
            f"用户原始要求：{command}\n"
            f"请至少包含这些一级标题：\n{section_text}"
        )

    def _persist_learning_outputs(self, content, save_plan, title_hint):
        save_plan = dict(save_plan or {})
        main_path = save_plan.get("main_path")
        mindmap_path = save_plan.get("mindmap_path")
        main_is_mindmap = bool(save_plan.get("main_is_mindmap"))

        saved_mindmap_path = None
        persisted_content = content
        mermaid_text = self._extract_mermaid_mindmap_block(content)

        if mindmap_path and mermaid_text:
            saved_mindmap_path = self._render_mindmap_with_model_fallback(
                mermaid_text,
                mindmap_path,
                title_hint or "思维导图",
            )
            if main_is_mindmap:
                return saved_mindmap_path, saved_mindmap_path, "# 思维导图\n已生成导图图片。"
            persisted_content = self._replace_mindmap_block_with_image_note(content, saved_mindmap_path)
        elif main_is_mindmap:
            raise StudyAssistantError("这次没有生成可渲染的思维导图，请再试一次。")

        saved_main_path = save_document(main_path, persisted_content)
        return saved_main_path, saved_mindmap_path, persisted_content

    def _extract_mermaid_mindmap_block(self, content):
        text = (content or "").strip()
        if not text:
            return ""
        match = re.search(r"```mermaid\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        block = match.group(1).strip()
        return block if "mindmap" in block.lower() else ""

    def _replace_mindmap_block_with_image_note(self, content, image_path):
        image_name = os.path.basename(image_path)
        replaced = re.sub(
            r"```mermaid\s*.*?```",
            f"思维导图图片已生成：{image_name}",
            content or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        return re.sub(r"\n{3,}", "\n\n", replaced).strip()

    def _parse_mermaid_mindmap_tree(self, mermaid_text):
        lines = []
        for raw_line in (mermaid_text or "").splitlines():
            if not raw_line.strip():
                continue
            stripped = raw_line.strip()
            if stripped.lower() == "mindmap" or stripped.startswith("%%") or stripped.startswith("::icon"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" \t"))
            lines.append((indent, stripped))
        if not lines:
            return None

        def clean_label(text):
            text = re.sub(r"::icon\([^)]*\)", "", text).strip()
            patterns = [
                r"^[A-Za-z0-9_]+\(\((.+)\)\)$",
                r"^[A-Za-z0-9_]+\[\[(.+)\]\]$",
                r"^[A-Za-z0-9_]+\[(.+)\]$",
                r"^[A-Za-z0-9_]+\{\{(.+)\}\}$",
                r"^\(\((.+)\)\)$",
                r"^\[\[(.+)\]\]$",
                r"^\[(.+)\]$",
                r"^\{\{(.+)\}\}$",
            ]
            for pattern in patterns:
                match = re.match(pattern, text)
                if match:
                    text = match.group(1).strip()
                    break
            text = re.sub(r"^[A-Za-z0-9_]+\s+", "", text).strip()
            return text or "节点"

        root = None
        stack = []
        for indent, raw_text in lines:
            node = {"text": clean_label(raw_text), "children": []}
            while stack and indent <= stack[-1][0]:
                stack.pop()
            if not stack:
                if root is None:
                    root = node
                else:
                    root.setdefault("children", []).append(node)
                stack.append((indent, node))
                continue
            stack[-1][1]["children"].append(node)
            stack.append((indent, node))
        return root

    def _render_mindmap_image(self, mermaid_text, output_path, title_hint):
        tree = self._parse_mermaid_mindmap_tree(mermaid_text)
        if not tree:
            raise StudyAssistantError("暂时没法把这次导图内容渲染成图片，请再试一次。")

        font = QFont("PingFang SC", 11)
        title_font = QFont("PingFang SC", 16, QFont.Bold)
        metrics = QFontMetrics(font)
        title_metrics = QFontMetrics(title_font)
        horizontal_gap = 34
        vertical_gap = 70
        margin = 28
        max_text_width = 220

        def measure(node):
            text_rect = metrics.boundingRect(0, 0, max_text_width, 1000, Qt.TextWordWrap, node["text"])
            node_width = max(108, text_rect.width() + 28)
            node_height = max(50, text_rect.height() + 22)
            node["_node_width"] = node_width
            node["_node_height"] = node_height
            children = node.get("children", [])
            if not children:
                node["_subtree_width"] = node_width
                node["_subtree_height"] = node_height
                return node["_subtree_width"], node["_subtree_height"]
            child_sizes = [measure(child) for child in children]
            total_children_width = sum(width for width, _ in child_sizes) + horizontal_gap * (len(children) - 1)
            node["_subtree_width"] = max(node_width, total_children_width)
            node["_subtree_height"] = node_height + vertical_gap + max(height for _, height in child_sizes)
            return node["_subtree_width"], node["_subtree_height"]

        def assign(node, left, top):
            subtree_width = node["_subtree_width"]
            node_width = node["_node_width"]
            node["_x"] = left + (subtree_width - node_width) / 2
            node["_y"] = top
            children = node.get("children", [])
            if not children:
                return
            total_children_width = sum(child["_subtree_width"] for child in children) + horizontal_gap * (len(children) - 1)
            child_left = left + (subtree_width - total_children_width) / 2
            child_top = top + node["_node_height"] + vertical_gap
            for child in children:
                assign(child, child_left, child_top)
                child_left += child["_subtree_width"] + horizontal_gap

        tree_width, tree_height = measure(tree)
        assign(tree, margin, margin + title_metrics.height() + 24)

        image_width = int(tree_width + margin * 2)
        image_height = int(tree_height + margin * 2 + title_metrics.height() + 24)
        image = QImage(image_width, image_height, QImage.Format_ARGB32)
        image.fill(QColor("#ffffff"))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setFont(title_font)
        painter.setPen(QColor("#1f2937"))
        painter.drawText(margin, margin + title_metrics.ascent(), title_hint or "思维导图")

        line_pen = QPen(QColor("#94a3b8"))
        line_pen.setWidth(2)
        painter.setPen(line_pen)

        def draw_edges(node):
            parent_center_x = node["_x"] + node["_node_width"] / 2
            parent_bottom_y = node["_y"] + node["_node_height"]
            for child in node.get("children", []):
                child_center_x = child["_x"] + child["_node_width"] / 2
                child_top_y = child["_y"]
                middle_y = parent_bottom_y + (child_top_y - parent_bottom_y) / 2
                painter.drawLine(int(parent_center_x), int(parent_bottom_y), int(parent_center_x), int(middle_y))
                painter.drawLine(int(parent_center_x), int(middle_y), int(child_center_x), int(middle_y))
                painter.drawLine(int(child_center_x), int(middle_y), int(child_center_x), int(child_top_y))
                draw_edges(child)

        def draw_nodes(node, depth=0):
            rect_x = int(node["_x"])
            rect_y = int(node["_y"])
            rect_w = int(node["_node_width"])
            rect_h = int(node["_node_height"])
            fill_color = QColor("#dbeafe") if depth == 0 else QColor("#eff6ff") if depth == 1 else QColor("#f8fafc")
            border_color = QColor("#3b82f6") if depth == 0 else QColor("#93c5fd") if depth == 1 else QColor("#cbd5e1")
            painter.setPen(QPen(border_color, 2 if depth == 0 else 1))
            painter.setBrush(QBrush(fill_color))
            painter.drawRoundedRect(rect_x, rect_y, rect_w, rect_h, 14, 14)
            painter.setFont(font)
            painter.setPen(QColor("#111827"))
            painter.drawText(rect_x + 14, rect_y + 11, rect_w - 28, rect_h - 22, Qt.AlignCenter | Qt.TextWordWrap, node["text"])
            for child in node.get("children", []):
                draw_nodes(child, depth + 1)

        draw_edges(tree)
        draw_nodes(tree)
        painter.end()

        target = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if not image.save(target, "PNG"):
            raise StudyAssistantError("思维导图图片保存失败，请再试一次。")
        return target

    def _render_mindmap_with_model_fallback(self, mermaid_text, output_path, title_hint):
        if DOUBAO_LLM_API_KEY:
            try:
                return self._render_mindmap_with_doubao(mermaid_text, output_path, title_hint)
            except (StudyAssistantError, VoiceLLMControllerError):
                pass
        return self._render_mindmap_image(mermaid_text, output_path, title_hint)

    def _render_mindmap_with_doubao(self, mermaid_text, output_path, title_hint):
        tree = self._parse_mermaid_mindmap_tree(mermaid_text)
        if not tree:
            raise StudyAssistantError("暂时没法把这次导图内容整理成可出图结构，请再试一次。")
        outline_lines = self._mindmap_tree_to_outline_lines(tree)
        prompt = (
            f"请生成一张中文思维导图图片，标题为《{title_hint or '思维导图'}》。\n"
            "要求：\n"
            "1. 白底，蓝灰色学术风，清晰、规整、适合答辩展示。\n"
            "2. 中心主题突出，分支层级明确，文本不要乱码。\n"
            "3. 用中文展示节点，避免过度装饰，整体像正式脑图而不是海报。\n"
            "4. 布局紧凑，边距合理，图片里不要出现水印说明文字。\n"
            "以下是需要体现在图中的导图结构：\n"
            f"{outline_lines}"
        )
        controller = self._build_learning_task_controller("doubao:doubao-seedream-5-0-lite-260128")
        try:
            image_result = controller.generate_image(prompt, size="2K")
        except VoiceLLMControllerError as exc:
            raise StudyAssistantError(str(exc)) from exc
        return self._persist_generated_image_result(image_result, output_path)

    def _mindmap_tree_to_outline_lines(self, tree):
        lines = []

        def walk(node, depth=0):
            text = (node or {}).get("text") or "节点"
            lines.append(f"{'  ' * depth}- {text}")
            for child in (node or {}).get("children", []):
                walk(child, depth + 1)

        walk(tree)
        return "\n".join(lines)

    def _build_learning_context(self, title_hint, scene):
        return {
            "project_profile": "小睿伴学是一套基于视觉感知的在线学习专注力智能识别和生成式伴学系统。",
            "scene": scene,
            "title_hint": title_hint,
            "assistant_capabilities": "你可以把当前资料整理成学习笔记、复习提纲和思维导图。",
        }

    def _answer_current_content_query(self, command, current_source):
        current_source = dict(current_source or {})
        source_kind = str(current_source.get("kind") or "").strip().lower()
        title_hint = current_source.get("title") or "当前内容"
        context = self._build_learning_context(title_hint, "当前内容问答")

        if source_kind == "attachment":
            attachment = dict(current_source.get("value") or {})
            if not attachment.get("path"):
                raise StudyAssistantError("暂时没有读取到可分析的当前图片或文档。")
            prompt = (command or "").strip() or "请帮我解释当前内容。"
            is_image = attachment.get("kind") == "image"
            if is_image and self._is_screen_capture_attachment(attachment):
                prompt = (
                    "这是一张用户授权截取的当前屏幕内容。"
                    "请优先理解屏幕中央或主要窗口中的页面、图片或文档内容，"
                    "忽略桌面背景、应用边框、通知、无关图标和小睿自身窗口。"
                    f"用户问题：{prompt}"
                )
            return self._chat_for_learning_task(
                prompt,
                context,
                attachment=attachment,
                max_tokens_override=320 if is_image else 900,
                retry_tokens_override=220 if is_image else 650,
            )

        if source_kind == "url":
            url = str(current_source.get("value") or "").strip()
            if not url:
                raise StudyAssistantError("暂时没有读取到可分析的当前页面。")
            page = fetch_page_content(url, timeout=8, max_chars=5200)
            prompt = (
                f"用户正在查看一个当前页面《{page.get('title') or title_hint}》。"
                f"用户问题：{(command or '').strip() or '请概括当前页面内容。'}。\n"
                f"页面标题：{page.get('title') or title_hint}\n"
                f"页面网址：{page.get('url') or url}\n"
                f"页面正文节选：\n{page.get('content') or ''}\n"
                "请只基于这些内容回答，回答尽量清楚、自然；如果信息不足，请直接说明。"
            )
            return self._chat_for_learning_task(
                prompt,
                context,
                max_tokens_override=900,
                retry_tokens_override=650,
            )

        raise StudyAssistantError("暂时没有读取到可分析的当前内容。")

    def _learning_task_model_name(self, needs_attachment=False):
        current_spec = self._normalized_remote_llm_model(self.selected_llm_model)
        provider, model_name = self._split_llm_spec(current_spec)

        if needs_attachment and DOUBAO_LLM_API_KEY:
            return "doubao:doubao-seed-1-6-vision-250815"
        if DEEPSEEK_LLM_API_KEY and not needs_attachment:
            return "deepseek:deepseek-chat"
        if provider == "deepseek" and model_name == "deepseek-reasoner":
            return "deepseek:deepseek-chat"
        return current_spec or f"{LOCAL_LLM_PROVIDER}:{LOCAL_LLM_MODEL}"

    def _build_learning_task_controller(self, model_name):
        controller = self.voice_llm_controller
        current_spec = self._normalized_remote_llm_model(self.selected_llm_model)
        target_spec = self._normalized_remote_llm_model(model_name)
        if not target_spec:
            return controller
        if target_spec == current_spec:
            return controller
        provider, bare_model = self._split_llm_spec(target_spec)
        return VoiceLLMController(
            local_enabled=True,
            local_provider=provider,
            local_base_url=self._base_url_for_provider(provider),
            local_model=bare_model,
            local_api_key=self._api_key_for_provider(provider),
            local_timeout=max(24, int(getattr(controller, "local_timeout", None) or 15)),
        )

    def _chat_for_learning_task(self, prompt, context, attachment=None, max_tokens_override=None, retry_tokens_override=None):
        model_name = self._learning_task_model_name(needs_attachment=bool(attachment))
        controller = self._build_learning_task_controller(model_name)
        attachment_kind = str((attachment or {}).get("kind") or "").strip().lower()
        is_doubao_attachment = bool(attachment) and model_name.startswith("doubao:")
        preferred_tokens = (
            int(max_tokens_override)
            if max_tokens_override is not None
            else (260 if is_doubao_attachment and attachment_kind == "image" else 1200 if attachment else 2200)
        )
        retry_tokens = (
            int(retry_tokens_override)
            if retry_tokens_override is not None
            else (180 if is_doubao_attachment and attachment_kind == "image" else 900 if attachment else 1600)
        )
        try:
            if attachment:
                return controller.chat_with_attachment(prompt, context, attachment, max_tokens_override=preferred_tokens)
            return controller.chat(prompt, context, max_tokens_override=preferred_tokens)
        except VoiceLLMControllerError as exc:
            error_text = str(exc)
            current_spec = self._normalized_remote_llm_model(
                f"{getattr(controller, 'local_provider', LOCAL_LLM_PROVIDER)}:{getattr(controller, 'local_model', LOCAL_LLM_MODEL)}"
            )
            if attachment and DOUBAO_LLM_API_KEY:
                stable_spec = "doubao:doubao-seed-1-6-vision-250815"
            elif any(token in error_text for token in ("超时", "timed out", "连接失败")) and DEEPSEEK_LLM_API_KEY and not attachment:
                retry_controller = self._build_learning_task_controller("deepseek:deepseek-chat")
                return retry_controller.chat(prompt, context, max_tokens_override=retry_tokens)
            elif "没有生成最终答案" not in error_text and "只返回了推理过程" not in error_text:
                raise
            elif DEEPSEEK_LLM_API_KEY:
                stable_spec = "deepseek:deepseek-chat"
            else:
                stable_spec = ""
            if not stable_spec or stable_spec == current_spec:
                raise
            retry_controller = self._build_learning_task_controller(stable_spec)
            if attachment:
                return retry_controller.chat_with_attachment(prompt, context, attachment, max_tokens_override=preferred_tokens)
            return retry_controller.chat(prompt, context, max_tokens_override=retry_tokens)

    def _shorten_reply(self, text, limit=180):
        plain = re.sub(r"\s+", " ", (text or "").strip())
        if len(plain) <= limit:
            return plain
        return plain[:limit] + "..."

    def _should_use_text_assist_mode(self, command):
        command = (command or "").strip()
        if not command:
            return False
        return any(
            checker(command)
            for checker in (
                self._is_start_detection_command,
                self._is_stop_detection_command,
                self._is_minimize_command,
                self._is_show_detection_command,
                self._is_open_analytics_command,
                self._is_focus_query,
                self._is_advice_query,
            )
        )

    def _build_voice_ai_context(self, command):
        status_key, status_text = self._get_detector_status(self.current_score)
        explanation = self.latest_runtime_info.get("score_explanation", {})
        reasons = explanation.get("reasons", [])
        metrics = explanation.get("metrics", {})
        avg_score = sum(self.attention_scores) / len(self.attention_scores) if self.attention_scores else 0.0
        return {
            "command": command,
            "is_detecting": self.is_detecting,
            "current_score": round(float(self.current_score), 1),
            "avg_score": round(float(avg_score), 1),
            "status_key": status_key,
            "status_text": status_text,
            "main_reason": reasons[0] if reasons else "",
            "all_reasons": reasons[:3],
            "face_detected": bool(metrics.get("face_detected")),
            "phone_detected": bool(metrics.get("phone_detected")),
            "pose": metrics.get("pose", "-"),
            "multimodal_status": (self.latest_runtime_info.get("multimodal") or {}).get("status", ""),
            "project_profile": "小睿伴学是一套在线学习状态感知与辅助干预系统，包含学生端实时检测、教师端班级管理、云端同步和语音助手。",
            "student_capabilities": [
                "实时检测专注度",
                "查看详细分析",
                "加入班级",
                "语音查询与页面控制",
            ],
            "teacher_capabilities": [
                "创建班级并生成班级码",
                "查看学生看板",
                "查看在线状态与分析数据",
            ],
            "classroom_flow": "教师先创建班级并生成班级码，学生输入班级码加入后，教师端即可看到学生状态与在线情况。",
            "assistant_capabilities": "小睿可以回答专注度、解释分数变化、给学习建议，并控制开始检测、停止检测、最小化、恢复窗口和打开详细数据页面。",
        }

    def _build_text_ai_context(self, command):
        status_key, status_text = self._get_detector_status(self.current_score)
        explanation = self.latest_runtime_info.get("score_explanation", {})
        reasons = explanation.get("reasons", [])
        avg_score = sum(self.attention_scores) / len(self.attention_scores) if self.attention_scores else 0.0
        return {
            "command": command,
            "is_detecting": self.is_detecting,
            "current_score": round(float(self.current_score), 1),
            "avg_score": round(float(avg_score), 1),
            "status_key": status_key,
            "status_text": status_text,
            "main_reason": reasons[0] if reasons else "",
            "project_profile": "小睿伴学是一套在线学习状态感知与辅助干预系统。",
            "assistant_capabilities": "实时检测专注度、解释分数变化、给学习建议、打开详细数据、开始或停止检测、最小化与恢复窗口。",
            "student_capabilities": "学生端支持实时检测、查看详细分析、加入班级和语音文本交互。",
            "teacher_capabilities": "教师端支持创建班级、查看学生状态、在线情况和分析数据。",
        }

    def _apply_voice_ai_result(self, request_id, result, command):
        self._finish_llm_request(request_id)
        if not self.voice_runtime_active:
            return
        if request_id != self.voice_request_id:
            return
        self._apply_assistant_result(request_id, result, command, source="voice")

    def _apply_assistant_result(self, request_id, result, command, source="voice"):
        intent = result.get("intent") or "none"
        reply = str(result.get("reply") or "").strip()
        advice_query = self._is_advice_query(command)

        if advice_query and intent in {"query_focus", "none"}:
            intent = "give_advice"

        if intent == "start_detection":
            reply = self._voice_start_detection_reply(reply)
            self._deliver_assistant_reply(reply, source=source)
            return

        if intent == "stop_detection":
            reply = self._voice_stop_detection_reply(reply)
            self._deliver_assistant_reply(reply, source=source)
            return

        if intent == "minimize_window":
            reply = self._voice_minimize_reply(reply)
            self._deliver_assistant_reply(reply, source=source)
            return

        if intent == "show_detection":
            reply = self._voice_show_detection_reply(reply)
            self._deliver_assistant_reply(reply, source=source)
            return

        if intent == "query_focus":
            if not reply:
                reply = self._build_focus_voice_reply()
            bubble_title = None
            bubble_level = None
            bubble_text = None
            if self.is_minimized and self.mini_robot:
                bubble_title, bubble_text = self._build_focus_bubble_payload()
                bubble_level = self._current_focus_level()
            self._deliver_assistant_reply(
                reply,
                source=source,
                bubble_title=bubble_title,
                bubble_text=bubble_text,
                bubble_level=bubble_level,
            )
            return

        if intent == "open_analytics":
            self.open_analytics_window()
            if not reply:
                reply = "已为你打开详细数据页面。"
            self._deliver_assistant_reply(reply, source=source)
            return

        if intent == "introduce_self":
            reply = reply or self._build_intro_voice_reply()
            self._deliver_assistant_reply(
                reply,
                source=source,
                bubble_title="认识一下小睿",
                bubble_level="focused",
            )
            return

        if intent == "general_chat":
            reply = reply or "我在呢，你可以继续问我和学习状态、功能操作相关的问题。"
            self._deliver_assistant_reply(reply, source=source, bubble_title="小睿回答", bubble_level="focused")
            return

        if intent == "give_advice":
            reply = reply or self._build_advice_voice_reply()
            self._deliver_assistant_reply(
                reply,
                source=source,
                bubble_title="学习建议",
                bubble_level=self._current_focus_level(),
            )
            return

        if source == "text":
            reply = reply or "我暂时没完全理解你的问题，你可以换一种说法再问我。"
            self._deliver_assistant_reply(reply, source="text", bubble_title="小睿回答", bubble_level="moderate")
            return

        self._handle_voice_command_locally(command, fallback_reply=reply, request_id=request_id, source=source)

    def _apply_text_ai_result(self, request_id, result, command):
        self._finish_llm_request(request_id)
        if not self.voice_runtime_active:
            return
        if request_id != self.voice_request_id:
            return
        source = (result or {}).get("source") or "text"
        self._apply_assistant_result(request_id, result, command, source=source)

    def _apply_study_task_result(self, request_id, result, command):
        self._finish_llm_request(request_id)
        if not self.voice_runtime_active:
            return
        if request_id != self.voice_request_id:
            return
        reply = (result or {}).get("reply") or "学习资料已经处理完成。"
        source = (result or {}).get("source") or "text"
        bubble_title = "学习资料已整理" if "资料" in command else "学习文档已生成"
        self._deliver_assistant_reply(reply, source=source, bubble_title=bubble_title, bubble_level="focused")

    def _handle_study_task_error(self, request_id, command, error_message):
        self._finish_llm_request(request_id)
        if not self.voice_runtime_active:
            return
        if request_id != self.voice_request_id:
            return
        self._set_assistant_status(error_message)
        self._deliver_assistant_reply(
            f"这次学习资料整理没有完成：{error_message}",
            source="text",
            bubble_title="学习整理失败",
            bubble_level="moderate",
        )

    def _handle_voice_ai_error(self, request_id, command, error_message):
        self._finish_llm_request(request_id)
        if not self.voice_runtime_active:
            return
        if request_id != self.voice_request_id:
            return
        self._set_assistant_status(f"{error_message}，已切回内置指令处理。")
        self._handle_voice_command_locally(command, request_id=request_id, source="voice")

    def _handle_text_ai_error(self, request_id, command, error_message):
        self._finish_llm_request(request_id)
        if not self.voice_runtime_active:
            return
        if request_id != self.voice_request_id:
            return
        self._set_assistant_status(error_message)
        self._deliver_assistant_reply(
            f"模型暂时没有连通：{error_message}",
            source="text",
            bubble_title="模型异常",
            bubble_level="moderate",
        )

    def _deliver_assistant_reply(self, reply, source="voice", bubble_title=None, bubble_text=None, bubble_level=None):
        reply = (reply or "").strip()
        if not reply:
            return
        self._append_assistant_message("assistant", reply)
        self._commit_assistant_log(reply)
        self._set_assistant_status(reply)
        if source == "voice":
            self.voice_speaker.speak(reply)
        mini_panel_visible = bool(self.mini_assistant_panel and self.mini_assistant_panel.isVisible())
        if self.is_minimized and self.mini_robot and not mini_panel_visible:
            title = bubble_title or "小睿回答"
            text = bubble_text or reply
            level = bubble_level or "focused"
            self.voice_hint_bubble.show_near(
                self.mini_robot,
                text,
                level=level,
                title=title,
            )

    def _handle_voice_command_locally(self, command, fallback_reply="", request_id=None, source="voice"):
        if not self.voice_runtime_active:
            return
        if request_id is not None and request_id != self.voice_request_id:
            return

        if self._is_start_detection_command(command):
            reply = self._voice_start_detection_reply(fallback_reply)
            self._deliver_assistant_reply(reply, source=source)
            return

        if self._is_stop_detection_command(command):
            reply = self._voice_stop_detection_reply(fallback_reply)
            self._deliver_assistant_reply(reply, source=source)
            return

        if self._is_minimize_command(command):
            reply = self._voice_minimize_reply(fallback_reply)
            self._deliver_assistant_reply(reply, source=source)
            return

        if self._is_show_detection_command(command):
            reply = self._voice_show_detection_reply(fallback_reply)
            self._deliver_assistant_reply(reply, source=source)
            return

        if self._is_open_analytics_command(command):
            self.open_analytics_window()
            reply = fallback_reply or "已为你打开详细数据页面。"
            self._deliver_assistant_reply(reply, source=source)
            return

        project_topic = self._match_project_info_topic(command)
        if project_topic:
            reply = fallback_reply or self._build_project_info_reply(project_topic)
            self._deliver_assistant_reply(reply, source=source, bubble_title="项目说明", bubble_level="focused")
            return

        if self._is_intro_query(command):
            reply = fallback_reply or self._build_intro_voice_reply()
            self._deliver_assistant_reply(reply, source=source, bubble_title="认识一下小睿", bubble_level="focused")
            return

        if self._is_advice_query(command):
            reply = fallback_reply or self._build_advice_voice_reply()
            self._deliver_assistant_reply(
                reply,
                source=source,
                bubble_title="学习建议",
                bubble_level=self._current_focus_level(),
            )
            return

        if self._is_focus_query(command):
            reply = fallback_reply or self._build_focus_voice_reply()
            bubble_title = None
            bubble_level = None
            bubble_text = None
            if self.is_minimized and self.mini_robot:
                bubble_title, bubble_text = self._build_focus_bubble_payload()
                bubble_level = self._current_focus_level()
            self._deliver_assistant_reply(
                reply,
                source=source,
                bubble_title=bubble_title,
                bubble_text=bubble_text,
                bubble_level=bubble_level,
            )
            return

        reply = fallback_reply or "我听到了，不过这次没有完全理解。你可以试试问我当前专注度、学习建议，或者让我打开详细数据。"
        self._deliver_assistant_reply(reply, source=source)

    def _is_focus_query(self, command):
        normalized = self._normalize_voice_command(command)
        return any(keyword in normalized for keyword in ["专注度", "专注", "分数", "状态", "我现在", "当前", "现在"])

    def _looks_like_meaningful_voice_command(self, command):
        normalized = self._normalize_voice_command(command)
        if len(normalized) < 2:
            return False
        if any(ch.isdigit() for ch in normalized):
            return True
        if self._is_start_detection_command(normalized):
            return True
        if self._is_stop_detection_command(normalized):
            return True
        if self._is_minimize_command(normalized):
            return True
        if self._is_show_detection_command(normalized):
            return True
        if self._is_open_analytics_command(normalized):
            return True
        if self._is_intro_query(normalized):
            return True
        if self._is_advice_query(normalized):
            return True
        if self._is_focus_query(normalized):
            return True

        hint_keywords = [
            "打开",
            "搜索",
            "查找",
            "检索",
            "网站",
            "网页",
            "哔哩",
            "b站",
            "百度",
            "知乎",
            "必应",
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
            "小睿",
            "返回",
            "恢复",
            "最小化",
            "为什么",
            "什么",
            "怎么",
            "谁",
            "能不能",
            "可以",
            "会不会",
            "介绍",
            "解释",
            "功能",
            "系统",
            "项目",
            "班级",
            "同步",
        ]
        if any(keyword in normalized for keyword in hint_keywords):
            return True

        chinese_chars = sum(1 for ch in normalized if "\u4e00" <= ch <= "\u9fff")
        return chinese_chars >= 4

    def _is_start_detection_command(self, command):
        normalized = self._normalize_voice_command(command)
        return any(keyword in normalized for keyword in ["开始检测", "开启检测", "启动检测", "打开检测", "开始专注检测", "启动摄像头", "打开摄像头"])

    def _is_stop_detection_command(self, command):
        normalized = self._normalize_voice_command(command)
        return any(keyword in normalized for keyword in ["停止检测", "关闭检测", "结束检测", "暂停检测", "停止专注检测", "关闭摄像头"])

    def _is_minimize_command(self, command):
        normalized = self._normalize_voice_command(command)
        return any(keyword in normalized for keyword in ["最小化", "程序最小化", "窗口最小化", "缩小窗口", "隐藏窗口"])

    def _is_show_detection_command(self, command):
        normalized = self._normalize_voice_command(command)
        keywords = [
            "返回检测界面",
            "回到检测界面",
            "打开检测界面",
            "进入检测界面",
            "返回检测页面",
            "回到检测页面",
            "返回主界面",
            "回到主界面",
            "恢复主界面",
            "恢复窗口",
            "显示主界面",
            "回到首页",
        ]
        return any(keyword in normalized for keyword in keywords)

    def _is_open_analytics_command(self, command):
        normalized = self._normalize_voice_command(command)
        action_words = ["打开", "查看", "显示", "进入", "切换", "展开"]
        target_words = [
            "详细数据",
            "详细分析",
            "数据页面",
            "分析页面",
            "详情页面",
            "详情页",
            "分析页",
            "数据页",
            "统计页面",
            "统计页",
            "报表页面",
            "报表页",
        ]
        if any(keyword in normalized for keyword in target_words):
            return True

        has_action = any(keyword in normalized for keyword in action_words)
        has_target = any(keyword in normalized for keyword in ["详细", "分析", "数据", "统计", "报表", "详情"])
        has_page = any(keyword in normalized for keyword in ["页面", "界面", "页"])
        return has_action and (has_target or has_page)

    def _match_project_info_topic(self, command):
        normalized = self._normalize_voice_command(command)
        topic_keywords = {
            "system_intro": [
                "这个系统是做什么的",
                "这个项目是做什么的",
                "项目介绍",
                "系统介绍",
                "介绍一下这个系统",
                "介绍一下这个项目",
                "这个软件是做什么的",
            ],
            "student_features": [
                "学生端有什么功能",
                "学生端能做什么",
                "学生端有哪些功能",
            ],
            "teacher_features": [
                "教师端有什么功能",
                "教师端能做什么",
                "教师端有哪些功能",
                "老师端有什么功能",
            ],
            "class_join": [
                "班级码怎么用",
                "怎么加入班级",
                "如何加入班级",
                "加入班级怎么操作",
            ],
            "cloud_sync": [
                "云端同步是什么",
                "账号会同步吗",
                "数据会同步吗",
                "云端有什么用",
            ],
            "assistant_features": [
                "你还能做什么",
                "语音助手能做什么",
                "小睿能做什么",
                "你可以帮我做什么",
            ],
        }
        for topic, keywords in topic_keywords.items():
            if any(keyword in normalized for keyword in keywords):
                return topic
        return ""

    def _is_intro_query(self, command):
        normalized = self._normalize_voice_command(command)
        keywords = [
            "介绍一下自己",
            "介绍你自己",
            "介绍下自己",
            "介绍一下你自己",
            "介绍下你自己",
            "简单介绍一下自己",
            "简单介绍一下你自己",
            "你是谁",
            "你能做什么",
            "你可以做什么",
            "你会做什么",
            "自我介绍",
        ]
        return any(keyword in normalized for keyword in keywords)

    def _is_advice_query(self, command):
        normalized = self._normalize_voice_command(command)
        keywords = [
            "建议",
            "怎么办",
            "怎么提高",
            "怎么提升",
            "为什么",
            "原因",
            "为什么低",
            "为什么分数低",
            "怎么学",
            "如何改进",
        ]
        return any(keyword in normalized for keyword in keywords)

    def _normalize_voice_command(self, command):
        normalized = (command or "").strip()
        replacements = {
            "小瑞": "小睿",
            "小锐": "小睿",
            "小蕊": "小睿",
            "小芮": "小睿",
            "晓睿": "小睿",
            "晓瑞": "小睿",
            "晓锐": "小睿",
            "晓蕊": "小睿",
            "晓芮": "小睿",
            "详请": "详情",
            "详青": "详情",
            "详情数据": "详细数据",
            "详细资料": "详细数据",
            "数据界面": "数据页面",
            "分析界面": "分析页面",
            "报表界面": "报表页面",
            "打卡": "打开",
            "达开": "打开",
            "岔开": "打开",
            "减测": "检测",
            "摄相头": "摄像头",
            "回到主页面": "回到主界面",
            "检测页面": "检测界面",
            "介绍下": "介绍一下",
            "自已": "自己",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _voice_start_detection_reply(self, fallback_reply=""):
        if self.is_detecting:
            return fallback_reply or "检测已经在运行了。"
        self.start_detection()
        if self.is_detecting:
            return fallback_reply or "已开始检测。"
        return fallback_reply or "我尝试开始检测了，但摄像头暂时还没有成功打开。"

    def _voice_stop_detection_reply(self, fallback_reply=""):
        if not self.is_detecting:
            return fallback_reply or "检测目前还没有启动。"
        self.stop_detection()
        return fallback_reply or "已停止检测。"

    def _voice_minimize_reply(self, fallback_reply=""):
        if self.is_minimized:
            return fallback_reply or "程序已经最小化了。"
        self.show_minimized()
        return fallback_reply or "已为你最小化程序。"

    def _voice_show_detection_reply(self, fallback_reply=""):
        was_minimized = self.is_minimized or not self.isVisible()
        had_analytics = bool(self.analytics_window and self.analytics_window.isVisible())

        if self.analytics_window and self.analytics_window.isVisible():
            self.analytics_window.hide()

        if was_minimized:
            self.show_normal()
        else:
            self.raise_()
            self.activateWindow()

        if had_analytics or was_minimized:
            return fallback_reply or "已返回检测界面。"
        return fallback_reply or "你现在就在检测界面。"

    def _build_focus_voice_reply(self):
        if not self.is_detecting or not self.attention_scores:
            return "现在还没有开始检测。请先点击开始检测，我就能播报你的专注度。"

        status_key, status_text = self._get_detector_status(self.current_score)
        avg_score = sum(self.attention_scores) / len(self.attention_scores)
        advice = self._voice_learning_advice(status_key)
        return f"你当前的专注度是{self.current_score:.0f}分，状态是{status_text}。本次平均专注度是{avg_score:.0f}分。{advice}"

    def _build_advice_voice_reply(self):
        if not self.is_detecting or not self.attention_scores:
            return "现在还没有开始检测。你先开始检测，我再根据当前状态给你学习建议。"

        status_key, status_text = self._get_detector_status(self.current_score)
        advice = self._voice_learning_advice(status_key)
        explanation = self.latest_runtime_info.get("score_explanation", {})
        reasons = explanation.get("reasons", [])
        reason_text = reasons[0] if reasons else ""
        if reason_text:
            return f"你当前处于{status_text}状态，主要原因是{reason_text}。{advice}"
        return f"你当前处于{status_text}状态。{advice}"

    def _build_intro_voice_reply(self):
        return "我是小睿伴学里的语音助手，可以帮你检测学习状态、查看数据并进行语音控制。"

    def _build_project_info_reply(self, topic):
        replies = {
            "system_intro": "小睿伴学是一套在线学习状态感知与辅助干预系统，包含学生端实时检测、教师端班级管理、云端同步和语音助手。",
            "student_features": "学生端可以实时检测专注度、查看详细分析、加入班级，还能通过小睿语音查询状态和控制页面。",
            "teacher_features": "教师端可以创建班级、生成班级码、查看学生看板、在线状态和班级分析数据。",
            "class_join": "教师先创建班级并生成班级码，学生在学生端输入班级码加入后，教师端就能看到对应学生的数据。",
            "cloud_sync": "云端同步可以让账号、班级关系、检测记录和在线状态在不同电脑之间保持一致，不再只保存在本地。",
            "assistant_features": "我可以帮你检测学习状态、查看数据并进行语音控制。",
        }
        return replies.get(topic, "你可以继续问我系统介绍、学生端功能、教师端功能、班级码怎么用，或者云端同步有什么用。")

    def _build_focus_bubble_payload(self):
        if not self.is_detecting or not self.attention_scores:
            return "等待检测", "还没有开始检测"

        status_key, status_text = self._get_detector_status(self.current_score)
        avg_score = sum(self.attention_scores) / len(self.attention_scores)
        advice_map = {
            "focused": "继续保持现在的节奏",
            "moderate": "再稳一点会更好",
            "distracted": "先把注意力拉回屏幕",
        }
        title_map = {
            "focused": "继续保持",
            "moderate": "小睿提醒",
            "distracted": "留意分心",
        }
        text = (
            f"当前专注 {self.current_score:.0f} 分（{status_text}）\n"
            f"平均 {avg_score:.0f} 分\n"
            f"{advice_map.get(status_key, '继续加油')}"
        )
        return title_map.get(status_key, "小睿播报"), text

    def _current_focus_level(self):
        if self.current_score >= 70:
            return "focused"
        if self.current_score >= 40:
            return "moderate"
        return "distracted"

    def _voice_learning_advice(self, status_key):
        explanation = self.latest_runtime_info.get("score_explanation", {})
        metrics = explanation.get("metrics", {})

        if metrics.get("phone_detected"):
            return "我检测到画面中可能有手机，建议先把手机移出视野，减少干扰。"
        if metrics and not metrics.get("face_detected"):
            return "我没有稳定检测到人脸，建议把脸放在画面中间，并保证光线更亮一些。"
        pose = metrics.get("pose")
        if pose and pose not in {"正视屏幕", "forward", "-"}:
            return "你的头部姿态有些偏离，建议坐正并让摄像头保持在视线附近。"
        if status_key == "focused":
            return "保持得不错，可以继续按现在的节奏学习。"
        if status_key == "moderate":
            return "建议先做一个小目标，比如专注五分钟完成当前题目。"
        return "建议暂停一下干扰源，整理桌面，然后重新开始一小段专注学习。"

    def _set_native_window_minimized(self, minimized: bool):
        if sys.platform.startswith("win"):
            SW_MINIMIZE = 6
            SW_RESTORE = 9
            handle = int(self.winId())
            ctypes.windll.user32.ShowWindow(handle, SW_MINIMIZE if minimized else SW_RESTORE)
        elif minimized:
            self.showMinimized()
        else:
            self.showNormal()

    def toggle_minimize(self):
        if self.is_minimized or self.windowState() & Qt.WindowMinimized:
            self.show_normal()
        else:
            self.show_minimized()

    def show_minimized(self):
        self.is_minimized = True
        if self.analytics_window and self.analytics_window.isVisible():
            self.analytics_window.hide()
        if self.voice_hint_bubble:
            self.voice_hint_bubble.hide()
        self.setWindowState(Qt.WindowMinimized)
        self._set_native_window_minimized(True)

        if self.mini_robot is None:
            self.mini_robot = MiniRobotWidget(size=80)
            self.mini_robot.clicked.connect(self._toggle_minimized_assistant_panel)
            self.mini_robot.moved.connect(self._reposition_minimized_overlays)

        self.mini_robot.set_attention(self.current_score)
        screen = self.screen().geometry()
        self.mini_robot.move(screen.width() - 120, 50)
        self.mini_robot.show()
        panel = self._ensure_minimized_assistant_panel()
        panel.hide()

    def show_normal(self):
        self.is_minimized = False
        if self.mini_robot:
            self.mini_robot.hide()
        if self.mini_assistant_panel:
            self.mini_assistant_panel.hide()
        if self.voice_hint_bubble:
            self.voice_hint_bubble.hide()
        self.setWindowState(Qt.WindowNoState)
        self._set_native_window_minimized(False)
        self.show()
        self.raise_()
        self.activateWindow()
        self.activateWindow()
        self.activateWindow()

    def _shutdown_voice_runtime(self):
        self.voice_runtime_active = False
        self._advance_voice_request()
        self.voice_interrupt_armed_until = 0.0
        try:
            self.stop_voice_assistant()
        except Exception:
            pass
        try:
            self.voice_speaker.stop()
        except Exception:
            pass
        if self.voice_hint_bubble:
            self.voice_hint_bubble.hide()
        if self.mini_assistant_panel:
            self.mini_assistant_panel.hide()
        try:
            self.voice_command_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def handle_logout(self):
        if hasattr(self, "presence_timer"):
            self.presence_timer.stop()
        self._set_presence(False)
        self._shutdown_voice_runtime()
        self.stop_detection()
        self.multimodal_stream.stop()
        if self.mini_robot:
            self.mini_robot.close()
        if self.mini_assistant_panel:
            self.mini_assistant_panel.close()
        if self.voice_hint_bubble:
            self.voice_hint_bubble.close()
        if self.analytics_window:
            self.analytics_window.close()
        self.logout_signal.emit()

    def closeEvent(self, event):
        if hasattr(self, "presence_timer"):
            self.presence_timer.stop()
        self._set_presence(False)
        self._shutdown_voice_runtime()
        self.stop_detection()
        self.multimodal_stream.stop()
        self.detector_executor.shutdown(wait=False, cancel_futures=True)
        self.detector_load_executor.shutdown(wait=False, cancel_futures=True)
        self.record_executor.shutdown(wait=False, cancel_futures=True)
        self.assistant_io_executor.shutdown(wait=False, cancel_futures=True)
        self.presence_executor.shutdown(wait=False, cancel_futures=True)
        if self.mini_robot:
            self.mini_robot.close()
        if self.mini_assistant_panel:
            self.mini_assistant_panel.close()
        if self.voice_hint_bubble:
            self.voice_hint_bubble.close()
        if self.analytics_window:
            self.analytics_window.close()
        event.accept()

    def format_duration(self, seconds):
        seconds = int(seconds)
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {sec}s"
        return f"{sec}s"
