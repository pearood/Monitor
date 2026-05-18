from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QTimer, QPoint, QSize, pyqtSignal, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QPen, QBrush, QPainterPath, QLinearGradient, QRadialGradient
from config.config import LOGO_PATH, COLORS, ROBOT_SIZE, ATTENTION_THRESHOLDS
import os

class RobotWidget(QWidget):
    clicked = pyqtSignal()
    
    def __init__(self, size=ROBOT_SIZE):
        super().__init__()
        self.robot_size = size
        self.attention_score = 50
        self.status = 'moderate'
        self.status_text = '一般'
        self.logo_pixmap = None
        self.drag_position = None
        
        self.load_logo()
        self.init_ui()
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate)
        self.animation_offset = 0
        self.animation_direction = 1
    
    def load_logo(self):
        if os.path.exists(LOGO_PATH):
            self.logo_pixmap = QPixmap(LOGO_PATH)
            if not self.logo_pixmap.isNull():
                self.logo_pixmap = self.logo_pixmap.scaled(
                    self.robot_size - 20, 
                    self.robot_size - 20,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
    
    def init_ui(self):
        self.setFixedSize(self.robot_size + 20, self.robot_size + 40)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)
    
    def set_attention(self, score):
        self.attention_score = score
        
        if score >= ATTENTION_THRESHOLDS['focused']:
            self.status = 'focused'
            self.status_text = '专注'
        elif score >= ATTENTION_THRESHOLDS['moderate']:
            self.status = 'moderate'
            self.status_text = '一般'
        else:
            self.status = 'distracted'
            self.status_text = '不专注'
        
        self.update()
    
    def get_status_color(self):
        return QColor(COLORS[self.status])
    
    def animate(self):
        self.animation_offset += 0.1 * self.animation_direction
        if abs(self.animation_offset) > 2:
            self.animation_direction *= -1
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        center_x = self.width() // 2
        
        color = QColor(COLORS[self.status])
        
        # 头和身体比例 - 头比身体大且宽
        head_height = self.robot_size * 0.7  # 头部占70%
        body_height = self.robot_size * 0.3  # 身体占30%
        head_y = 0
        body_y = head_y + head_height - 5
        
        # 绘制圆形身体（圆角矩形）
        body_width = self.robot_size * 0.5  # 身体宽度较小
        body_height = self.robot_size * 0.3
        body_x = center_x - body_width // 2
        
        # 身体阴影
        shadow_brush = QBrush(QColor(0, 0, 0, 60))
        painter.setBrush(shadow_brush)
        painter.drawRoundedRect(
            body_x + 3, body_y + 3,
            body_width, body_height,
            15, 15
        )
        
        # 身体渐变
        body_gradient = QRadialGradient(center_x, body_y + body_height//2, body_height)
        body_gradient.setColorAt(0, color.lighter(140))
        body_gradient.setColorAt(0.5, color.lighter(110))
        body_gradient.setColorAt(1, color.darker(120))
        
        painter.setBrush(body_gradient)
        painter.drawRoundedRect(
            body_x, body_y,
            body_width, body_height,
            15, 15
        )
        
        # 身体高光
        body_highlight = QLinearGradient(
            body_x, body_y,
            body_x, body_y + body_height
        )
        body_highlight.setColorAt(0, QColor(255, 255, 255, 60))
        body_highlight.setColorAt(0.3, QColor(255, 255, 255, 30))
        body_highlight.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setBrush(body_highlight)
        painter.drawRoundedRect(
            body_x, body_y,
            body_width, body_height,
            15, 15
        )
        
        # 绘制圆形头部（比身体大且宽）
        head_radius = head_height // 2
        head_center_x = center_x
        head_center_y = head_y + head_radius
        
        # 头部阴影
        head_shadow = QBrush(QColor(0, 0, 0, 50))
        painter.setBrush(head_shadow)
        painter.drawEllipse(head_center_x - head_radius + 2, head_center_y - head_radius + 2, 
                           head_radius * 2, head_radius * 2)
        
        # 头部渐变
        head_gradient = QRadialGradient(head_center_x, head_center_y, head_radius)
        head_gradient.setColorAt(0, color.lighter(160))
        head_gradient.setColorAt(0.6, color.lighter(120))
        head_gradient.setColorAt(1, color.darker(110))
        
        painter.setBrush(head_gradient)
        painter.drawEllipse(head_center_x - head_radius, head_center_y - head_radius, 
                          head_radius * 2, head_radius * 2)
        
        # 头部高光
        head_highlight = QLinearGradient(
            head_center_x - head_radius, head_center_y - head_radius,
            head_center_x - head_radius, head_center_y + head_radius
        )
        head_highlight.setColorAt(0, QColor(255, 255, 255, 80))
        head_highlight.setColorAt(0.4, QColor(255, 255, 255, 30))
        head_highlight.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setBrush(head_highlight)
        painter.drawEllipse(head_center_x - head_radius, head_center_y - head_radius, 
                          head_radius * 2, head_radius * 2)
        
        # 绘制眼睛椭圆区域（更大）
        eye_area_width = head_radius * 1.2  # 椭圆区域更大
        eye_area_height = head_radius * 0.7
        eye_area_x = head_center_x - eye_area_width // 2
        eye_area_y = head_center_y - eye_area_height // 2
        
        # 眼睛区域渐变
        eye_area_gradient = QRadialGradient(head_center_x, head_center_y, eye_area_width//2)
        eye_area_gradient.setColorAt(0, QColor(255, 255, 255, 180))
        eye_area_gradient.setColorAt(1, QColor(200, 200, 200, 150))
        painter.setBrush(eye_area_gradient)
        painter.drawEllipse(eye_area_x, eye_area_y, eye_area_width, eye_area_height)
        
        # 绘制两个动态眼睛（左右移动）
        eye_size = head_radius * 0.25
        
        # 根据专注度计算眼睛左右位置（动态效果）
        eye_horizontal_offset = (100 - self.attention_score) / 100 * head_radius * 0.15
        eye_y = head_center_y - head_radius * 0.05
        
        # 左眼向左移动，右眼向右移动
        left_eye_x = head_center_x - head_radius * 0.35 - eye_horizontal_offset
        right_eye_x = head_center_x + head_radius * 0.35 + eye_horizontal_offset
        
        # 左眼
        left_eye_gradient = QRadialGradient(left_eye_x, eye_y, eye_size)
        left_eye_gradient.setColorAt(0, QColor(255, 255, 255, 230))
        left_eye_gradient.setColorAt(1, QColor(180, 180, 180, 200))
        painter.setBrush(left_eye_gradient)
        painter.drawEllipse(left_eye_x - eye_size//2, eye_y - eye_size//2, eye_size, eye_size)
        
        # 右眼
        right_eye_gradient = QRadialGradient(right_eye_x, eye_y, eye_size)
        right_eye_gradient.setColorAt(0, QColor(255, 255, 255, 230))
        right_eye_gradient.setColorAt(1, QColor(180, 180, 180, 200))
        painter.setBrush(right_eye_gradient)
        painter.drawEllipse(right_eye_x - eye_size//2, eye_y - eye_size//2, eye_size, eye_size)
        
        # 绘制瞳孔（左右移动）
        pupil_size = eye_size * 0.6
        pupil_horizontal_offset = (100 - self.attention_score) / 100 * eye_size * 0.3
        
        for eye_x in [left_eye_x, right_eye_x]:
            pupil_gradient = QRadialGradient(eye_x, eye_y, pupil_size//2)
            pupil_gradient.setColorAt(0, QColor(60, 60, 60))
            pupil_gradient.setColorAt(1, QColor(20, 20, 20))
            painter.setBrush(pupil_gradient)
            painter.drawEllipse(eye_x - pupil_size//2, eye_y - pupil_size//2, pupil_size, pupil_size)
        
        # 在身体内部显示专注度分数（百分号，大字体）
        body_center_y = body_y + body_height // 2
        
        painter.setPen(QPen(QColor(255, 255, 255), 3))
        font = QFont('Arial', 16, QFont.Bold)
        painter.setFont(font)
        
        score_text = f'{int(self.attention_score)}%'
        text_rect = painter.fontMetrics().boundingRect(score_text)
        text_x = center_x - text_rect.width() // 2
        text_y = body_center_y + text_rect.height() // 3
        
        # 数字阴影
        painter.setPen(QPen(QColor(0, 0, 0, 180), 3))
        painter.drawText(text_x + 2, text_y + 2, score_text)
        
        # 白色数字
        painter.setPen(QPen(QColor(255, 255, 255), 3))
        painter.drawText(text_x, text_y, score_text)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_position:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drag_position:
                self.clicked.emit()
            self.drag_position = None

class MiniRobotWidget(QWidget):
    clicked = pyqtSignal()
    moved = pyqtSignal()
    
    def __init__(self, size=80):
        super().__init__()
        self.robot_size = size
        self.attention_score = 50
        self.status = 'moderate'
        self.drag_position = None
        self.is_dragging = False
        self.press_position = None
        
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate)
        self.animation_frame = 0
        self.animation_speed = 20
        
        self.init_ui()
    
    def init_ui(self):
        self.setFixedSize(self.robot_size + 30, self.robot_size + 50)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.animation_timer.start(50)
    
    def set_attention(self, score):
        self.attention_score = score
        
        if score >= ATTENTION_THRESHOLDS['focused']:
            self.status = 'focused'
        elif score >= ATTENTION_THRESHOLDS['moderate']:
            self.status = 'moderate'
        else:
            self.status = 'distracted'
        
        self.update()
    
    def animate(self):
        self.animation_frame = (self.animation_frame + 1) % self.animation_speed
        self.update()
    
    def get_animation_offset(self):
        if self.status == 'focused':
            return int(3 * abs(self.animation_frame - self.animation_speed//2) / (self.animation_speed//2))
        elif self.status == 'moderate':
            return int(1 * abs(self.animation_frame - self.animation_speed//2) / (self.animation_speed//2))
        else:
            return 0
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        color = QColor(COLORS[self.status])
        
        robot_radius = self.robot_size // 2
        
        outer_shadow_radius = robot_radius + 8
        painter.setBrush(QBrush(color.lighter(140)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            center_x - outer_shadow_radius,
            center_y - outer_shadow_radius,
            outer_shadow_radius * 2,
            outer_shadow_radius * 2
        )
        
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(
            center_x - robot_radius - 4,
            center_y - robot_radius - 4,
            (robot_radius + 4) * 2,
            (robot_radius + 4) * 2
        )
        
        main_gradient = QRadialGradient(center_x, center_y, robot_radius)
        main_gradient.setColorAt(0, color.lighter(130))
        main_gradient.setColorAt(0.5, color)
        main_gradient.setColorAt(1, color.darker(110))
        
        painter.setBrush(main_gradient)
        painter.drawEllipse(
            center_x - robot_radius,
            center_y - robot_radius,
            robot_radius * 2,
            robot_radius * 2
        )
        
        highlight = QLinearGradient(
            center_x - robot_radius, center_y - robot_radius,
            center_x - robot_radius, center_y + robot_radius
        )
        highlight.setColorAt(0, QColor(255, 255, 255, 100))
        highlight.setColorAt(0.3, QColor(255, 255, 255, 40))
        highlight.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setBrush(highlight)
        painter.drawEllipse(
            center_x - robot_radius,
            center_y - robot_radius,
            robot_radius * 2,
            robot_radius * 2
        )
        
        screen_width = robot_radius * 1.1
        screen_height = robot_radius * 0.55
        screen_x = center_x - screen_width // 2
        screen_y = center_y - screen_height // 2
        
        screen_gradient = QRadialGradient(center_x, center_y, screen_width // 2)
        screen_gradient.setColorAt(0, color.darker(140))
        screen_gradient.setColorAt(1, color.darker(180))
        painter.setBrush(screen_gradient)
        painter.drawRoundedRect(screen_x, screen_y, screen_width, screen_height, 8, 8)
        
        antenna_width = robot_radius * 0.18
        antenna_height = robot_radius * 0.45
        antenna_offset_x = robot_radius + 2
        antenna_offset_y = robot_radius * 0.1
        
        left_antenna_x = center_x - antenna_offset_x - antenna_width // 2
        right_antenna_x = center_x + antenna_offset_x - antenna_width // 2
        antenna_y = center_y - antenna_offset_y
        
        antenna_gradient = QRadialGradient(center_x, center_y, robot_radius)
        antenna_gradient.setColorAt(0, color.lighter(130))
        antenna_gradient.setColorAt(1, color)
        
        painter.setBrush(antenna_gradient)
        painter.drawRoundedRect(left_antenna_x, antenna_y, antenna_width, antenna_height, antenna_width // 2, antenna_width // 2)
        painter.drawRoundedRect(right_antenna_x, antenna_y, antenna_width, antenna_height, antenna_width // 2, antenna_width // 2)
        
        rod_width = antenna_width * 0.4
        rod_height = robot_radius * 0.6
        left_rod_x = left_antenna_x + antenna_width // 2 - rod_width // 2
        right_rod_x = right_antenna_x + antenna_width // 2 - rod_width // 2
        rod_y = antenna_y - rod_height
        
        painter.setBrush(antenna_gradient)
        painter.drawRoundedRect(left_rod_x, rod_y, rod_width, rod_height, rod_width // 2, rod_width // 2)
        painter.drawRoundedRect(right_rod_x, rod_y, rod_width, rod_height, rod_width // 2, rod_width // 2)
        
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        font = QFont('Arial', max(10, int(robot_radius * 0.35)), QFont.Bold)
        painter.setFont(font)
        
        score_text = f'{int(self.attention_score)}%'
        text_rect = painter.fontMetrics().boundingRect(score_text)
        text_x = center_x - text_rect.width() // 2
        text_y = center_y + text_rect.height() // 4
        
        painter.setPen(QPen(QColor(0, 0, 0, 150), 2))
        painter.drawText(text_x + 1, text_y + 1, score_text)
        
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawText(text_x, text_y, score_text)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.press_position = event.globalPos()
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            self.is_dragging = False
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_position:
            self.move(event.globalPos() - self.drag_position)
            self.is_dragging = True
            self.moved.emit()
            event.accept()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self.is_dragging and self.press_position:
                distance = (event.globalPos() - self.press_position).manhattanLength()
                if distance < 5:
                    self.clicked.emit()
            self.drag_position = None
            self.press_position = None
            self.is_dragging = False
