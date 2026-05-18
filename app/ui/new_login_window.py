from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QComboBox, QStackedWidget, QMessageBox,
    QFrame, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
from config.config import APP_NAME, COLORS

class NewLoginRegisterWindow(QWidget):
    login_success = pyqtSignal(dict)
    auth_finished = pyqtSignal(str, bool, object)
    
    def __init__(self, db):
        super().__init__()
        self.db = db
        self.auth_busy = False
        self.auth_executor = ThreadPoolExecutor(max_workers=1)
        self.auth_finished.connect(self._handle_auth_finished)
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle(f'{APP_NAME} - 登录')
        self.setMinimumSize(620, 760)
        self.resize(620, 760)
        
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 登录/注册表单
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: white;")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(40, 40, 40, 40)
        
        # Logo和标题
        brand_layout = QVBoxLayout()
        brand_layout.setAlignment(Qt.AlignCenter)
        brand_layout.setSpacing(8)

        title_label = QLabel(APP_NAME)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont('Microsoft YaHei UI', 36, QFont.Bold))
        title_label.setStyleSheet("""
            color: #2563EB;
            font-weight: 800;
            letter-spacing: 1px;
            margin-top: 4px;
        """)
        brand_layout.addWidget(title_label)

        brand_subtitle = QLabel('AI学习状态感知与陪伴平台')
        brand_subtitle.setAlignment(Qt.AlignCenter)
        brand_subtitle.setFont(QFont('Microsoft YaHei', 11))
        brand_subtitle.setStyleSheet("""
            color: #64748B;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        """)
        brand_layout.addWidget(brand_subtitle)

        left_layout.addLayout(brand_layout)
        left_layout.addSpacing(32)
        
        # 标题
        self.page_title = QLabel('欢迎回来')
        self.page_title.setFont(QFont('Microsoft YaHei', 24, QFont.Bold))
        self.page_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.page_title)
        
        self.page_subtitle = QLabel('请登录您的账号')
        self.page_subtitle.setFont(QFont('Microsoft YaHei', 14))
        self.page_subtitle.setStyleSheet("color: #64748b; margin-bottom: 30px;")
        self.page_subtitle.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.page_subtitle)
        
        # 堆叠窗口
        self.stacked_widget = QStackedWidget()
        
        login_widget = self.create_login_widget()
        self.stacked_widget.addWidget(login_widget)
        
        register_widget = self.create_register_widget()
        self.stacked_widget.addWidget(register_widget)
        
        left_layout.addWidget(self.stacked_widget)
        
        # 切换按钮
        switch_layout = QHBoxLayout()
        switch_layout.setAlignment(Qt.AlignCenter)
        
        self.switch_label = QLabel('还没有账号？')
        self.switch_label.setFont(QFont('Microsoft YaHei', 14))
        self.switch_label.setStyleSheet("color: #64748b;")
        switch_layout.addWidget(self.switch_label)
        
        self.switch_btn = QPushButton('立即注册')
        self.switch_btn.setFont(QFont('Microsoft YaHei', 14, QFont.Medium))
        self.switch_btn.setFlat(True)
        self.switch_btn.setCursor(Qt.PointingHandCursor)
        self.switch_btn.setStyleSheet("""
            QPushButton {
                color: #2563eb;
                border: none;
                text-decoration: underline;
            }
            QPushButton:hover {
                color: #1d4ed8;
            }
        """)
        self.switch_btn.clicked.connect(self.toggle_page)
        switch_layout.addWidget(self.switch_btn)
        
        left_layout.addLayout(switch_layout)
        
        # 只保留左侧登录表单，移除右侧区域
        main_layout.addWidget(left_widget)
    
    def create_login_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # 用户类型选择 - 简洁设计
        switch_layout = QHBoxLayout()
        switch_layout.setSpacing(10)
        
        # 学生按钮
        self.student_btn = QPushButton('🎓 学 生')
        self.student_btn.setCheckable(True)
        self.student_btn.setChecked(True)
        self.student_btn.setFixedHeight(50)
        self.student_btn.setFont(QFont('Microsoft YaHei', 12, QFont.Bold))
        self.student_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
        """)
        self.student_btn.clicked.connect(lambda: self.select_user_type('student'))
        switch_layout.addWidget(self.student_btn)
        
        # 教师按钮
        self.teacher_btn = QPushButton('👨‍🏫 教 师')
        self.teacher_btn.setCheckable(True)
        self.teacher_btn.setFixedHeight(50)
        self.teacher_btn.setFont(QFont('Microsoft YaHei', 12, QFont.Bold))
        self.teacher_btn.setStyleSheet("""
            QPushButton {
                background-color: #94A3B8;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #64748B;
            }
        """)
        self.teacher_btn.clicked.connect(lambda: self.select_user_type('teacher'))
        switch_layout.addWidget(self.teacher_btn)

        self.parent_btn = QPushButton('👨‍👩‍👧 家 长')
        self.parent_btn.setCheckable(True)
        self.parent_btn.setFixedHeight(50)
        self.parent_btn.setFont(QFont('Microsoft YaHei', 12, QFont.Bold))
        self.parent_btn.setStyleSheet("""
            QPushButton {
                background-color: #94A3B8;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #64748B;
            }
        """)
        self.parent_btn.clicked.connect(lambda: self.select_user_type('parent'))
        switch_layout.addWidget(self.parent_btn)
        
        layout.addLayout(switch_layout)
        
        # 当前选择的用户类型
        self.selected_user_type = 'student'
        
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText('用户名')
        self.login_username.setFont(QFont('Microsoft YaHei', 11))
        self.login_username.setStyleSheet(self.get_input_style())
        self.login_username.setMinimumHeight(45)
        layout.addWidget(self.login_username)
        
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText('密码')
        self.login_password.setEchoMode(QLineEdit.Password)
        self.login_password.setFont(QFont('Microsoft YaHei', 11))
        self.login_password.setStyleSheet(self.get_input_style())
        self.login_password.setMinimumHeight(45)
        layout.addWidget(self.login_password)
        
        self.login_btn = QPushButton('登 录')
        self.login_btn.setFont(QFont('Microsoft YaHei', 14, QFont.Bold))
        self.login_btn.setMinimumHeight(50)
        self.login_btn.setCursor(Qt.PointingHandCursor)
        self.login_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: #2980B9;
            }}
            QPushButton:pressed {{
                background-color: #1F6F9F;
            }}
        """)
        self.login_btn.clicked.connect(self.handle_login)
        layout.addWidget(self.login_btn)
        
        demo_label = QLabel('演示账号: student1/123456、teacher1/123456、parent1/123456')
        demo_label.setFont(QFont('Microsoft YaHei', 9))
        demo_label.setStyleSheet(f"color: #95A5A6;")
        demo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(demo_label)
        
        layout.addStretch()
        return widget
    
    def create_register_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # 用户类型选择 - 简洁设计
        user_type_layout = QHBoxLayout()
        user_type_layout.setSpacing(10)
        
        # 学生按钮
        self.reg_student_btn = QPushButton('🎓 学 生')
        self.reg_student_btn.setCheckable(True)
        self.reg_student_btn.setChecked(True)
        self.reg_student_btn.setFixedHeight(50)
        self.reg_student_btn.setFont(QFont('Microsoft YaHei', 12, QFont.Bold))
        self.reg_student_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
        """)
        self.reg_student_btn.clicked.connect(lambda: self.select_reg_user_type('student'))
        user_type_layout.addWidget(self.reg_student_btn)
        
        # 教师按钮
        self.reg_teacher_btn = QPushButton('👨‍🏫 教 师')
        self.reg_teacher_btn.setCheckable(True)
        self.reg_teacher_btn.setFixedHeight(50)
        self.reg_teacher_btn.setFont(QFont('Microsoft YaHei', 12, QFont.Bold))
        self.reg_teacher_btn.setStyleSheet("""
            QPushButton {
                background-color: #94A3B8;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #64748B;
            }
        """)
        self.reg_teacher_btn.clicked.connect(lambda: self.select_reg_user_type('teacher'))
        user_type_layout.addWidget(self.reg_teacher_btn)

        self.reg_parent_btn = QPushButton('👨‍👩‍👧 家 长')
        self.reg_parent_btn.setCheckable(True)
        self.reg_parent_btn.setFixedHeight(50)
        self.reg_parent_btn.setFont(QFont('Microsoft YaHei', 12, QFont.Bold))
        self.reg_parent_btn.setStyleSheet("""
            QPushButton {
                background-color: #94A3B8;
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #64748B;
            }
        """)
        self.reg_parent_btn.clicked.connect(lambda: self.select_reg_user_type('parent'))
        user_type_layout.addWidget(self.reg_parent_btn)
        
        layout.addLayout(user_type_layout)
        
        # 当前选择的注册用户类型
        self.selected_reg_user_type = 'student'
        
        self.reg_username = QLineEdit()
        self.reg_username.setPlaceholderText('用户名')
        self.reg_username.setFont(QFont('Microsoft YaHei', 11))
        self.reg_username.setStyleSheet(self.get_input_style())
        self.reg_username.setMinimumHeight(45)
        layout.addWidget(self.reg_username)
        
        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText('密码')
        self.reg_password.setEchoMode(QLineEdit.Password)
        self.reg_password.setFont(QFont('Microsoft YaHei', 11))
        self.reg_password.setStyleSheet(self.get_input_style())
        self.reg_password.setMinimumHeight(45)
        layout.addWidget(self.reg_password)
        
        self.reg_confirm = QLineEdit()
        self.reg_confirm.setPlaceholderText('确认密码')
        self.reg_confirm.setEchoMode(QLineEdit.Password)
        self.reg_confirm.setFont(QFont('Microsoft YaHei', 11))
        self.reg_confirm.setStyleSheet(self.get_input_style())
        self.reg_confirm.setMinimumHeight(45)
        layout.addWidget(self.reg_confirm)
        
        self.register_btn = QPushButton('注 册')
        self.register_btn.setFont(QFont('Microsoft YaHei', 14, QFont.Bold))
        self.register_btn.setMinimumHeight(50)
        self.register_btn.setCursor(Qt.PointingHandCursor)
        self.register_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['secondary']};
                color: white;
                border: none;
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: #8E44AD;
            }}
            QPushButton:pressed {{
                background-color: #7D3C98;
            }}
        """)
        self.register_btn.clicked.connect(self.handle_register)
        layout.addWidget(self.register_btn)
        
        layout.addStretch()
        return widget
    
    def get_switch_button_style(self, user_type, is_selected):
        if user_type == 'student':
            if is_selected:
                return f"""
                    QPushButton {{
                        background-color: {COLORS['primary']};
                        border: none;
                        border-radius: 15px;
                        color: white;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: #2980B9;
                    }}
                    QPushButton:pressed {{
                        background-color: #1F6F9F;
                    }}
                """
            else:
                return f"""
                    QPushButton {{
                        background-color: transparent;
                        border: none;
                        border-radius: 15px;
                        color: #5D6D7E;
                    }}
                    QPushButton:hover {{
                        background-color: rgba(52, 152, 219, 0.1);
                        color: {COLORS['primary']};
                    }}
                """
        else:  # teacher
            if is_selected:
                return f"""
                    QPushButton {{
                        background-color: {COLORS['secondary']};
                        border: none;
                        border-radius: 15px;
                        color: white;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: #8E44AD;
                    }}
                    QPushButton:pressed {{
                        background-color: #7D3C98;
                    }}
                """
            else:
                return f"""
                    QPushButton {{
                        background-color: transparent;
                        border: none;
                        border-radius: 15px;
                        color: #5D6D7E;
                    }}
                    QPushButton:hover {{
                        background-color: rgba(155, 89, 182, 0.1);
                        color: {COLORS['secondary']};
                    }}
                """
    
    def select_user_type(self, user_type):
        self.selected_user_type = user_type

        button_specs = {
            "student": (self.student_btn, "#2563EB", "#1D4ED8"),
            "teacher": (self.teacher_btn, "#8B5CF6", "#7C3AED"),
            "parent": (self.parent_btn, "#10B981", "#059669"),
        }
        for key, (button, active_color, hover_color) in button_specs.items():
            selected = key == user_type
            button.setChecked(selected)
            if selected:
                button.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {active_color};
                        color: white;
                        border: none;
                        border-radius: 10px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: {hover_color};
                    }}
                """)
            else:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #94A3B8;
                        color: white;
                        border: none;
                        border-radius: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #64748B;
                    }
                """)
    
    def select_reg_user_type(self, user_type):
        self.selected_reg_user_type = user_type

        button_specs = {
            "student": (self.reg_student_btn, "#2563EB", "#1D4ED8"),
            "teacher": (self.reg_teacher_btn, "#8B5CF6", "#7C3AED"),
            "parent": (self.reg_parent_btn, "#10B981", "#059669"),
        }
        for key, (button, active_color, hover_color) in button_specs.items():
            selected = key == user_type
            button.setChecked(selected)
            if selected:
                button.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {active_color};
                        color: white;
                        border: none;
                        border-radius: 10px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: {hover_color};
                    }}
                """)
            else:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #94A3B8;
                        color: white;
                        border: none;
                        border-radius: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #64748B;
                    }
                """)
    
    def get_input_style(self):
        return f"""
            QLineEdit {{
                background-color: white;
                border: 2px solid #D5DBDB;
                border-radius: 10px;
                padding: 0 15px;
                color: {COLORS['text']};
            }}
            QLineEdit:focus {{
                border: 2px solid {COLORS['primary']};
            }}
            QLineEdit::placeholder {{
                color: #BDC3C7;
            }}
        """
    
    def toggle_page(self):
        current = self.stacked_widget.currentIndex()
        if current == 0:
            self.stacked_widget.setCurrentIndex(1)
            self.page_title.setText('创建账号')
            self.page_subtitle.setText('请注册您的账号')
            self.switch_label.setText('已有账号？')
            self.switch_btn.setText('立即登录')
        else:
            self.stacked_widget.setCurrentIndex(0)
            self.page_title.setText('欢迎回来')
            self.page_subtitle.setText('请登录您的账号')
            self.switch_label.setText('还没有账号？')
            self.switch_btn.setText('立即注册')
    
    def handle_login(self):
        if self.auth_busy:
            return

        username = self.login_username.text().strip()
        password = self.login_password.text().strip()
        user_type = self.selected_user_type
        
        if not username or not password:
            QMessageBox.warning(self, '提示', '请输入用户名和密码')
            return

        self._set_auth_busy(True, "login")
        self.auth_executor.submit(self._run_login, username, password, user_type)
    
    def handle_register(self):
        if self.auth_busy:
            return

        username = self.reg_username.text().strip()
        password = self.reg_password.text().strip()
        confirm = self.reg_confirm.text().strip()
        user_type = self.selected_reg_user_type
        name = ''  # 不再需要姓名输入
        email = ''  # 不再需要邮箱输入
        
        if not username or not password:
            QMessageBox.warning(self, '提示', '用户名和密码不能为空')
            return
        
        if password != confirm:
            QMessageBox.warning(self, '提示', '两次输入的密码不一致')
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, '提示', '密码长度至少6位')
            return

        self._set_auth_busy(True, "register")
        self.auth_executor.submit(self._run_register, username, password, user_type, name, email)

    def _run_login(self, username, password, user_type):
        try:
            success, result = self.db.login_user(username, password, user_type)
        except Exception as exc:
            success, result = False, f"登录异常：{exc}"
        self.auth_finished.emit("login", success, result)

    def _run_register(self, username, password, user_type, name, email):
        try:
            success, result = self.db.register_user(username, password, user_type, name, email)
        except Exception as exc:
            success, result = False, f"注册异常：{exc}"
        self.auth_finished.emit("register", success, result)

    def _handle_auth_finished(self, action, success, result):
        self._set_auth_busy(False, action)
        if action == "login":
            if success:
                self.login_success.emit(result)
                self.close()
            else:
                QMessageBox.warning(self, '登录失败', str(result))
            return

        if success:
            QMessageBox.information(self, '注册成功', str(result))
            self.toggle_page()
        else:
            QMessageBox.warning(self, '注册失败', str(result))

    def _set_auth_busy(self, busy, action="login"):
        self.auth_busy = busy
        widgets = [
            getattr(self, "login_username", None),
            getattr(self, "login_password", None),
            getattr(self, "reg_username", None),
            getattr(self, "reg_password", None),
            getattr(self, "reg_confirm", None),
            getattr(self, "student_btn", None),
            getattr(self, "teacher_btn", None),
            getattr(self, "parent_btn", None),
            getattr(self, "reg_student_btn", None),
            getattr(self, "reg_teacher_btn", None),
            getattr(self, "reg_parent_btn", None),
            getattr(self, "switch_btn", None),
        ]
        for widget in widgets:
            if widget is not None:
                widget.setEnabled(not busy)

        if hasattr(self, "login_btn"):
            self.login_btn.setEnabled(not busy)
            self.login_btn.setText("登录中..." if busy and action == "login" else "登 录")
        if hasattr(self, "register_btn"):
            self.register_btn.setEnabled(not busy)
            self.register_btn.setText("注册中..." if busy and action == "register" else "注 册")

        if busy:
            self.setCursor(Qt.WaitCursor)
        else:
            self.unsetCursor()

    def closeEvent(self, event):
        try:
            self.auth_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        super().closeEvent(event)
