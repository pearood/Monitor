from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.analytics_windows import StudentDetailDialog, TeacherAnalyticsWindow
from config.config import APP_NAME, COLORS, WINDOW_HEIGHT, WINDOW_WIDTH


SURFACE = "#FFFFFF"
SOFT = "#F4F8FC"
INK = "#1F2D3D"
MUTED = "#6B7A90"
LINE = "#E6EDF5"


class TeacherMetricCard(QFrame):
    def __init__(self, title, accent):
        super().__init__()
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {SURFACE};
                border: 1px solid {LINE};
                border-radius: 20px;
            }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        accent_label = QLabel("●")
        accent_label.setStyleSheet(f"color: {accent};")
        accent_label.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(accent_label, alignment=Qt.AlignLeft)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {MUTED};")
        title_label.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(title_label)

        self.value_label = QLabel("--")
        self.value_label.setStyleSheet(f"color: {accent};")
        self.value_label.setFont(QFont("Arial", 26, QFont.Bold))
        layout.addWidget(self.value_label)

        self.hint_label = QLabel("")
        self.hint_label.setStyleSheet(f"color: {MUTED};")
        self.hint_label.setFont(QFont("Microsoft YaHei", 9))
        layout.addWidget(self.hint_label)

    def set_value(self, value, hint=""):
        self.value_label.setText(value)
        self.hint_label.setText(hint)


class TeacherMainWindow(QWidget):
    logout_signal = pyqtSignal()
    dashboard_loaded = pyqtSignal(int, object)
    dashboard_failed = pyqtSignal(int, str)
    class_action_finished = pyqtSignal(bool, str, object)
    class_options_loaded = pyqtSignal(list, object)

    def __init__(self, user_info, db):
        super().__init__()
        self.user_info = user_info
        self.db = db
        self.analytics_window = None
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.current_class = ""
        self.current_class_info = None
        self.refresh_request_id = 0
        self.class_action_inflight = False

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refresh_dashboard)
        self.update_timer.start(8000)

        self.dashboard_loaded.connect(self._apply_dashboard_payload)
        self.dashboard_failed.connect(self._handle_dashboard_error)
        self.class_action_finished.connect(self._handle_class_action_finished)
        self.class_options_loaded.connect(self._apply_class_options)

        self.init_ui()
        self.refresh_class_options_async()

    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} - 教师端 - {self.user_info.get('name', self.user_info['username'])}")
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"QWidget {{ background-color: {SOFT}; color: {INK}; }}")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(22, 22, 22, 22)
        main_layout.setSpacing(18)

        hero_frame = QFrame()
        hero_frame.setStyleSheet(
            f"""
            QFrame {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FFFFFF,
                    stop:1 #F1F8FF
                );
                border: 1px solid {LINE};
                border-radius: 24px;
            }}
            """
        )
        hero_layout = QHBoxLayout(hero_frame)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(22)

        hero_left = QVBoxLayout()
        hero_left.setSpacing(8)

        title = QLabel("教师分析中心")
        title.setFont(QFont("Microsoft YaHei", 28, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['primary']};")
        hero_left.addWidget(title)

        subtitle = QLabel("围绕班级、学生与专注趋势的实时管理界面")
        subtitle.setFont(QFont("Microsoft YaHei", 11))
        subtitle.setStyleSheet(f"color: {MUTED};")
        hero_left.addWidget(subtitle)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(10)

        self.class_name_label = QLabel("未选择班级")
        self.class_name_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        self.class_name_label.setStyleSheet(
            """
            QLabel {
                background: rgba(52, 152, 219, 0.12);
                color: #1D4ED8;
                border-radius: 14px;
                padding: 8px 14px;
            }
            """
        )
        badge_row.addWidget(self.class_name_label)

        self.class_code_label = QLabel("班级码 --")
        self.class_code_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.class_code_label.setStyleSheet(
            """
            QLabel {
                background: rgba(31, 45, 61, 0.06);
                color: #334155;
                border-radius: 14px;
                padding: 8px 14px;
            }
            """
        )
        badge_row.addWidget(self.class_code_label)
        badge_row.addStretch()
        hero_left.addLayout(badge_row)

        self.class_meta_label = QLabel("创建班级后，可将班级码发给学生加入。")
        self.class_meta_label.setFont(QFont("Microsoft YaHei", 10))
        self.class_meta_label.setStyleSheet(f"color: {MUTED};")
        hero_left.addWidget(self.class_meta_label)

        hero_layout.addLayout(hero_left, stretch=3)

        hero_right = QVBoxLayout()
        hero_right.setSpacing(10)

        selector_label = QLabel("当前班级")
        selector_label.setFont(QFont("Microsoft YaHei", 10))
        selector_label.setStyleSheet(f"color: {MUTED};")
        hero_right.addWidget(selector_label)

        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(210)
        self.class_combo.setMinimumHeight(40)
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        self.class_combo.setStyleSheet(self._combo_style())
        hero_right.addWidget(self.class_combo)

        self.create_class_btn = QPushButton("创建班级")
        self.create_class_btn.clicked.connect(self.create_class)
        self.create_class_btn.setStyleSheet(self._button_style("#2563EB", "#1D4ED8"))
        hero_right.addWidget(self.create_class_btn)

        analytics_btn = QPushButton("打开分析中心")
        analytics_btn.clicked.connect(self.open_teacher_analytics)
        analytics_btn.setStyleSheet(self._button_style("#0EA5E9", "#0284C7"))
        hero_right.addWidget(analytics_btn)

        logout_btn = QPushButton("退出登录")
        logout_btn.clicked.connect(self.handle_logout)
        logout_btn.setStyleSheet(self._button_style("#EF4444", "#DC2626"))
        hero_right.addWidget(logout_btn)
        hero_right.addStretch()

        hero_layout.addLayout(hero_right, stretch=1)
        main_layout.addWidget(hero_frame)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(16)
        self.avg_card = TeacherMetricCard("班级平均专注度", COLORS["focused"])
        self.online_card = TeacherMetricCard("当前在线学生", "#10B981")
        self.max_card = TeacherMetricCard("当前最高专注度", COLORS["primary"])
        self.min_card = TeacherMetricCard("当前最低专注度", COLORS["danger"])
        self.records_card = TeacherMetricCard("累计有效记录", COLORS["warning"])
        for card in (self.avg_card, self.online_card, self.max_card, self.min_card, self.records_card):
            stats_row.addWidget(card)
        main_layout.addLayout(stats_row)

        table_frame = QFrame()
        table_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: {SURFACE};
                border: 1px solid {LINE};
                border-radius: 22px;
            }}
            """
        )
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(22, 20, 22, 20)
        table_layout.setSpacing(14)

        table_header = QHBoxLayout()
        header_text_layout = QVBoxLayout()
        header_text_layout.setSpacing(4)

        table_title = QLabel("班级学生实时看板")
        table_title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        table_title.setStyleSheet(f"color: {INK};")
        header_text_layout.addWidget(table_title)

        self.table_subtitle = QLabel("选择班级后，这里会展示学生平均专注度、最新状态与活跃情况。")
        self.table_subtitle.setFont(QFont("Microsoft YaHei", 10))
        self.table_subtitle.setStyleSheet(f"color: {MUTED};")
        header_text_layout.addWidget(self.table_subtitle)

        table_header.addLayout(header_text_layout)
        table_header.addStretch()

        detail_btn = QPushButton("查看所选学生")
        detail_btn.clicked.connect(self.open_selected_student)
        detail_btn.setStyleSheet(self._button_style("#8B5CF6", "#7C3AED"))
        table_header.addWidget(detail_btn)
        table_layout.addLayout(table_header)

        self.students_table = QTableWidget()
        self.students_table.setColumnCount(8)
        self.students_table.setHorizontalHeaderLabels(
            ["ID", "姓名", "班级", "在线状态", "平均专注度", "最新状态", "记录数", "最近时间"]
        )
        self.students_table.verticalHeader().setVisible(False)
        self.students_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.students_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.students_table.setAlternatingRowColors(True)
        self.students_table.setShowGrid(False)
        self.students_table.itemDoubleClicked.connect(lambda _: self.open_selected_student())
        self.students_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.students_table.horizontalHeader().setMinimumSectionSize(80)
        self.students_table.setStyleSheet(self._table_style())
        table_layout.addWidget(self.students_table)

        self.empty_label = QLabel("当前还没有班级数据，先创建班级并让学生加入后，这里就会自动刷新。")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setFont(QFont("Microsoft YaHei", 10))
        self.empty_label.setStyleSheet(f"color: {MUTED}; padding: 12px 0 4px 0;")
        table_layout.addWidget(self.empty_label)

        main_layout.addWidget(table_frame, stretch=1)

    def _button_style(self, background, hover):
        return f"""
            QPushButton {{
                background-color: {background};
                color: white;
                border: none;
                border-radius: 12px;
                padding: 10px 18px;
                font: 11pt 'Microsoft YaHei';
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """

    def _combo_style(self):
        return f"""
            QComboBox {{
                background: white;
                border: 1px solid {LINE};
                border-radius: 12px;
                padding: 0 12px;
                color: {INK};
                font: 10pt 'Microsoft YaHei';
            }}
            QComboBox::drop-down {{
                border: none;
                width: 28px;
            }}
        """

    def _table_style(self):
        return f"""
            QTableWidget {{
                background-color: {SURFACE};
                border: 1px solid {LINE};
                border-radius: 16px;
                alternate-background-color: #F8FBFE;
                color: {INK};
                selection-background-color: #DCEEFF;
                selection-color: {INK};
            }}
            QHeaderView::section {{
                background-color: #F2F7FC;
                color: {INK};
                border: none;
                padding: 10px;
                font-weight: 600;
            }}
            QTableWidget::item {{
                padding: 10px;
                border-bottom: 1px solid {LINE};
            }}
        """

    def refresh_class_options(self, preferred_class=None):
        options = self.db.get_class_options() or []
        self._apply_class_options(options, preferred_class)

    def refresh_class_options_async(self, preferred_class=None):
        self.executor.submit(self._load_class_options, preferred_class)

    def _load_class_options(self, preferred_class=None):
        try:
            options = self.db.get_class_options() or []
        except Exception:
            options = []
        self.class_options_loaded.emit(options, preferred_class)

    def _apply_class_options(self, options, preferred_class=None):
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(options)
        self.class_combo.blockSignals(False)

        if not options:
            self.current_class = ""
            return

        target_class = preferred_class if preferred_class in options else options[0]
        self.class_combo.setCurrentText(target_class)
        self.current_class = target_class
        self.refresh_dashboard()

    def create_class(self):
        if self.class_action_inflight:
            return
        class_name, accepted = QInputDialog.getText(self, "创建班级", "请输入新的班级名称：")
        if not accepted:
            return

        self.class_action_inflight = True
        self.create_class_btn.setEnabled(False)
        self.create_class_btn.setText("创建中...")
        self.executor.submit(self._run_create_class, class_name)

    def on_class_changed(self):
        self.current_class = self.class_combo.currentText().strip()
        self.refresh_dashboard()

    def refresh_dashboard(self):
        if not self.current_class:
            self.current_class_info = None
            self.class_name_label.setText("还没有班级")
            self.class_code_label.setText("班级码 --")
            self.class_meta_label.setText("点击右侧“创建班级”后，即可生成班级并分发班级码。")
            self.table_subtitle.setText("当前没有可展示的班级，请先创建班级。")
            self.avg_card.set_value("--", "等待数据接入")
            self.online_card.set_value("--", "等待数据接入")
            self.max_card.set_value("--", "等待数据接入")
            self.min_card.set_value("--", "等待数据接入")
            self.records_card.set_value("--", "等待数据接入")
            self.students_table.setRowCount(0)
            self.empty_label.setVisible(True)
            return

        self.refresh_request_id += 1
        request_id = self.refresh_request_id
        current_class = self.current_class

        self.class_meta_label.setText("正在同步班级数据，请稍候...")
        self.table_subtitle.setText(f"当前查看：{current_class}，正在从云端刷新最新数据。")
        self.executor.submit(self._load_dashboard_payload, request_id, current_class)

    def _load_dashboard_payload(self, request_id, class_name):
        try:
            payload = {
                "class_name": class_name,
                "summary": self.db.get_class_summary(class_name),
                "student_rows": self.db.get_class_student_summaries(class_name),
                "class_info": self.db.get_class_info(class_name) or {},
            }
        except Exception as exc:
            self.dashboard_failed.emit(request_id, str(exc))
            return
        self.dashboard_loaded.emit(request_id, payload)

    def _apply_dashboard_payload(self, request_id, payload):
        if request_id != self.refresh_request_id:
            return

        summary = payload["summary"]
        student_rows = payload["student_rows"]
        self.current_class_info = payload["class_info"]

        class_code = self.current_class_info.get("class_code", "--")
        student_count = summary.get("student_count", 0)
        online_count = summary.get("online_count", sum(1 for item in student_rows if item.get("online")))
        focused_rate = summary.get("focused_rate", 0.0) * 100

        self.class_name_label.setText(self.current_class)
        self.class_code_label.setText(f"班级码 {class_code}")
        self.class_meta_label.setText(
            f"班级人数 {student_count} 人 · 在线 {online_count} 人 · 专注率 {focused_rate:.1f}% · 将班级码分享给学生即可加入。"
        )
        self.table_subtitle.setText(f"当前查看：{self.current_class}，双击学生行可进入个人详细分析。")

        self.avg_card.set_value(f"{summary['avg_score']:.1f}%", "班级整体专注水平")
        self.online_card.set_value(str(online_count), "最近 20 秒在线学生数")
        self.max_card.set_value(f"{summary['max_score']:.1f}%", "当前班级最佳表现")
        self.min_card.set_value(f"{summary['min_score']:.1f}%", "当前班级最低值")
        self.records_card.set_value(str(summary["record_count"]), "累计采集记录条数")

        self.students_table.setRowCount(len(student_rows))
        self.empty_label.setVisible(len(student_rows) == 0)

        for row_index, row in enumerate(student_rows):
            latest_time = (row["latest_timestamp"] or "-").replace("T", " ")
            online_text = "在线" if row.get("online") else "离线"
            values = [
                row["id"],
                row["name"],
                row["class_name"],
                online_text,
                f"{row['avg_score']:.1f}%",
                row["latest_status"],
                row["record_count"],
                latest_time,
            ]

            for column_index, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                if column_index == 3:
                    item.setForeground(QColor("#10B981" if row.get("online") else "#94A3B8"))
                if column_index == 4:
                    score = row["avg_score"]
                    if score >= 70:
                        item.setForeground(QColor(COLORS["focused"]))
                    elif score >= 40:
                        item.setForeground(QColor(COLORS["moderate"]))
                    else:
                        item.setForeground(QColor(COLORS["distracted"]))
                self.students_table.setItem(row_index, column_index, item)

        if self.analytics_window is not None:
            self.analytics_window.refresh_view()

    def _handle_dashboard_error(self, request_id, error_message):
        if request_id != self.refresh_request_id:
            return
        self.class_meta_label.setText("班级数据同步失败，请稍后重试。")
        self.table_subtitle.setText(error_message)

    def _run_create_class(self, class_name):
        success, message, class_info = self.db.create_class(self.user_info["id"], class_name)
        self.class_action_finished.emit(success, message, class_info)

    def _handle_class_action_finished(self, success, message, class_info):
        self.class_action_inflight = False
        self.create_class_btn.setEnabled(True)
        self.create_class_btn.setText("创建班级")
        if success:
            created_class = (class_info or {}).get("class_name", "")
            class_code = (class_info or {}).get("class_code", "--")
            self.refresh_class_options_async(preferred_class=created_class)
            QMessageBox.information(
                self,
                "班级创建成功",
                f"{message}\n\n将班级码 {class_code} 发给学生后，他们就可以在学生端加入班级。",
            )
            return

        QMessageBox.warning(self, "创建失败", message)

    def open_selected_student(self):
        selected_row = self.students_table.currentRow()
        if selected_row < 0:
            return

        user_id_item = self.students_table.item(selected_row, 0)
        if user_id_item is None:
            return

        student_info = self.db.get_user_by_id(int(user_id_item.text()))
        if not student_info:
            return

        dialog = StudentDetailDialog(student_info, self.db, self)
        dialog.exec_()

    def open_teacher_analytics(self):
        if self.analytics_window is None:
            self.analytics_window = TeacherAnalyticsWindow(
                self.user_info,
                self.db,
                self.current_class,
            )
        self.analytics_window.show()
        self.analytics_window.raise_()
        self.analytics_window.activateWindow()
        self.analytics_window.refresh_view()

    def handle_logout(self):
        if self.analytics_window:
            self.analytics_window.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.logout_signal.emit()
