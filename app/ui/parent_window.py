from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from app.ui.analytics_windows import ChartCanvas, SummaryCard, panel_style, score_color, style_table
from config.config import APP_NAME, COLORS, WINDOW_HEIGHT, WINDOW_WIDTH


SURFACE = "#FFFFFF"
SOFT = "#F4F8FC"
INK = "#1F2D3D"
MUTED = "#6B7A90"
LINE = "#E6EDF5"


class ParentMainWindow(QWidget):
    logout_signal = pyqtSignal()
    dashboard_loaded = pyqtSignal(object)
    dashboard_failed = pyqtSignal(str)
    bind_finished = pyqtSignal(bool, str, object)

    def __init__(self, user_info, db):
        super().__init__()
        self.user_info = user_info
        self.db = db
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.refresh_request_id = 0

        self.dashboard_loaded.connect(self._apply_dashboard)
        self.dashboard_failed.connect(self._handle_dashboard_error)
        self.bind_finished.connect(self._handle_bind_finished)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refresh_dashboard)
        self.update_timer.start(10000)

        self.current_child = None
        self.last_successful_payload = None

        self.init_ui()
        self.refresh_dashboard()

    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} - 家长端 - {self.user_info.get('name', self.user_info['username'])}")
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"QWidget {{ background-color: {SOFT}; color: {INK}; }}")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(22, 22, 22, 22)
        main_layout.setSpacing(18)

        hero = QFrame()
        hero.setStyleSheet(
            f"""
            QFrame {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FFFFFF,
                    stop:1 #F5F9FF
                );
                border: 1px solid {LINE};
                border-radius: 24px;
            }}
            """
        )
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_layout.setSpacing(24)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        title = QLabel("家长关怀中心")
        title.setFont(QFont("Microsoft YaHei", 28, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['primary']};")
        left_layout.addWidget(title)

        subtitle = QLabel("查看孩子专注度、每日学习情况和与小睿的互动记录")
        subtitle.setFont(QFont("Microsoft YaHei", 11))
        subtitle.setStyleSheet(f"color: {MUTED};")
        left_layout.addWidget(subtitle)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(10)

        self.child_name_label = QLabel("暂未绑定孩子")
        self.child_name_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        self.child_name_label.setStyleSheet(
            """
            QLabel {
                background: rgba(52, 152, 219, 0.12);
                color: #1D4ED8;
                border-radius: 14px;
                padding: 8px 14px;
            }
            """
        )
        badge_row.addWidget(self.child_name_label)

        self.child_meta_label = QLabel("绑定后可查看孩子专注数据与提问记录")
        self.child_meta_label.setFont(QFont("Microsoft YaHei", 10))
        self.child_meta_label.setStyleSheet(
            """
            QLabel {
                background: rgba(31, 45, 61, 0.06);
                color: #334155;
                border-radius: 14px;
                padding: 8px 14px;
            }
            """
        )
        badge_row.addWidget(self.child_meta_label)
        badge_row.addStretch()
        left_layout.addLayout(badge_row)

        self.hero_hint_label = QLabel("家长端会展示孩子最新专注状态、每日专注趋势和当天向小睿提出的问题。")
        self.hero_hint_label.setFont(QFont("Microsoft YaHei", 10))
        self.hero_hint_label.setStyleSheet(f"color: {MUTED};")
        left_layout.addWidget(self.hero_hint_label)

        hero_layout.addLayout(left_layout, stretch=1)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        self.bind_btn = QPushButton("绑定孩子")
        self.bind_btn.clicked.connect(self.bind_child)
        self.bind_btn.setStyleSheet(self._button_style("#2563EB", "#1D4ED8"))
        right_layout.addWidget(self.bind_btn)

        refresh_btn = QPushButton("刷新数据")
        refresh_btn.clicked.connect(self.refresh_dashboard)
        refresh_btn.setStyleSheet(self._button_style("#0EA5E9", "#0284C7"))
        right_layout.addWidget(refresh_btn)

        logout_btn = QPushButton("退出登录")
        logout_btn.clicked.connect(self.handle_logout)
        logout_btn.setStyleSheet(self._button_style("#EF4444", "#DC2626"))
        right_layout.addWidget(logout_btn)
        right_layout.addStretch(1)

        hero_layout.addLayout(right_layout)
        main_layout.addWidget(hero)

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(12)
        self.current_card = SummaryCard("当前状态", COLORS["primary"])
        self.avg_card = SummaryCard("平均专注度", COLORS["success"])
        self.focus_card = SummaryCard("专注占比", COLORS["warning"])
        self.today_question_card = SummaryCard("今日小睿提问", "#8B5CF6")
        self.records_card = SummaryCard("有效记录数", "#546A7B")
        for card in [
            self.current_card,
            self.avg_card,
            self.focus_card,
            self.today_question_card,
            self.records_card,
        ]:
            cards_layout.addWidget(card)
        main_layout.addLayout(cards_layout)

        upper_layout = QHBoxLayout()
        upper_layout.setSpacing(16)

        trend_panel = self._create_panel("最近学习趋势", "查看最近 14 天孩子的平均专注度变化")
        self.trend_canvas = ChartCanvas(width=5.8, height=2.8)
        trend_panel.layout().addWidget(self.trend_canvas)
        upper_layout.addWidget(trend_panel, stretch=3)

        insight_panel = self._create_panel("家长提醒", "这里会给出当前更适合关注的学习信号")
        self.insight_label = QLabel("绑定孩子后，这里会显示专注度变化、在线情况和提问活跃度提示。")
        self.insight_label.setWordWrap(True)
        self.insight_label.setAlignment(Qt.AlignTop)
        self.insight_label.setFont(QFont("Microsoft YaHei", 10))
        self.insight_label.setStyleSheet(
            f"""
            QLabel {{
                color: {INK};
                background-color: #F8FBFA;
                border: 1px solid {LINE};
                border-radius: 14px;
                padding: 14px;
                line-height: 150%;
            }}
            """
        )
        insight_panel.layout().addWidget(self.insight_label, stretch=1)
        upper_layout.addWidget(insight_panel, stretch=2)
        main_layout.addLayout(upper_layout, stretch=2)

        lower_layout = QHBoxLayout()
        lower_layout.setSpacing(16)

        daily_panel = self._create_panel("每日学习情况", "按天汇总平均专注度、记录数和专注占比")
        self.daily_table = QTableWidget()
        self.daily_table.setColumnCount(4)
        self.daily_table.setHorizontalHeaderLabels(["日期", "平均专注度", "记录数", "专注占比"])
        style_table(self.daily_table)
        daily_panel.layout().addWidget(self.daily_table)
        lower_layout.addWidget(daily_panel, stretch=2)

        ask_panel = self._create_panel("孩子向小睿提问记录", "展示最近提问的时间、方式和问题内容")
        self.ask_table = QTableWidget()
        self.ask_table.setColumnCount(4)
        self.ask_table.setHorizontalHeaderLabels(["时间", "方式", "问题", "模型"])
        style_table(self.ask_table)
        ask_panel.layout().addWidget(self.ask_table)
        lower_layout.addWidget(ask_panel, stretch=3)

        main_layout.addLayout(lower_layout, stretch=3)

    def _create_panel(self, title, subtitle=""):
        panel = QFrame()
        panel.setStyleSheet(panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_label.setStyleSheet(f"color: {INK};")
        layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setWordWrap(True)
            subtitle_label.setFont(QFont("Microsoft YaHei", 9))
            subtitle_label.setStyleSheet(f"color: {MUTED};")
            layout.addWidget(subtitle_label)

        return panel

    def _button_style(self, normal, hover):
        return f"""
            QPushButton {{
                background-color: {normal};
                color: white;
                border: none;
                border-radius: 14px;
                padding: 12px 16px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """

    def bind_child(self):
        username, accepted = QInputDialog.getText(self, "绑定孩子", "请输入孩子的学生账号：")
        if not accepted:
            return
        self.bind_btn.setEnabled(False)
        self.executor.submit(self._run_bind_child, username)

    def _run_bind_child(self, username):
        try:
            success, message, child = self.db.bind_parent_to_student(self.user_info["id"], username)
            self.bind_finished.emit(success, message, child)
        except Exception as exc:
            self.bind_finished.emit(False, f"绑定失败：{exc}", None)

    def _handle_bind_finished(self, success, message, child):
        self.bind_btn.setEnabled(True)
        if success:
            self.current_child = child
            QMessageBox.information(self, "绑定成功", message)
            self.refresh_dashboard()
            return
        QMessageBox.warning(self, "绑定失败", message)

    def refresh_dashboard(self):
        self.refresh_request_id += 1
        request_id = self.refresh_request_id
        self.executor.submit(self._load_dashboard, request_id)

    def _load_dashboard(self, request_id):
        try:
            payload = self.db.get_parent_dashboard(self.user_info["id"])
            self.dashboard_loaded.emit({"request_id": request_id, "payload": payload})
        except Exception as exc:
            self.dashboard_failed.emit(str(exc))

    def _apply_dashboard(self, event):
        if event["request_id"] != self.refresh_request_id:
            return
        payload = event["payload"] or {}
        if payload.get("_fetch_failed"):
            if self.last_successful_payload:
                message = payload.get("_error_message") or "网络刷新暂时失败，已保留上一版数据。"
                self.insight_label.setText(f"家长端刚刚刷新失败：{message}\n\n当前先为你保留上一版已加载的数据。")
                return
            self.insight_label.setText(
                f"家长端数据暂时加载失败：{payload.get('_error_message') or '请稍后重试。'}"
            )
            return

        child = payload.get("child")
        summary = payload.get("summary") or {}
        daily_rows = payload.get("daily_rows") or []
        ask_logs = payload.get("assistant_logs") or []
        today_query_count = int(payload.get("today_query_count") or 0)

        self.current_child = child
        if not child:
            self.child_name_label.setText("暂未绑定孩子")
            self.child_meta_label.setText("请输入孩子的学生账号完成绑定")
            self.hero_hint_label.setText("绑定后，家长端可以查看孩子专注趋势、学习记录和每天向小睿提问的内容。")
            self.current_card.set_value("--", "等待绑定")
            self.avg_card.set_value("--", "暂无平均数据")
            self.focus_card.set_value("--", "暂无专注占比")
            self.today_question_card.set_value("0", "今日暂无提问")
            self.records_card.set_value("0", "暂无有效记录")
            self._render_empty_trend()
            self._render_daily_rows([])
            self._render_ask_logs([])
            self.insight_label.setText("暂未绑定孩子。点击右上角“绑定孩子”，输入孩子的学生账号后即可开始查看。")
            return

        self.last_successful_payload = payload

        child_name = child.get("name") or child.get("username") or "未命名学生"
        class_name = child.get("class_name") or "未加入班级"
        online_text = "在线" if child.get("online") else "离线"
        self.child_name_label.setText(f"孩子：{child_name}")
        self.child_meta_label.setText(f"{class_name} · {online_text}")
        self.hero_hint_label.setText(
            f"已绑定学生账号 {child.get('username')}。可持续查看孩子最近专注状态、每日学习趋势和小睿提问记录。"
        )

        latest_score = float(summary.get("latest_score") or 0.0)
        latest_status = summary.get("latest_status") or "暂无数据"
        self.current_card.set_value(f"{latest_score:.0f}%", latest_status)
        self.avg_card.set_value(f"{float(summary.get('avg_score') or 0.0):.1f}%", "最近 300 条记录")
        self.focus_card.set_value(f"{float(summary.get('focused_rate') or 0.0) * 100:.0f}%", "专注记录占比")
        self.today_question_card.set_value(str(today_query_count), "今日向小睿提问次数")
        self.records_card.set_value(str(int(summary.get("record_count") or 0)), summary.get("latest_timestamp") or "暂无最近时间")

        self._render_trend(daily_rows)
        self._render_daily_rows(daily_rows)
        self._render_ask_logs(ask_logs)

        insight_lines = [
            f"孩子当前状态：{latest_status}，最近平均专注度 {float(summary.get('avg_score') or 0.0):.1f}%。",
            f"今日向小睿提问 {today_query_count} 次，{('说明学习互动较活跃' if today_query_count >= 3 else '今天提问不多，可以适当鼓励孩子主动求助。')}",
        ]
        if ask_logs:
            insight_lines.append(f"最近一次提问：{ask_logs[0].get('query_text', '')[:36]}")
        else:
            insight_lines.append("最近还没有记录到孩子向小睿提问。")
        self.insight_label.setText("\n\n".join(insight_lines))

    def _render_empty_trend(self):
        self.trend_canvas.clear()
        self.trend_canvas.axes.text(0.5, 0.5, "暂无趋势数据", ha="center", va="center", color=MUTED)
        self.trend_canvas.draw()

    def _render_trend(self, daily_rows):
        self.trend_canvas.clear()
        if not daily_rows:
            self._render_empty_trend()
            return
        rows = list(reversed(daily_rows[-10:]))
        labels = [str(item.get("date", ""))[5:] for item in rows]
        values = [float(item.get("avg_score") or 0.0) for item in rows]
        colors = [score_color(value) for value in values]
        self.trend_canvas.axes.plot(labels, values, color=COLORS["primary"], linewidth=2.2, marker="o")
        self.trend_canvas.axes.scatter(labels, values, color=colors, s=42, zorder=3)
        self.trend_canvas.axes.set_ylim(0, 100)
        self.trend_canvas.axes.set_ylabel("平均专注度", color=MUTED, fontsize=9)
        self.trend_canvas.draw()

    def _render_daily_rows(self, daily_rows):
        self.daily_table.setRowCount(len(daily_rows))
        for row_index, row in enumerate(daily_rows):
            values = [
                row.get("date", "-"),
                f"{float(row.get('avg_score') or 0.0):.1f}%",
                str(int(row.get("record_count") or 0)),
                f"{float(row.get('focused_rate') or 0.0) * 100:.0f}%",
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column_index == 1:
                    item.setForeground(QColor(score_color(float(row.get("avg_score") or 0.0))))
                self.daily_table.setItem(row_index, column_index, item)
        self.daily_table.resizeRowsToContents()

    def _render_ask_logs(self, ask_logs):
        self.ask_table.setRowCount(len(ask_logs))
        for row_index, row in enumerate(ask_logs):
            source_label = "语音" if row.get("source") == "voice" else "文本"
            values = [
                str(row.get("timestamp", "")).replace("T", " "),
                source_label,
                row.get("query_text", ""),
                row.get("model_name", "") or "内置/默认",
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                self.ask_table.setItem(row_index, column_index, item)
        self.ask_table.resizeRowsToContents()

    def _handle_dashboard_error(self, message):
        self.insight_label.setText(f"家长端数据暂时加载失败：{message}")

    def handle_logout(self):
        self.update_timer.stop()
        self.executor.shutdown(wait=False)
        self.logout_signal.emit()
        self.close()
