import os
import tempfile

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "focus_mpl"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config.config import APP_NAME, COLORS, WINDOW_HEIGHT, WINDOW_WIDTH


rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


SURFACE = "#FFFFFF"
INK = "#1F2D3D"
MUTED = "#667085"
SOFT = "#F4F7FB"
LINE = "#E5EAF1"


def panel_style(radius=18):
    return f"""
        QFrame {{
            background-color: {SURFACE};
            border: 1px solid {LINE};
            border-radius: {radius}px;
        }}
    """


def style_table(table):
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.setStyleSheet(
        f"""
        QTableWidget {{
            background-color: {SURFACE};
            alternate-background-color: {SOFT};
            border: 1px solid {LINE};
            border-radius: 12px;
            gridline-color: {LINE};
            color: {INK};
            selection-background-color: #DFF3EA;
            selection-color: {INK};
        }}
        QHeaderView::section {{
            background-color: #ECF3F1;
            color: {INK};
            border: none;
            padding: 8px;
            font-weight: 600;
        }}
        QTableWidget::item {{
            padding: 8px;
        }}
        """
    )


def score_color(score):
    if score >= 70:
        return COLORS["focused"]
    if score >= 40:
        return COLORS["moderate"]
    return COLORS["distracted"]


def status_label_from_score(score):
    if score >= 70:
        return "专注"
    if score >= 40:
        return "一般"
    return "不专注"


def compact_time_label(timestamp):
    text = str(timestamp or "").strip()
    if not text:
        return "-"

    text = text.replace("T", " ")
    if " " in text:
        text = text.split()[-1]
    if "." in text:
        text = text.split(".", 1)[0]
    return text[:8]


def sparse_tick_indexes(total, max_ticks=6):
    if total <= 0:
        return []
    if total <= max_ticks:
        return list(range(total))

    step = (total - 1) / float(max_ticks - 1)
    return sorted({int(round(index * step)) for index in range(max_ticks)})


class ChartCanvas(FigureCanvas):
    def __init__(self, width=5, height=3):
        self.figure = Figure(figsize=(width, height))
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.figure.patch.set_facecolor(SURFACE)
        self.figure.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.18)
        self.setMinimumSize(int(width * 70), int(height * 64))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self):
        self.axes.cla()
        self.axes.set_facecolor(SURFACE)
        self.axes.grid(alpha=0.18, color="#97A3B6", linewidth=0.8)
        self.axes.tick_params(colors=MUTED, labelsize=8)
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.axes.spines["left"].set_color(LINE)
        self.axes.spines["bottom"].set_color(LINE)


class SummaryCard(QFrame):
    def __init__(self, title, color):
        super().__init__()
        self.color = color
        self.setMinimumHeight(118)
        self.setStyleSheet(
            f"""
            QFrame {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FFFFFF,
                    stop:1 #F5FAF8
                );
                border: 1px solid {LINE};
                border-left: 6px solid {color};
                border-radius: 18px;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {MUTED}; border: none; background: transparent;")
        self.title_label.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self.title_label)

        self.value_label = QLabel("--")
        self.value_label.setStyleSheet(f"color: {color}; border: none; background: transparent;")
        self.value_label.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        layout.addWidget(self.value_label)

        self.subtitle_label = QLabel("")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet(f"color: {MUTED}; border: none; background: transparent;")
        self.subtitle_label.setFont(QFont("Microsoft YaHei", 9))
        layout.addWidget(self.subtitle_label)

    def set_value(self, value, subtitle=""):
        self.value_label.setText(value)
        self.subtitle_label.setText(subtitle)


class DonutMetricCard(QFrame):
    def __init__(self, title, color):
        super().__init__()
        self.color = color
        self.setMinimumHeight(158)
        self.setMaximumHeight(178)
        self.setStyleSheet(panel_style(radius=16))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(f"color: {MUTED};")
        self.title_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        layout.addWidget(self.title_label)

        self.chart = ChartCanvas(width=2.35, height=1.65)
        self.chart.setMinimumSize(158, 112)
        layout.addWidget(self.chart, stretch=1)

        self.value_label = QLabel("--")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet(f"color: {color};")
        self.value_label.setFont(QFont("Microsoft YaHei", 1))
        self.value_label.setVisible(False)
        layout.addWidget(self.value_label)

        self.subtitle_label = QLabel("")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet(f"color: {MUTED};")
        self.subtitle_label.setFont(QFont("Microsoft YaHei", 10))
        layout.addWidget(self.subtitle_label)

    def set_metric(self, percent, center_text, subtitle="", color=None):
        color = color or self.color
        percent = max(0.0, min(100.0, float(percent)))
        self.value_label.setText(center_text)
        self.value_label.setStyleSheet(f"color: {color};")
        self.subtitle_label.setText(subtitle)
        self.chart.setVisible(True)
        self.chart.clear()
        self.chart.axes.grid(False)
        self.chart.axes.axis("equal")
        self.chart.axes.axis("off")
        self.chart.axes.set_xticks([])
        self.chart.axes.set_yticks([])
        self.chart.figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        self.chart.axes.pie(
            [percent, 100 - percent],
            startangle=90,
            colors=[color, "#E8EEF5"],
            counterclock=False,
            radius=0.9,
            wedgeprops={"width": 0.28, "edgecolor": SURFACE},
        )
        self.chart.axes.text(
            0,
            0,
            center_text,
            ha="center",
            va="center",
            color=color,
            fontsize=17,
            weight="bold",
        )
        self.chart.draw()

    def set_empty(self, subtitle):
        self.value_label.setText("--")
        self.value_label.setStyleSheet(f"color: {MUTED};")
        self.subtitle_label.setText(subtitle)
        self.chart.setVisible(False)


class StudentAnalyticsWindow(QWidget):
    def __init__(self, user_info, db, snapshot_provider):
        super().__init__()
        self.user_info = user_info
        self.db = db
        self.snapshot_provider = snapshot_provider

        self.setWindowTitle(f"{APP_NAME} - 学习分析中心 - {user_info.get('name') or user_info['username']}")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(f"background-color: {SOFT};")

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_view)
        self.refresh_timer.start(1200)

        self._build_ui()
        self.refresh_view()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        hero = QFrame()
        hero.setMinimumHeight(96)
        hero.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #102A43,
                    stop:0.55 #176B65,
                    stop:1 #2FB47C
                );
                border-radius: 24px;
            }
            """
        )
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(20, 10, 20, 10)
        hero_layout.setSpacing(12)

        title_box = QVBoxLayout()
        title = QLabel("学生学习分析")
        title.setFont(QFont("Microsoft YaHei", 21, QFont.Bold))
        title.setStyleSheet("color: white; background: transparent;")
        title_box.addWidget(title)

        self.hero_subtitle = QLabel("实时摄像头专注度、手机、人脸和姿态状态会在这里汇总。")
        self.hero_subtitle.setFont(QFont("Microsoft YaHei", 9))
        self.hero_subtitle.setStyleSheet("color: #DDF7EE; background: transparent;")
        self.hero_subtitle.setWordWrap(True)
        title_box.addWidget(self.hero_subtitle)
        hero_layout.addLayout(title_box, stretch=1)

        self.live_badge = QLabel("等待检测")
        self.live_badge.setAlignment(Qt.AlignCenter)
        self.live_badge.setMinimumWidth(160)
        self.live_badge.setStyleSheet(
            """
            QLabel {
                color: white;
                background-color: rgba(255, 255, 255, 0.18);
                border: 1px solid rgba(255, 255, 255, 0.28);
                border-radius: 18px;
                padding: 12px 18px;
                font-weight: 700;
            }
            """
        )
        self.live_badge.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        hero_layout.addWidget(self.live_badge)
        main_layout.addWidget(hero)

        card_grid = QHBoxLayout()
        card_grid.setSpacing(12)
        self.current_card = DonutMetricCard("当前专注", COLORS["primary"])
        self.avg_card = DonutMetricCard("平均专注", COLORS["success"])
        self.focus_card = DonutMetricCard("专注占比", COLORS["warning"])
        self.phone_card = DonutMetricCard("手机状态", COLORS["danger"])

        cards = [
            self.current_card,
            self.avg_card,
            self.focus_card,
            self.phone_card,
        ]
        for card in cards:
            card_grid.addWidget(card)
        main_layout.addLayout(card_grid)

        chart_layout = QHBoxLayout()
        chart_layout.setSpacing(16)

        trend_panel = self._create_panel("专注度趋势")
        self.trend_canvas = ChartCanvas(width=6.6, height=2.45)
        self.trend_canvas.setMaximumHeight(238)
        trend_panel.layout().addWidget(self.trend_canvas)
        chart_layout.addWidget(trend_panel, stretch=2)

        distribution_panel = self._create_panel("状态与视觉信号")
        distribution_content = QHBoxLayout()
        distribution_content.setSpacing(12)
        self.status_canvas = ChartCanvas(width=3.05, height=2.45)
        self.signal_canvas = ChartCanvas(width=3.5, height=2.45)
        self.status_canvas.setMaximumHeight(238)
        self.signal_canvas.setMaximumHeight(238)
        distribution_content.addWidget(self.status_canvas)
        distribution_content.addWidget(self.signal_canvas)
        distribution_panel.layout().addLayout(distribution_content)
        chart_layout.addWidget(distribution_panel, stretch=2)

        main_layout.addLayout(chart_layout, stretch=3)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(16)

        history_panel = self._create_panel("最近检测明细")
        self.history_table = QTableWidget()
        self.history_table.setMaximumHeight(176)
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels(
            ["时间", "分数", "状态", "姿态", "人脸", "手机", "主要原因"]
        )
        style_table(self.history_table)
        history_panel.layout().addWidget(self.history_table)
        bottom_layout.addWidget(history_panel, stretch=3)

        insight_panel = self._create_panel("诊断建议")
        self.insight_label = QLabel("开始检测后这里会显示分析建议。")
        self.insight_label.setMaximumHeight(176)
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
        bottom_layout.addWidget(insight_panel, stretch=1)

        main_layout.addLayout(bottom_layout, stretch=2)

    def _create_panel(self, title, subtitle=""):
        panel = QFrame()
        panel.setStyleSheet(panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(6)

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

    def refresh_view(self):
        snapshot = self.snapshot_provider()
        live_records = snapshot.get("live_records") or []
        scores = snapshot.get("attention_scores") or [row["attention_score"] for row in live_records]

        if not snapshot.get("is_detecting") or not scores:
            message = "检测中，等待第一帧数据" if snapshot.get("is_detecting") else "未开始检测"
            self._render_idle_state(message)
            return

        self._set_analysis_visible(True)
        summary = self._session_summary(live_records, scores)

        current_score = self._current_score(snapshot, summary)
        avg_score = snapshot.get("avg_score") or summary["avg_score"]
        focused_seconds = snapshot.get("focused_duration_seconds", 0.0)
        distraction_count = snapshot.get("distraction_count", 0)
        live_metrics = self._snapshot_runtime_metrics(snapshot)
        recent_delta = self._recent_delta(scores)

        self.live_badge.setText(f"{status_label_from_score(current_score)} · {current_score:.0f}%")
        self.live_badge.setStyleSheet(
            f"""
            QLabel {{
                color: white;
                background-color: {score_color(current_score)};
                border-radius: 18px;
                padding: 12px 18px;
                font-weight: 700;
            }}
            """
        )

        self.current_card.set_metric(
            current_score,
            f"{current_score:.0f}%",
            self._score_subtitle(current_score, recent_delta),
            score_color(current_score),
        )
        self.avg_card.set_metric(
            avg_score,
            f"{avg_score:.0f}%",
            f"本次检测 · {summary['record_count']} 条",
            score_color(avg_score),
        )
        self.focus_card.set_metric(
            summary["focused_rate"] * 100,
            f"{summary['focused_rate'] * 100:.0f}%",
            f"专注时长 {self._format_duration(focused_seconds)}",
            COLORS["warning"],
        )

        phone_detected = bool(live_metrics.get("phone_detected"))
        phone_conf = float(live_metrics.get("phone_confidence") or 0.0)
        phone_source = live_metrics.get("phone_source") or "-"
        self.phone_card.set_metric(
            phone_conf * 100 if phone_detected else 0,
            "有手机" if phone_detected else "安全",
            f"置信度 {phone_conf:.2f} · {phone_source}" if phone_detected else "未检测到手机",
            COLORS["danger"] if phone_detected else COLORS["success"],
        )

        self._render_trend_chart(scores, live_records)
        self._render_status_chart(summary)
        self._render_signal_chart(live_metrics)
        self._render_history_table(live_records, snapshot)
        self._render_insights(snapshot, summary, live_metrics, recent_delta)

    def _set_analysis_visible(self, visible):
        for widget in [self.trend_canvas, self.status_canvas, self.signal_canvas, self.history_table]:
            widget.setVisible(visible)

    def _render_idle_state(self, message):
        self._set_analysis_visible(False)
        self.live_badge.setText(message)
        self.live_badge.setStyleSheet(
            f"""
            QLabel {{
                color: white;
                background-color: #8A99A8;
                border-radius: 18px;
                padding: 12px 18px;
                font-weight: 700;
            }}
            """
        )
        self.current_card.set_empty("开始检测后显示当前分数")
        self.avg_card.set_empty("开始检测后统计本次平均")
        self.focus_card.set_empty("开始检测后统计专注占比")
        self.phone_card.set_empty("开始检测后显示手机识别")
        self.history_table.setRowCount(0)
        self.insight_label.setText("开始检测后，这里会展示本次会话的实时图表、手机/人脸/姿态判断和诊断建议。")

    def _session_summary(self, records, scores):
        scores = [float(score) for score in scores]
        distribution = {"专注": 0, "一般": 0, "不专注": 0}
        for record in records:
            status = record.get("status", status_label_from_score(record["attention_score"]))
            distribution[status] = distribution.get(status, 0) + 1

        if not records:
            for score in scores:
                distribution[status_label_from_score(score)] += 1

        focused_count = distribution.get("专注", 0) + distribution.get("focused", 0)
        latest_score = scores[-1] if scores else 0.0
        return {
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "record_count": len(scores),
            "focused_rate": focused_count / len(scores) if scores else 0.0,
            "latest_status": status_label_from_score(latest_score),
            "latest_score": latest_score,
            "status_distribution": distribution,
        }

    def _current_score(self, snapshot, summary):
        scores = snapshot.get("attention_scores") or []
        if scores:
            return float(snapshot.get("current_score", scores[-1]))
        return float(summary.get("latest_score", 0.0))

    def _snapshot_runtime_metrics(self, snapshot):
        latest_info = snapshot.get("latest_runtime_info", {})
        explanation = latest_info.get("score_explanation", {})
        metrics = dict(explanation.get("metrics", {}))
        if "calibration" not in metrics:
            calibration = latest_info.get("calibration") or {}
            metrics["calibration"] = calibration.get("message", "-")
        return metrics

    def _recent_delta(self, scores):
        if len(scores) < 4:
            return 0.0
        previous = scores[-4:-1]
        return float(scores[-1] - (sum(previous) / len(previous)))

    def _score_subtitle(self, score, recent_delta):
        direction = "上升" if recent_delta > 3 else "下降" if recent_delta < -3 else "稳定"
        return f"{status_label_from_score(score)} · 近段时间{direction} {recent_delta:+.1f}"

    def _render_trend_chart(self, scores, live_records=None):
        self.trend_canvas.clear()
        live_records = live_records or []

        if live_records:
            records = live_records[-80:]
            scores = [float(record["attention_score"]) for record in records]
            labels = [compact_time_label(record.get("timestamp")) for record in records]
        else:
            scores = [float(score) for score in scores[-80:]]
            labels = [str(index + 1) for index in range(len(scores))]

        if not scores:
            self.trend_canvas.axes.text(0.5, 0.5, "暂无数据", ha="center", va="center", color=MUTED)
            self.trend_canvas.draw()
            return

        self.trend_canvas.figure.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.24)
        x_values = list(range(len(scores)))
        tick_indexes = sparse_tick_indexes(len(scores), max_ticks=6)
        self.trend_canvas.axes.plot(x_values, scores, color="#176B65", linewidth=2.6)
        self.trend_canvas.axes.fill_between(x_values, scores, color="#2FB47C", alpha=0.12)
        self.trend_canvas.axes.scatter([x_values[-1]], [scores[-1]], color=score_color(scores[-1]), s=55, zorder=3)
        self.trend_canvas.axes.axhline(70, color=COLORS["focused"], linestyle="--", linewidth=1.1, alpha=0.8)
        self.trend_canvas.axes.axhline(40, color=COLORS["moderate"], linestyle="--", linewidth=1.1, alpha=0.8)
        self.trend_canvas.axes.set_ylim(0, 100)
        if len(scores) > 1:
            self.trend_canvas.axes.set_xlim(-0.5, len(scores) - 0.5)
        self.trend_canvas.axes.set_xticks(tick_indexes)
        self.trend_canvas.axes.set_xticklabels(
            [labels[index] for index in tick_indexes],
            rotation=18,
            ha="right",
        )
        self.trend_canvas.axes.set_xlabel("检测时间", color=MUTED)
        self.trend_canvas.axes.set_ylabel("专注度", color=MUTED)
        self.trend_canvas.draw()

    def _render_status_chart(self, summary):
        self.status_canvas.clear()
        self.status_canvas.figure.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04)
        distribution = summary.get("status_distribution", {})
        labels = ["专注", "一般", "不专注"]
        values = [
            distribution.get("专注", distribution.get("focused", 0)),
            distribution.get("一般", distribution.get("moderate", 0)),
            distribution.get("不专注", distribution.get("distracted", 0)),
        ]
        if sum(values) == 0:
            self.status_canvas.axes.text(0.5, 0.5, "暂无状态", ha="center", va="center", color=MUTED)
            self.status_canvas.draw()
            return

        colors = [COLORS["focused"], COLORS["moderate"], COLORS["distracted"]]
        self.status_canvas.axes.pie(
            values,
            startangle=90,
            colors=colors,
            wedgeprops={"width": 0.45, "edgecolor": SURFACE},
        )
        total = max(sum(values), 1)
        legend_text = "  ".join(f"{label}{value / total * 100:.0f}%" for label, value in zip(labels, values))
        self.status_canvas.axes.text(
            0.5,
            0.02,
            legend_text,
            transform=self.status_canvas.axes.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color=MUTED,
        )
        self.status_canvas.axes.set_title("状态占比", fontsize=11, color=INK)
        self.status_canvas.draw()

    def _render_signal_chart(self, metrics):
        self.signal_canvas.clear()
        self.signal_canvas.figure.subplots_adjust(left=0.24, right=0.96, top=0.86, bottom=0.14)
        if not metrics:
            self.signal_canvas.axes.text(0.5, 0.5, "等待实时信号", ha="center", va="center", color=MUTED)
            self.signal_canvas.draw()
            return

        face_value = 100 if metrics.get("face_detected") else 0
        eye_score = float(metrics.get("eye_open_score") or 0.0)
        eye_value = max(0.0, min(100.0, eye_score * 100.0))
        if metrics.get("eyes_closed"):
            eye_value = min(eye_value, 25.0)
        perclos = float(metrics.get("perclos") or 0.0)
        if perclos:
            eye_value = min(eye_value, max(0.0, 100.0 - perclos * 140.0))
        mouth_open_strength = float(metrics.get("mouth_open_strength") or metrics.get("mouth_open_ratio") or 0.0)
        mouth_value = max(0.0, min(100.0, 100.0 - mouth_open_strength * 100.0))
        if metrics.get("yawn_detected"):
            mouth_value = min(mouth_value, 20.0)
        posture_pressure = max(
            float(metrics.get("body_slouch") or 0.0),
            float(metrics.get("body_lean") or 0.0),
            float(metrics.get("head_pose_pressure") or 0.0),
        )
        posture_value = max(0.0, min(100.0, 100.0 - posture_pressure * 100.0))
        phone_conf = float(metrics.get("phone_confidence") or 0.0)
        phone_value = max(0.0, 100.0 - phone_conf * 100.0) if metrics.get("phone_detected") else 100.0
        movement = float(metrics.get("movement_norm") or 0.0)
        stability_value = max(0.0, min(100.0, 100.0 - movement * 450.0))
        final_score = float(metrics.get("final_score") or metrics.get("smoothed_score") or 0.0)

        labels = ["人脸", "眼睛", "嘴部", "坐姿", "无手机", "稳定", "当前分"]
        values = [face_value, eye_value, mouth_value, posture_value, phone_value, stability_value, final_score]
        colors = [COLORS["primary"], "#3A7CA5", "#C97C5D", "#8E7CC3", COLORS["success"], COLORS["warning"], "#176B65"]
        self.signal_canvas.axes.barh(labels, values, color=colors, alpha=0.88)
        self.signal_canvas.axes.set_xlim(0, 100)
        self.signal_canvas.axes.set_title("实时信号", fontsize=10, color=INK)
        self.signal_canvas.draw()

    def _render_history_table(self, live_records, snapshot):
        records = list(reversed(live_records[-10:]))
        self.history_table.setRowCount(len(records))
        for row_index, record in enumerate(records):
            metrics = record.get("metrics", {})
            runtime = metrics.get("runtime", {})
            reasons = metrics.get("reasons") or []
            phone_text = "是" if runtime.get("phone_detected") else "否" if runtime else "-"
            if runtime.get("phone_detected") and runtime.get("phone_confidence") is not None:
                phone_text = f"是 {runtime.get('phone_confidence'):.2f}"
            values = [
                record["timestamp"],
                f"{record['attention_score']:.1f}%",
                record["status"],
                runtime.get("pose", "-"),
                "是" if runtime.get("face_detected") else "否" if runtime else "-",
                phone_text,
                reasons[0] if reasons else "-",
            ]

            for column_index, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter if column_index != 6 else Qt.AlignVCenter | Qt.AlignLeft)
                if column_index == 1:
                    item.setForeground(QColor(score_color(record["attention_score"])))
                    item.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
                if column_index == 5 and "是" in str(value):
                    item.setForeground(QColor(COLORS["danger"]))
                self.history_table.setItem(row_index, column_index, item)
        self.history_table.resizeRowsToContents()

    def _render_insights(self, snapshot, summary, metrics, recent_delta):
        latest_info = snapshot.get("latest_runtime_info", {})
        reasons = latest_info.get("score_explanation", {}).get("reasons", [])
        current_score = self._current_score(snapshot, summary)

        lines = []
        if current_score >= 70:
            lines.append("当前整体处于专注区间，可以把这段姿态作为较稳定的个人基线。")
        elif current_score >= 40:
            lines.append("当前处于一般区间，建议结合右侧实时信号看是姿态、手机还是人脸稳定性导致。")
        else:
            lines.append("当前处于不专注区间，优先检查手机、人脸丢失和低头/偏头三个原因。")

        if metrics.get("phone_detected"):
            lines.append("画面中检测到手机，系统会快速降低专注度。")
        if metrics and not metrics.get("face_detected"):
            lines.append("人脸未稳定检测到，建议提高光线、让脸保持在画面中间。")
        pose = metrics.get("pose")
        if pose and pose not in {"正视屏幕", "forward", "-"}:
            lines.append(f"当前姿态为 {pose}，如果你在看屏幕但被判偏头，可以重新做个人校准。")
        if recent_delta < -12:
            lines.append("最近分数下降较快，可能发生了手机进入画面、转头或人脸短暂丢失。")
        elif recent_delta > 12:
            lines.append("最近分数恢复较快，说明实时响应已经跟上当前画面变化。")
        if reasons:
            lines.append(f"最新判分原因: {reasons[0]}")

        self.insight_label.setText("\n".join(f"{index + 1}. {line}" for index, line in enumerate(lines[:6])))

    def _format_duration(self, seconds):
        seconds = int(seconds)
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {sec}s"
        return f"{sec}s"


class TeacherAnalyticsWindow(QWidget):
    def __init__(self, user_info, db, default_class):
        super().__init__()
        self.user_info = user_info
        self.db = db
        self.default_class = default_class

        self.setWindowTitle(f"{APP_NAME} - 教师分析中心")
        self.resize(1180, 820)
        self.setStyleSheet(f"background-color: {SOFT};")

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_view)
        self.refresh_timer.start(5000)

        self._build_ui()
        self.refresh_view()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header = QFrame()
        header.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #183B56, stop:1 #3A7CA5);
                border-radius: 22px;
            }
            """
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 18, 22, 18)
        title = QLabel("教师分析中心")
        title.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        title.setStyleSheet("color: white; background: transparent;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.class_combo = QComboBox()
        self.class_combo.addItems(self.db.get_class_options())
        if self.default_class:
            index = self.class_combo.findText(self.default_class)
            if index >= 0:
                self.class_combo.setCurrentIndex(index)
        self.class_combo.currentIndexChanged.connect(self.refresh_view)
        self.class_combo.setStyleSheet(
            """
            QComboBox {
                background: white;
                border-radius: 12px;
                padding: 8px 14px;
                min-width: 160px;
            }
            """
        )
        header_layout.addWidget(self.class_combo)
        layout.addWidget(header)

        card_layout = QHBoxLayout()
        card_layout.setSpacing(12)
        self.avg_card = SummaryCard("班级平均专注度", COLORS["success"])
        self.max_card = SummaryCard("最高专注度", COLORS["primary"])
        self.min_card = SummaryCard("最低专注度", COLORS["danger"])
        self.focus_card = SummaryCard("专注占比", COLORS["warning"])
        self.records_card = SummaryCard("有效记录数", "#546A7B")
        for card in [self.avg_card, self.max_card, self.min_card, self.focus_card, self.records_card]:
            card_layout.addWidget(card)
        layout.addLayout(card_layout)

        chart_layout = QHBoxLayout()
        chart_layout.setSpacing(16)

        trend_panel = self._create_panel("班级趋势")
        self.trend_canvas = ChartCanvas(width=6.5, height=3.2)
        trend_panel.layout().addWidget(self.trend_canvas)
        chart_layout.addWidget(trend_panel, stretch=3)

        top_panel = self._create_panel("学生平均分")
        self.student_canvas = ChartCanvas(width=4.4, height=3.2)
        top_panel.layout().addWidget(self.student_canvas)
        chart_layout.addWidget(top_panel, stretch=2)

        layout.addLayout(chart_layout)

        table_panel = self._create_panel("学生明细")
        table_header = QHBoxLayout()
        table_header.addStretch()
        self.detail_btn = QPushButton("查看所选学生详情")
        self.detail_btn.clicked.connect(self.open_selected_student)
        self.detail_btn.setStyleSheet(self._button_style(COLORS["primary"], "#2980B9"))
        table_header.addWidget(self.detail_btn)
        table_panel.layout().addLayout(table_header)

        self.students_table = QTableWidget()
        self.students_table.setColumnCount(8)
        self.students_table.setHorizontalHeaderLabels(
            ["ID", "姓名", "班级", "平均分", "专注占比", "最新状态", "记录数", "最近时间"]
        )
        style_table(self.students_table)
        self.students_table.itemDoubleClicked.connect(lambda _: self.open_selected_student())
        table_panel.layout().addWidget(self.students_table)

        layout.addWidget(table_panel, stretch=1)

    def _create_panel(self, title):
        panel = QFrame()
        panel.setStyleSheet(panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        title_label.setStyleSheet(f"color: {INK};")
        layout.addWidget(title_label)
        return panel

    def _button_style(self, background, hover):
        return f"""
            QPushButton {{
                background-color: {background};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 9px 16px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """

    def refresh_view(self):
        class_name = self.class_combo.currentText()
        summary = self.db.get_class_summary(class_name)
        trend = self.db.get_class_trend(class_name, limit=120)
        student_rows = self.db.get_class_student_summaries(class_name)

        self.avg_card.set_value(f"{summary['avg_score']:.1f}%", f"{summary['student_count']} 名学生")
        self.max_card.set_value(f"{summary['max_score']:.1f}%", "班级最高")
        self.min_card.set_value(f"{summary['min_score']:.1f}%", "班级最低")
        self.focus_card.set_value(f"{summary['focused_rate'] * 100:.0f}%", "专注记录占比")
        self.records_card.set_value(str(summary["record_count"]), "真实采集记录")

        self._render_class_trend(trend)
        self._render_student_bars(student_rows)
        self._render_student_table(student_rows)

    def _render_class_trend(self, trend):
        self.trend_canvas.clear()
        scores = [row["attention_score"] for row in trend]
        if not scores:
            self.trend_canvas.axes.text(0.5, 0.5, "暂无数据", ha="center", va="center", color=MUTED)
            self.trend_canvas.draw()
            return

        labels = [compact_time_label(row.get("timestamp")) for row in trend]
        x_values = list(range(len(scores)))
        tick_indexes = sparse_tick_indexes(len(scores), max_ticks=6)
        self.trend_canvas.figure.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.24)
        self.trend_canvas.axes.plot(x_values, scores, color="#3A7CA5", linewidth=2.4)
        self.trend_canvas.axes.fill_between(x_values, scores, color="#3A7CA5", alpha=0.1)
        self.trend_canvas.axes.axhline(70, color=COLORS["focused"], linestyle="--", linewidth=1.0, alpha=0.7)
        self.trend_canvas.axes.axhline(40, color=COLORS["moderate"], linestyle="--", linewidth=1.0, alpha=0.7)
        self.trend_canvas.axes.set_ylim(0, 100)
        if len(scores) > 1:
            self.trend_canvas.axes.set_xlim(-0.5, len(scores) - 0.5)
        self.trend_canvas.axes.set_xticks(tick_indexes)
        self.trend_canvas.axes.set_xticklabels(
            [labels[index] for index in tick_indexes],
            rotation=18,
            ha="right",
        )
        self.trend_canvas.axes.set_xlabel("检测时间", color=MUTED)
        self.trend_canvas.axes.set_ylabel("专注度", color=MUTED)
        self.trend_canvas.draw()

    def _render_student_bars(self, student_rows):
        self.student_canvas.clear()
        top_rows = student_rows[:8]
        if not top_rows:
            self.student_canvas.axes.text(0.5, 0.5, "暂无数据", ha="center", va="center", color=MUTED)
            self.student_canvas.draw()
            return

        labels = [row["name"][:6] for row in top_rows]
        values = [row["avg_score"] for row in top_rows]
        colors = [score_color(value) for value in values]
        self.student_canvas.axes.bar(labels, values, color=colors, alpha=0.88)
        self.student_canvas.axes.set_ylim(0, 100)
        self.student_canvas.axes.tick_params(axis="x", rotation=20)
        self.student_canvas.draw()

    def _render_student_table(self, student_rows):
        self.students_table.setRowCount(len(student_rows))
        for row_index, row in enumerate(student_rows):
            values = [
                row["id"],
                row["name"],
                row["class_name"],
                f"{row['avg_score']:.1f}%",
                f"{row['focused_rate'] * 100:.0f}%",
                row["latest_status"],
                row["record_count"],
                row["latest_timestamp"] or "-",
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                if column_index == 3:
                    item.setForeground(QColor(score_color(row["avg_score"])))
                    item.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
                self.students_table.setItem(row_index, column_index, item)
        self.students_table.resizeRowsToContents()

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


class StudentDetailDialog(QDialog):
    def __init__(self, student_info, db, parent=None):
        super().__init__(parent)
        self.student_info = student_info
        self.db = db

        self.setWindowTitle(f"{APP_NAME} - 学生详情 - {student_info.get('name') or student_info['username']}")
        self.resize(980, 760)
        self.setStyleSheet(f"background-color: {SOFT};")

        self._build_ui()
        self.refresh_view()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        header = QFrame()
        header.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #102A43, stop:1 #176B65);
                border-radius: 22px;
            }
            """
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(22, 18, 22, 18)
        title = QLabel(self.student_info.get("name") or self.student_info["username"])
        title.setFont(QFont("Microsoft YaHei", 24, QFont.Bold))
        title.setStyleSheet("color: white; background: transparent;")
        header_layout.addWidget(title)
        meta = QLabel(
            f"用户名: {self.student_info['username']}    班级: {self.student_info.get('class_name') or '-'}    邮箱: {self.student_info.get('email') or '-'}"
        )
        meta.setFont(QFont("Microsoft YaHei", 10))
        meta.setStyleSheet("color: #DDF7EE; background: transparent;")
        meta.setWordWrap(True)
        header_layout.addWidget(meta)
        layout.addWidget(header)

        card_layout = QHBoxLayout()
        card_layout.setSpacing(12)
        self.avg_card = SummaryCard("平均专注度", COLORS["success"])
        self.max_card = SummaryCard("最高专注度", COLORS["primary"])
        self.min_card = SummaryCard("最低专注度", COLORS["danger"])
        self.focus_card = SummaryCard("专注占比", COLORS["warning"])
        self.records_card = SummaryCard("记录数", "#546A7B")
        for card in [self.avg_card, self.max_card, self.min_card, self.focus_card, self.records_card]:
            card_layout.addWidget(card)
        layout.addLayout(card_layout)

        chart_layout = QHBoxLayout()
        chart_layout.setSpacing(16)

        trend_panel = self._create_panel("学生最近趋势")
        self.trend_canvas = ChartCanvas(width=6.3, height=3.1)
        trend_panel.layout().addWidget(self.trend_canvas)
        chart_layout.addWidget(trend_panel, stretch=3)

        status_panel = self._create_panel("状态构成")
        self.status_canvas = ChartCanvas(width=3.5, height=3.1)
        status_panel.layout().addWidget(self.status_canvas)
        self.detail_insight_label = QLabel("暂无诊断建议")
        self.detail_insight_label.setWordWrap(True)
        self.detail_insight_label.setStyleSheet(f"color: {MUTED};")
        self.detail_insight_label.setFont(QFont("Microsoft YaHei", 9))
        status_panel.layout().addWidget(self.detail_insight_label)
        chart_layout.addWidget(status_panel, stretch=2)

        layout.addLayout(chart_layout)

        table_panel = self._create_panel("最近记录")
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels(
            ["时间", "专注度", "状态", "姿态", "人脸", "手机", "模型场景"]
        )
        style_table(self.history_table)
        table_panel.layout().addWidget(self.history_table)
        layout.addWidget(table_panel, stretch=1)

    def _create_panel(self, title):
        panel = QFrame()
        panel.setStyleSheet(panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
        title_label = QLabel(title)
        title_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        title_label.setStyleSheet(f"color: {INK};")
        layout.addWidget(title_label)
        return panel

    def refresh_view(self):
        summary = self.db.get_student_summary(self.student_info["id"], limit=300)
        trend = self.db.get_recent_student_trend(self.student_info["id"], limit=80)

        self.avg_card.set_value(f"{summary['avg_score']:.1f}%", f"最新状态 {summary['latest_status']}")
        self.max_card.set_value(f"{summary['max_score']:.1f}%", "历史最高")
        self.min_card.set_value(f"{summary['min_score']:.1f}%", "历史最低")
        self.focus_card.set_value(f"{summary['focused_rate'] * 100:.0f}%", "专注记录占比")
        self.records_card.set_value(str(summary["record_count"]), summary["latest_timestamp"] or "暂无最近时间")

        self._render_trend(trend)
        self._render_status(summary)
        self._render_history(trend)
        self._render_detail_insight(summary, trend)

    def _render_trend(self, trend):
        self.trend_canvas.clear()
        scores = [row["attention_score"] for row in trend]
        if not scores:
            self.trend_canvas.axes.text(0.5, 0.5, "暂无数据", ha="center", va="center", color=MUTED)
            self.trend_canvas.draw()
            return

        labels = [compact_time_label(row.get("timestamp")) for row in trend]
        x_values = list(range(len(scores)))
        tick_indexes = sparse_tick_indexes(len(scores), max_ticks=6)
        self.trend_canvas.figure.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.24)
        self.trend_canvas.axes.plot(x_values, scores, color="#176B65", linewidth=2.3)
        self.trend_canvas.axes.fill_between(x_values, scores, color="#2FB47C", alpha=0.12)
        self.trend_canvas.axes.axhline(70, color=COLORS["focused"], linestyle="--", linewidth=1.0, alpha=0.7)
        self.trend_canvas.axes.axhline(40, color=COLORS["moderate"], linestyle="--", linewidth=1.0, alpha=0.7)
        self.trend_canvas.axes.set_ylim(0, 100)
        if len(scores) > 1:
            self.trend_canvas.axes.set_xlim(-0.5, len(scores) - 0.5)
        self.trend_canvas.axes.set_xticks(tick_indexes)
        self.trend_canvas.axes.set_xticklabels(
            [labels[index] for index in tick_indexes],
            rotation=18,
            ha="right",
        )
        self.trend_canvas.axes.set_xlabel("检测时间", color=MUTED)
        self.trend_canvas.axes.set_ylabel("专注度", color=MUTED)
        self.trend_canvas.draw()

    def _render_status(self, summary):
        self.status_canvas.clear()
        distribution = summary.get("status_distribution", {})
        labels = ["专注", "一般", "不专注"]
        values = [
            distribution.get("专注", distribution.get("focused", 0)),
            distribution.get("一般", distribution.get("moderate", 0)),
            distribution.get("不专注", distribution.get("distracted", 0)),
        ]
        if sum(values) == 0:
            self.status_canvas.axes.text(0.5, 0.5, "暂无数据", ha="center", va="center", color=MUTED)
            self.status_canvas.draw()
            return

        wedges, _texts = self.status_canvas.axes.pie(
            values,
            startangle=90,
            colors=[COLORS["focused"], COLORS["moderate"], COLORS["distracted"]],
            wedgeprops={"width": 0.45, "edgecolor": SURFACE},
        )
        self.status_canvas.axes.legend(
            wedges,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        self.status_canvas.draw()

    def _render_history(self, trend):
        records = list(reversed(trend[-30:]))
        self.history_table.setRowCount(len(records))
        for row_index, record in enumerate(records):
            metrics = record.get("metrics", {})
            runtime = metrics.get("runtime", {})
            backend = metrics.get("backend", {})
            phone_text = "是" if runtime.get("phone_detected") else "否" if runtime else "-"
            if runtime.get("phone_detected") and runtime.get("phone_confidence") is not None:
                phone_text = f"是 {runtime.get('phone_confidence'):.2f}"
            values = [
                record["timestamp"],
                f"{record['attention_score']:.1f}%",
                record["status"],
                runtime.get("pose", "-"),
                "是" if runtime.get("face_detected") else "否" if runtime else "-",
                phone_text,
                backend.get("scene") or "-",
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                if column_index == 1:
                    item.setForeground(QColor(score_color(record["attention_score"])))
                    item.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
                if column_index == 5 and "是" in str(value):
                    item.setForeground(QColor(COLORS["danger"]))
                self.history_table.setItem(row_index, column_index, item)
        self.history_table.resizeRowsToContents()

    def _render_detail_insight(self, summary, trend):
        if not trend:
            self.detail_insight_label.setText("暂无记录，学生端开始检测后会自动生成趋势和明细。")
            return

        latest = trend[-1]
        lines = [
            f"最近状态: {latest['status']}，分数 {latest['attention_score']:.1f}%。",
            f"专注占比: {summary['focused_rate'] * 100:.0f}%，记录数 {summary['record_count']}。",
        ]
        metrics = latest.get("metrics", {}).get("runtime", {})
        if metrics.get("phone_detected"):
            lines.append("最近一次检测到手机，需要重点关注。")
        if metrics and not metrics.get("face_detected"):
            lines.append("最近一次人脸不稳定，可能影响评分准确性。")
        self.detail_insight_label.setText("\n".join(lines))
