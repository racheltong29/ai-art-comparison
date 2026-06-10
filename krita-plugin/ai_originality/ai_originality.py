"""Krita dock panel — incremental originality feedback while you work."""

from __future__ import annotations

from krita import DockWidget, DockWidgetFactory, DockWidgetFactoryBase, Krita
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .api_client import DEFAULT_API_URL, analyze_png
from .canvas_export import export_active_document_png


class OriginalityDock(DockWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Originality Check")
        self._busy = False
        self._history: list[float] = []

        root = QWidget()
        layout = QVBoxLayout(root)

        self.status_label = QLabel("Start the webapp server, then check your canvas.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.score_label = QLabel("Originality: —")
        self.score_label.setAlignment(Qt.AlignCenter)
        font = self.score_label.font()
        font.setPointSize(18)
        font.setBold(True)
        self.score_label.setFont(font)
        layout.addWidget(self.score_label)

        self.ai_label = QLabel("AI-like: —")
        self.ai_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.ai_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Originality %p%")
        layout.addWidget(self.progress)

        self.trend_label = QLabel("Trend: —")
        self.trend_label.setWordWrap(True)
        layout.addWidget(self.trend_label)

        self.live_check = QCheckBox("Live feedback")
        self.live_check.setToolTip("Re-check the canvas on a timer while you paint.")
        layout.addWidget(self.live_check)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Every"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(15, 300)
        self.interval_spin.setValue(45)
        self.interval_spin.setSuffix(" sec")
        interval_row.addWidget(self.interval_spin)
        layout.addLayout(interval_row)

        self.check_button = QPushButton("Check now")
        self.check_button.clicked.connect(self.check_now)
        layout.addWidget(self.check_button)

        layout.addStretch()
        self.setWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_now)
        self.live_check.toggled.connect(self._on_live_toggled)
        self.interval_spin.valueChanged.connect(self._reset_timer_interval)

    def canvasChanged(self, canvas) -> None:  # noqa: ARG002 - required Krita hook
        pass

    def _on_live_toggled(self, enabled: bool) -> None:
        if enabled:
            self._reset_timer_interval()
            self.timer.start()
            self.check_now()
        else:
            self.timer.stop()

    def _reset_timer_interval(self) -> None:
        if self.live_check.isChecked():
            self.timer.setInterval(self.interval_spin.value() * 1000)

    def check_now(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.check_button.setEnabled(False)
        self.status_label.setText("Analyzing canvas…")

        try:
            png_bytes = export_active_document_png()
            if png_bytes is None:
                self.status_label.setText("Open a document with a paint layer first.")
                return

            result = analyze_png(png_bytes, DEFAULT_API_URL)
            originality = float(result["originality_score"])
            ai_like = float(result["ai_likeness_percent"])

            self._history.append(originality)
            if len(self._history) > 12:
                self._history.pop(0)

            self.score_label.setText(f"Originality: {originality:.0f}%")
            self.ai_label.setText(f"AI-like: {ai_like:.0f}%")
            self.progress.setValue(int(round(originality)))
            self.status_label.setText(f"Last check OK · {result.get('device', 'cpu')}")
            self.trend_label.setText(self._trend_text())
        except RuntimeError as exc:
            self.status_label.setText(str(exc))
        finally:
            self._busy = False
            self.check_button.setEnabled(True)

    def _trend_text(self) -> str:
        if len(self._history) < 2:
            return "Trend: need one more check to show direction."
        delta = self._history[-1] - self._history[-2]
        if delta >= 3:
            return f"Trend: originality up (+{delta:.0f} pts) — nice."
        if delta <= -3:
            return f"Trend: originality down ({delta:.0f} pts) — try rougher texture or bolder choices."
        return "Trend: holding steady."


instance = Krita.instance()
dock_factory = DockWidgetFactory(
    "originalityCheckDock",
    DockWidgetFactoryBase.DockRight,
    OriginalityDock,
)
instance.addDockWidgetFactory(dock_factory)
