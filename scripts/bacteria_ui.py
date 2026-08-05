from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QImage,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bacteria_assistant.config import MODEL_PATH
from bacteria_assistant.features import read_image
from bacteria_assistant.inference import predict_bacteria_image

# ── Win95 / Win98 Classic Light Palette ─────────────────────────────────────
W_DESKTOP = "#008080"
W_BG = "#D4D0C8"
W_PANEL = "#D4D0C8"
W_WHITE = "#FFFFFF"
W_TITLE_1 = "#000080"
W_TITLE_2 = "#1084D0"
W_TITLE_TEXT = "#FFFFFF"
W_TEXT = "#000000"
W_TEXT_GRAY = "#444444"
W_TEXT_DISABLED = "#808080"
W_SHADOW = "#808080"
W_HIGHLIGHT = "#FFFFFF"
W_MIDLIGHT = "#E8E4DC"
W_INSET_BG = "#FFFFFF"
W_INSET_BORDER = "#808080"
W_STATUS_BG = "#D4D0C8"
W_GREEN = "#008000"
W_RED = "#CC0000"
W_ORANGE = "#CC6600"
W_BLUE_LINK = "#000080"
W_BTN_FACE = "#D4D0C8"
W_MENU_BAR = "#D4D0C8"

GLOBAL_QSS = f"""
* {{
    font-family: 'MS Sans Serif', 'Tahoma', 'Segoe UI', sans-serif;
    font-size: 11px;
    color: {W_TEXT};
}}

QMainWindow, QWidget#root {{
    background-color: {W_DESKTOP};
}}

QWidget#window_frame {{
    background-color: {W_BG};
    border: 2px solid;
    border-top-color:    {W_HIGHLIGHT};
    border-left-color:   {W_HIGHLIGHT};
    border-bottom-color: {W_SHADOW};
    border-right-color:  {W_SHADOW};
}}

QPushButton {{
    background-color: {W_BTN_FACE};
    color: {W_TEXT};
    border: 2px solid;
    border-top-color:    {W_HIGHLIGHT};
    border-left-color:   {W_HIGHLIGHT};
    border-bottom-color: {W_SHADOW};
    border-right-color:  {W_SHADOW};
    padding: 3px 12px 4px 12px;
    min-height: 23px;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: #E0DCD4;
}}
QPushButton:pressed {{
    border-top-color:    {W_SHADOW};
    border-left-color:   {W_SHADOW};
    border-bottom-color: {W_HIGHLIGHT};
    border-right-color:  {W_HIGHLIGHT};
    padding: 4px 11px 3px 13px;
    background-color: {W_BG};
}}
QPushButton:disabled {{
    color: {W_TEXT_DISABLED};
    background-color: {W_BTN_FACE};
}}
QPushButton#titleBtn {{
    background-color: {W_BTN_FACE};
    min-width: 16px; max-width: 16px;
    min-height: 14px; max-height: 14px;
    padding: 0 2px;
    font-size: 9px; font-weight: bold;
    border: 2px solid;
    border-top-color:    {W_HIGHLIGHT};
    border-left-color:   {W_HIGHLIGHT};
    border-bottom-color: {W_SHADOW};
    border-right-color:  {W_SHADOW};
}}
QPushButton#titleBtn:pressed {{
    border-top-color:    {W_SHADOW};
    border-left-color:   {W_SHADOW};
    border-bottom-color: {W_HIGHLIGHT};
    border-right-color:  {W_HIGHLIGHT};
}}

QTextEdit {{
    background-color: {W_INSET_BG};
    color: {W_TEXT};
    border: 2px solid;
    border-top-color:    {W_SHADOW};
    border-left-color:   {W_SHADOW};
    border-bottom-color: {W_HIGHLIGHT};
    border-right-color:  {W_HIGHLIGHT};
    font-family: 'Courier New', monospace;
    font-size: 11px;
    padding: 4px;
    selection-background-color: {W_TITLE_1};
    selection-color: white;
}}

QScrollBar:vertical {{
    background: {W_BG};
    width: 16px;
    border: 1px solid {W_SHADOW};
}}
QScrollBar::handle:vertical {{
    background: {W_BTN_FACE};
    border: 2px solid;
    border-top-color:    {W_HIGHLIGHT};
    border-left-color:   {W_HIGHLIGHT};
    border-bottom-color: {W_SHADOW};
    border-right-color:  {W_SHADOW};
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background: {W_BTN_FACE};
    border: 2px solid;
    border-top-color:    {W_HIGHLIGHT};
    border-left-color:   {W_HIGHLIGHT};
    border-bottom-color: {W_SHADOW};
    border-right-color:  {W_SHADOW};
    height: 16px;
    subcontrol-origin: margin;
}}

QStatusBar {{
    background-color: {W_STATUS_BG};
    color: {W_TEXT};
    border-top: 1px solid {W_SHADOW};
    font-size: 11px;
}}
QStatusBar::item {{
    border: 2px solid;
    border-top-color:    {W_SHADOW};
    border-left-color:   {W_SHADOW};
    border-bottom-color: {W_HIGHLIGHT};
    border-right-color:  {W_HIGHLIGHT};
    padding: 0 4px;
}}

QLabel {{ background: transparent; color: {W_TEXT}; }}
"""


def raised_frame(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setStyleSheet(f"""
        QFrame {{
            background-color: {W_BG};
            border: 2px solid;
            border-top-color:    {W_HIGHLIGHT};
            border-left-color:   {W_HIGHLIGHT};
            border-bottom-color: {W_SHADOW};
            border-right-color:  {W_SHADOW};
        }}
    """)
    return f


def sunken_frame(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setStyleSheet(f"""
        QFrame {{
            background-color: {W_INSET_BG};
            border: 2px solid;
            border-top-color:    {W_SHADOW};
            border-left-color:   {W_SHADOW};
            border-bottom-color: {W_HIGHLIGHT};
            border-right-color:  {W_HIGHLIGHT};
        }}
    """)
    return f


def etched_separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setFixedHeight(4)
    sep.setStyleSheet(f"""
        QFrame {{
            border: none;
            border-top: 1px solid {W_SHADOW};
            border-bottom: 1px solid {W_HIGHLIGHT};
            background: transparent;
        }}
    """)
    return sep


class Win95TitleBar(QWidget):
    def __init__(self, title: str, icon: str = "🔬", parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._dragging = False
        self._drag_pos = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 2, 2, 2)
        layout.setSpacing(2)

        icon_lbl = QLabel(icon)
        icon_lbl.setFixedSize(16, 16)
        icon_lbl.setStyleSheet("font-size: 13px; border: none; background: transparent; color: white;")
        layout.addWidget(icon_lbl)

        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet(
            "color: white; font-weight: bold; font-size: 11px; border: none; background: transparent;"
        )
        layout.addWidget(self._title_lbl)
        layout.addStretch(1)

        for symbol, slot in [("_", None), ("□", None), ("✕", QApplication.quit)]:
            btn = QPushButton(symbol)
            btn.setObjectName("titleBtn")
            if slot:
                btn.clicked.connect(slot)
            layout.addWidget(btn)

    def paintEvent(self, event):
        painter = QPainter(self)
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0, QColor(W_TITLE_1))
        grad.setColorAt(0.55, QColor("#0050AA"))
        grad.setColorAt(1.0, QColor(W_TITLE_2))
        painter.fillRect(self.rect(), QBrush(grad))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPos() - self.window().frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_pos:
            self.window().move(event.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self._drag_pos = None


class GroupBox(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        header.setFixedHeight(16)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 8, 0)
        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color: {W_TEXT}; font-size: 11px; font-weight: bold; background: transparent; border: none;"
        )
        h_layout.addWidget(lbl)
        h_layout.addStretch(1)
        outer.addWidget(header)

        self._inner = QFrame()
        self._inner.setStyleSheet(f"""
            QFrame {{
                background-color: {W_BG};
                border: 2px solid;
                border-top-color:    {W_SHADOW};
                border-left-color:   {W_SHADOW};
                border-bottom-color: {W_HIGHLIGHT};
                border-right-color:  {W_HIGHLIGHT};
            }}
        """)
        self._content_layout = QVBoxLayout(self._inner)
        self._content_layout.setContentsMargins(8, 6, 8, 8)
        self._content_layout.setSpacing(6)
        outer.addWidget(self._inner)

    def content_layout(self):
        return self._content_layout


class BlinkLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._base = text
        self._on = True
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(700)

    def set_text(self, text: str):
        self._base = text
        self._refresh()

    def _tick(self):
        self._on = not self._on
        self._refresh()

    def _refresh(self):
        self.setText(self._base + (" ▌" if self._on else "  "))


def make_result_row(label: str) -> tuple:
    row_w = QWidget()
    row_w.setStyleSheet("background: transparent;")
    hl = QHBoxLayout(row_w)
    hl.setContentsMargins(0, 1, 0, 1)
    hl.setSpacing(6)

    key_lbl = QLabel(label + ":")
    key_lbl.setFixedWidth(110)
    key_lbl.setStyleSheet(
        f"color: {W_TEXT_GRAY}; font-size: 11px; font-weight: bold; background: transparent; border: none;"
    )

    val_frame = sunken_frame()
    val_frame.setFixedHeight(22)
    vfl = QHBoxLayout(val_frame)
    vfl.setContentsMargins(4, 1, 4, 1)

    val_lbl = QLabel("—")
    val_lbl.setStyleSheet(f"color: {W_TEXT}; font-size: 11px; background: transparent; border: none;")
    vfl.addWidget(val_lbl)
    vfl.addStretch(1)

    hl.addWidget(key_lbl)
    hl.addWidget(val_frame, 1)
    return row_w, val_lbl


class BacteriaPredictorUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.resize(1020, 740)

        self.image_path: Path | None = None
        self._detailed_payload: dict | None = None

        root = QWidget()
        root.setObjectName("root")
        root.setStyleSheet(f"background-color: {W_DESKTOP};")
        self.setCentralWidget(root)

        root_vbox = QVBoxLayout(root)
        root_vbox.setContentsMargins(6, 6, 6, 6)
        root_vbox.setSpacing(0)

        wf = QWidget()
        wf.setObjectName("window_frame")
        wf.setStyleSheet(f"""
            QWidget#window_frame {{
                background-color: {W_BG};
                border: 2px solid;
                border-top-color:    {W_HIGHLIGHT};
                border-left-color:   {W_HIGHLIGHT};
                border-bottom-color: {W_SHADOW};
                border-right-color:  {W_SHADOW};
            }}
        """)
        wf_vbox = QVBoxLayout(wf)
        wf_vbox.setContentsMargins(0, 0, 0, 0)
        wf_vbox.setSpacing(0)
        root_vbox.addWidget(wf)

        self.title_bar = Win95TitleBar("Bacteria Predictor — Lab Assistant")
        wf_vbox.addWidget(self.title_bar)

        # Menu bar (decorative)
        menu_bar = QWidget()
        menu_bar.setFixedHeight(22)
        menu_bar.setStyleSheet(f"background-color: {W_BG}; border: none;")
        mb_layout = QHBoxLayout(menu_bar)
        mb_layout.setContentsMargins(4, 0, 4, 0)
        mb_layout.setSpacing(0)
        for item in ["File", "Edit", "View", "Tools", "Help"]:
            mb_btn = QPushButton(item)
            mb_btn.setFixedHeight(20)
            mb_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: none;
                    padding: 0 8px; font-size: 11px; color: {W_TEXT};
                }}
                QPushButton:hover {{
                    background-color: {W_TITLE_1}; color: white;
                }}
            """)
            mb_layout.addWidget(mb_btn)
        mb_layout.addStretch(1)
        wf_vbox.addWidget(menu_bar)
        wf_vbox.addWidget(etched_separator())

        # Toolbar
        toolbar = raised_frame()
        toolbar.setFixedHeight(40)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(6, 4, 6, 4)
        tb_layout.setSpacing(6)

        self.choose_btn = QPushButton("  Open Image...")
        self.predict_btn = QPushButton("  Analyze")
        self.predict_btn.setEnabled(False)
        self.choose_btn.setMinimumWidth(130)
        self.predict_btn.setMinimumWidth(110)

        tb_layout.addWidget(self.choose_btn)

        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setFixedWidth(6)
        div.setStyleSheet(f"""
            QFrame {{
                border: none;
                border-left:  1px solid {W_SHADOW};
                border-right: 1px solid {W_HIGHLIGHT};
                background: transparent;
            }}
        """)
        tb_layout.addWidget(div)
        tb_layout.addWidget(self.predict_btn)
        tb_layout.addStretch(1)

        self.status_light = QLabel("●")
        self.status_light.setStyleSheet(f"color: {W_GREEN}; font-size: 16px; border: none;")
        self.blink_lbl = BlinkLabel("Ready")
        self.blink_lbl.setStyleSheet(f"color: {W_TEXT}; font-size: 11px; border: none;")
        tb_layout.addWidget(self.status_light)
        tb_layout.addWidget(self.blink_lbl)
        wf_vbox.addWidget(toolbar)
        wf_vbox.addWidget(etched_separator())

        # Content
        content = QWidget()
        content.setStyleSheet(f"background-color: {W_BG}; border: none;")
        content_hl = QHBoxLayout(content)
        content_hl.setContentsMargins(8, 8, 8, 6)
        content_hl.setSpacing(8)
        wf_vbox.addWidget(content, 1)

        # Left: image viewer
        img_group = GroupBox("Image Viewer")
        img_cl = img_group.content_layout()

        img_frame = sunken_frame()
        img_frame.setMinimumSize(440, 320)
        img_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_inner = QVBoxLayout(img_frame)
        img_inner.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet(
            f"background-color: {W_INSET_BG}; border: none; color: {W_TEXT_DISABLED}; font-size: 11px;"
        )
        self.image_label.setText("(No image loaded)\n\nClick 'Open Image...' to begin")
        img_inner.addWidget(self.image_label)
        img_cl.addWidget(img_frame, 1)

        self.filename_lbl = QLabel("File:  (none)")
        self.filename_lbl.setStyleSheet(f"color: {W_TEXT_GRAY}; font-size: 10px; border: none;")
        img_cl.addWidget(self.filename_lbl)
        content_hl.addWidget(img_group, 3)

        # Right: results + details
        right_vbox = QVBoxLayout()
        right_vbox.setSpacing(8)
        content_hl.addLayout(right_vbox, 2)

        results_group = GroupBox("Analysis Results")
        res_cl = results_group.content_layout()
        res_cl.setSpacing(4)

        self.result_vals: dict[str, QLabel] = {}
        for key, label in [
            ("NAME", "Bacteria Name"),
            ("TYPE", "Bacteria Type"),
            ("SHAPE", "Dominant Shape"),
            ("COLONIES", "Total Colonies"),
            ("CONFIDENCE", "Confidence"),
        ]:
            row_w, val_lbl = make_result_row(label)
            self.result_vals[key] = val_lbl
            res_cl.addWidget(row_w)

        right_vbox.addWidget(results_group)

        details_group = GroupBox("Detailed Output")
        det_cl = details_group.content_layout()
        det_cl.setSpacing(4)

        det_header = QHBoxLayout()
        det_header.setContentsMargins(0, 0, 0, 0)
        self.toggle_details_btn = QPushButton("Show Details")
        self.toggle_details_btn.setEnabled(False)
        self.toggle_details_btn.setFixedHeight(24)
        det_header.addStretch(1)
        det_header.addWidget(self.toggle_details_btn)
        det_cl.addLayout(det_header)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setVisible(False)
        self.output_text.setMinimumHeight(140)
        det_cl.addWidget(self.output_text)

        self.output_placeholder = QLabel("Run analysis to view raw output.")
        self.output_placeholder.setStyleSheet(f"color: {W_TEXT_DISABLED}; font-size: 11px; border: none; padding: 8px;")
        self.output_placeholder.setAlignment(Qt.AlignCenter)
        det_cl.addWidget(self.output_placeholder)

        right_vbox.addWidget(details_group, 1)

        # Status bar
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("  BPA-22  |  Ready  |  Open an image to begin")
        wf_vbox.addWidget(status)

        self.choose_btn.clicked.connect(self._choose_image)
        self.predict_btn.clicked.connect(self._predict)
        self.toggle_details_btn.clicked.connect(self._toggle_details)

        self.setStyleSheet(GLOBAL_QSS)

    def _preview_pixmap_from_array(self, image_bgr) -> QPixmap:
        """Create a preview pixmap from a cv2/numpy BGR image."""
        rgb = image_bgr[:, :, ::-1].copy()
        h, w, c = rgb.shape
        bytes_per_line = c * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def _choose_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(PROJECT_ROOT),
            "Images (*.png *.PNG *.jpg *.JPG *.jpeg *.JPEG *.bmp *.BMP "
            "*.tif *.TIF *.tiff *.TIFF *.webp *.WEBP);;All Files (*)",
        )
        if not file_name:
            return

        self.image_path = Path(file_name)

        # Validate image using backend reader first (same path used for prediction).
        try:
            backend_image = read_image(str(self.image_path))
        except Exception:
            QMessageBox.warning(self, "Error", "Could not open image file for analysis.")
            self.image_path = None
            self.predict_btn.setEnabled(False)
            return

        pix = QPixmap(str(self.image_path))
        if pix.isNull():
            # Some files can be read by backend but not directly by QPixmap.
            pix = self._preview_pixmap_from_array(backend_image)

        w = max(self.image_label.width() - 4, 100)
        h = max(self.image_label.height() - 4, 100)
        scaled = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if scaled.isNull():
            self.image_label.setText("(Preview unavailable)\nImage is loaded and ready for analysis")
        else:
            self.image_label.setPixmap(scaled)
        self.predict_btn.setEnabled(True)
        self.filename_lbl.setText(f"File:  {self.image_path.name}")
        self.blink_lbl.set_text("Image loaded")
        self.status_light.setStyleSheet(f"color: {W_ORANGE}; font-size: 16px; border: none;")

        for v in self.result_vals.values():
            v.setText("—")
        self._detailed_payload = None
        self.toggle_details_btn.setEnabled(False)
        self.toggle_details_btn.setText("Show Details")
        self.output_text.setVisible(False)
        self.output_placeholder.setVisible(True)
        self.output_text.clear()

    def _toggle_details(self) -> None:
        if self._detailed_payload is None:
            return
        vis = self.output_text.isVisible()
        self.output_text.setVisible(not vis)
        self.output_placeholder.setVisible(vis)
        self.toggle_details_btn.setText("Hide Details" if not vis else "Show Details")

    def _predict(self) -> None:
        if self.image_path is None:
            return

        model_path = PROJECT_ROOT / MODEL_PATH
        if not model_path.exists():
            QMessageBox.warning(
                self,
                "Model not found",
                "Trained model not found.\nPlease run train_model.py first.",
            )
            return

        self.blink_lbl.set_text("Analyzing...")
        self.status_light.setStyleSheet(f"color: {W_ORANGE}; font-size: 16px; border: none;")
        QApplication.processEvents()

        try:
            basic = predict_bacteria_image(self.image_path, model_path=model_path, mode="basic")
            advanced = predict_bacteria_image(self.image_path, model_path=model_path, mode="advanced")
        except Exception as exc:
            QMessageBox.critical(self, "Prediction Failed", str(exc))
            return

        self.result_vals["NAME"].setText(basic.get("predicted_bacteria_name", "—"))
        self.result_vals["TYPE"].setText(basic.get("bacteria_type", "—"))
        self.result_vals["SHAPE"].setText(basic.get("dominant_shape", "—"))
        self.result_vals["COLONIES"].setText(str(basic.get("total_colonies_detected", "—")))

        conf_raw = basic.get("confidence", "—")
        conf_str = str(conf_raw)
        self.result_vals["CONFIDENCE"].setText(conf_str)

        try:
            cf = float(conf_str.strip("%")) / 100 if "%" in conf_str else float(conf_str)
            conf_color = W_GREEN if cf > 0.75 else W_ORANGE if cf > 0.5 else W_RED
        except Exception:
            conf_color = W_TEXT
        self.result_vals["CONFIDENCE"].setStyleSheet(
            f"color: {conf_color}; font-weight: bold; font-size: 11px; background: transparent; border: none;"
        )

        self._detailed_payload = {
            "input_image": str(self.image_path),
            "basic_output": basic,
            "advanced_output": advanced,
        }
        self.output_text.setPlainText(json.dumps(self._detailed_payload, indent=2))
        self.output_text.setVisible(False)
        self.output_placeholder.setVisible(True)
        self.toggle_details_btn.setText("Show Details")
        self.toggle_details_btn.setEnabled(True)

        self.blink_lbl.set_text("Done")
        self.status_light.setStyleSheet(f"color: {W_GREEN}; font-size: 16px; border: none;")
        self.statusBar().showMessage(
            f"  Analysis complete  |  File: {self.image_path.name}"
            f"  |  Colonies detected: {basic.get('total_colonies_detected', '—')}"
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(W_BG))
    pal.setColor(QPalette.WindowText, QColor(W_TEXT))
    pal.setColor(QPalette.Base, QColor(W_INSET_BG))
    pal.setColor(QPalette.AlternateBase, QColor(W_MIDLIGHT))
    pal.setColor(QPalette.ToolTipBase, QColor(W_INSET_BG))
    pal.setColor(QPalette.ToolTipText, QColor(W_TEXT))
    pal.setColor(QPalette.Text, QColor(W_TEXT))
    pal.setColor(QPalette.Button, QColor(W_BTN_FACE))
    pal.setColor(QPalette.ButtonText, QColor(W_TEXT))
    pal.setColor(QPalette.BrightText, QColor(W_RED))
    pal.setColor(QPalette.Link, QColor(W_BLUE_LINK))
    pal.setColor(QPalette.Highlight, QColor(W_TITLE_1))
    pal.setColor(QPalette.HighlightedText, QColor(W_WHITE))
    pal.setColor(QPalette.Light, QColor(W_HIGHLIGHT))
    pal.setColor(QPalette.Midlight, QColor(W_MIDLIGHT))
    pal.setColor(QPalette.Dark, QColor(W_SHADOW))
    pal.setColor(QPalette.Mid, QColor("#A0A09A"))
    pal.setColor(QPalette.Shadow, QColor("#404040"))
    app.setPalette(pal)

    window = BacteriaPredictorUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
