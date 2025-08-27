import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QAction, QMessageBox,
    QVBoxLayout, QInputDialog, QLabel, QStatusBar, QSlider
)

from app.model_wrapper import ModelWrapper
from app.utils import rgba_overlay_from_mask, largest_polygon_from_mask, labelme_like_shape


class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image_np: np.ndarray = None  # BGR
        self.image_qpix: QPixmap = None
        self.overlay_qimage: QImage = None  # RGBA overlay
        self.polygons: List[List[Tuple[float, float]]] = []
        self.labels: List[str] = []

        self._scaled_pix: QPixmap = None
        self._scale: float = 1.0
        self._base_fit_scale: float = 1.0
        self._offset_x: int = 0
        self._offset_y: int = 0
        self._user_scaled: bool = False
        self._is_panning: bool = False
        self._last_pan_pos: QPoint = None
        self.overlay_opacity: float = 0.35  # 0..1

    def has_image(self) -> bool:
        return self.image_np is not None

    def set_image(self, image_np_bgr: np.ndarray) -> None:
        self.image_np = image_np_bgr
        h, w = self.image_np.shape[:2]
        # Convert to QImage (BGR -> RGB)
        rgb = self.image_np[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.image_qpix = QPixmap.fromImage(qimg)
        self.overlay_qimage = None
        self.polygons = []
        self.labels = []
        self._rescale_to_fit()
        self.update()

    def set_overlay_mask(self, mask: np.ndarray) -> None:
        if mask is None:
            self.overlay_qimage = None
        else:
            overlay_rgba = rgba_overlay_from_mask(mask, color=(0, 255, 0, 90))
            h, w = overlay_rgba.shape[:2]
            qimg = QImage(overlay_rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
            self.overlay_qimage = qimg.copy()  # copy to own the data
        self.update()

    def set_overlay_opacity(self, opacity: float) -> None:
        self.overlay_opacity = float(max(0.0, min(1.0, opacity)))
        self.update()

    def add_annotation(self, polygon: List[Tuple[float, float]], label: str) -> None:
        if polygon:
            self.polygons.append(polygon)
            self.labels.append(label)
            self.update()

    def clear_last(self) -> None:
        if self.polygons:
            self.polygons.pop()
            self.labels.pop()
            self.update()

    def clear_all(self) -> None:
        self.polygons.clear()
        self.labels.clear()
        self.overlay_qimage = None
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale_to_fit()

    def _rescale_to_fit(self) -> None:
        if self.image_qpix is None:
            return
        iw = self.image_qpix.width()
        ih = self.image_qpix.height()
        ww = max(1, self.width())
        wh = max(1, self.height())
        self._base_fit_scale = min(ww / iw, wh / ih)
        if not self._user_scaled:
            self._scale = self._base_fit_scale
            sw = int(iw * self._scale)
            sh = int(ih * self._scale)
            self._offset_x = (ww - sw) // 2
            self._offset_y = (wh - sh) // 2
        # Always refresh scaled pix with current scale
        sw = int(iw * self._scale)
        sh = int(ih * self._scale)
        self._scaled_pix = self.image_qpix.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _widget_to_image_xy(self, pos: QPoint) -> Tuple[int, int]:
        if self._scaled_pix is None:
            return (0, 0)
        x = int((pos.x() - self._offset_x) / max(self._scale, 1e-6))
        y = int((pos.y() - self._offset_y) / max(self._scale, 1e-6))
        # clamp
        h, w = self.image_np.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return (x, y)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self._last_pan_pos = event.pos()
            return
        if event.button() == Qt.LeftButton:
            self.parent().on_canvas_mouse_press(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if self._scaled_pix is not None:
            p.drawPixmap(self._offset_x, self._offset_y, self._scaled_pix)
            if self.overlay_qimage is not None:
                ow = int(self.overlay_qimage.width() * self._scale)
                oh = int(self.overlay_qimage.height() * self._scale)
                overlay_scaled = self.overlay_qimage.scaled(ow, oh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                p.save()
                p.setOpacity(self.overlay_opacity)
                p.drawImage(self._offset_x, self._offset_y, overlay_scaled)
                p.restore()
            # draw polygons
            pen = QPen(QColor(0, 255, 0), 2)
            p.setPen(pen)
            for poly in self.polygons:
                if len(poly) < 2:
                    continue
                for i in range(len(poly)):
                    x1, y1 = poly[i]
                    x2, y2 = poly[(i + 1) % len(poly)]
                    sx1 = int(x1 * self._scale) + self._offset_x
                    sy1 = int(y1 * self._scale) + self._offset_y
                    sx2 = int(x2 * self._scale) + self._offset_x
                    sy2 = int(y2 * self._scale) + self._offset_y
                    p.drawLine(sx1, sy1, sx2, sy2)
        p.end()

    def mouseMoveEvent(self, event) -> None:
        if self._is_panning and self._last_pan_pos is not None:
            dx = event.pos().x() - self._last_pan_pos.x()
            dy = event.pos().y() - self._last_pan_pos.y()
            self._offset_x += dx
            self._offset_y += dy
            self._last_pan_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._is_panning = False
            self._last_pan_pos = None

    def wheelEvent(self, event) -> None:
        if self.image_qpix is None:
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        steps = angle / 120.0
        factor = 1.15 ** steps
        self._zoom_at(event.pos(), factor)

    def _zoom_at(self, widget_pos: QPoint, factor: float) -> None:
        # Compute image coords under cursor before zoom
        ix, iy = self._widget_to_image_xy(widget_pos)
        old_scale = self._scale
        new_scale = max(0.05, min(20.0, old_scale * factor))
        if abs(new_scale - old_scale) < 1e-6:
            return
        self._scale = new_scale
        self._user_scaled = True
        # Keep cursor anchored: wx = ox' + ix*s'
        wx = widget_pos.x()
        wy = widget_pos.y()
        self._offset_x = int(wx - ix * self._scale)
        self._offset_y = int(wy - iy * self._scale)
        # Recompute scaled pix
        iw = self.image_qpix.width()
        ih = self.image_qpix.height()
        sw = int(iw * self._scale)
        sh = int(ih * self._scale)
        self._scaled_pix = self.image_qpix.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Click-Segment Labeling (SAM2-ready)")
        self.viewer = ImageViewer(self)

        self.model = ModelWrapper(checkpoint_path="/home/tao/sam2.1_hiera_base_plus.pt")

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage(f"Backend: {self.model.get_backend_info()}")
        # Opacity slider
        self.opacity_label = QLabel("Opacity:")
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(self.viewer.overlay_opacity * 100))
        self.opacity_slider.setFixedWidth(160)
        self.opacity_slider.setToolTip("Mask opacity")
        self.opacity_slider.valueChanged.connect(lambda v: self.viewer.set_overlay_opacity(v / 100.0))
        self.status.addPermanentWidget(self.opacity_label)
        self.status.addPermanentWidget(self.opacity_slider)

        self._image_path: str = ""

        self._make_actions()
        self._make_menu()

        self.resize(1200, 800)

    def _make_actions(self):
        self.act_open = QAction("Open Image", self)
        self.act_open.triggered.connect(self.on_open)
        self.act_save = QAction("Save Annotations", self)
        self.act_save.triggered.connect(self.on_save)
        self.act_clear_last = QAction("Clear Last", self)
        self.act_clear_last.triggered.connect(self.on_clear_last)
        self.act_clear_all = QAction("Clear All", self)
        self.act_clear_all.triggered.connect(self.on_clear_all)
        self.act_backend = QAction("Backend Info", self)
        self.act_backend.triggered.connect(self.on_backend_info)

    def _make_menu(self):
        m_file = self.menuBar().addMenu("File")
        m_file.addAction(self.act_open)
        m_file.addAction(self.act_save)
        m_file.addSeparator()
        m_file.addAction(self.act_clear_last)
        m_file.addAction(self.act_clear_all)
        m_file.addSeparator()
        m_file.addAction(self.act_backend)

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", os.getcwd(), "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        self._image_path = path
        try:
            img = Image.open(path).convert("RGB")
            image_np = np.array(img)[:, :, ::-1]  # to BGR
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open image: {e}")
            return
        self.viewer.set_image(image_np)
        self.status.showMessage(f"Opened: {os.path.basename(path)} | Backend: {self.model.get_backend_info()}")

    def on_save(self):
        if not self.viewer.has_image():
            QMessageBox.information(self, "Info", "Open an image first.")
            return
        if not self.viewer.polygons:
            QMessageBox.information(self, "Info", "No annotations to save.")
            return
        default_name = os.path.splitext(os.path.basename(self._image_path or "annotations"))[0] + ".json"
        path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", default_name, "JSON (*.json)")
        if not path:
            return
        h, w = self.viewer.image_np.shape[:2]
        shapes = [labelme_like_shape(label, poly) for label, poly in zip(self.viewer.labels, self.viewer.polygons)]
        data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(self._image_path),
            "imageData": None,
            "imageHeight": int(h),
            "imageWidth": int(w),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save JSON: {e}")
            return
        self.status.showMessage(f"Saved: {os.path.basename(path)}")

    def on_clear_last(self):
        self.viewer.clear_last()

    def on_clear_all(self):
        self.viewer.clear_all()

    def on_backend_info(self):
        msg = f"Using backend: {self.model.get_backend_info()}\n"
        if self.model.get_backend_info() != "SAM2":
            msg += "Install SAM2 and place checkpoint at /home/tao/sam2.1_hiera_base_plus.pt to enable AI segmentation."
        QMessageBox.information(self, "Backend", msg)

    def on_canvas_mouse_press(self, event) -> None:
        if not self.viewer.has_image():
            return
        if event.button() != Qt.LeftButton:
            return
        # Map widget pos to image pixel coordinates
        pos = event.pos()
        x, y = self.viewer._widget_to_image_xy(pos)
        # Run segmentation
        try:
            mask = self.model.segment(self.viewer.image_np, (x, y))
        except Exception as e:
            QMessageBox.critical(self, "Segmentation Error", str(e))
            return
        self.viewer.set_overlay_mask(mask)
        # Prompt for label
        label, ok = QInputDialog.getText(self, "Label", "Enter class label:")
        if not ok or not label:
            return
        polygon = largest_polygon_from_mask(mask)
        if not polygon:
            QMessageBox.information(self, "Info", "No polygon extracted from mask.")
            return
        self.viewer.add_annotation(polygon, label)


def main():
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()