"""Export the active Krita document as PNG bytes."""

from __future__ import annotations

from krita import Krita
from PyQt5.QtCore import QBuffer, QIODevice
from PyQt5.QtGui import QImage


def export_active_document_png(max_edge: int = 512) -> bytes | None:
    instance = Krita.instance()
    document = instance.activeDocument()
    if document is None:
        return None

    node = document.activeNode() or document.topNode()
    if node is None:
        return None

    width = document.width()
    height = document.height()
    if width <= 0 or height <= 0:
        return None

    pixel_data = node.projectionPixelData(0, 0, width, height)
    image = QImage(pixel_data, width, height, QImage.Format_ARGB32).copy()

    longest = max(image.width(), image.height())
    if longest > max_edge:
        if image.width() >= image.height():
            scaled_width = max_edge
            scaled_height = int(height * (max_edge / width))
        else:
            scaled_height = max_edge
            scaled_width = int(width * (max_edge / height))
        image = image.scaled(scaled_width, scaled_height)

    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    if not image.save(buffer, "PNG"):
        return None
    return bytes(buffer.data())
