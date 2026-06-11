"""Import the PyQt version that matches this Krita build."""

import pykrita

if pykrita.qt_major_version() == 6:
    from PyQt6.QtCore import QBuffer, QIODevice, Qt, QTimer
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import (
        QCheckBox,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
else:
    from PyQt5.QtCore import QBuffer, QIODevice, Qt, QTimer
    from PyQt5.QtGui import QImage
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
