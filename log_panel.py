"""Log panel. Qt is imported on first use, not when AVLite scans the plugin."""

from __future__ import annotations

import logging
import queue
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import VisualizerQtWindow

_LOG_READY = False


def _install_log() -> None:
    """Bind Qt classes on first use so community-plugin import does not init Qt."""
    global _LOG_READY, QtCore, QtGui, QtWidgets, LogPanel
    if _LOG_READY:
        return
    from .qt_binding import import_qt

    QtCore, QtGui, QtWidgets = import_qt()
    _LOG_READY = True

    class _QtLogHandler(logging.Handler):
        def __init__(self, sink: "LogPanel"):
            super().__init__()
            self.sink = sink

        def emit(self, record: logging.LogRecord) -> None:
            try:
                if not self.sink.allows(record.name):
                    return
                self.sink.enqueue(self.format(record), record.levelno)
            except Exception:
                self.handleError(record)

    class LogPanel(QtWidgets.QGroupBox):
        """Live log view with the same filters as the Tk visualizer."""

        def __init__(self, root: "VisualizerQtWindow"):
            super().__init__("Log")
            self.setObjectName("logPanel")
            self.root = root
            self._queue: queue.Queue[tuple[str, int]] = queue.Queue()
            self._handler: logging.Handler | None = None
            self._file_handler: logging.Handler | None = None
            self._stdout = sys.stdout
            self._closed = False

            outer = QtWidgets.QVBoxLayout(self)
            outer.setContentsMargins(6, 6, 6, 6)
            outer.setSpacing(4)

            filters = QtWidgets.QHBoxLayout()
            self._checks: dict[str, QtWidgets.QCheckBox] = {}
            for key, label in (
                ("core", "Core"),
                ("plugins", "Plugins"),
                ("perceive", "Perception"),
                ("plan", "Planning"),
                ("control", "Control"),
                ("execute", "Execution"),
                ("vis", "Visualization"),
                ("common", "Common"),
            ):
                box = QtWidgets.QCheckBox(label)
                box.setObjectName(f"logFilter_{key}")
                box.setChecked(self._flag(key))
                box.stateChanged.connect(lambda state, k=key: self._set_flag(k, state))
                filters.addWidget(box)
                self._checks[key] = box
            filters.addStretch(1)
            outer.addLayout(filters)

            levels = QtWidgets.QHBoxLayout()
            self._level = QtWidgets.QButtonGroup(self)
            for value in ("DEBUG", "INFO", "WARN", "STDOUT"):
                btn = QtWidgets.QRadioButton(value)
                btn.setObjectName(f"logLevel_{value}")
                self._level.addButton(btn)
                levels.addWidget(btn)
                if value == str(self.root.setting.log_level.get() or "INFO"):
                    btn.setChecked(True)
            self._level.buttonClicked.connect(self._on_level)
            levels.addStretch(1)
            self._file = QtWidgets.QCheckBox("File")
            self._file.setObjectName("logToFile")
            self._file.setChecked(bool(self.root.setting.log_to_file.get()))
            self._file.stateChanged.connect(self._on_file)
            levels.addWidget(self._file)
            clear_btn = QtWidgets.QPushButton("Clear")
            clear_btn.setObjectName("logClear")
            clear_btn.clicked.connect(self.clear)
            levels.addWidget(clear_btn)
            copy_btn = QtWidgets.QPushButton("Copy")
            copy_btn.setObjectName("logCopy")
            copy_btn.clicked.connect(self.copy_log)
            levels.addWidget(copy_btn)
            self._expand_btn = QtWidgets.QPushButton("▴")
            self._expand_btn.setObjectName("logExpand")
            self._expand_btn.clicked.connect(self.toggle_height)
            levels.addWidget(self._expand_btn)
            outer.addLayout(levels)

            self.view = QtWidgets.QTextEdit()
            self.view.setObjectName("logView")
            self.view.setReadOnly(True)
            self.view.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
            self.view.setFont(QtGui.QFont("monospace", 11))
            self.view.setStyleSheet(
                "QTextEdit { background: #101410; color: #b6f5b6; "
                "selection-background-color: #345c34; }"
            )
            # Always-on vertical bar: AsNeeded appears once the log fills and
            # steals width, which rewraps lines and flickers the pane.
            self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.view.setFixedHeight(140)
            self.view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            outer.addWidget(self.view, 1)
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

            self._handler = _QtLogHandler(self)
            self._handler.setFormatter(
                logging.Formatter("%(levelname).4s %(name)s: %(message)s")
            )
            root_logger = logging.getLogger()
            root_logger.addHandler(self._handler)
            # Stay at INFO through startup. The profile level is applied after
            # the first settings load so those lines stay on screen, as in Tk.
            if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
                root_logger.setLevel(logging.INFO)

            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._flush)
            self._timer.start(int(self.root.setting.p68_log_pull_time))
            logging.getLogger(__name__).info("Log initialized.")

        def _flag(self, key: str) -> bool:
            s = self.root.setting
            return bool(
                {
                    "core": s.p68_show_core_logs,
                    "plugins": s.p68_show_plugins_logs,
                    "perceive": s.p68_show_perceive_logs,
                    "plan": s.p68_show_plan_logs,
                    "control": s.p68_show_control_logs,
                    "execute": s.p68_show_execute_logs,
                    "vis": s.p68_show_vis_logs,
                    "common": s.p68_show_common_logs,
                }[key].get()
            )

        def _set_flag(self, key: str, state: int) -> None:
            s = self.root.setting
            var = {
                "core": s.p68_show_core_logs,
                "plugins": s.p68_show_plugins_logs,
                "perceive": s.p68_show_perceive_logs,
                "plan": s.p68_show_plan_logs,
                "control": s.p68_show_control_logs,
                "execute": s.p68_show_execute_logs,
                "vis": s.p68_show_vis_logs,
                "common": s.p68_show_common_logs,
            }[key]
            var.set(state == QtCore.Qt.Checked)
            logging.getLogger(__name__).info("Log filter updated.")

        def allows(self, name: str) -> bool:
            """Same layer rules as the Tk log view."""
            if self.root.setting.p68_disable_log.get():
                return False
            s = self.root.setting
            layers = (
                ("avlite.c10_perception", s.p68_show_perceive_logs),
                ("avlite.c20_planning", s.p68_show_plan_logs),
                ("avlite.c30_control", s.p68_show_control_logs),
                ("avlite.c40_execution", s.p68_show_execute_logs),
                ("avlite.c50_common", s.p68_show_common_logs),
                ("avlite.plugins.p60_visualizer_tk", s.p68_show_vis_logs),
                ("avlite.plugins.avlite_QT5", s.p68_show_vis_logs),
            )
            for prefix, flag in layers:
                if name == prefix or name.startswith(prefix + "."):
                    if not s.p68_show_core_logs.get() and not name.startswith("avlite.plugins."):
                        return False
                    if name.startswith("avlite.plugins.") and not s.p68_show_plugins_logs.get():
                        return False
                    return bool(flag.get())
            if name.startswith("avlite.plugins.") or ".plugins." in name:
                return bool(s.p68_show_plugins_logs.get())
            return bool(s.p68_show_core_logs.get())

        def enqueue(self, text: str, level: int) -> None:
            self._queue.put((text, level))

        def _color(self, level: int) -> str:
            if level >= logging.ERROR:
                return "#ff6b6b"
            if level >= logging.WARNING:
                return "#ffe66b"
            if level >= logging.INFO:
                return "#b6f5b6"
            return "#9ad1ff"

        def _flush(self) -> None:
            if self._closed:
                return
            lines: list[tuple[str, int]] = []
            while True:
                try:
                    lines.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if not lines or self.root.setting.p68_disable_log.get():
                return
            cursor = self.view.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            for text, level in lines:
                fmt = QtGui.QTextCharFormat()
                fmt.setForeground(QtGui.QColor(self._color(level)))
                cursor.setCharFormat(fmt)
                cursor.insertText(text + "\n")
            self.view.setTextCursor(cursor)
            self.view.ensureCursorVisible()
            max_lines = int(self.root.setting.p68_max_log_lines)
            doc = self.view.document()
            if max_lines > 0 and doc.blockCount() > max_lines:
                extra = doc.blockCount() - max_lines
                cur = self.view.textCursor()
                cur.movePosition(QtGui.QTextCursor.Start)
                cur.movePosition(QtGui.QTextCursor.Down, QtGui.QTextCursor.KeepAnchor, extra)
                cur.removeSelectedText()

        def _on_level(self, button) -> None:
            self._apply_level(button.text())

        def _apply_level(self, level: str) -> None:
            self.root.setting.log_level.set(level)
            logger = logging.getLogger()
            if level == "DEBUG":
                logger.setLevel(logging.DEBUG)
            elif level == "WARN":
                logger.setLevel(logging.WARNING)
            elif level == "STDOUT":
                logger.setLevel(logging.CRITICAL)
                sys.stdout = _StdoutToLog(self)
            else:
                logger.setLevel(logging.INFO)
                sys.stdout = self._stdout
            # WARN so the line stays visible after switching to WARN, matching Tk.
            note = logging.getLogger(__name__)
            if level == "STDOUT":
                note.critical("Log level updated to: %s", level)
            else:
                note.warning("Log level updated to: %s", level)

        def _on_file(self, state: int) -> None:
            self.root.setting.log_to_file.set(state == QtCore.Qt.Checked)
            if self._file_handler is not None:
                logging.getLogger().removeHandler(self._file_handler)
                self._file_handler.close()
                self._file_handler = None
            if not self.root.setting.log_to_file.get():
                logging.getLogger(__name__).info("File logging stopped.")
                return
            log_dir = Path.cwd() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"avlite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname).4s %(name)s: %(message)s")
            )
            logging.getLogger().addHandler(handler)
            self._file_handler = handler
            logging.getLogger(__name__).info("Logging to file: %s", path)

        def scroll_lines(self, steps: int) -> None:
            bar = self.view.verticalScrollBar()
            bar.setValue(bar.value() + int(steps) * max(1, bar.singleStep()))

        def scroll_half_page(self, direction: int) -> None:
            bar = self.view.verticalScrollBar()
            step = max(1, bar.pageStep() // 2)
            bar.setValue(bar.value() + int(direction) * step)

        def scroll_top(self) -> None:
            self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().minimum())

        def scroll_bottom(self) -> None:
            bar = self.view.verticalScrollBar()
            bar.setValue(bar.maximum())

        def toggle_height(self) -> None:
            expanded = not bool(self.root.setting.p68_log_view_expanded.get())
            self.root.setting.p68_log_view_expanded.set(expanded)
            if expanded:
                self.view.setFixedHeight(320)
                self._expand_btn.setText("▾")
            else:
                self.view.setFixedHeight(140)
                self._expand_btn.setText("▴")

        def copy_log(self) -> None:
            QtWidgets.QApplication.clipboard().setText(self.view.toPlainText())
            logging.getLogger(__name__).info("Log copied to clipboard.")

        def clear(self) -> None:
            self.view.clear()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        def shutdown(self) -> None:
            self._closed = True
            self._timer.stop()
            if self._handler is not None:
                logging.getLogger().removeHandler(self._handler)
            if self._file_handler is not None:
                logging.getLogger().removeHandler(self._file_handler)
                self._file_handler.close()
                self._file_handler = None
            sys.stdout = self._stdout

    class _StdoutToLog:
        def __init__(self, panel: LogPanel):
            self.panel = panel

        def write(self, buf: str) -> None:
            for line in str(buf).splitlines():
                if line:
                    self.panel.enqueue(line, logging.INFO)

        def flush(self) -> None:
            pass


def __getattr__(name: str):
    if name == "LogPanel":
        _install_log()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
