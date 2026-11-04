"""Settings toolbar that matches the Tk visualizer's SettingShortcutView.

Qt is imported on first use so registering the plugin does not initialize it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from avlite.c60_apps.c65_setting_utils import list_profiles, save_setting
from avlite.c60_apps.c69_settings import AppSettings

from .shortcuts import SHORTCUTS

if TYPE_CHECKING:
    from .main_window import VisualizerQtWindow

log = logging.getLogger(__name__)

_READY = False


def _install() -> None:
    global _READY, QtCore, QtWidgets, ConfigBar
    if _READY:
        return
    from .qt_binding import import_qt

    QtCore, _QtGui, QtWidgets = import_qt()
    _READY = True

    class ConfigBar(QtWidgets.QGroupBox):
        """Profile, view toggles, and settings / plugin launchers."""

        def __init__(self, root: "VisualizerQtWindow"):
            super().__init__("Settings")
            self.setObjectName("configBar")
            self.root = root
            self._syncing = False

            row = QtWidgets.QHBoxLayout(self)
            row.setContentsMargins(6, 4, 6, 4)
            row.setSpacing(6)

            self.chk_shortcuts = self._check("Shortcuts", "chkShortcuts")
            self.chk_dark = self._check("Dark", "chkDark")
            self.chk_global = self._check("Global", "chkGlobal")
            self.chk_frenet = self._check("Local Frenet", "chkLocalFrenet")
            self.chk_local = self._check("Local Global", "chkLocalGlobal")
            for box in (
                self.chk_shortcuts, self.chk_dark, self.chk_global, self.chk_frenet, self.chk_local,
            ):
                row.addWidget(box)
            self.chk_shortcuts.stateChanged.connect(self._on_shortcuts)
            self.chk_dark.stateChanged.connect(self._on_dark)
            self.chk_global.stateChanged.connect(self._on_global)
            self.chk_frenet.stateChanged.connect(self._on_frenet)
            self.chk_local.stateChanged.connect(self._on_local)

            self.status = QtWidgets.QLabel("Click plot")
            self.status.setObjectName("plotStatus")
            self.status.setMinimumWidth(140)
            row.addWidget(self.status)
            row.addStretch(1)

            # Tk packs the right side from the right: gear is rightmost, profile is leftmost of that group.
            self.profile = QtWidgets.QComboBox()
            self.profile.setObjectName("profileCombo")
            self.profile.setMinimumWidth(140)
            self.profile.currentTextChanged.connect(self._on_profile)
            row.addWidget(self.profile)

            self.btn_save = self._button("Save Settings", "btnSaveSettings", self.save_settings)
            self.btn_reset = self._button("Reset Settings", "btnResetSettings", self.reset_settings)
            self.btn_reload = self._button("Reload Stack", "btnReloadStack", self.reload_stack)
            self.btn_plugins = self._button("Plugins", "btnPlugins", self.open_plugins)
            self.btn_settings = self._button("⚙", "btnSettings", self.open_settings)
            for btn in (self.btn_save, self.btn_reset, self.btn_reload, self.btn_plugins, self.btn_settings):
                row.addWidget(btn)

            self.shortcut_frame = QtWidgets.QGroupBox("Shortcuts")
            self.shortcut_frame.setObjectName("shortcutFrame")
            self._build_shortcuts()
            self.shortcut_frame.setVisible(False)

            self.root.setting.perception_status_text.bind(
                lambda: self.status.setText(str(self.root.setting.perception_status_text.get()))
            )
            self.sync_from_settings()

        def _check(self, text: str, name: str) -> QtWidgets.QCheckBox:
            box = QtWidgets.QCheckBox(text)
            box.setObjectName(name)
            return box

        def _button(self, text: str, name: str, slot) -> QtWidgets.QPushButton:
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName(name)
            btn.clicked.connect(slot)
            return btn

        def _build_shortcuts(self) -> None:
            outer = QtWidgets.QVBoxLayout(self.shortcut_frame)
            outer.setContentsMargins(4, 4, 4, 4)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMaximumHeight(150)
            inner = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(inner)
            grid.setContentsMargins(4, 4, 4, 4)
            row = 0
            last = None
            for entry in SHORTCUTS:
                if entry.group != last:
                    heading = QtWidgets.QLabel(entry.group)
                    heading.setStyleSheet("font-weight: bold;")
                    grid.addWidget(heading, row, 0)
                    last = entry.group
                key = QtWidgets.QLabel(entry.default)
                key.setMinimumWidth(90)
                grid.addWidget(key, row, 1)
                grid.addWidget(QtWidgets.QLabel(entry.label), row, 2)
                row += 1
            scroll.setWidget(inner)
            outer.addWidget(scroll)

        def sync_from_settings(self) -> None:
            s = self.root.setting
            self._syncing = True
            try:
                self.chk_shortcuts.setChecked(bool(s.p60_shortcut_mode.get()))
                self.chk_dark.setChecked(bool(s.p60_dark_mode.get()))
                self.chk_global.setChecked(bool(s.p67_global_plan_view.get()))
                self.chk_frenet.setChecked(bool(s.p67_show_local_frenet_view.get()))
                self.chk_local.setChecked(bool(s.p67_show_local_global_view.get()))
                profiles = list(getattr(s, "profile_list", None) or []) or list_profiles(AppSettings) or ["default"]
                current = str(s.c60_selected_profile.get() or "default")
                self.profile.blockSignals(True)
                self.profile.clear()
                self.profile.addItems([str(p) for p in profiles])
                idx = self.profile.findText(current)
                self.profile.setCurrentIndex(idx if idx >= 0 else 0)
                self.profile.blockSignals(False)
                self.status.setText(str(s.perception_status_text.get()))
            finally:
                self._syncing = False

        def _checked(self, state: int) -> bool:
            return state == QtCore.Qt.Checked

        def _on_shortcuts(self, state: int) -> None:
            if self._syncing:
                return
            self.root.setting.p60_shortcut_mode.set(self._checked(state))
            self.root.update_shortcut_mode()

        def _on_dark(self, state: int) -> None:
            if self._syncing:
                return
            self.root.setting.p60_dark_mode.set(self._checked(state))
            self.root._apply_theme()

        def _on_global(self, state: int) -> None:
            if self._syncing:
                return
            self.root.setting.p67_global_plan_view.set(self._checked(state))
            self.root.update_views()

        def _on_frenet(self, state: int) -> None:
            if self._syncing:
                return
            self.root.setting.p67_show_local_frenet_view.set(self._checked(state))
            self.root.update_views()

        def _on_local(self, state: int) -> None:
            if self._syncing:
                return
            self.root.setting.p67_show_local_global_view.set(self._checked(state))
            self.root.update_views()

        def _on_profile(self, name: str) -> None:
            if self._syncing or not name:
                return
            if name == str(self.root.setting.c60_selected_profile.get() or ""):
                return
            log.info("Selected profile: %s", name)
            self.root.setting.c60_selected_profile.set(name)
            self.root.load_settings()
            self.root.reload_stack(reload_code=False)

        def save_settings(self) -> None:
            self.root.setting.push_plugin_schema()
            profile = str(self.root.setting.c60_selected_profile.get() or "default")
            save_setting(self.root.setting.schema, profile=profile)
            save_setting(AppSettings, profile=profile)
            log.info("Saved settings to profile %s", profile)

        def reset_settings(self) -> None:
            self.root.load_settings()
            self.root.reload_stack(reload_code=False)

        def reload_stack(self) -> None:
            self.root.reload_stack(reload_code=True)

        def open_plugins(self) -> None:
            dlg = QtWidgets.QDialog(self.root.win)
            dlg.setWindowTitle("Community Plugins")
            dlg.resize(640, 360)
            layout = QtWidgets.QVBoxLayout(dlg)
            layout.addWidget(QtWidgets.QLabel("Registered community plugins (c62_community_plugins):"))
            table = QtWidgets.QTableWidget(0, 2)
            table.setHorizontalHeaderLabels(["Name", "Path"])
            plugins = dict(AppSettings.c62_community_plugins or {})
            table.setRowCount(len(plugins))
            for row, (name, path) in enumerate(plugins.items()):
                table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name)))
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(path)))
            table.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(table)
            close = QtWidgets.QPushButton("Close")
            close.clicked.connect(dlg.accept)
            layout.addWidget(close)
            dlg.exec_()

        def open_settings(self) -> None:
            dlg = QtWidgets.QDialog(self.root.win)
            dlg.setWindowTitle("Settings")
            dlg.resize(420, 280)
            form = QtWidgets.QFormLayout(dlg)
            s = self.root.setting
            nxt = QtWidgets.QComboBox()
            profiles = list_profiles(AppSettings) or ["default"]
            nxt.addItems(profiles)
            idx = nxt.findText(str(s.p60_next_profile.get() or "default"))
            nxt.setCurrentIndex(idx if idx >= 0 else 0)

            def _next(name: str) -> None:
                if name:
                    s.p60_next_profile.set(name)

            nxt.currentTextChanged.connect(_next)
            form.addRow("Next profile (F)", nxt)

            def _spin(attr: str, lo: float, hi: float) -> QtWidgets.QDoubleSpinBox:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setValue(float(getattr(s, attr)))
                spin.valueChanged.connect(lambda v, a=attr: setattr(s, a, float(v)))
                return spin

            form.addRow("XY zoom", _spin("p66_xy_zoom", 5, 500))
            form.addRow("Frenet zoom", _spin("p66_frenet_zoom", 5, 500))
            form.addRow("Global zoom", _spin("p66_global_zoom", 5, 5000))
            slow = QtWidgets.QDoubleSpinBox()
            slow.setRange(0.05, 2.0)
            slow.setSingleStep(0.05)
            slow.setValue(float(s.p69_mouse_drag_slowdown_factor))
            slow.valueChanged.connect(lambda v: setattr(s, "p69_mouse_drag_slowdown_factor", float(v)))
            form.addRow("Drag slowdown", slow)
            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            buttons.rejected.connect(dlg.reject)
            buttons.accepted.connect(dlg.accept)
            form.addRow(buttons)
            dlg.exec_()


def __getattr__(name: str):
    if name == "ConfigBar":
        _install()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
