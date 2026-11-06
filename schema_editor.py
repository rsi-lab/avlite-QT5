"""Schema-driven settings window. Qt is imported only when the dialog opens."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_SKIP = {"filepath", "schema", "exclude"}


def open_schema_settings(root) -> None:
    from .qt_binding import import_qt

    QtCore, QtGui, QtWidgets = import_qt()
    root.setting.push_plugin_schema()
    dlg = _SchemaDialog(root, QtCore, QtGui, QtWidgets)
    dlg.exec_()


def _model_fields(setting) -> dict:
    model = setting if isinstance(setting, type) else type(setting)
    return dict(getattr(model, "model_fields", {}) or {})


def _sections(root) -> list[tuple[str, Any]]:
    from avlite.c10_perception.c19_settings import PerceptionSettings
    from avlite.c20_planning.c29_settings import PlanningSettings
    from avlite.c30_control.c39_settings import ControlSettings
    from avlite.c40_execution.c49_settings import ExecutionSettings
    from avlite.c60_apps.c63_plugins import load_builtin_plugin_settings, load_community_plugin_setting
    from avlite.c60_apps.c69_settings import AppSettings

    from .settings import PluginSettings

    sections: list[tuple[str, Any]] = [
        ("Perception", PerceptionSettings),
        ("Planning", PlanningSettings),
        ("Control", ControlSettings),
        ("Execution", ExecutionSettings),
        ("Visualizer", PluginSettings),
    ]
    if not bool(root.setting.c62_load_plugins.get()):
        return sections
    for name in list(AppSettings.c62_default_plugins or []):
        try:
            cls = load_builtin_plugin_settings(name)
        except Exception:
            log.debug("builtin plugin settings skipped: %s", name, exc_info=True)
            continue
        if cls is None or cls is PluginSettings:
            continue
        sections.append((f"Plugin {name}", cls))
    profile = str(root.setting.c60_selected_profile.get() or "default")
    for name, stored in dict(AppSettings.c62_community_plugins or {}).items():
        if name == getattr(root, "hosting_plugin_name", "avlite-QT5"):
            continue
        try:
            cls = load_community_plugin_setting(name, stored, profile=profile)
        except Exception:
            log.debug("community plugin settings skipped: %s", name, exc_info=True)
            continue
        if cls is None or cls is PluginSettings:
            continue
        sections.append((f"Community {name}", cls))
    return sections


def _fit_window(win, QtWidgets, QtCore) -> None:
    """Keep the dialog inside the screen, with room for the title bar."""
    screen = QtWidgets.QApplication.primaryScreen()
    avail = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1280, 800)
    margin = 48
    win.resize(
        min(1080, max(720, avail.width() - margin * 2)),
        min(840, max(560, avail.height() - margin * 2)),
    )
    win.setMinimumSize(
        min(640, max(480, avail.width() - margin)),
        min(420, max(320, avail.height() - margin)),
    )
    # Stay on the screen. Extra content scrolls instead of drawing past the edge.
    win.setMaximumSize(max(640, avail.width()), max(480, avail.height()))


def _make_resize_filter(dialog, QtCore):
    class _Filter(QtCore.QObject):
        def eventFilter(self, obj, event):  # noqa: N802
            if event.type() == QtCore.QEvent.Resize and obj is dialog.win:
                dialog._fit_profile_scroll()
            return False

    return _Filter(dialog.win)


class _SchemaDialog:
    def __init__(self, root, QtCore, QtGui, QtWidgets):
        self.root = root
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self.QtGui = QtGui
        self.win = QtWidgets.QDialog(root.win)
        self.win.setWindowTitle("Settings")
        self.win.setObjectName("schemaSettings")
        _fit_window(self.win, QtWidgets, QtCore)
        self._fitting_profile = False
        outer = QtWidgets.QVBoxLayout(self.win)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        outer.addWidget(self._profile_bar())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("schemaTabs")
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(QtCore.Qt.ElideNone)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.setStyleSheet(
            "QTabBar::tab { padding: 6px 12px; margin: 1px 2px 0 0; }"
        )
        self._editors: list[tuple[Any, str, Any, Any]] = []
        self._sections: list[tuple[str, Any]] = []
        for title, setting in _sections(root):
            self._sections.append((title, setting))
            self.tabs.addTab(self._section(title, setting), title)
        self.tabs.addTab(self._shortcuts_tab(), "Shortcuts")
        self.tabs.addTab(self._plugins_tab(), "Plugins")
        outer.addWidget(self.tabs, 1)
        self.win.finished.connect(self._end_capture)
        self._resize_fit = _make_resize_filter(self, QtCore)
        self.win.installEventFilter(self._resize_fit)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Close
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.win.reject)
        outer.addWidget(buttons)

    def exec_(self):
        return self.win.exec_()

    def _profile_bar(self):
        QtWidgets = self.QtWidgets
        from avlite.c60_apps.c65_setting_utils import list_profiles
        from avlite.c60_apps.c69_settings import AppSettings

        bar = QtWidgets.QWidget()
        bar.setObjectName("profileBar")
        rows = QtWidgets.QVBoxLayout(bar)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(6)
        names = list_profiles(AppSettings) or ["default"]
        current = str(self.root.setting.c60_selected_profile.get() or "default")

        def row():
            host = QtWidgets.QWidget()
            box = QtWidgets.QHBoxLayout(host)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(6)
            rows.addWidget(host)
            return box

        self.profile = QtWidgets.QComboBox()
        self.profile.setObjectName("profileCombo")
        self.profile.setMinimumContentsLength(12)
        self.profile.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.profile.addItems(names)
        idx = self.profile.findText(current)
        self.profile.setCurrentIndex(idx if idx >= 0 else 0)
        self.profile.currentTextChanged.connect(self._load_profile)
        actions = (
            ("New", self._new_profile, "profileNew"),
            ("Rename", self._rename_profile, "profileRename"),
            ("Delete", self._delete_profile, "profileDelete"),
            ("Import", self._import_profile, "profileImport"),
            ("Export", self._export_profile, "profileExport"),
            ("Reload", lambda: self._load_profile(self.profile.currentText()), "profileReload"),
        )
        self._profile_buttons = {}
        for label, slot, name in actions:
            btn = QtWidgets.QPushButton(label)
            btn.setObjectName(name)
            btn.clicked.connect(slot)
            self._profile_buttons[name] = btn
        is_default = current == "default"
        self._profile_buttons["profileRename"].setEnabled(not is_default)
        self._profile_buttons["profileDelete"].setEnabled(not is_default)

        top = row()
        top.addWidget(QtWidgets.QLabel("Profile"))
        top.addWidget(self.profile, 1)
        for name in ("profileNew", "profileRename", "profileDelete"):
            top.addWidget(self._profile_buttons[name])

        self.next_profile = QtWidgets.QComboBox()
        self.next_profile.setObjectName("nextProfileCombo")
        self.next_profile.addItems(names)
        nxt = str(self.root.setting.p60_next_profile.get() or "default")
        nidx = self.next_profile.findText(nxt)
        self.next_profile.setCurrentIndex(nidx if nidx >= 0 else 0)
        self.next_profile.currentTextChanged.connect(
            lambda text: self.root.setting.p60_next_profile.set(text) if text else None
        )
        mid = row()
        mid.addWidget(QtWidgets.QLabel("Next profile"))
        mid.addWidget(self.next_profile, 1)
        for name in ("profileImport", "profileExport", "profileReload"):
            mid.addWidget(self._profile_buttons[name])

        reset_all = QtWidgets.QPushButton("Reset all to defaults")
        reset_all.setObjectName("resetAllDefaults")
        reset_all.clicked.connect(lambda: self._reset_stack(exclude_execution=False))
        reset_keep = QtWidgets.QPushButton("Reset except execution")
        reset_keep.setObjectName("resetExceptExecution")
        reset_keep.clicked.connect(lambda: self._reset_stack(exclude_execution=True))
        self.note = QtWidgets.QLineEdit(str(getattr(AppSettings, "c60_profile_note", "") or ""))
        self.note.setObjectName("profileNote")
        self.note.setMinimumWidth(80)
        bottom = row()
        bottom.addWidget(reset_all)
        bottom.addWidget(reset_keep)
        bottom.addWidget(QtWidgets.QLabel("Note"))
        bottom.addWidget(self.note, 1)

        from avlite.c60_apps.c68_paths import ConfigPaths

        if ConfigPaths.can_edit_bundled():
            repo = QtWidgets.QCheckBox("Edit repository configs")
            repo.setObjectName("editRepoConfigs")
            repo.setChecked(ConfigPaths.is_repo_target())
            repo.stateChanged.connect(self._toggle_repo_configs)
            extra = row()
            extra.addWidget(repo)
            extra.addStretch(1)

        scroll = self._scroll(bar)
        scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        self._profile_bar = bar
        self._profile_scroll = scroll
        self._fit_profile_scroll()
        return scroll

    def _fit_profile_scroll(self) -> None:
        """Keep every profile row on screen. A scrollbar must not cover a row."""
        if self._fitting_profile:
            return
        bar = getattr(self, "_profile_bar", None)
        scroll = getattr(self, "_profile_scroll", None)
        if bar is None or scroll is None:
            return
        self._fitting_profile = True
        try:
            hint = bar.sizeHint()
            bar.setMinimumHeight(hint.height())
            bar.setMinimumWidth(hint.width())
            needs_hbar = hint.width() > max(320, self.win.width() - 36)
            extra = scroll.horizontalScrollBar().sizeHint().height() + 6 if needs_hbar else 4
            height = hint.height() + extra
            if scroll.maximumHeight() != height:
                scroll.setMaximumHeight(height)
        finally:
            self._fitting_profile = False

    def _scroll(self, body):
        """Scroll area that keeps its contents reachable instead of clipping them."""
        QtWidgets = self.QtWidgets
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(self.QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(self.QtCore.Qt.ScrollBarAsNeeded)
        scroll.setAlignment(self.QtCore.Qt.AlignTop | self.QtCore.Qt.AlignLeft)
        scroll.setWidget(body)
        return scroll

    def _section(self, title: str, setting):
        QtWidgets = self.QtWidgets
        body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(body)
        form.setContentsMargins(8, 8, 16, 12)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(self.QtCore.Qt.AlignLeft | self.QtCore.Qt.AlignVCenter)
        heading = QtWidgets.QLabel(title)
        heading.setStyleSheet("font-weight: bold;")
        form.addRow(heading)
        for name, field in _model_fields(setting).items():
            if name in _SKIP or name.startswith("_"):
                continue
            value = self._runtime_value(setting, name)
            editor = self._editor(name, value, field)
            if editor is None:
                continue
            label = QtWidgets.QLabel(name)
            label.setMinimumWidth(label.sizeHint().width())
            tip = getattr(field, "description", "") or ""
            if tip:
                label.setToolTip(tip)
                editor.setToolTip(tip)
            editor.setObjectName(f"field_{title}_{name}")
            editor.setMinimumWidth(160)
            form.addRow(label, editor)
            self._editors.append((setting, name, value, editor))
        reset = QtWidgets.QPushButton("Reset this section to defaults")
        reset.setObjectName(f"resetSection_{title}")
        reset.clicked.connect(lambda _checked=False, target=setting: self._reset_one(target))
        form.addRow(reset)
        body.setMinimumWidth(form.totalMinimumSize().width())
        return self._scroll(body)

    def _runtime_value(self, setting, name: str):
        from .setting_model import Var

        if setting is self.root.setting.schema:
            current = getattr(self.root.setting, name, None)
            if isinstance(current, Var):
                return current.get()
            if current is not None and not callable(current):
                return current
        return getattr(setting, name)

    def _editor(self, name: str, value, field):
        QtWidgets = self.QtWidgets
        if isinstance(value, bool):
            box = QtWidgets.QCheckBox()
            box.setChecked(bool(value))
            return box
        if isinstance(value, int) and not isinstance(value, bool):
            spin = QtWidgets.QSpinBox()
            spin.setRange(-1_000_000_000, 1_000_000_000)
            spin.setValue(int(value))
            return spin
        if isinstance(value, float):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(4)
            spin.setValue(float(value))
            return spin
        if isinstance(value, str) or value is None:
            edit = QtWidgets.QLineEdit("" if value is None else str(value))
            if value is None:
                edit.setPlaceholderText("None")
            return edit
        if isinstance(value, (list, tuple)):
            edit = QtWidgets.QLineEdit(", ".join(str(v) for v in value))
            return edit
        if name == "p60_shortcuts":
            return None
        if isinstance(value, dict):
            edit = QtWidgets.QPlainTextEdit()
            edit.setPlainText("\n".join(f"{k}={v}" for k, v in value.items()))
            edit.setFixedHeight(80)
            return edit
        return None

    def _parsed(self, original, editor):
        QtWidgets = self.QtWidgets
        if isinstance(editor, QtWidgets.QCheckBox):
            return bool(editor.isChecked())
        if isinstance(editor, QtWidgets.QSpinBox) and not isinstance(editor, QtWidgets.QDoubleSpinBox):
            return int(editor.value())
        if isinstance(editor, QtWidgets.QDoubleSpinBox):
            return float(editor.value())
        if isinstance(editor, QtWidgets.QPlainTextEdit):
            previous = original if isinstance(original, dict) else {}
            out = {}
            for line in editor.toPlainText().splitlines():
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                sample = previous.get(key)
                out[key] = val.strip() if sample is None else _coerce_scalar(val, sample)
            return out
        text = editor.text().strip()
        if isinstance(original, (list, tuple)):
            if text.lower() in ("", "none"):
                return [] if isinstance(original, list) else ()
            sample = next((item for item in original if item is not None), None)
            return [_coerce_scalar(part, sample) for part in text.split(",") if part.strip()]
        if isinstance(original, dict):
            return original
        if original is None:
            return None if text.lower() in ("", "none") else text
        return _coerce_scalar(text, original)

    def _apply_editors(self) -> None:
        from .setting_model import Var

        for setting, name, original, editor in self._editors:
            try:
                value = self._parsed(original, editor)
                setattr(setting, name, value)
            except Exception:
                log.exception("Could not apply %s", name)
                continue
            if setting is self.root.setting.schema:
                current = getattr(self.root.setting, name, None)
                if isinstance(current, Var):
                    current.set(value)
                elif current is not None and not callable(current):
                    setattr(self.root.setting, name, value)

    def _save(self) -> None:
        from avlite.c60_apps.c65_setting_utils import save_setting
        from avlite.c60_apps.c69_settings import AppSettings

        if not self._store_shortcuts(quiet=True):
            return
        self._apply_editors()
        profile = str(self.root.setting.c60_selected_profile.get() or "default")
        AppSettings.c60_profile_note = self.note.text()
        AppSettings.c60_selected_profile = profile
        AppSettings.c62_load_plugins = bool(self.root.setting.c62_load_plugins.get())
        saved = set()
        for setting, _name, _orig, _editor in self._editors:
            if id(setting) in saved:
                continue
            saved.add(id(setting))
            try:
                save_setting(setting, profile=profile)
            except Exception:
                log.exception("save_setting failed for %s", type(setting).__name__)
        try:
            save_setting(AppSettings, profile=profile)
        except Exception:
            log.exception("save_setting AppSettings failed")
        self.root.setting.pull_plugin_schema()
        self.root.update_views()
        self.root.update_ui()
        log.info("Saved schema settings to profile %s", profile)
        self.win.accept()

    def _load_profile(self, name: str) -> None:
        if not name:
            return
        self.root.load_settings(profile=name)
        self.root.reload_stack(reload_code=False)
        self.win.accept()
        open_schema_settings(self.root)

    def _new_profile(self) -> None:
        name, ok = self.QtWidgets.QInputDialog.getText(self.win, "Profile", "Name")
        if not ok or not str(name).strip():
            return
        self.root.setting.c60_selected_profile.set(str(name).strip())
        self._save()

    def _rename_profile(self) -> None:
        from avlite.c60_apps.c65_setting_utils import rename_profile

        old = str(self.root.setting.c60_selected_profile.get() or "default")
        if old == "default":
            self.QtWidgets.QMessageBox.warning(self.win, "Rename", "Cannot rename the default profile.")
            return
        name, ok = self.QtWidgets.QInputDialog.getText(self.win, "Rename profile", "New name", text=old)
        if not ok or not str(name).strip() or str(name).strip() == old:
            return
        if not rename_profile(old, str(name).strip()):
            self.QtWidgets.QMessageBox.warning(self.win, "Rename", f"Could not rename {old}.")
            return
        self._load_profile(str(name).strip())

    def _delete_profile(self) -> None:
        from avlite.c60_apps.c65_setting_utils import delete_profile

        profile = str(self.root.setting.c60_selected_profile.get() or "default")
        if profile == "default":
            self.QtWidgets.QMessageBox.warning(self.win, "Delete", "Cannot delete the default profile.")
            return
        if self.QtWidgets.QMessageBox.question(self.win, "Delete", f"Delete profile {profile}?") != self.QtWidgets.QMessageBox.Yes:
            return
        if not delete_profile(profile):
            self.QtWidgets.QMessageBox.warning(self.win, "Delete", f"Could not delete {profile}.")
            return
        self._load_profile("default")

    def _export_profile(self) -> None:
        from avlite.c60_apps.c65_setting_utils import export_profile

        profile = str(self.root.setting.c60_selected_profile.get() or "default")
        path, _ = self.QtWidgets.QFileDialog.getSaveFileName(
            self.win, "Export profile", f"{profile}.yaml", "YAML (*.yaml)"
        )
        if not path:
            return
        try:
            count = export_profile(profile, path, include_stack=True, include_app=True, include_plugins=True)
        except Exception as exc:
            self.QtWidgets.QMessageBox.warning(self.win, "Export", str(exc))
            return
        self.QtWidgets.QMessageBox.information(self.win, "Export", f"Exported {count} section(s) to\n{path}")

    def _import_profile(self) -> None:
        from avlite.c60_apps.c65_setting_utils import import_profile

        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(self.win, "Import profile", "", "YAML (*.yaml)")
        if not path:
            return
        try:
            name = import_profile(path, overwrite=True)
        except Exception as exc:
            self.QtWidgets.QMessageBox.warning(self.win, "Import", str(exc))
            return
        self._load_profile(name)

    def _plugins_tab(self):
        from .plugin_browser import _Browser

        page = self.QtWidgets.QWidget()
        page.setObjectName("settingsPluginsPage")
        _Browser(self.root, self.QtCore, self.QtWidgets, host=page)
        return self._scroll(page)

    def _toggle_repo_configs(self, state: int) -> None:
        from avlite.c60_apps.c68_paths import ConfigPaths

        enabled = state == self.QtCore.Qt.Checked
        box = self.win.findChild(self.QtWidgets.QCheckBox, "editRepoConfigs")
        if enabled:
            answer = self.QtWidgets.QMessageBox.question(
                self.win,
                "Edit repository configs",
                "Core stack and built-in plugin settings will be read and written\n"
                "from the bundled config directory instead of your user config.\n\n"
                "Continue?",
            )
            if answer != self.QtWidgets.QMessageBox.Yes:
                if box is not None:
                    box.blockSignals(True)
                    box.setChecked(False)
                    box.blockSignals(False)
                return
        ConfigPaths.set_repo_target(enabled)
        self._load_profile(self.profile.currentText())

    def _reset_one(self, setting) -> None:
        from avlite.c60_apps.c64_settings_schema import apply_validated_to_setting, schema_of

        schema = schema_of(setting)
        if schema is None:
            log.warning("No schema to reset")
            return
        apply_validated_to_setting(setting, schema.model_validate({}))
        if setting is self.root.setting.schema:
            self.root.setting.pull_plugin_schema()
            self._fill_shortcut_table()
        self._sync_editors(setting)
        self.root.setting.sync_stack_from_singletons()
        self.root.stack_panels.refresh()
        self.root.update_views()
        self.root._apply_theme()
        log.info("Reset %s to source defaults", type(setting).__name__)

    def _reset_stack(self, *, exclude_execution: bool) -> None:
        from avlite.c10_perception.c19_settings import PerceptionSettings
        from avlite.c20_planning.c29_settings import PlanningSettings
        from avlite.c30_control.c39_settings import ControlSettings
        from avlite.c40_execution.c49_settings import ExecutionSettings

        for setting in (PerceptionSettings, PlanningSettings, ControlSettings):
            self._reset_one(setting)
        if not exclude_execution:
            self._reset_one(ExecutionSettings)

    def _sync_editors(self, setting) -> None:
        for index, (owner, name, _original, editor) in enumerate(self._editors):
            if owner is not setting:
                continue
            value = self._runtime_value(owner, name)
            self._editors[index] = (owner, name, value, editor)
            self._write_widget(editor, value)

    def _write_widget(self, editor, value) -> None:
        QtWidgets = self.QtWidgets
        if isinstance(editor, QtWidgets.QCheckBox):
            editor.setChecked(bool(value))
            return
        if isinstance(editor, QtWidgets.QSpinBox) and not isinstance(editor, QtWidgets.QDoubleSpinBox):
            editor.setValue(int(value or 0))
            return
        if isinstance(editor, QtWidgets.QDoubleSpinBox):
            editor.setValue(float(value or 0.0))
            return
        if isinstance(editor, QtWidgets.QPlainTextEdit):
            pairs = value.items() if isinstance(value, dict) else []
            editor.setPlainText("\n".join(f"{k}={v}" for k, v in pairs))
            return
        if isinstance(value, (list, tuple)):
            editor.setText(", ".join(str(v) for v in value))
            return
        editor.setText("" if value is None else str(value))

    def _shortcuts_tab(self):
        QtWidgets = self.QtWidgets
        from .shortcuts import SHORTCUTS

        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._shortcut_table = QtWidgets.QTableWidget(len(SHORTCUTS), 4)
        table = self._shortcut_table
        table.setObjectName("shortcutTable")
        table.setHorizontalHeaderLabels(["Group", "Action", "Default", "Binding"])
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        table.setMinimumHeight(160)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self._fill_shortcut_table()
        layout.addWidget(table, 1)

        self._shortcut_status = QtWidgets.QLabel("")
        self._shortcut_status.setObjectName("shortcutStatus")
        self._shortcut_status.setWordWrap(True)
        layout.addWidget(self._shortcut_status)

        buttons = QtWidgets.QHBoxLayout()
        capture = QtWidgets.QPushButton("Capture key")
        capture.setObjectName("captureShortcut")
        capture.clicked.connect(self._begin_capture)
        apply = QtWidgets.QPushButton("Apply shortcut bindings")
        apply.setObjectName("applyShortcuts")
        apply.clicked.connect(lambda: self._store_shortcuts(quiet=False))
        reset = QtWidgets.QPushButton("Reset to defaults")
        reset.setObjectName("resetShortcuts")
        reset.clicked.connect(self._reset_shortcuts)
        for btn in (capture, apply, reset):
            buttons.addWidget(btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self._capture_filter = None
        return self._scroll(page)

    def _fill_shortcut_table(self) -> None:
        from .settings import PluginSettings
        from .shortcuts import SHORTCUTS

        QtWidgets = self.QtWidgets
        table = self._shortcut_table
        overrides = dict(getattr(PluginSettings, "p60_shortcuts", {}) or {})
        table.setRowCount(len(SHORTCUTS))
        for row, shortcut in enumerate(SHORTCUTS):
            binding = str(overrides.get(shortcut.id) or shortcut.default)
            for col, text in enumerate((shortcut.group, shortcut.label, shortcut.default, binding)):
                item = QtWidgets.QTableWidgetItem(text)
                if col < 3:
                    item.setFlags(item.flags() & ~self.QtCore.Qt.ItemIsEditable)
                table.setItem(row, col, item)
            table.item(row, 3).setData(self.QtCore.Qt.UserRole, shortcut.id)

    def _store_shortcuts(self, quiet: bool = False) -> bool:
        from .settings import PluginSettings
        from .shortcuts import SHORTCUTS

        table = getattr(self, "_shortcut_table", None)
        if table is None:
            return True
        seen: dict[str, str] = {}
        out = {}
        for row, shortcut in enumerate(SHORTCUTS):
            item = table.item(row, 3)
            binding = (item.text() if item is not None else "").strip() or shortcut.default
            if binding in seen:
                message = f"{binding} is already used by {seen[binding]}."
                self._shortcut_status.setText(message)
                if not quiet:
                    self.QtWidgets.QMessageBox.warning(self.win, "Shortcuts", message)
                return False
            seen[binding] = shortcut.label
            if binding != shortcut.default:
                out[shortcut.id] = binding
        PluginSettings.p60_shortcuts = out
        refresh = getattr(self.root.config_bar, "refresh_shortcut_labels", None)
        if refresh is not None:
            refresh()
        self._shortcut_status.setText("Shortcut bindings applied." if out else "Shortcuts are at their defaults.")
        log.info("Shortcut overrides: %s", out)
        return True

    def _reset_shortcuts(self) -> None:
        from .settings import PluginSettings

        PluginSettings.p60_shortcuts = {}
        self._fill_shortcut_table()
        self._store_shortcuts(quiet=True)
        self._shortcut_status.setText("Shortcuts reset to defaults.")

    def _begin_capture(self) -> None:
        row = self._shortcut_table.currentRow()
        if row < 0:
            self._shortcut_status.setText("Select a shortcut row, then capture a key.")
            return
        self._capture_row = row
        self._shortcut_status.setText("Press a key. Esc cancels.")
        if self._capture_filter is None:
            dialog = self

            class _Grab(self.QtCore.QObject):
                def eventFilter(self, obj, event):  # noqa: N802
                    if event.type() != dialog.QtCore.QEvent.KeyPress or event.isAutoRepeat():
                        return False
                    dialog._finish_capture(event)
                    return True

            self._capture_filter = _Grab(self.win)
        app = self.QtWidgets.QApplication.instance()
        app.removeEventFilter(self._capture_filter)
        app.installEventFilter(self._capture_filter)

    def _end_capture(self, *_args) -> None:
        app = self.QtWidgets.QApplication.instance()
        filt = getattr(self, "_capture_filter", None)
        if filt is not None and app is not None:
            app.removeEventFilter(filt)

    def _finish_capture(self, event) -> None:
        app = self.QtWidgets.QApplication.instance()
        if self._capture_filter is not None:
            app.removeEventFilter(self._capture_filter)
        if event.key() == self.QtCore.Qt.Key_Escape:
            self._shortcut_status.setText("Capture cancelled.")
            return
        sequence = _sequence_from_event(event, self.QtCore)
        if not sequence:
            self._shortcut_status.setText("That key cannot be used as a shortcut.")
            return
        item = self._shortcut_table.item(self._capture_row, 3)
        if item is None:
            return
        item.setText(sequence)
        self._store_shortcuts(quiet=False)


def _coerce_scalar(text: str, sample):
    """Keep the type a schema field already had when a text box is saved."""
    text = str(text).strip()
    if isinstance(sample, bool):
        return text.lower() in ("1", "true", "yes")
    if isinstance(sample, int) and not isinstance(sample, bool):
        try:
            return int(text)
        except ValueError:
            return text
    if isinstance(sample, float):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _sequence_from_event(event, QtCore) -> str:
    """Same binding text the main window uses for shortcut overrides."""
    Qt = QtCore.Qt
    key = event.key()
    text = event.text()
    ctrl = bool(event.modifiers() & Qt.ControlModifier)
    special = {
        Qt.Key_Up: "Up",
        Qt.Key_Down: "Down",
        Qt.Key_Escape: "Esc",
        Qt.Key_Plus: "+",
        Qt.Key_Equal: "+",
        Qt.Key_Minus: "-",
    }
    if ctrl:
        name = special.get(key)
        if name is None and Qt.Key_A <= key <= Qt.Key_Z:
            name = chr(key)
        return ("Ctrl+" + name) if name else ""
    if text and text.isprintable():
        return text
    return special.get(key, "")
