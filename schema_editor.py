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


class _SchemaDialog:
    def __init__(self, root, QtCore, QtGui, QtWidgets):
        self.root = root
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self.win = QtWidgets.QDialog(root.win)
        self.win.setWindowTitle("Settings")
        self.win.setObjectName("schemaSettings")
        self.win.resize(920, 720)
        outer = QtWidgets.QVBoxLayout(self.win)
        outer.addWidget(self._profile_bar())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("schemaTabs")
        self._editors: list[tuple[Any, str, Any, Any]] = []
        for title, setting in _sections(root):
            self.tabs.addTab(self._section(title, setting), title)
        self.tabs.addTab(self._shortcuts_tab(), "Shortcuts")
        outer.addWidget(self.tabs, 1)
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
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel("Profile"))
        self.profile = QtWidgets.QComboBox()
        self.profile.setObjectName("profileCombo")
        names = list_profiles(AppSettings) or ["default"]
        self.profile.addItems(names)
        current = str(self.root.setting.c60_selected_profile.get() or "default")
        idx = self.profile.findText(current)
        self.profile.setCurrentIndex(idx if idx >= 0 else 0)
        self.profile.currentTextChanged.connect(self._load_profile)
        row.addWidget(self.profile)
        for label, slot in (
            ("New", self._new_profile),
            ("Rename", self._rename_profile),
            ("Delete", self._delete_profile),
            ("Import", self._import_profile),
            ("Export", self._export_profile),
            ("Reload", lambda: self._load_profile(self.profile.currentText())),
        ):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        row.addWidget(QtWidgets.QLabel("Note"))
        self.note = QtWidgets.QLineEdit(str(getattr(AppSettings, "c60_profile_note", "") or ""))
        self.note.setObjectName("profileNote")
        row.addWidget(self.note, 1)
        return bar

    def _section(self, title: str, setting):
        QtWidgets = self.QtWidgets
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(body)
        form.addRow(QtWidgets.QLabel(title))
        for name, field in _model_fields(setting).items():
            if name in _SKIP or name.startswith("_"):
                continue
            value = self._runtime_value(setting, name)
            editor = self._editor(name, value, field)
            if editor is None:
                continue
            label = QtWidgets.QLabel(name)
            tip = getattr(field, "description", "") or ""
            if tip:
                label.setToolTip(tip)
                editor.setToolTip(tip)
            editor.setObjectName(f"field_{title}_{name}")
            form.addRow(label, editor)
            self._editors.append((setting, name, value, editor))
        scroll.setWidget(body)
        return scroll

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
            out = {}
            for line in editor.toPlainText().splitlines():
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if key:
                    out[key] = val.strip()
            return out
        text = editor.text().strip()
        if isinstance(original, (list, tuple)):
            if text.lower() in ("", "none"):
                return [] if original is not None else None
            return [part.strip() for part in text.split(",") if part.strip()]
        if original is None:
            return None if text.lower() in ("", "none") else text
        return text

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

    def _shortcuts_tab(self):
        QtWidgets = self.QtWidgets
        from .settings import PluginSettings
        from .shortcuts import SHORTCUTS

        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        table = QtWidgets.QTableWidget(len(SHORTCUTS), 4)
        table.setObjectName("shortcutTable")
        table.setHorizontalHeaderLabels(["Group", "Action", "Default", "Binding"])
        overrides = dict(getattr(PluginSettings, "p60_shortcuts", {}) or {})
        sequences: dict[str, str] = {}
        for row, shortcut in enumerate(SHORTCUTS):
            binding = str(overrides.get(shortcut.id, shortcut.default))
            sequences[shortcut.id] = binding
            for col, text in enumerate((shortcut.group, shortcut.label, shortcut.default, binding)):
                item = QtWidgets.QTableWidgetItem(text)
                if col < 3:
                    item.setFlags(item.flags() & ~self.QtCore.Qt.ItemIsEditable)
                table.setItem(row, col, item)
            table.item(row, 3).setData(self.QtCore.Qt.UserRole, shortcut.id)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)

        def _store() -> None:
            seen: dict[str, str] = {}
            out = {}
            for row, shortcut in enumerate(SHORTCUTS):
                binding = table.item(row, 3).text().strip() or shortcut.default
                if binding in seen:
                    self.QtWidgets.QMessageBox.warning(
                        self.win, "Shortcuts", f"{binding} is already used by {seen[binding]}.",
                    )
                    return
                seen[binding] = shortcut.label
                if binding != shortcut.default:
                    out[shortcut.id] = binding
            PluginSettings.p60_shortcuts = out
            log.info("Shortcut overrides: %s", out)

        apply = QtWidgets.QPushButton("Apply shortcut bindings")
        apply.setObjectName("applyShortcuts")
        apply.clicked.connect(_store)
        layout.addWidget(apply)
        return page
