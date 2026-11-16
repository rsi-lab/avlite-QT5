"""Community and member plugin browser.

Install, uninstall, update, and profile registration go through
``_PluginOperations`` — the same operations the Tk browser uses. This module
only draws the window. Member sign-in calls ``_GitHubClient`` because AVLite
has no public wrapper around that device flow.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

log = logging.getLogger(__name__)

_COLUMNS = ("Name", "Category", "Author", "Version", "Status", "Update", "Path")
_UPDATE_LABELS = {
    "up-to-date": "Up to date \u2713",
    "update-available": "Update \u2191",
    "unknown": "\u2014",
}
_COMMUNITY_DISCLAIMER = (
    "Community plugins are third-party code. AV-Lab does not vet or guarantee their safety. "
    "For research and development only (use at your own risk)."
)
_MEMBERS_DISCLAIMER = (
    "Member plugins are AV-Lab–listed but not safety-certified. "
    "For research and development only (use at your own risk)."
)


def open_plugin_browser(root) -> None:
    from .qt_binding import import_qt

    QtCore, _QtGui, QtWidgets = import_qt()
    _Browser(root, QtCore, QtWidgets).exec_()


def _ops():
    from avlite.plugins.p60_visualizer_tk.p63_plugins_app import (
        GitHubApiError,
        _GitHubClient,
        _PluginOperations,
        can_add_to_profile,
    )

    return _PluginOperations, _GitHubClient, GitHubApiError, can_add_to_profile


def _plugins_dir(*, private: bool) -> Path:
    from avlite.c60_apps.c68_paths import PluginPaths

    if PluginPaths.is_dev_mode():
        return PluginPaths.private_dev_dir() if private else PluginPaths.community_dev_dir()
    return PluginPaths.install_dir()


def _display_name(entry, name: str) -> str:
    return str((entry or {}).get("display_name") or "").strip() or name


def _category(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(part) for part in value)
    return str(value or "")


class _Bridge:
    """Created after Qt is imported. ``type`` is filled in by :func:`_bridge`."""

    type = None


def _bridge(QtCore):
    if _Bridge.type is None:
        class Bridge(QtCore.QObject):
            ready = QtCore.pyqtSignal(object, str)
            result = QtCore.pyqtSignal(object, object)
            readme = QtCore.pyqtSignal(str)
            code = QtCore.pyqtSignal(str, str)
            failed = QtCore.pyqtSignal(str)
            authed = QtCore.pyqtSignal(str, str)
            updates = QtCore.pyqtSignal(object)

        _Bridge.type = Bridge
    return _Bridge.type


class _Browser:
    def __init__(self, root, QtCore, QtWidgets, host=None):
        self.root = root
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self._owns_window = host is None
        if host is None:
            self.win = QtWidgets.QDialog(root.win)
            self.win.setWindowTitle("AVLite Plugins")
            self.win.setObjectName("pluginBrowser")
            screen = QtWidgets.QApplication.primaryScreen()
            avail = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1280, 800)
            self.win.resize(min(1100, max(800, avail.width() - 80)), min(640, max(460, avail.height() - 80)))
        else:
            self.win = host
        layout = QtWidgets.QVBoxLayout(self.win)
        layout.setContentsMargins(8, 8, 8, 8)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("pluginTabs")
        self.community = _Page(self, private=False)
        self.members = _Page(self, private=True)
        self.tabs.addTab(self.community.widget, "Community")
        self.tabs.addTab(self.members.widget, "Members")
        layout.addWidget(self.tabs, 1)
        footer = QtWidgets.QHBoxLayout()
        self.footer = QtWidgets.QLabel()
        self.footer.setObjectName("pluginFooter")
        footer.addWidget(self.footer, 1)
        self.dev_mode = QtWidgets.QCheckBox("Dev mode")
        self.dev_mode.setObjectName("chkPluginDev")
        from avlite.c60_apps.c68_paths import PluginPaths

        self.dev_mode.setChecked(PluginPaths.is_dev_mode())
        self.dev_mode.toggled.connect(self._on_dev_mode)
        footer.addWidget(self.dev_mode)
        layout.addLayout(footer)
        self._refresh_footer()
        self.community.start()
        self.members.start()

    def exec_(self):
        return self.win.exec_()

    def _refresh_footer(self) -> None:
        from avlite.c60_apps.c68_paths import PluginPaths

        if PluginPaths.is_dev_mode():
            text = (
                "Dev mode: "
                f"{PluginPaths.format_display(PluginPaths.community_dev_dir())} | "
                f"{PluginPaths.format_display(PluginPaths.private_dev_dir())}"
            )
        else:
            text = f"Install location: {PluginPaths.format_display(PluginPaths.install_dir())}"
        self.footer.setText(text)

    def _on_dev_mode(self, enabled: bool) -> None:
        from avlite.c60_apps.c68_paths import PluginPaths

        PluginPaths.set_dev_mode(bool(enabled))
        self._refresh_footer()
        self.community.use_dir()
        self.members.use_dir()

    def notify(self) -> None:
        """Reload stack settings after a plugin is added or removed, as Tk does."""
        from avlite.c60_apps.c65_setting_utils import list_profiles
        from avlite.c60_apps.c69_settings import AppSettings

        self.root.setting.profile_list = list_profiles(AppSettings)
        self.root.load_settings(only_stack=True)
        if self.root.setting.c62_load_plugins.get():
            self.root.reload_stack(reload_code=True)
        else:
            self.root.stack_panels.refresh()
            self.root.update_ui()


class _Page:
    def __init__(self, browser: _Browser, *, private: bool):
        QtWidgets = browser.QtWidgets
        self.browser = browser
        self.private = private
        self.plugins_dir = _plugins_dir(private=private)
        self._registry: list[dict] = []
        self._rows: dict[str, dict] = {}
        self._updates: dict[str, str] = {}
        self._busy = False
        self._closed = False
        self._token = ""
        self._login = ""
        self.widget = QtWidgets.QWidget()
        self.widget.setObjectName("membersPage" if private else "communityPage")
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        if private:
            auth = QtWidgets.QHBoxLayout()
            self.auth_label = QtWidgets.QLabel("Sign in with GitHub to browse member plugins.")
            self.auth_label.setObjectName("memberAuth")
            auth.addWidget(self.auth_label, 1)
            self.btn_sign_in = QtWidgets.QPushButton("Sign in with GitHub")
            self.btn_sign_in.setObjectName("btnSignIn")
            self.btn_sign_in.clicked.connect(self._sign_in)
            self.btn_sign_out = QtWidgets.QPushButton("Sign out")
            self.btn_sign_out.setObjectName("btnSignOut")
            self.btn_sign_out.clicked.connect(self._sign_out)
            auth.addWidget(self.btn_sign_in)
            auth.addWidget(self.btn_sign_out)
            layout.addLayout(auth)
        disclaimer = QtWidgets.QLabel(_MEMBERS_DISCLAIMER if private else _COMMUNITY_DISCLAIMER)
        disclaimer.setWordWrap(True)
        layout.addWidget(disclaimer)
        filters = QtWidgets.QHBoxLayout()
        self.only_installed = QtWidgets.QCheckBox("Show Installed only")
        self.only_installed.setObjectName("chkInstalledOnly" if not private else "chkMembersInstalled")
        self.only_active = QtWidgets.QCheckBox("Show Active only")
        self.only_active.setObjectName("chkActiveOnly" if not private else "chkMembersActive")
        self.only_installed.toggled.connect(self._fill)
        self.only_active.toggled.connect(self._fill)
        filters.addWidget(self.only_installed)
        filters.addWidget(self.only_active)
        filters.addStretch(1)
        layout.addLayout(filters)
        self.table = QtWidgets.QTableWidget(0, len(_COLUMNS))
        self.table.setObjectName("membersTable" if private else "communityTable")
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(220)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.itemDoubleClicked.connect(lambda *_: self._details())
        layout.addWidget(self.table, 1)
        buttons = QtWidgets.QHBoxLayout()
        self._buttons = {}
        for label, slot, name in (
            ("Refresh", self.refresh, "btnRefresh"),
            ("Install", self._install, "btnInstall"),
            ("Add to Profile", self._add, "btnAddProfile"),
            ("Uninstall", self._uninstall, "btnUninstall"),
            ("Update", self._update, "btnUpdate"),
            ("Update All", self._update_all, "btnUpdateAll"),
            ("Open on GitHub", self._github, "btnGitHub"),
            ("Open Folder", self._folder, "btnOpenFolder"),
            ("Details", self._details, "btnDetails"),
        ):
            if private:
                name = name + "Members"
            btn = QtWidgets.QPushButton(label)
            btn.setObjectName(name)
            btn.clicked.connect(slot)
            self._buttons[label] = btn
            buttons.addWidget(btn)
        if browser._owns_window and not private:
            close = QtWidgets.QPushButton("Close")
            close.setObjectName("closePlugins")
            close.clicked.connect(browser.win.accept)
            buttons.addWidget(close)
        layout.addLayout(buttons)
        self.status = QtWidgets.QLabel("Ready")
        self.status.setObjectName("membersStatus" if private else "pluginStatus")
        layout.addWidget(self.status)
        self._signal = _bridge(browser.QtCore)()
        self._signal.ready.connect(self._on_registry)
        self._signal.result.connect(self._on_result)
        self._signal.updates.connect(self._on_updates)
        self._task_done = None
        browser.win.destroyed.connect(self._mark_closed)

    def start(self) -> None:
        self._fill()
        if self.private:
            self._set_auth_visible()
            loaded = _ops()[1]._load_github_token()
            if loaded:
                self._token, self._login = loaded
                self._set_auth_visible()
                self.refresh()
            else:
                self.status.setText("Sign in to browse member plugins.")
            return
        self.refresh()

    def _mark_closed(self, *_args) -> None:
        self._closed = True

    def _set_auth_visible(self) -> None:
        if not self.private:
            return
        signed = bool(self._token)
        self.auth_label.setText(
            f"Signed in as @{self._login}" if signed else "Sign in with GitHub to browse member plugins."
        )
        self.btn_sign_in.setVisible(not signed)
        self.btn_sign_out.setVisible(signed)

    def _signed_in(self) -> bool:
        return not self.private or bool(self._token)

    def use_dir(self) -> None:
        self.plugins_dir = _plugins_dir(private=self.private)
        self._fill()

    def _profile(self) -> str:
        return str(self.browser.root.setting.c60_selected_profile.get() or "default")

    def _run(self, fn, done) -> None:
        self._task_done = done

        def worker():
            try:
                result = fn()
                error = None
            except Exception as exc:
                log.exception("Plugin task failed")
                result, error = None, exc
            if not self._closed:
                self._signal.result.emit(result, error)

        threading.Thread(target=worker, daemon=True).start()

    def _on_result(self, result, error) -> None:
        if self._closed:
            return
        done = self._task_done
        self._task_done = None
        if done is not None:
            done(result, None if error is None else str(error))

    def refresh(self) -> None:
        if not self._signed_in():
            self._registry = []
            self._fill()
            self.status.setText("Sign in to browse member plugins.")
            return
        self._set_busy(True, "Fetching registry…")
        private = self.private
        token = self._token or None

        def task():
            return _ops()[0].fetch_registry(private=private, token=token)

        def worker():
            try:
                rows = task()
                err = ""
            except Exception as exc:
                rows, err = [], str(exc)
            if not self._closed:
                self._signal.ready.emit(rows, err)

        threading.Thread(target=worker, daemon=True).start()

    def _on_registry(self, rows, err: str) -> None:
        if self._closed:
            return
        self._registry = list(rows or [])
        self._set_busy(False, err or "Ready")
        self._fill()
        if self._registry:
            self._check_updates()

    def _records(self) -> list[dict]:
        ops = _ops()[0]
        from avlite.c60_apps.c68_paths import PluginPaths
        from avlite.c60_apps.c69_settings import AppSettings

        installed = {item["name"]: item for item in ops.list_installed(self.plugins_dir)}
        registered = dict(AppSettings.c62_community_plugins or {})
        seen: set[str] = set()
        records = []
        for entry in self._registry:
            name = str(entry.get("name") or "")
            if not name:
                continue
            seen.add(name)
            records.append(self._record(name, entry, installed.pop(name, None), registered.get(name), ops, PluginPaths))
        for name, inst in installed.items():
            seen.add(name)
            records.append(self._record(name, None, inst, registered.get(name), ops, PluginPaths))
        for name, stored in registered.items():
            if name in seen:
                continue
            records.append(self._record(name, None, None, stored, ops, PluginPaths))
        return records

    def _record(self, name, entry, inst, stored, ops, PluginPaths) -> dict:
        load = PluginPaths.load_path(name, stored) if stored is not None else None
        if load is None and inst is not None:
            load = inst["path"]
        if entry is None:
            status = "Active \u2713" if stored is not None else "Installed"
            path = PluginPaths.format_display(load) if load is not None else ""
            return {
                "name": name, "entry": None, "status": status, "path": path,
                "load": load, "label": name, "category": "", "author": "", "version": "",
            }
        if stored is not None and load is not None:
            status = "Active \u2713"
        elif inst is not None:
            status = "Installed"
        else:
            status = "Available"
        ok, _current, _required = ops.meets_min_avlite(entry)
        if not ok and status == "Available":
            status = "Incompatible"
        if load is not None:
            path = PluginPaths.format_display(load)
        else:
            path = str(entry.get("repository") or "")
        return {
            "name": name,
            "entry": entry,
            "status": status,
            "path": path,
            "load": load,
            "label": _display_name(entry, name),
            "category": _category(entry.get("category", "")),
            "author": str(entry.get("author") or ""),
            "version": str(entry.get("version") or ""),
        }

    def _visible(self, status: str) -> bool:
        if self.only_active.isChecked():
            return status.startswith("Active")
        if self.only_installed.isChecked():
            return status.startswith("Installed") or status.startswith("Active")
        return True

    def _fill(self, *_args) -> None:
        QtWidgets = self.browser.QtWidgets
        QtCore = self.browser.QtCore
        records = [row for row in self._records() if self._visible(row["status"])]
        self._rows = {row["name"]: row for row in records}
        self.table.setRowCount(len(records))
        for index, row in enumerate(records):
            values = (
                row["label"], row["category"], row["author"], row["version"],
                row["status"],
                _UPDATE_LABELS.get(self._updates.get(row["name"], ""), "Checking\u2026" if row["load"] else ""),
                row["path"],
            )
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if col == 0:
                    item.setData(QtCore.Qt.UserRole, row["name"])
                if row["status"] == "Incompatible":
                    item.setForeground(QtCore.Qt.gray)
                self.table.setItem(index, col, item)
        self._update_buttons()

    def _selected(self) -> dict | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        return self._rows.get(str(item.data(self.browser.QtCore.Qt.UserRole) or ""))

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        if message:
            self.status.setText(message)
        self._update_buttons()

    def _update_buttons(self) -> None:
        ops, _client, _err, can_add = _ops()
        row = self._selected()
        status = "" if row is None else row["status"]
        installed = status.startswith("Installed") or status.startswith("Active")
        available = status == "Available"
        enabled = self._signed_in() and not self._busy
        from avlite.c60_apps.c69_settings import AppSettings

        in_profile = row is not None and row["name"] in dict(AppSettings.c62_community_plugins or {})
        has_update = installed and row is not None and self._updates.get(row["name"]) == "update-available"
        has_any = any(value == "update-available" for value in self._updates.values())
        has_github = False
        if row is not None:
            has_github = ops.get_plugin_repository_url(row["entry"], row["load"]) is not None
        self._buttons["Install"].setEnabled(available and enabled)
        self._buttons["Add to Profile"].setEnabled(
            can_add(installed=installed, in_profile=in_profile) and enabled
        )
        self._buttons["Uninstall"].setEnabled(installed and enabled)
        self._buttons["Update"].setEnabled(has_update and enabled)
        self._buttons["Update All"].setEnabled(has_any and enabled)
        self._buttons["Open on GitHub"].setEnabled(has_github and enabled)
        self._buttons["Refresh"].setEnabled(enabled)
        self._buttons["Details"].setEnabled(row is not None)
        self._buttons["Open Folder"].setEnabled(True)

    def _install(self) -> None:
        row = self._selected()
        if row is None or row["status"] != "Available" or row["entry"] is None:
            return
        ops = _ops()[0]
        entry = row["entry"]
        name = row["name"]
        ok, current, required = ops.meets_min_avlite(entry)
        if not ok:
            text = required if required and "ROS" in required else (
                f"'{name}' requires AVLite >= {required} (current {current})."
            )
            self.browser.QtWidgets.QMessageBox.critical(self.widget, "Cannot install", text)
            return
        self._set_busy(True, f"Installing {name}…")
        profile = self._profile()
        token = self._token or None
        plugins_dir = self.plugins_dir
        private = self.private

        def task():
            path = ops.install_plugin(entry, plugins_dir, token=token)
            ops.register_in_profile(name, path, profile=profile, private=private)
            return path

        def done(path, err):
            if err:
                self._set_busy(False, f"Install failed: {err}")
                self.browser.QtWidgets.QMessageBox.critical(self.widget, "Install failed", err)
                return
            self._requirements(name, Path(path), entry)
            notes = ops.dependency_notes(entry)
            if notes:
                self.browser.QtWidgets.QMessageBox.information(
                    self.widget, "Dependency notes", f"'{name}'\n\nDependency notes:\n{notes}",
                )
            self._set_busy(False, f"Installed {name}")
            self._fill()
            self.browser.notify()

        self._run(task, done)

    def _requirements(self, name: str, plugin_path: Path, entry: dict | None) -> None:
        ops = _ops()[0]
        req = ops.requirements_path(plugin_path, entry)
        if req.is_file():
            missing, mismatched = ops.check_requirements(req)
            box = self.browser.QtWidgets.QMessageBox
            if mismatched:
                box.warning(
                    self.widget, "Dependency version mismatch",
                    f"'{name}' requires:\n  " + "\n  ".join(mismatched),
                )
            if missing and box.question(
                self.widget, "Install missing dependencies?",
                f"'{name}' needs:\n  " + "\n  ".join(missing) + "\n\nInstall them into the current Python environment?",
            ) == box.Yes:
                try:
                    ops.pip_install(req, cwd=plugin_path)
                except Exception as exc:
                    box.critical(self.widget, "pip install failed", str(exc))
        src = plugin_path / f"{name}.yaml"
        if not src.is_file():
            return
        box = self.browser.QtWidgets.QMessageBox
        if box.question(
            self.widget, "Add recommended profile?",
            f"'{name}' includes a recommended settings profile.\n\nAdd it as '{name}'?",
        ) != box.Yes:
            return
        from avlite.c60_apps.c65_setting_utils import import_profile, load_setting, profile_file_path
        from avlite.c60_apps.c69_settings import AppSettings

        dest = profile_file_path(name, for_write=True)
        overwrite = False
        if os.path.exists(dest):
            if box.question(
                self.widget, "Overwrite existing profile?",
                f"Profile '{name}' already exists.\n\nOverwrite it?",
            ) != box.Yes:
                return
            overwrite = True
        current = self._profile()
        try:
            import_profile(src, profile=name, overwrite=overwrite)
            ops.register_in_profile(name, plugin_path, profile=name, private=self.private)
        except Exception as exc:
            box.critical(self.widget, "Import profile failed", str(exc))
        finally:
            load_setting(AppSettings, profile=current)
        from avlite.c60_apps.c65_setting_utils import list_profiles

        self.browser.root.setting.profile_list = list_profiles(AppSettings)

    def _add(self) -> None:
        row = self._selected()
        if row is None or row["load"] is None:
            return
        try:
            _ops()[0].register_in_profile(
                row["name"], Path(row["load"]), profile=self._profile(), private=self.private,
            )
        except Exception as exc:
            self.browser.QtWidgets.QMessageBox.critical(self.widget, "Add to profile failed", str(exc))
            return
        self.status.setText(f"Added {row['name']} to profile")
        self._fill()
        self.browser.notify()

    def _uninstall(self) -> None:
        row = self._selected()
        if row is None or not (row["status"].startswith("Installed") or row["status"].startswith("Active")):
            return
        box = self.browser.QtWidgets.QMessageBox
        name = row["name"]
        if box.question(
            self.widget, "Uninstall plugin",
            f"Remove '{name}' from the profile and delete its files?",
        ) != box.Yes:
            return
        from avlite.c60_apps.c65_setting_utils import dev_mode_uninstall_warning

        warning = dev_mode_uninstall_warning(self.plugins_dir, name)
        if warning and box.question(self.widget, "Uninstall plugin", warning + "\n\nContinue uninstall?") != box.Yes:
            return
        ops = _ops()[0]
        profile = self._profile()
        plugins_dir = self.plugins_dir
        self._set_busy(True, f"Uninstalling {name}…")

        def task():
            ops.unregister_from_profile(name, profile=profile)
            ops.uninstall_plugin(name, plugins_dir)
            return name

        def done(_result, err):
            if err:
                self._set_busy(False, f"Uninstall failed: {err}")
                box.critical(self.widget, "Uninstall failed", err)
                return
            self._set_busy(False, f"Uninstalled {name}")
            self._fill()
            self.browser.notify()

        self._run(task, done)

    def _update(self) -> None:
        row = self._selected()
        if row is None or self._updates.get(row["name"]) != "update-available":
            return
        self._update_one(row["name"])

    def _update_one(self, name: str) -> None:
        ops = _ops()[0]
        installed = {item["name"]: item for item in ops.list_installed(self.plugins_dir)}
        inst = installed.get(name)
        if inst is None:
            return
        entry = next((item for item in self._registry if item.get("name") == name), None)
        version = str(entry.get("version") or "latest") if entry else "latest"
        token = self._token or None
        path = Path(inst["path"])
        self._set_busy(True, f"Updating {name}…")

        def task():
            ops.update_plugin(path, version, token=token)
            return path

        def done(result, err):
            if err:
                self._set_busy(False, f"Update failed for {name}: {err}")
                self.browser.QtWidgets.QMessageBox.critical(self.widget, "Update failed", err)
                return
            self._updates.pop(name, None)
            self._requirements(name, Path(result), entry)
            self._set_busy(False, f"Updated {name}")
            self._fill()
            self.browser.notify()

        self._run(task, done)

    def _update_all(self) -> None:
        names = [name for name, state in self._updates.items() if state == "update-available"]
        if not names:
            return
        ops = _ops()[0]
        installed = {item["name"]: item for item in ops.list_installed(self.plugins_dir)}
        token = self._token or None
        self._set_busy(True, "Updating plugins…")

        def task():
            done = []
            for name in names:
                inst = installed.get(name)
                if inst is None:
                    continue
                entry = next((item for item in self._registry if item.get("name") == name), None)
                version = str(entry.get("version") or "latest") if entry else "latest"
                ops.update_plugin(Path(inst["path"]), version, token=token)
                done.append(name)
            return done

        def finished(done_names, err):
            if err:
                self._set_busy(False, f"Update failed: {err}")
                self.browser.QtWidgets.QMessageBox.critical(self.widget, "Update failed", err)
                return
            for name in done_names or []:
                self._updates.pop(name, None)
            self._set_busy(False, f"Updated {len(done_names or [])} plugin(s).")
            self._fill()
            self.browser.notify()

        self._run(task, finished)

    def _github(self) -> None:
        row = self._selected()
        if row is None:
            return
        url = _ops()[0].get_plugin_repository_url(row["entry"], row["load"])
        if url:
            webbrowser.open(url)

    def _folder(self) -> None:
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(self.plugins_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.plugins_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self.plugins_dir)])
        except Exception as exc:
            self.status.setText(f"Could not open folder: {exc}")

    def _details(self) -> None:
        row = self._selected()
        if row is None:
            return
        QtWidgets = self.browser.QtWidgets
        dlg = QtWidgets.QDialog(self.widget)
        dlg.setWindowTitle(row["label"])
        dlg.resize(700, 520)
        layout = QtWidgets.QVBoxLayout(dlg)
        entry = row["entry"] or {}
        ops = _ops()[0]
        lines = [
            f"Author: {entry.get('author') or '—'}",
            f"Version: {entry.get('version') or '—'}",
            f"Description: {entry.get('description') or '—'}",
            f"Website: {entry.get('site_url') or '—'}",
            f"Status: {row['status']}",
            f"Path: {row['path'] or '—'}",
        ]
        if entry:
            ok, current, required = ops.meets_min_avlite(entry)
            if not ok:
                text = required if required and "ROS" in required else f"Requires AVLite >= {required} (current {current})"
                lines.append(f"Compatibility: {text}")
        meta = QtWidgets.QLabel("\n".join(lines))
        meta.setWordWrap(True)
        layout.addWidget(meta)
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setPlainText("Loading README…")
        layout.addWidget(view, 1)
        token = self._token or None
        signal = _bridge(self.browser.QtCore)(dlg)

        def show(body: str) -> None:
            if hasattr(view, "setMarkdown"):
                view.setMarkdown(body)
            else:
                view.setPlainText(body)

        signal.readme.connect(show)

        def load():
            try:
                body = ops.load_plugin_readme(row["name"], row["entry"], row["load"], token=token)
            except Exception as exc:
                body = str(exc)
            signal.readme.emit(body)

        threading.Thread(target=load, daemon=True).start()
        buttons = QtWidgets.QHBoxLayout()
        site = ops.site_url(entry)
        if site:
            web = QtWidgets.QPushButton("Open Website")
            web.clicked.connect(lambda: webbrowser.open(site))
            buttons.addWidget(web)
        repo = ops.get_plugin_repository_url(row["entry"], row["load"])
        if repo:
            github = QtWidgets.QPushButton("Open on GitHub")
            github.clicked.connect(lambda: webbrowser.open(repo))
            buttons.addWidget(github)
        for label, slot in (
            ("Install", self._install),
            ("Add to Profile", self._add),
            ("Uninstall", self._uninstall),
            ("Update", self._update),
        ):
            btn = QtWidgets.QPushButton(label)
            btn.setEnabled(self._buttons[label].isEnabled())
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dlg.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        dlg.exec_()

    def _check_updates(self) -> None:
        ops = _ops()[0]
        token = self._token or None
        pending = {
            name: (row["load"], row["entry"])
            for name, row in self._rows.items()
            if row["load"] is not None
        }

        def worker():
            found = {}
            for name, (path, entry) in pending.items():
                try:
                    found[name] = ops.check_plugin_update(Path(path), entry, token=token)
                except Exception:
                    found[name] = "unknown"
            if not self._closed:
                self._signal.updates.emit(found)

        threading.Thread(target=worker, daemon=True).start()

    def _on_updates(self, found) -> None:
        if self._closed:
            return
        self._updates.update(dict(found or {}))
        self._fill()

    def _sign_in(self) -> None:
        client = _ops()[1]
        QtWidgets = self.browser.QtWidgets
        dlg = QtWidgets.QDialog(self.widget)
        dlg.setWindowTitle("Sign in with GitHub")
        dlg.resize(460, 200)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addWidget(QtWidgets.QLabel("1. Open the GitHub authorization page"))
        uri = QtWidgets.QLabel("Starting…")
        uri.setWordWrap(True)
        uri.setObjectName("githubUri")
        layout.addWidget(uri)
        layout.addWidget(QtWidgets.QLabel("2. Enter this code:"))
        code = QtWidgets.QLabel("…")
        code.setObjectName("githubCode")
        layout.addWidget(code)
        state = QtWidgets.QLabel("Waiting for authorization…")
        layout.addWidget(state)
        buttons = QtWidgets.QHBoxLayout()
        open_btn = QtWidgets.QPushButton("Open in browser")
        open_btn.setEnabled(False)
        copy_btn = QtWidgets.QPushButton("Copy")
        copy_btn.setEnabled(False)
        cancel = QtWidgets.QPushButton("Cancel")
        buttons.addWidget(open_btn)
        buttons.addWidget(copy_btn)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        cancelled = {"value": False}

        def cancel_flow():
            cancelled["value"] = True
            dlg.reject()

        cancel.clicked.connect(cancel_flow)
        holder = {"uri": ""}

        def open_page():
            if holder["uri"]:
                webbrowser.open(holder["uri"])

        open_btn.clicked.connect(open_page)

        def copy_code():
            QtWidgets.QApplication.clipboard().setText(code.text())
            state.setText("Code copied to clipboard.")

        copy_btn.clicked.connect(copy_code)
        signal = _bridge(self.browser.QtCore)(dlg)

        def on_code(user_code: str, verification: str):
            code.setText(user_code)
            uri.setText(verification)
            holder["uri"] = verification
            open_btn.setEnabled(bool(verification))
            copy_btn.setEnabled(bool(user_code))

        def on_auth(token: str, login: str):
            if cancelled["value"] or self._closed:
                return
            self._token = token
            self._login = login
            self._set_auth_visible()
            dlg.accept()
            self.refresh()

        def on_fail(message: str):
            if not cancelled["value"]:
                state.setText(message)

        signal.code.connect(on_code)
        signal.authed.connect(on_auth)
        signal.failed.connect(on_fail)

        def worker():
            try:
                flow = client._start_device_flow()
                signal.code.emit(str(flow.get("user_code") or ""), str(flow.get("verification_uri") or ""))
                token = client._poll_device_flow(
                    str(flow.get("device_code") or ""),
                    int(flow.get("interval") or 5),
                    int(flow.get("expires_in") or 600),
                )
                login = client._github_user(token) or ""
                client._save_github_token(token, login)
                signal.authed.emit(token, login)
            except Exception as exc:
                signal.failed.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()
        dlg.exec_()

    def _sign_out(self) -> None:
        _ops()[1]._clear_github_token()
        self._token = ""
        self._login = ""
        self._registry = []
        self._set_auth_visible()
        self._fill()
        self.status.setText("Signed out.")
