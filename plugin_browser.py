"""Built-in and community plugin browser. Qt is imported only when the dialog opens.

The Tk browser also drives a GitHub device-flow installer (p63_plugins_app.py).
That network install flow is not part of this dialog.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def open_plugin_browser(root) -> None:
    from .qt_binding import import_qt

    QtCore, QtGui, QtWidgets = import_qt()
    dlg = _Browser(root, QtCore, QtWidgets)
    dlg.exec_()


class _Browser:
    def __init__(self, root, QtCore, QtWidgets, host=None):
        self.root = root
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore
        self._owns_window = host is None
        if host is None:
            self.win = QtWidgets.QDialog(root.win)
            self.win.setWindowTitle("Plugins")
            self.win.setObjectName("pluginBrowser")
            screen = QtWidgets.QApplication.primaryScreen()
            avail = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1280, 800)
            self.win.resize(min(860, max(640, avail.width() - 96)), min(640, max(420, avail.height() - 96)))
            self.win.setMinimumSize(min(520, max(360, avail.width() - 80)), min(360, max(280, avail.height() - 80)))
        else:
            self.win = host
        layout = QtWidgets.QVBoxLayout(self.win)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        splitter = QtWidgets.QSplitter()
        self.builtin = QtWidgets.QListWidget()
        self.builtin.setObjectName("builtinPlugins")
        self.builtin.setMinimumHeight(120)
        self.community = QtWidgets.QListWidget()
        self.community.setObjectName("communityPlugins")
        self.community.setMinimumHeight(120)
        left = QtWidgets.QWidget()
        left_l = QtWidgets.QVBoxLayout(left)
        left_l.addWidget(QtWidgets.QLabel("Built-in plugins"))
        left_l.addWidget(self.builtin)
        right = QtWidgets.QWidget()
        right_l = QtWidgets.QVBoxLayout(right)
        right_l.addWidget(QtWidgets.QLabel("Community plugins"))
        right_l.addWidget(self.community)
        splitter.addWidget(left)
        splitter.addWidget(right)
        layout.addWidget(splitter, 1)

        add = QtWidgets.QHBoxLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Package name")
        self.name_edit.setObjectName("pluginName")
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Package directory")
        self.path_edit.setObjectName("pluginPath")
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self._browse)
        add_btn = QtWidgets.QPushButton("Add community")
        add_btn.setObjectName("addCommunityPlugin")
        add_btn.clicked.connect(self._add)
        add.addWidget(self.name_edit)
        add.addWidget(self.path_edit, 1)
        add.addWidget(browse)
        add.addWidget(add_btn)
        layout.addLayout(add)

        buttons = QtWidgets.QHBoxLayout()
        actions = [
            ("Details", self._details, "pluginDetails"),
            ("Remove built-in", self._remove_builtin, "removeBuiltin"),
            ("Remove community", self._remove_community, "removeCommunity"),
            ("Reset built-in", self._reset_builtin, "resetBuiltin"),
            ("Reset community", self._reset_community, "resetCommunity"),
        ]
        if self._owns_window:
            actions.append(("Close", self.win.accept, "closePlugins"))
        for label, slot, name in actions:
            btn = QtWidgets.QPushButton(label)
            btn.setObjectName(name)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        layout.addLayout(buttons)
        self._reload()

    def exec_(self):
        return self.win.exec_()

    def _reload(self) -> None:
        from avlite.c60_apps.c69_settings import AppSettings

        self.builtin.clear()
        for name in list(AppSettings.c62_default_plugins or []):
            self.builtin.addItem(str(name))
        self.community.clear()
        for name, path in dict(AppSettings.c62_community_plugins or {}).items():
            item = self.QtWidgets.QListWidgetItem(f"{name}  —  {path}")
            item.setData(self.QtCore.Qt.UserRole, str(name))
            self.community.addItem(item)

    def _selected_builtin(self) -> str:
        item = self.builtin.currentItem()
        return item.text() if item is not None else ""

    def _selected_community(self) -> str:
        item = self.community.currentItem()
        if item is None:
            return ""
        return str(item.data(self.QtCore.Qt.UserRole) or "")

    def _browse(self) -> None:
        path = self.QtWidgets.QFileDialog.getExistingDirectory(self.win, "Plugin directory")
        if path:
            self.path_edit.setText(path)

    def _add(self) -> None:
        from avlite.c60_apps.c68_paths import PluginPaths
        from avlite.c60_apps.c69_settings import AppSettings

        name = self.name_edit.text().strip()
        directory = self.path_edit.text().strip()
        if not name:
            return
        plugins = dict(AppSettings.c62_community_plugins or {})
        plugins[name] = PluginPaths.normalize_stored(name, directory)
        AppSettings.c62_community_plugins = plugins
        log.info("Added community plugin %s", name)
        self._reload()

    def _remove_community(self) -> None:
        from avlite.c60_apps.c69_settings import AppSettings

        name = self._selected_community()
        if not name:
            return
        if name == getattr(self.root, "hosting_plugin_name", ""):
            self.QtWidgets.QMessageBox.warning(self.win, "Remove", "The running visualizer plugin stays loaded.")
            return
        plugins = dict(AppSettings.c62_community_plugins or {})
        plugins.pop(name, None)
        AppSettings.c62_community_plugins = plugins
        log.info("Removed community plugin %s", name)
        self._reload()

    def _remove_builtin(self) -> None:
        from avlite.c60_apps.c63_plugins import unregister_plugin_package
        from avlite.c60_apps.c65_setting_utils import can_remove_builtin_plugin
        from avlite.c60_apps.c69_settings import AppSettings

        name = self._selected_builtin()
        if not name:
            return
        blocked = can_remove_builtin_plugin(name, getattr(self.root, "hosting_plugin_name", ""))
        if blocked:
            self.QtWidgets.QMessageBox.warning(self.win, "Remove Plugin", blocked)
            return
        plugins = list(AppSettings.c62_default_plugins or [])
        if name in plugins:
            plugins.remove(name)
            AppSettings.c62_default_plugins = plugins
            try:
                unregister_plugin_package(name)
            except Exception:
                log.debug("unregister failed for %s", name, exc_info=True)
            self.root.reload_stack(reload_code=False)
        self._reload()

    def _reset_builtin(self) -> None:
        from avlite.c60_apps.c63_plugins import list_plugins
        from avlite.c60_apps.c69_settings import AppSettings

        AppSettings.c62_default_plugins = list_plugins()
        self.root.reload_stack(reload_code=False)
        self._reload()

    def _reset_community(self) -> None:
        from avlite.c60_apps.c68_paths import PluginPaths
        from avlite.c60_apps.c69_settings import AppSettings

        AppSettings.c62_community_plugins = PluginPaths.installed_map()
        self.root._pin_self_as_community_plugin()
        self._reload()

    def _details(self) -> None:
        from avlite.c60_apps.c65_setting_utils import profile_file_path
        from avlite.c60_apps.c68_paths import PluginPaths
        from avlite.c60_apps.c69_settings import AppSettings

        builtin = self._selected_builtin()
        community = self._selected_community()
        profile = str(self.root.setting.c60_selected_profile.get() or "default")
        lines = []
        if builtin:
            source = PluginPaths.builtin_dir() / builtin
            lines.append(f"Built-in: {builtin}")
            lines.append(f"Source: {source}")
            lines.append(f"Settings: {profile_file_path(profile, for_write=False)} (plugins.{builtin})")
        if community:
            stored = dict(AppSettings.c62_community_plugins or {}).get(community, community)
            path = PluginPaths.load_path(community, stored)
            lines.append(f"Community: {community}")
            lines.append(f"Path: {path}")
        if not lines:
            lines.append("Select a plugin.")
        self.QtWidgets.QMessageBox.information(self.win, "Plugin", "\n".join(lines))
