"""Load this checkout through AVLite's community plugin importer."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
os.environ.setdefault("AVLITE_QT_NO_OPENGL", "1")

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PLUGIN_NAME = "avlite-QT5"


def ensure_plugin_loaded() -> None:
    from avlite.c60_apps.c63_plugins import import_plugin_modules

    import_plugin_modules(str(PLUGIN_DIR), pkg_name=PLUGIN_NAME)


@pytest.fixture(scope="session", autouse=True)
def _load_plugin():
    ensure_plugin_loaded()


def pytest_runtest_setup(item):
    import sys

    if "avlite.plugins.avlite_QT5" not in sys.modules:
        ensure_plugin_loaded()


@pytest.fixture(scope="module")
def qapp():
    ensure_plugin_loaded()
    from avlite.plugins.avlite_QT5.qt_binding import configure_qt_env, import_qt

    configure_qt_env(offscreen=os.environ.get("QT_QPA_PLATFORM") == "offscreen")
    _c, _g, QtWidgets = import_qt()
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from avlite.c60_apps.c62_factory import load_stack_settings
    from avlite.c60_apps.c65_setting_utils import load_setting
    from avlite.c60_apps.c69_settings import AppSettings
    from avlite.plugins.avlite_QT5.main_window import VisualizerQtWindow

    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    load_setting(AppSettings, profile="default")
    AppSettings.c62_load_plugins = True
    AppSettings.c62_community_plugins = {PLUGIN_NAME: str(PLUGIN_DIR)}
    load_stack_settings(profile="default", load_plugins=True)
    ensure_plugin_loaded()

    win = VisualizerQtWindow()
    yield win
    win.stack_panels.execution.stop_exec()
    win.log_panel.shutdown()
    win._ui_timer.stop()
    win.win.close()
    ensure_plugin_loaded()
