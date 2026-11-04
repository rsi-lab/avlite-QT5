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
