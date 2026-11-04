"""Community-plugin import must not initialize Qt.

AVLite's Tk visualizer imports every registered plugin while its loading
screen is up. Importing PyQt or pyqtgraph there switches Matplotlib onto the
Qt backend and can block on a second QApplication.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def test_community_import_does_not_initialize_qt():
    script = textwrap.dedent(
        f"""
        import matplotlib
        matplotlib.use("Agg", force=True)
        backend = matplotlib.get_backend()

        from avlite.c60_apps.c63_plugins import import_plugin_modules
        import_plugin_modules({str(PLUGIN_DIR)!r}, pkg_name="avlite-QT5")

        import sys
        loaded = [name for name in sys.modules if name == "PyQt5" or name.startswith("PyQt5.") or name == "pyqtgraph" or name.startswith("pyqtgraph.")]
        assert not loaded, loaded
        assert matplotlib.get_backend() == backend

        from avlite.plugins.avlite_QT5.plot_widgets import GlobalPlotWidget
        assert "PyQt5" in sys.modules
        assert GlobalPlotWidget.__name__ == "GlobalPlotWidget"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
