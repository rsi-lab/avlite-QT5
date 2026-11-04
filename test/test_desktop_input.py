"""Real widget input: QTest clicks and visible toolbar / log controls.

These run against a shown window. On Xvfb use the xcb platform; offscreen
still delivers QTest events to the widgets.
"""

from __future__ import annotations

import logging

import pytest

from conftest import ensure_plugin_loaded


def _show(window, qapp):
    from PyQt5 import QtWidgets

    window.win.resize(1400, 1000)
    window.show()
    qapp.processEvents()
    QtWidgets.QApplication.processEvents()
    window.update_ui()
    qapp.processEvents()


def _click(widget, button, pos):
    from PyQt5 import QtTest

    QtTest.QTest.mouseClick(widget, button, pos=pos)


def _view_point(plot, fx: float, fy: float):
    """Pixel on the plot viewport that maps to the given data coordinate."""
    from PyQt5 import QtCore

    import pyqtgraph as pg

    scene = plot.getViewBox().mapViewToScene(pg.Point(fx, fy))
    pix = plot.mapFromScene(scene)
    return QtCore.QPoint(int(pix.x()), int(pix.y()))


@pytest.mark.requires_data
def test_config_bar_and_log_are_visible(window, qapp):
    _show(window, qapp)
    bar = window.config_bar
    assert bar.objectName() == "configBar"
    assert bar.isVisible()
    assert bar.height() > 20
    for name in (
        "chkShortcuts", "chkDark", "chkGlobal", "chkLocalFrenet", "chkLocalGlobal",
        "profileCombo", "btnSaveSettings", "btnResetSettings", "btnReloadStack",
        "btnPlugins", "btnSettings", "plotStatus",
    ):
        widget = bar.findChild(window.QtWidgets.QWidget, name)
        assert widget is not None, name
        assert widget.isVisible(), name

    log_panel = window.log_panel
    assert log_panel.isVisible()
    assert log_panel.height() >= 140
    assert log_panel.view.isVisible()
    assert log_panel.view.height() >= 80
    for key in ("core", "plugins", "perceive", "plan", "control", "execute", "vis", "common"):
        box = log_panel.findChild(window.QtWidgets.QCheckBox, f"logFilter_{key}")
        assert box is not None and box.isVisible()

    from PyQt5 import QtCore, QtTest

    text = log_panel.view.toPlainText()
    assert "Log initialized." in text
    assert "Loaded settings from profile" in text
    info = window.win.findChild(QtCore.QObject, "logLevel_INFO")
    assert info is not None and info.isVisible()
    QtTest.QTest.mouseClick(info, QtCore.Qt.LeftButton)
    logging.getLogger("avlite.plugins.avlite_QT5.desktop").info("desktop-log-marker")
    QtTest.QTest.qWait(80)
    window.log_panel._flush()
    text = log_panel.view.toPlainText()
    assert "desktop-log-marker" in text
    assert "Log level updated to: INFO" in text


@pytest.mark.requires_data
def test_right_click_sets_start_then_goal(window, qapp):
    from PyQt5 import QtCore

    _show(window, qapp)
    plot = window.global_plot
    assert plot.getViewBox().menuEnabled() is False
    ego = window.exec.ego_state
    first = _view_point(plot, ego.x + 15.0, ego.y + 8.0)
    second = _view_point(plot, ego.x + 40.0, ego.y - 6.0)
    _click(plot.viewport(), QtCore.Qt.RightButton, first)
    qapp.processEvents()
    xs, ys = plot.start_marker.getData()
    assert xs is not None and len(xs) == 1
    assert plot._start_point is not None
    _click(plot.viewport(), QtCore.Qt.RightButton, second)
    qapp.processEvents()
    assert plot._start_point is None
    gxs, gys = plot.goal_marker.getData()
    assert gxs is not None and len(gxs) == 1


@pytest.mark.requires_data
def test_left_click_teleports_ego(window, qapp):
    from PyQt5 import QtCore

    _show(window, qapp)
    plot = window.global_plot
    ego = window.exec.ego_state
    target_x, target_y = ego.x + 12.0, ego.y - 9.0
    _click(plot.viewport(), QtCore.Qt.LeftButton, _view_point(plot, target_x, target_y))
    qapp.processEvents()
    moved = window.exec.ego_state
    assert abs(moved.x - target_x) < 1.5
    assert abs(moved.y - target_y) < 1.5


@pytest.mark.requires_data
def test_local_right_click_spawns_npc(window, qapp):
    from PyQt5 import QtCore

    _show(window, qapp)
    pm = getattr(window.exec.world, "pm", None) or window.exec.pm
    before = len(list(getattr(pm, "agent_vehicles", []) or []))
    plot = window.local_plot.xy_plot
    ego = window.exec.ego_state
    _click(plot.viewport(), QtCore.Qt.RightButton, _view_point(plot, ego.x + 6.0, ego.y + 4.0))
    qapp.processEvents()
    after = len(list(getattr(pm, "agent_vehicles", []) or []))
    assert after == before + 1
    window.local_plot.redraw()
    xs, _ys = window.local_plot.xy_agents[0].getData()
    assert xs is not None and len(xs) > 2


@pytest.mark.requires_data
def test_ctrl_drag_pans_global_plot(window, qapp):
    from PyQt5 import QtCore, QtGui, QtWidgets

    _show(window, qapp)
    plot = window.global_plot
    vp = plot.viewport()
    origin = QtCore.QPoint(max(vp.width() // 2, 20), max(vp.height() // 2, 20))
    end = QtCore.QPoint(origin.x() + 50, origin.y() + 20)
    before = (float(plot._center_delta[0]), float(plot._center_delta[1]))

    def _send(kind, button, buttons, pos):
        ev = QtGui.QMouseEvent(
            kind, pos, vp.mapToGlobal(pos), button, buttons, QtCore.Qt.ControlModifier,
        )
        QtWidgets.QApplication.sendEvent(vp, ev)

    _send(QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, origin)
    _send(QtCore.QEvent.MouseMove, QtCore.Qt.NoButton, QtCore.Qt.LeftButton, end)
    _send(QtCore.QEvent.MouseButtonRelease, QtCore.Qt.LeftButton, QtCore.Qt.NoButton, end)
    qapp.processEvents()
    after = (float(plot._center_delta[0]), float(plot._center_delta[1]))
    assert after != before


@pytest.mark.requires_data
def test_wheel_changes_zoom(window, qapp):
    from PyQt5 import QtCore, QtGui

    _show(window, qapp)
    plot = window.global_plot
    before = float(window.setting.p66_global_zoom)
    vp = plot.viewport()
    pos = QtCore.QPoint(vp.width() // 2, vp.height() // 2)
    event = QtGui.QWheelEvent(
        pos, vp.mapToGlobal(pos), QtCore.QPoint(0, 0), QtCore.QPoint(0, 120),
        120, QtCore.Qt.Vertical, QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
    )
    qapp.sendEvent(vp, event)
    qapp.processEvents()
    assert float(window.setting.p66_global_zoom) < before


@pytest.mark.requires_data
def test_shift_s_shows_shortcut_table(window, qapp):
    from PyQt5 import QtCore, QtTest

    _show(window, qapp)
    assert window.shortcut_help.isHidden()
    window.global_plot.viewport().setFocus()
    qapp.processEvents()
    QtTest.QTest.keyClick(window.global_plot.viewport(), QtCore.Qt.Key_S, QtCore.Qt.ShiftModifier)
    qapp.processEvents()
    assert not window.shortcut_help.isHidden()
    assert window.shortcut_help.isVisible()
    assert window.win.findChild(QtCore.QObject, "configBar").isVisible()
    assert window.win.findChild(QtCore.QObject, "logPanel").isVisible()
