"""Widget geometry must stay put across UI refresh ticks.

Running BasicSim used to resize pyqtgraph axes with the tick labels and
resize proportional-font readouts with the digits. Both changed size hints,
which grew the window and shifted the plot panes. These ticks cover idle
refreshes, ego jumps across digit boundaries, and a running executor.
"""

from __future__ import annotations

import logging

import pytest

from conftest import ensure_plugin_loaded


def _rect(widget):
    geo = widget.geometry()
    return (geo.x(), geo.y(), geo.width(), geo.height())


def _axis_width(plot) -> int:
    return int(round(plot.getAxis("left").width()))


def _span(plot) -> tuple[float, float]:
    (x0, x1), (y0, y1) = plot.getViewBox().viewRange()
    return (round(float(x1 - x0), 2), round(float(y1 - y0), 2))


def _snapshot(window):
    panels = window.stack_panels
    return {
        "window": (window.win.width(), window.win.height()),
        "minimum": (
            window.win.centralWidget().minimumSizeHint().width(),
            window.win.centralWidget().minimumSizeHint().height(),
        ),
        "splitter": _rect(window.plots_splitter),
        "local": _rect(window.local_plot),
        "global": _rect(window.global_plot),
        "xy": _rect(window.local_plot.xy_plot),
        "frenet": _rect(window.local_plot.frenet_plot),
        "config": _rect(window.config_bar),
        "status": _rect(window.config_bar.status),
        "stack": _rect(panels),
        "perception": _rect(panels.perception),
        "planning": _rect(panels.planning),
        "control": _rect(panels.control),
        "execution": _rect(panels.execution),
        "state": _rect(panels.perception.state_lbl),
        "wp": _rect(panels.planning.wp_lbl),
        "stats": _rect(panels.execution.stats_lbl),
        "caps": _rect(panels.execution.cap_label),
        "vel": _rect(panels.control.gauges["vel_cte"]),
        "accel": _rect(panels.control.gauges["accel"]),
        "log": _rect(window.log_panel),
        "log_view": _rect(window.log_panel.view),
        "axis_global": _axis_width(window.global_plot),
        "axis_xy": _axis_width(window.local_plot.xy_plot),
        "axis_frenet": _axis_width(window.local_plot.frenet_plot),
        "span_global": _span(window.global_plot),
        "span_xy": _span(window.local_plot.xy_plot),
        "span_frenet": _span(window.local_plot.frenet_plot),
    }


@pytest.mark.requires_data
def test_geometry_stable_across_refresh_ticks(window, qapp):
    ensure_plugin_loaded()
    from PyQt5 import QtCore

    window._ui_timer.stop()
    window.win.resize(1600, 1000)
    window.show()
    qapp.processEvents()

    # Let the first layout and the axis pin settle, then everything must hold.
    for _ in range(4):
        window.update_ui()
        qapp.processEvents()
    baseline = _snapshot(window)
    assert baseline["log_view"][3] >= 80
    assert window.log_panel.view.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOn
    assert window.log_panel.view.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff
    assert not any(window.global_plot.getViewBox().autoRangeEnabled())

    poses = (
        (5.0, 3.0, 0.0),
        (12.0, -80.0, -0.4),
        (999.0, -999.0, 1.2),
        (9999.0, -9999.0, -2.5),
        (50000.0, 12345.0, 0.1),
        (5.2, 3.1, 0.0),
        (250.0, -710.0, 0.3),
    )
    logger = logging.getLogger("avlite.plugins.avlite_QT5.layout")
    for i, (x, y, theta) in enumerate(poses):
        window.teleport_ego(x, y, theta)
        window.setting.cte_vel.set(float(i * 10))
        window.setting.cte_steer.set(-float(i * 3))
        window.setting.cmd_accel.set(float(i) - 2.0)
        window.setting.cmd_steer.set(-0.25 * i)
        logger.info("layout tick %s at %.1f,%.1f %s", i, x, y, "x" * (8 * (i + 1)))
        window.log_panel._flush()
        window.update_ui()
        qapp.processEvents()
        got = _snapshot(window)
        assert got == baseline, _first_diff(baseline, got)

    for _ in range(6):
        window.stack_panels.execution.step_exec()
        logger.info("layout step %s", "y" * 40)
        window.log_panel._flush()
        qapp.processEvents()
        got = _snapshot(window)
        # The camera center follows the ego; the zoom span and the widgets must not.
        assert got == baseline, _first_diff(baseline, got)


def _first_diff(before: dict, after: dict) -> str:
    lines = []
    for key in before:
        if before[key] != after[key]:
            lines.append(f"{key}: {before[key]} -> {after[key]}")
    return "\n".join(lines)
