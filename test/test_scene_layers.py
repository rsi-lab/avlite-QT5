"""Plot layers, capability checks, schema settings, and the plugin browser."""

from __future__ import annotations

import numpy as np


def test_plan_prediction_lattice_occupancy_and_clusters(window):
    from avlite.c10_perception.c11_perception_model import OccupancyMap, SingleTrajectory

    ego = window.exec.ego_state
    path = np.array(
        [[ego.x, ego.y], [ego.x + 6.0, ego.y + 1.0], [ego.x + 12.0, ego.y + 2.0]],
        dtype=float,
    )
    window.exec.pm.prediction = SingleTrajectory(trajectories={1: path})
    grid = np.zeros((16, 16), dtype=float)
    grid[8, 8] = 1.0
    window.exec.pm.occupancy_map = OccupancyMap(
        grid=grid, origin_x=float(ego.x) - 4.0, origin_y=float(ego.y) - 4.0, resolution=0.5,
    )
    window.exec.pm.detection_clusters = np.array(
        [[ego.x + 1.0, ego.y + 1.0], [ego.x + 2.0, ego.y - 1.0]], dtype=float,
    )
    window.setting.p67_show_occupancy_map.set(True)
    window.setting.p67_show_prediction.set(True)
    window.setting.p66_show_lidar_clusters.set(True)
    window.setting.p66_show_local_lattice.set(True)

    class _Traj:
        path_x = [float(ego.x), float(ego.x) + 4.0]
        path_y = [float(ego.y), float(ego.y) + 1.0]
        path_s_from_parent = [0.0, 4.0]
        path_d_from_parent = [0.0, 0.4]

    class _Edge:
        collision = True
        boundary_violation = False
        local_trajectory = _Traj()

    window.exec.local_planner.lattice_edges = [_Edge()]
    window.setting.p66_show_legend.set(True)
    window.update_ui()

    counts = window.local_plot.layer_point_counts
    assert counts["prediction_xy"] >= 3
    assert counts["occupancy"] == 16 * 16
    assert counts["clusters_xy"] == 2
    assert counts["lattice_xy"] >= 2
    assert not window.local_plot.legend.isHidden()
    assert window.local_plot.xy_occ.isVisible()

    from avlite.c10_perception.c11_perception_model import OccupancyFlow

    flow = np.zeros((10, 12), dtype=float)
    flow[3:7, 4:8] = 0.75
    window.exec.pm.prediction = OccupancyFlow(
        occupancy_flow={1: [flow]},
        origin_x=float(ego.x),
        origin_y=float(ego.y),
        resolution=0.5,
    )
    window.update_ui()
    assert window.local_plot.layer_point_counts["occupancy_flow"] > 0
    assert window.local_plot.xy_flow.isVisible()

    xs, _ys = window.global_plot.global_plan.getData()
    assert xs is not None and len(xs) > 2
    wx, _wy = window.global_plot.g_wp_current.getData()
    assert wx is not None and len(wx) == 1
    hx, _hy = window.global_plot.heading_line.getData()
    assert hx is not None and len(hx) == 2
    assert window.global_plot.global_plan_speed.getData() is not None


def test_capability_checks_filter_world_sensors(window):
    from avlite.c40_execution.c49_settings import ExecutionSettings

    previous = ExecutionSettings.c41_world_capabilities
    try:
        ExecutionSettings.c41_world_capabilities = None
        window.stack_panels.refresh()
        box = window.win.findChild(window.QtWidgets.QCheckBox, "worldCap_LIDAR_2D")
        stack = window.win.findChild(window.QtWidgets.QCheckBox, "stackCap_DETECTION")
        assert box is not None and box.isChecked()
        assert stack is not None and stack.isChecked()
        box.setChecked(False)
        assert ExecutionSettings.c41_world_capabilities is not None
        assert "LIDAR_2D" not in ExecutionSettings.c41_world_capabilities
        assert "DETECTION" in (ExecutionSettings.c41_world_stack_capabilities or [])
        gauge = window.win.findChild(window.QtWidgets.QWidget, "valueGauge_vel_cte")
        assert gauge is not None
        window.setting.cte_vel.set(4.0)
        assert gauge._value == 4.0
        assert window.win.findChild(window.QtWidgets.QWidget, "capabilityPanel") is not None
        assert window.win.findChild(window.QtWidgets.QWidget, "mapFile") is not None
        assert window.win.findChild(window.QtWidgets.QWidget, "contractInfo") is not None
    finally:
        ExecutionSettings.c41_world_capabilities = previous
        ExecutionSettings.c41_world_stack_capabilities = None


def test_control_readouts_keep_their_instruments(window, qapp):
    """CTE bars sit on their numbers. Steer is a wheel and speed is a dial."""
    window._ui_timer.stop()
    window.show()
    qapp.processEvents()
    control = window.stack_panels.control

    def _placed(label_name: str, instrument_name: str):
        label = control.gauges[label_name]
        instrument = window.win.findChild(window.QtWidgets.QWidget, instrument_name)
        assert instrument is not None and instrument.isVisible()
        origin = label.mapTo(control, label.rect().topLeft())
        label_box = window.QtCore.QRect(origin, label.size())
        inst_origin = instrument.mapTo(control, instrument.rect().topLeft())
        inst_box = window.QtCore.QRect(inst_origin, instrument.size())
        return label_box, inst_box

    for label_name, instrument_name in (
        ("vel_cte", "valueGauge_vel_cte"),
        ("pos_cte", "valueGauge_pos_cte"),
    ):
        label_box, bar_box = _placed(label_name, instrument_name)
        assert abs(bar_box.left() - label_box.left()) <= 2, (label_name, label_box, bar_box)
        assert 0 <= bar_box.top() - label_box.bottom() <= 8, (label_name, label_box, bar_box)
        assert bar_box.width() >= label_box.width(), (label_name, label_box, bar_box)

    for label_name, instrument_name in (("accel", "speedGauge"), ("steer", "steerWheel")):
        label_box, inst_box = _placed(label_name, instrument_name)
        assert 0 <= inst_box.top() - label_box.bottom() <= 8, (label_name, label_box, inst_box)
        assert inst_box.left() <= label_box.right() and inst_box.right() >= label_box.left()

    assert window.win.findChild(window.QtWidgets.QWidget, "valueGauge_steer") is None
    assert window.win.findChild(window.QtWidgets.QWidget, "valueGauge_accel") is None
    window.setting.cmd_steer.set(0.35)
    assert window.win.findChild(window.QtWidgets.QWidget, "steerWheel")._steer == 0.35
    window.exec.ego_state.velocity = 12.5
    window.setting.vehicle_state.set("speed-check")
    dial = window.win.findChild(window.QtWidgets.QWidget, "speedGauge")
    assert dial._speed == 12.5
    assert dial.caption() == "45.0 km/h (12.5 m/s)"
    assert "Accel" in control.gauges["accel"].text()
    assert "Steer" in control.gauges["steer"].text()


def test_schema_settings_and_plugin_browser_build(window, qapp):
    from avlite.c10_perception.c19_settings import PerceptionSettings
    from avlite.plugins.avlite_QT5.plugin_browser import _Browser
    from avlite.plugins.avlite_QT5.qt_binding import import_qt
    from avlite.plugins.avlite_QT5.schema_editor import _SchemaDialog

    QtCore, QtGui, QtWidgets = import_qt()
    original = PerceptionSettings.c11_max_agents
    dlg = _SchemaDialog(window, QtCore, QtGui, QtWidgets)
    dlg.win.show()
    qapp.processEvents()
    try:
        assert dlg.win.findChild(QtWidgets.QTabWidget, "schemaTabs") is not None
        field = dlg.win.findChild(QtWidgets.QSpinBox, "field_Perception_c11_max_agents")
        assert field is not None
        assert field.value() == int(original)
        field.setValue(int(original) + 1)
        dlg._apply_editors()
        assert PerceptionSettings.c11_max_agents == int(original) + 1
        assert dlg.win.findChild(QtWidgets.QTableWidget, "shortcutTable") is not None
        assert dlg.win.findChild(QtWidgets.QLineEdit, "profileNote") is not None
    finally:
        PerceptionSettings.c11_max_agents = original
        dlg.win.close()

    browser = _Browser(window, QtCore, QtWidgets)
    browser.win.show()
    qapp.processEvents()
    try:
        assert browser.win.findChild(QtWidgets.QListWidget, "communityPlugins") is not None
        assert browser.community.count() >= 1
        assert browser.win.findChild(QtWidgets.QPushButton, "addCommunityPlugin") is not None
    finally:
        browser.win.close()
