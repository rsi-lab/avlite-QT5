"""Plot layers, capability checks, schema settings, and the plugin browser."""

from __future__ import annotations

import numpy as np


def test_plan_prediction_lattice_occupancy_and_clusters(window):
    from avlite.c10_perception.c11_perception_model import AgentState, OccupancyMap, SingleTrajectory

    ego = window.exec.ego_state
    window.setting.p67_show_local_global_view.set(True)
    path = np.array(
        [[ego.x, ego.y], [ego.x + 6.0, ego.y + 1.0], [ego.x + 12.0, ego.y + 2.0]],
        dtype=float,
    )
    window.exec.pm.prediction = SingleTrajectory(trajectories={1: path})
    window.exec.pm.agent_vehicles = [
        AgentState(x=float(ego.x) + 6.0, y=float(ego.y) + 1.0, velocity=4.0, agent_id=1),
    ]
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
    assert counts["prediction_ellipses"] == 0
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
    """CTE bars sit on their numbers. Speed and steer live on the control plot."""
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

    assert control.findChild(window.QtWidgets.QWidget, "speedGauge") is None
    assert control.findChild(window.QtWidgets.QWidget, "steerWheel") is None
    assert window.win.findChild(window.QtWidgets.QWidget, "valueGauge_steer") is None
    assert window.win.findChild(window.QtWidgets.QWidget, "valueGauge_accel") is None
    window.drive(steer=0.35)
    wheel = window.control_plot.steer_wheel
    dial = window.control_plot.speed_gauge
    assert wheel.steer == 0.35
    window.exec.ego_state.velocity = 12.5
    window.update_ui()
    assert dial.speed == 12.5
    assert dial.caption() == "45.0 km/h"
    assert dial.caption_mps() == "12.5 m/s"
    assert "Accel" in control.gauges["accel"].text()
    assert "Steer" in control.gauges["steer"].text()
    assert window.control_plot.speed_gauge.isVisible()
    assert window.control_plot.steer_wheel.isVisible()


def test_gmm_and_gp_prediction_geometry():
    from avlite.c10_perception.c11_perception_model import AgentState, GMM, GP, State
    from avlite.plugins.avlite_QT5.plot_widgets import _ellipse_steps, _prediction_drawings

    ego = State(x=0.0, y=0.0, theta=0.0)
    ahead = AgentState(x=10.0, y=0.0, velocity=5.0, theta=0.0, agent_id=3)
    steps = np.stack([np.linspace(10.0, 16.0, 20), np.zeros(20)], axis=1)
    gp = GP(means={3: steps}, covariance={3: np.eye(40) * 0.25})
    lines, rings = _prediction_drawings(gp, [ahead], ego)
    assert len(lines) == 1
    assert lines[0][1][:3] == (255, 140, 0)
    assert lines[0][1][3] == 1.0
    assert len(rings) == len(_ellipse_steps(20, 0.1)) * 2
    assert {round(item[1][3], 2) for item in rings} == {0.5, 0.95}

    modes = np.stack([steps, steps + np.array([0.0, 1.0])])
    cov = np.tile(np.eye(2) * 0.25, (20, 1, 1))
    gmm = GMM(
        trajectories={3: modes},
        covariances={3: np.stack([cov, cov])},
        weights={3: np.array([0.25, 0.75])},
    )
    lines, rings = _prediction_drawings(gmm, [ahead], ego)
    assert len(lines) == 2
    assert {round(item[1][3], 2) for item in lines} == {0.25, 0.75}
    assert len(rings) == len(_ellipse_steps(20, 0.1)) * 4

    behind = AgentState(x=-8.0, y=0.0, velocity=5.0, theta=0.0, agent_id=3)
    lines, _rings = _prediction_drawings(gp, [behind], ego)
    assert lines[0][1][:3] == (147, 112, 219)


def test_forecast_alpha_stays_true_on_a_dark_plot(qapp):
    """A translucent stroke must keep Tk's blend. Antialiasing washes it out."""
    from PyQt5 import QtGui
    from avlite.plugins.avlite_QT5.plot_widgets import LocalPlotWidget

    class _Setting:
        p60_bg_color = "#333333"
        p60_fg_color = "white"

    class _Root:
        setting = _Setting()

    widget = LocalPlotWidget(_Root())
    widget.resize(480, 220)
    widget.xy_plot.resize(480, 180)
    widget.update_theme("#333333", "white")
    widget.xy_pred.set_drawings(
        [(np.column_stack((np.linspace(0.0, 20.0, 40), np.zeros(40))), (255, 140, 0, 0.25))],
        [],
    )
    widget.xy_plot.setRange(xRange=(0, 20), yRange=(-4, 4), padding=0)
    widget.show()
    qapp.processEvents()
    image = widget.xy_plot.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB888)
    ptr = image.bits()
    ptr.setsize(image.byteCount())
    raw = np.frombuffer(ptr, np.uint8).reshape(image.height(), image.bytesPerLine())
    rgb = raw[:, : image.width() * 3].reshape(image.height(), image.width(), 3).astype(int)
    # 0.25 * darkorange + 0.75 * #333333.
    target = np.array([102, 73, 38])
    close = np.abs(rgb - target).sum(axis=2) <= 18
    assert int(close.sum()) > 20
    text = widget.xy_plot.getAxis("bottom").textPen().color()
    assert text.red() > 240 and text.green() > 240 and text.blue() > 240
    assert "white" in widget.legend.styleSheet()
    assert "#333333" in widget.legend.styleSheet()
    widget.update_theme("white", "black")
    text = widget.xy_plot.getAxis("left").textPen().color()
    assert text.red() < 20 and text.green() < 20 and text.blue() < 20
    assert "black" in widget.legend.styleSheet()
    assert "white" in widget.legend.styleSheet().split("background")[-1]
    widget.close()


def test_constant_velocity_gmm_and_gp_paint(window):
    from avlite.c10_perception.c11_perception_model import AgentState
    from avlite.c10_perception.c15_perception_algs import ConstantVelocityGMM, ConstantVelocityGP

    window.setting.p67_show_local_global_view.set(True)
    window.setting.p67_show_prediction.set(True)
    ego = window.exec.ego_state
    window.exec.pm.agent_vehicles = [
        AgentState(
            x=float(ego.x) + 18.0, y=float(ego.y) + 1.5,
            theta=float(ego.theta), velocity=8.0, agent_id=7,
        ),
    ]
    ConstantVelocityGMM().predict(window.exec.pm)
    window.update_ui()
    counts = window.local_plot.layer_point_counts
    assert counts["prediction_xy"] > 20
    assert counts["prediction_ellipses"] > counts["prediction_xy"]
    ConstantVelocityGP().predict(window.exec.pm)
    window.update_ui()
    counts = window.local_plot.layer_point_counts
    assert counts["prediction_xy"] > 10
    assert counts["prediction_ellipses"] > 0


def _table_names(table):
    return [table.item(row, 0).text() for row in range(table.rowCount())]


def test_schema_settings_and_plugin_browser_build(window, qapp, monkeypatch):
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

    from avlite.c60_apps.c69_settings import AppSettings
    from avlite.plugins.p60_visualizer_tk import p63_plugins_app as plugins

    sample = {
        "name": "sample-plugin",
        "display_name": "Sample Plugin",
        "author": "Ada",
        "version": "1.2.3",
        "category": "perception",
        "description": "A sample",
        "repository": "https://github.com/example/sample-plugin",
    }
    monkeypatch.setattr(plugins._PluginOperations, "fetch_registry", lambda **_kwargs: [sample])
    calls = []
    monkeypatch.setattr(
        plugins._PluginOperations, "install_plugin",
        lambda entry, plugins_dir, token=None: calls.append(("install", entry["name"])) or "/tmp/sample-plugin",
    )
    monkeypatch.setattr(
        plugins._PluginOperations, "register_in_profile",
        lambda name, path, profile=None, private=False: calls.append(("register", name, str(path))),
    )

    import time

    browser = _Browser(window, QtCore, QtWidgets)
    browser.notify = lambda: calls.append("notify")
    browser.win.show()
    table = browser.win.findChild(QtWidgets.QTableWidget, "communityTable")
    names = []
    for _ in range(50):
        qapp.processEvents()
        names = _table_names(table)
        if "Sample Plugin" in names and "avlite-QT5" in names:
            break
        time.sleep(0.02)
    try:
        assert table.columnCount() == 7
        assert [table.horizontalHeaderItem(col).text() for col in range(7)] == [
            "Name", "Category", "Author", "Version", "Status", "Update", "Path",
        ]
        assert "avlite-QT5" in names
        assert "Sample Plugin" in names
        sample_row = names.index("Sample Plugin")
        assert table.item(sample_row, 2).text() == "Ada"
        assert table.item(sample_row, 3).text() == "1.2.3"
        assert table.item(sample_row, 4).text() == "Available"
        table.selectRow(sample_row)
        qapp.processEvents()
        assert browser.win.findChild(QtWidgets.QPushButton, "btnInstall").isEnabled()
        assert not browser.win.findChild(QtWidgets.QPushButton, "btnAddProfile").isEnabled()
        browser.win.findChild(QtWidgets.QPushButton, "btnInstall").click()
        for _ in range(50):
            qapp.processEvents()
            if "notify" in calls:
                break
            time.sleep(0.02)
        assert ("install", "sample-plugin") in calls
        assert ("register", "sample-plugin", "/tmp/sample-plugin") in calls
        assert "notify" in calls
        assert browser.win.findChild(QtWidgets.QPushButton, "btnSignIn") is not None
        assert "avlite-QT5" in dict(AppSettings.c62_community_plugins or {})
    finally:
        browser.win.close()
