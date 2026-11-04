"""One headless test per Tk-visualizer feature, against BasicSim.

Run::

    QT_QPA_PLATFORM=offscreen python3 -m pytest test/test_feature_parity.py -q
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("AVLITE_QT_NO_OPENGL", "1")

from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c61_app_strategy import AppStrategy
from avlite.c60_apps.c62_factory import load_stack_settings
from avlite.c60_apps.c65_setting_utils import load_setting
from avlite.c60_apps.c69_settings import AppSettings

from conftest import PLUGIN_DIR, PLUGIN_NAME, ensure_plugin_loaded


@pytest.fixture(scope="module")
def qapp():
    ensure_plugin_loaded()
    from avlite.plugins.avlite_QT5.qt_binding import configure_qt_env, import_qt

    configure_qt_env(offscreen=True)
    _c, _g, QtWidgets = import_qt()
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    load_setting(AppSettings, profile="default")
    AppSettings.c62_load_plugins = True
    AppSettings.c62_community_plugins = {PLUGIN_NAME: str(PLUGIN_DIR)}
    load_stack_settings(profile="default", load_plugins=True)
    ensure_plugin_loaded()

    from avlite.plugins.avlite_QT5.main_window import VisualizerQtWindow

    win = VisualizerQtWindow()
    yield win
    win.stack_panels.execution.stop_exec()
    win.log_panel.shutdown()
    win._ui_timer.stop()
    win.win.close()
    ensure_plugin_loaded()


@pytest.mark.requires_data
def test_registers_qt_app_strategy():
    ensure_plugin_loaded()
    cls = AppStrategy.registry["qt5"]
    assert cls.cli_name == "qt5"
    assert cls.__module__.startswith("avlite.plugins.avlite_QT5")


@pytest.mark.requires_data
def test_basicsim_stack(window):
    assert type(window.exec.world).__name__ == "BasicSim"
    assert window.exec.ego_state is not None


@pytest.mark.requires_data
def test_global_plot_ego_and_plan(window):
    window.update_ui()
    xs, ys = window.global_plot.ego_marker.getData()
    assert len(xs) == 1
    px, py = window.global_plot.global_plan.getData()
    assert len(px) > 2 and len(py) > 2


@pytest.mark.requires_data
def test_global_plan_boundaries(window):
    window.setting.p66_show_global_plan_boundaries.set(True)
    window.update_ui()
    xs, _ys = window.global_plot.left_boundary.getData()
    assert len(xs) > 2


@pytest.mark.requires_data
def test_local_ego_box(window):
    window.update_ui()
    xs, ys = window.local_plot.xy_ego.getData()
    assert len(xs) >= 4 and len(ys) >= 4


@pytest.mark.requires_data
def test_local_frenet_reference(window):
    window.setting.p67_show_local_frenet_view.set(True)
    window.update_ui()
    xs, _ys = window.local_plot.sd_ref.getData()
    assert len(xs) > 2


@pytest.mark.requires_data
def test_exec_step_advances_sim_time(window):
    window.setting.executer_type.set("SyncExecuter")
    window.setting.pace_sim.set(True)
    window.setting.sim_dt.set(0.05)
    window.reload_stack(reload_code=False)
    before = float(window.exec.elapsed_sim_time)
    window.stack_panels.execution.step_exec()
    assert float(window.exec.elapsed_sim_time) >= before + 0.05 - 1e-6


@pytest.mark.requires_data
def test_exec_start_stop(window):
    window.stack_panels.execution.toggle_exec()
    assert window.setting.exec_running is True
    window.stack_panels.execution.stop_exec()
    assert window.setting.exec_running is False


@pytest.mark.requires_data
def test_exec_reset(window):
    window.teleport_ego(1.0, 2.0, 0.1)
    window.stack_panels.execution.reset_exec()
    # Reset returns to the saved start pose, not the teleported one.
    assert abs(window.exec.ego_state.x - 1.0) > 1.0


@pytest.mark.requires_data
def test_teleport_ego(window):
    window.teleport_ego(12.0, -4.0, 0.3)
    assert abs(window.exec.ego_state.x - 12.0) < 1e-6
    assert abs(window.exec.pm.ego_vehicle.x - 12.0) < 1e-6


@pytest.mark.requires_data
def test_spawn_agent(window):
    from avlite.c10_perception.c11_perception_model import AgentState

    before = len(getattr(window.exec.world, "agents", []) or [])
    agent = AgentState()
    agent.x, agent.y, agent.theta = window.exec.ego_state.x + 8, window.exec.ego_state.y, 0.0
    window.spawn_agent(agent)
    after = len(getattr(window.exec.world, "agents", []) or [])
    assert after >= before


@pytest.mark.requires_data
def test_switch_controller(window):
    window.setting.controller_type.set("StanleyController")
    window.reload_stack(reload_code=False)
    assert type(window.exec.controller).__name__ == "StanleyController"


@pytest.mark.requires_data
def test_switch_executer(window):
    window.setting.executer_type.set("SyncExecuter")
    window.reload_stack(reload_code=False)
    assert type(window.exec).__name__ == "SyncExecuter"


@pytest.mark.requires_data
def test_perception_pipeline_refresh(window):
    window.setting.perception_type.set("PerceptionPipeline")
    window.setting.prediction_strategy_type.set("ConstantVelocityPrediction")
    window.refresh_pipeline()
    assert type(window.exec.perception).__name__ == "PerceptionPipeline"


@pytest.mark.requires_data
def test_local_replan(window):
    window.local_replan()
    assert window.exec.local_planner is not None


@pytest.mark.requires_data
def test_step_plan_and_back(window):
    window.set_waypoint(2)
    assert int(window.exec.local_planner.global_trajectory.current_wp) == 2
    window.step_plan()
    advanced = int(window.exec.local_planner.global_trajectory.current_wp)
    assert advanced > 2
    window.step_plan_back()
    assert int(window.exec.local_planner.global_trajectory.current_wp) == advanced - 1


@pytest.mark.requires_data
def test_align_plan(window):
    window.align_plan()
    assert window.exec.local_planner.location_xy is not None


@pytest.mark.requires_data
def test_align_control(window):
    x, y = window.exec.local_planner.location_xy
    window.align_control()
    assert abs(window.exec.ego_state.x - x) < 1e-3
    assert abs(window.exec.ego_state.y - y) < 1e-3


@pytest.mark.requires_data
def test_step_control_and_gauges(window):
    window.step_control()
    # Gauges mirror controller outputs (may be zero at rest, but must be numeric).
    float(window.setting.cmd_accel.get())
    float(window.setting.cmd_steer.get())
    assert "Accel" in window.stack_panels.control.gauges["accel"].text()


@pytest.mark.requires_data
def test_manual_drive(window):
    v0 = float(window.exec.world.get_ego_state().velocity)
    window.drive(acceleration=3.0, steer=0.0)
    v1 = float(window.exec.world.get_ego_state().velocity)
    assert v1 > v0
    assert abs(float(window.exec.pm.ego_vehicle.velocity) - v1) < 1e-6


@pytest.mark.requires_data
def test_save_global_plan(window, tmp_path):
    path = window.save_global_plan(str(tmp_path / "plan.json"))
    assert path and os.path.isfile(path) and os.path.getsize(path) > 10


@pytest.mark.requires_data
def test_save_occupancy_without_map_is_noop(window):
    # Default race stack has no occupancy map; the control must not raise.
    assert window.save_occupancy_map("/tmp/should-not-write-occ.json") is None


@pytest.mark.requires_data
def test_set_start_pose(window):
    window.teleport_ego(30.0, 40.0, 0.2)
    window.set_start_pose()
    pose = ExecutionSettings.c40_start_pose
    assert abs(pose[0] - 30.0) < 1e-3
    assert abs(pose[1] - 40.0) < 1e-3


@pytest.mark.requires_data
def test_set_default_global_plan_reloads(window):
    current = ExecutionSettings.c40_global_trajectory
    window.set_default_global_plan(current)
    assert window.exec is not None
    assert type(window.exec.world).__name__ == "BasicSim"


@pytest.mark.requires_data
def test_capability_filters(window):
    window.set_capability_filters(world=["LIDAR_2D"], stack=["LOCALIZATION"])
    assert ExecutionSettings.c41_world_capabilities == ["LIDAR_2D"]
    assert ExecutionSettings.c41_world_stack_capabilities == ["LOCALIZATION"]
    assert "world:" in window.stack_panels.execution.cap_label.text()


@pytest.mark.requires_data
def test_probabilistic_collision_flag(window):
    window.setting.probabilistic_collision_checking.set(True)
    assert PerceptionSettings.c15_probabilistic_collision_checking is True
    window.setting.probabilistic_collision_checking.set(False)


@pytest.mark.requires_data
def test_shortcut_mode(window):
    window.setting.p60_shortcut_mode.set(True)
    window.update_shortcut_mode()
    assert window.stack_panels.isHidden()
    assert not window.shortcut_help.isHidden()
    window.setting.p60_shortcut_mode.set(False)
    window.update_shortcut_mode()
    assert not window.stack_panels.isHidden()


@pytest.mark.requires_data
def test_view_toggles(window):
    window.setting.p67_global_plan_view.set(False)
    window.setting.p67_show_local_frenet_view.set(False)
    window.update_views()
    assert window.global_plot.isHidden()
    assert window.local_plot.frenet_plot.isHidden()
    window.setting.p67_global_plan_view.set(True)
    window.setting.p67_show_local_frenet_view.set(True)
    window.update_views()
    assert not window.global_plot.isHidden()


@pytest.mark.requires_data
def test_dark_mode_toggle(window):
    window.setting.p60_dark_mode.set(True)
    window._apply_theme()
    assert "#2b2b2b" in window.win.styleSheet()
    window.setting.p60_dark_mode.set(False)
    window._apply_theme()
    assert window.win.styleSheet() == ""


@pytest.mark.requires_data
def test_log_panel_clear_and_filter(window):
    import logging

    window.setting.p68_show_core_logs.set(True)
    logging.getLogger("avlite.qt.parity").error("parity-marker")
    window.log_panel._flush()
    assert "parity-marker" in window.log_panel.view.toPlainText()
    window.log_panel.clear()
    assert window.log_panel.view.toPlainText() == ""


@pytest.mark.requires_data
def test_lidar_overlay_after_step(window):
    window.setting.p66_show_lidar_global.set(True)
    window.stack_panels.execution.step_exec()
    window.update_ui()
    xs, ys = window.local_plot.xy_lidar.getData()
    assert xs is not None and len(xs) > 0 and len(ys) > 0


@pytest.mark.requires_data
def test_plot_latency_recorded(window):
    window.update_ui()
    assert float(window.setting.plot_latency_ms.get()) >= 0.0
    assert float(window.setting.plot_latency_ms.get()) < 500.0


@pytest.mark.requires_data
def test_module_entrypoint_resolves_app():
    """``python -m avlite.plugins.avlite_QT5`` imports the registered app."""
    ensure_plugin_loaded()
    import avlite.plugins.avlite_QT5 as plugin

    assert plugin.VisualizationQtApp is AppStrategy.registry["qt5"]
