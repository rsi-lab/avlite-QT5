"""Behavior that has to match AVLite dev: settings key, plan step, control plot."""

from __future__ import annotations

import pytest
import yaml

from conftest import PLUGIN_DIR, ensure_plugin_loaded


def test_plugin_settings_section_uses_import_name():
    ensure_plugin_loaded()
    from avlite.c60_apps.c65_setting_utils import setting_section
    from avlite.plugins.avlite_QT5.settings import PluginSettings

    assert setting_section(PluginSettings) == ("plugins", "avlite_QT5")


def test_shipped_profile_uses_import_name():
    profile = yaml.safe_load((PLUGIN_DIR / "avlite-QT5.yaml").read_text(encoding="utf-8"))
    assert "avlite_QT5" in profile["plugins"]
    assert "avlite-QT5" not in profile["plugins"]
    assert profile["c69_apps"]["c62_community_plugins"]["avlite-QT5"] == "avlite-QT5"
    assert "avlite>=0.6.7" in (PLUGIN_DIR / "requirements.txt").read_text(encoding="utf-8")


@pytest.mark.requires_data
def test_plan_step_shortcut_moves_the_waypoint(window):
    window.set_waypoint(2)
    window._run_shortcut("step_plan")
    assert int(window.exec.local_planner.global_trajectory.current_wp) > 2


@pytest.mark.requires_data
def test_manual_steer_reaches_the_wheel(window, qapp):
    window._ui_timer.stop()
    window.show()
    qapp.processEvents()
    window.drive(steer=0.4)
    assert window.applied_cmd.steer == 0.4
    assert window.exec.controller.cmd.steer == 0.4
    assert window.control_plot.steer_wheel.steer == 0.4


@pytest.mark.requires_data
def test_gg_trail_samples_a_speed_change(window, qapp):
    window._ui_timer.stop()
    window.show()
    qapp.processEvents()
    plot = window.control_plot
    plot.previous = (0.0, 10.0, 0.0)
    window.exec.elapsed_sim_time = 0.1
    window.exec.ego_state.velocity = 12.0
    window.exec.ego_state.theta = 0.2
    plot.redraw()
    assert plot.trail
    lateral, longitudinal = plot.trail[-1]
    assert longitudinal == pytest.approx(20.0)
    assert lateral == pytest.approx(12.0 * 0.2 / 0.1)


@pytest.mark.requires_data
def test_exec_step_receives_localization_dt(window, monkeypatch):
    seen = {}

    def _step(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(window.exec, "step", _step)
    window.setting.localization_dt.set(0.07)
    window.stack_panels.execution.step_exec()
    assert seen["localization_dt"] == pytest.approx(0.07)


def test_profile_note_stays_quiet_when_empty(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        window.QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: shown.append(args),
    )
    window.loaded_profile = "other"
    window.load_settings(profile="default")
    assert shown == []
