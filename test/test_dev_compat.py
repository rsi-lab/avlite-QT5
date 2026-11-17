"""Behavior that has to match AVLite dev: settings key, plan step, control plot."""

from __future__ import annotations

import pytest
import yaml

from conftest import PLUGIN_DIR, ensure_plugin_loaded


def test_raceline_colors_follow_the_tk_stops():
    from avlite.plugins.avlite_QT5.plot_widgets import _velocity_colors

    slow, mid, fast = _velocity_colors([0.0, 5.0, 10.0], "relative")
    assert slow == (57, 255, 20, 255)
    assert mid == (255, 215, 0, 255)
    assert fast == (220, 20, 60, 255)
    assert _velocity_colors([float("nan")], "relative") == [(80, 80, 80, 0)]


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


def test_gauge_labels_stay_clear_as_the_pane_changes(qapp):
    """Tick, caption, and g-g text stay inside the widget and off the arc and the spokes."""
    import math

    from avlite.plugins.avlite_QT5.control_plot import ControlPlot

    def closest(rect, cx, cy):
        x = min(max(cx, rect.left()), rect.right())
        y = min(max(cy, rect.top()), rect.bottom())
        return math.hypot(x - cx, y - cy)

    for width, height in ((160, 150), (200, 180), (420, 640), (531, 383), (820, 560), (240, 720), (860, 220)):
        plot = ControlPlot(None)
        plot.resize(width, height)
        plot.placed_size = (0, 0)
        plot.place()
        plot.speed_gauge.set(100.0, 10.0)
        plot.steer_wheel.set(0.7)
        plot.gg.show_trail([(12.0, -18.0), (6.0, 9.0), (-8.0, 14.0)], 20.0)
        plot.grab()
        for gauge in (plot.speed_gauge, plot.steer_wheel, plot.gg):
            rects = gauge.label_rects
            assert rects, (width, height, gauge.objectName())
            for rect in rects:
                assert rect.left() >= -1.0
                assert rect.top() >= -1.0
                assert rect.right() <= gauge.width() + 1.0
                assert rect.bottom() <= gauge.height() + 1.0
            for index, left in enumerate(rects):
                for right in rects[index + 1 :]:
                    assert not left.intersects(right), (width, height, gauge.objectName())
        dial = plot.speed_gauge
        cx, cy = dial.arc_center
        captions = [rect for rect in dial.label_rects if rect.center().y() >= dial.dial_px]
        assert len(captions) == 2
        assert all(rect.top() >= dial.dial_px - 1.0 for rect in captions)
        for rect in dial.label_rects:
            if rect.center().y() >= dial.dial_px:
                continue
            assert closest(rect, cx, cy) + 0.6 >= dial.arc_outer
        wheel = plot.steer_wheel
        assert wheel.label_inner > wheel.spoke_outer + 0.5
        assert dial.caption().endswith("km/h")
        assert dial.caption_mps().endswith("m/s")
        assert wheel.caption().endswith("rad")
        if min(width, height) >= 180:
            assert len(dial.label_rects) >= 9
            assert len(wheel.label_rects) >= 4
            assert len(plot.gg.label_rects) >= 6


def test_gg_marker_paints_over_a_label(qapp):
    """The live g-g sample stays visible when it sits on a number."""
    from avlite.plugins.avlite_QT5.control_plot import ControlPlot
    from avlite.plugins.avlite_QT5.qt_binding import import_qt

    _QtCore, QtGui, _QtWidgets = import_qt()
    marker = QtGui.QColor("#3d8bfd")

    def is_marker(color) -> bool:
        return (
            abs(color.red() - marker.red()) <= 12
            and abs(color.green() - marker.green()) <= 12
            and abs(color.blue() - marker.blue()) <= 12
        )

    for background in ("#2b2b2b", "#efefef"):
        plot = ControlPlot(None)
        plot.resize(420, 360)
        plot.placed_size = (0, 0)
        plot.place()
        palette = plot.gg.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(background))
        plot.gg.setPalette(palette)
        # The origin sample is drawn on the center "0.0" label.
        plot.gg.show_trail([(0.0, 0.0)], 20.0)
        image = plot.gg.grab().toImage()
        scale = image.devicePixelRatio() or 1.0
        covered = 0
        for rect in plot.gg.label_rects:
            left = max(0, int(rect.left() * scale))
            top = max(0, int(rect.top() * scale))
            right = min(image.width() - 1, int(rect.right() * scale))
            bottom = min(image.height() - 1, int(rect.bottom() * scale))
            for y in range(top, bottom + 1):
                for x in range(left, right + 1):
                    if is_marker(image.pixelColor(x, y)):
                        covered += 1
        assert covered >= 20, background


@pytest.mark.requires_data
def test_pipeline_sub_strategies_follow_the_selected_strategy(window, qapp, monkeypatch):
    """Detect/Track/Predict and Behavior/Path/Speed appear only for their pipeline."""
    errors = []
    monkeypatch.setattr(
        window.QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: errors.append(args),
    )
    window._ui_timer.stop()
    window.show()
    qapp.processEvents()
    sizes = window.plots_splitter.sizes()
    total = sum(sizes) or 1
    # default.yaml stores Frenet 0.6144, global 0.2061, control 0.1795.
    assert sizes[0] / total == pytest.approx(0.6144, abs=0.08)
    assert sizes[1] / total == pytest.approx(0.2061, abs=0.08)
    assert window.local_plot.xy_plot.isHidden()
    assert not window.local_plot.frenet_plot.isHidden()
    perc = window.stack_panels.perception
    plan = window.stack_panels.planning

    assert perc.perc.currentText() == "PerceptionPipeline"
    assert not perc.det.isHidden() and not perc.lbl_predict.isHidden()
    # Detect sits beside the strategy combo, the way Tk packs that block.
    # Combo geometry is relative to its own parent, so compare in the panel.
    detect_x = perc.det.mapTo(perc, perc.det.rect().topLeft()).x()
    assert detect_x >= perc.perc.geometry().right() - 8
    assert plan.lplan.currentText() != "LocalPlanningPipeline"
    assert plan.beh.isHidden() and plan.lbl_behavior.isHidden()
    assert plan.path.isHidden() and plan.vel.isHidden()
    map_y = perc.mapping.mapTo(perc, perc.mapping.rect().topLeft()).y()
    predict_bottom = perc.prd.mapTo(perc, perc.prd.rect().bottomLeft()).y()
    assert map_y > predict_bottom
    # Unfolding shows every row, then the pipeline rule hides its own again.
    window.setting.p67_plan_folded.set(True)
    qapp.processEvents()
    window.setting.p67_plan_folded.set(False)
    qapp.processEvents()
    assert plan.beh.isHidden() and not plan.lplan.isHidden()
    window.setting.p67_perception_folded.set(True)
    qapp.processEvents()
    window.setting.p67_perception_folded.set(False)
    qapp.processEvents()
    assert not perc.det.isHidden() and not perc.lbl_predict.isHidden()

    window.setting.perception_type.set("")
    qapp.processEvents()
    assert perc.perc.currentText() == ""
    assert perc.det.isHidden() and perc.trk.isHidden() and perc.prd.isHidden()
    assert perc.lbl_detect.isHidden() and perc.lbl_track.isHidden() and perc.lbl_predict.isHidden()
    assert not perc.mapping.isHidden() and not perc.loc.isHidden()
    # The hidden rows give their space back, the way Tk's grid_remove does.
    assert perc.mapping.geometry().top() - perc.perc.geometry().bottom() < 36

    window.setting.perception_type.set("PerceptionPipeline")
    qapp.processEvents()
    assert not perc.det.isHidden() and not perc.prd.isHidden()

    original_local = plan.lplan.currentText()
    plan.lplan.setCurrentText("LocalPlanningPipeline")
    qapp.processEvents()
    assert plan.lplan.currentText() == "LocalPlanningPipeline"
    assert not plan.beh.isHidden() and not plan.lbl_behavior.isHidden()
    assert not plan.path.isHidden() and not plan.lbl_path.isHidden()
    assert not plan.vel.isHidden() and not plan.lbl_speed.isHidden()
    assert abs(plan.beh.geometry().top() - plan.path.geometry().top()) < 8
    assert plan.path.geometry().left() > plan.beh.geometry().right() - 4
    prob_y = plan.prob.mapTo(plan, plan.prob.rect().topLeft()).y()
    speed_bottom = plan.vel.mapTo(plan, plan.vel.rect().bottomLeft()).y()
    assert prob_y > speed_bottom
    assert not perc.det.isHidden()

    plan.lplan.setCurrentText(original_local)
    qapp.processEvents()
    assert errors == []
    assert plan.beh.isHidden() and plan.path.isHidden() and plan.vel.isHidden()
    assert plan.prob.geometry().top() - plan.lplan.geometry().bottom() < 36
    # The occupancy save button belongs on Map, and only for OccupancyMapper.
    previous_map = window.setting.mapping_type.get()
    assert perc.save_occ.isHidden()
    window.setting.mapping_type.set("OccupancyMapper")
    qapp.processEvents()
    assert not perc.save_occ.isHidden()
    window.setting.mapping_type.set(previous_map)
    qapp.processEvents()
    assert perc.save_occ.isHidden()
