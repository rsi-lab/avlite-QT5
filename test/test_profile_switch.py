"""Profile changes reload the stack and the visualizer block Tk stores."""

from __future__ import annotations

import pytest


def _quiet(window, monkeypatch):
    notes = []
    errors = []
    monkeypatch.setattr(window.QtWidgets.QMessageBox, "information", lambda *args, **kwargs: notes.append(args))
    monkeypatch.setattr(window.QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: errors.append(args))
    return notes, errors


def _select(window, name: str) -> None:
    combo = window.config_bar.profile
    assert combo.findText(name) >= 0, list(combo.itemText(i) for i in range(combo.count()))
    combo.setCurrentText(name)


@pytest.mark.requires_data
def test_profile_combo_reloads_stack_views_and_notes(window, monkeypatch):
    from avlite.c10_perception.c19_settings import PerceptionSettings
    from avlite.c50_common.c51_capabilities import StackCapability
    from avlite.c60_apps.c69_settings import AppSettings

    notes, errors = _quiet(window, monkeypatch)
    assert window.setting.c60_selected_profile.get() == "default"
    assert window.setting.p66_xy_zoom == pytest.approx(130.0)
    assert window.setting.p67_show_local_global_view.get() is False
    assert not window.global_plot.isHidden()
    assert PerceptionSettings.c12_prediction_strategy == "ConstantVelocityPrediction"
    assert "avlite-QT5" in dict(AppSettings.c62_community_plugins or {})

    _select(window, "Prediction")
    assert errors == []
    assert notes == []
    assert window.setting.c60_selected_profile.get() == "Prediction"
    assert PerceptionSettings.c12_prediction_strategy == "ConstantVelocityGMM"
    assert StackCapability.PREDICTION_GMM in window.exec.perception.stack_capabilities
    assert window.setting.p66_xy_zoom == pytest.approx(170.0)
    assert window.setting.p66_frenet_zoom == pytest.approx(80.0)
    assert window.setting.p67_global_plan_view.get() is False
    assert window.global_plot.isHidden()
    # The file says ControlPlot is closed and p67_show_control_view is still true.
    # Tk follows the pane list, so the control plot stays closed.
    assert window.setting.p67_show_control_view.get() is False
    assert window.control_plot.isHidden()
    assert window.setting.p67_show_local_global_view.get() is True
    assert not window.local_plot.xy_plot.isHidden()
    assert not window.local_plot.frenet_plot.isHidden()
    assert window.stack_panels.perception.prd.currentText() == "ConstantVelocityGMM"
    assert window.local_plot.xy_plot.backgroundBrush().color().name() == "#333333"
    assert "avlite-QT5" in dict(AppSettings.c62_community_plugins or {})

    window.switch_profile()
    assert errors == []
    assert notes == []
    assert window.setting.c60_selected_profile.get() == "default"
    assert window.setting.p66_xy_zoom == pytest.approx(130.0)
    assert not window.global_plot.isHidden()
    assert PerceptionSettings.c12_prediction_strategy == "ConstantVelocityPrediction"

    _select(window, "Mapping")
    assert errors == []
    assert len(notes) == 1
    assert window.control_plot.isHidden()
    assert window.global_plot.isHidden()
    assert window.local_plot.frenet_plot.isHidden()
    assert not window.local_plot.xy_plot.isHidden()
    assert "occupancy mapping" in str(notes[0][2]).lower() or "occupancy mapping" in str(notes[0][-1]).lower()
    window.load_settings(profile="Mapping")
    assert len(notes) == 1

    _select(window, "default")
    assert len(notes) == 1
    assert window.setting.p66_xy_zoom == pytest.approx(130.0)
    assert not window.global_plot.isHidden()


@pytest.mark.requires_data
def test_settings_window_load_profile(window, monkeypatch):
    from avlite.c10_perception.c19_settings import PerceptionSettings
    from avlite.plugins.avlite_QT5 import schema_editor
    from avlite.plugins.avlite_QT5.qt_binding import import_qt

    notes, errors = _quiet(window, monkeypatch)
    opened = []
    monkeypatch.setattr(schema_editor, "open_schema_settings", lambda root: opened.append(root))
    _QtCore, _QtGui, QtWidgets = import_qt()
    dlg = schema_editor._SchemaDialog(window, window.QtCore, window.QtGui, QtWidgets)
    assert dlg.profile.findText("Prediction") >= 0
    dlg.profile.setCurrentText("Prediction")
    assert errors == []
    assert notes == []
    assert opened == [window]
    assert window.setting.c60_selected_profile.get() == "Prediction"
    assert PerceptionSettings.c12_prediction_strategy == "ConstantVelocityGMM"
    assert window.setting.p66_xy_zoom == pytest.approx(170.0)
    assert window.global_plot.isHidden()
