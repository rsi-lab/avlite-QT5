"""Open panes have to draw, and the bridge combo sits beside Execution."""

from __future__ import annotations

import itertools
import math

import pytest


def _ink(widget) -> int:
    image = widget.grab().toImage()
    step = 6
    ink = 0
    for y in range(0, image.height(), step):
        for x in range(0, image.width(), step):
            color = image.pixelColor(x, y)
            if max(color.red(), color.green(), color.blue()) - min(color.red(), color.green(), color.blue()) > 18:
                ink += 1
    return ink


def _span(plot) -> tuple[float, float]:
    (x0, x1), (y0, y1) = plot.getViewBox().viewRange()
    return float(x1 - x0), float(y1 - y0)


def _show(window, qapp) -> None:
    window._ui_timer.stop()
    window.win.resize(1400, 1000)
    window.show()
    qapp.processEvents()
    window.update_ui()
    qapp.processEvents()


@pytest.mark.requires_data
def test_every_pane_combo_draws(window, qapp):
    _show(window, qapp)
    plots = {
        "xy": window.local_plot.xy_plot,
        "frenet": window.local_plot.frenet_plot,
        "global": window.global_plot,
    }
    flags = ("p67_show_local_global_view", "p67_show_local_frenet_view", "p67_global_plan_view", "p67_show_control_view")
    for bits in itertools.product((False, True), repeat=4):
        if not any(bits):
            continue
        for field, visible in zip(flags, bits):
            getattr(window.setting, field).set(visible)
        window.update_views()
        qapp.processEvents()
        window.update_ui()
        qapp.processEvents()
        xy, fr, glob, ctrl = bits
        wanted = []
        if xy or fr:
            wanted.append(window.local_plot)
        if glob:
            wanted.append(window.global_plot)
        if ctrl:
            wanted.append(window.control_plot)
        got = [window.plots_splitter.widget(i) for i in range(window.plots_splitter.count())]
        assert got == wanted, bits
        assert all(size > 40 for size in window.plots_splitter.sizes()), (bits, window.plots_splitter.sizes())
        inner = []
        if xy:
            inner.append(window.local_plot.xy_plot)
        if fr:
            inner.append(window.local_plot.frenet_plot)
        assert [window.local_plot.panes.widget(i) for i in range(window.local_plot.panes.count())] == inner
        for name, plot, visible in (
            ("xy", plots["xy"], xy),
            ("frenet", plots["frenet"], fr),
            ("global", plots["global"], glob),
        ):
            assert plot.isHidden() is not visible
            if not visible:
                continue
            assert plot.width() > 40 and plot.height() > 40, (bits, name, plot.size())
            span = _span(plot)
            assert all(math.isfinite(value) and value > 1 for value in span), (bits, name, span)
            assert _ink(plot) > 12, (bits, name, _ink(plot), plot.size())
        if ctrl:
            assert not window.control_plot.isHidden()
            assert window.control_plot.width() > 40 and window.control_plot.height() > 40
            assert _ink(window.control_plot) > 12
        else:
            assert window.control_plot.isHidden()


@pytest.mark.requires_data
def test_software_renderer_does_not_replace_the_viewport(window, qapp):
    from avlite.plugins.avlite_QT5.plot_widgets import restore_plot_surface

    _show(window, qapp)
    plot = window.global_plot
    viewport = plot.viewport()
    called = []
    plot.useOpenGL = lambda *args, **kwargs: called.append(args)
    assert restore_plot_surface(plot) == "ready"
    window.setting.p67_show_local_global_view.set(True)
    window.setting.p67_show_control_view.set(False)
    window.update_views()
    qapp.processEvents()
    assert called == []
    assert plot.viewport() is viewport
    assert window.plots_splitter.indexOf(window.global_plot) >= 0
    assert window.plots_splitter.indexOf(window.control_plot) < 0


@pytest.mark.requires_data
def test_profile_switch_still_draws_the_open_panes(window, qapp, monkeypatch):
    monkeypatch.setattr(window.QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(window.QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: None)
    _show(window, qapp)
    combo = window.config_bar.profile
    combo.setCurrentText("Prediction")
    qapp.processEvents()
    window.update_ui()
    qapp.processEvents()
    assert window.global_plot.isHidden()
    assert window.control_plot.isHidden()
    assert not window.local_plot.xy_plot.isHidden()
    assert not window.local_plot.frenet_plot.isHidden()
    assert _ink(window.local_plot.xy_plot) > 12
    assert _ink(window.local_plot.frenet_plot) > 12
    combo.setCurrentText("default")
    qapp.processEvents()
    window.update_ui()
    qapp.processEvents()
    assert window.local_plot.xy_plot.isHidden()
    assert not window.global_plot.isHidden()
    assert _ink(window.global_plot) > 12


@pytest.mark.requires_data
def test_bridge_combo_sits_beside_execution(window, qapp):
    _show(window, qapp)
    execution = window.stack_panels.execution
    panel = execution.bridge_panel
    bridge = execution.bridge
    assert panel.objectName() == "worldBridge"
    assert panel.title() == "World Bridge"
    assert bridge.objectName() == "bridgeCombo"
    assert panel.isAncestorOf(bridge)
    assert panel.parent() is window.stack_panels
    assert bridge.currentText() == "BasicSim"
    assert panel.geometry().x() >= execution.geometry().right() - 8
    assert "world:" in execution.cap_label.text()
    text = execution.stats_lbl.text()
    assert text.count("FPS") == 4
    assert "perc=" in text and "plan=" in text and "ctrl=" in text and "ui=" in text
    assert "plot=" in text and text.endswith("ms")
    assert "sim=" in text and "FPS" not in text.split("perc=")[0]
