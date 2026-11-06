"""Settings dialog fit, and the actions Tk exposes on that window."""

from __future__ import annotations

import avlite.c60_apps.c65_setting_utils as setting_utils
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c30_control.c39_settings import ControlSettings
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c64_settings_schema import schema_of
from avlite.c60_apps.c68_paths import ConfigPaths


def _snap(cls) -> dict:
    schema = schema_of(cls)
    return {name: getattr(cls, name) for name in schema.model_fields}


def _restore(cls, saved: dict) -> None:
    for name, value in saved.items():
        setattr(cls, name, value)


def _show(window, qapp):
    window._ui_timer.stop()
    window.win.resize(1400, 1000)
    window.show()
    qapp.processEvents()


def _text_fits(widget) -> bool:
    text = widget.text() if hasattr(widget, "text") else ""
    if not text:
        return True
    if getattr(widget, "wordWrap", lambda: False)():
        return True
    return widget.width() + 2 >= widget.fontMetrics().horizontalAdvance(text)


def test_settings_dialog_fits_and_actions(window, qapp):
    from PyQt5 import QtCore, QtGui, QtWidgets

    from avlite.plugins.avlite_QT5.schema_editor import _SchemaDialog
    from avlite.plugins.avlite_QT5.settings import PluginSettings

    _show(window, qapp)
    saved_settings = [
        (PerceptionSettings, _snap(PerceptionSettings)),
        (PlanningSettings, _snap(PlanningSettings)),
        (ControlSettings, _snap(ControlSettings)),
        (ExecutionSettings, _snap(ExecutionSettings)),
    ]
    saved_shortcuts = dict(PluginSettings.p60_shortcuts or {})
    saved_view = bool(window.setting.p67_global_plan_view.get())
    dlg = _SchemaDialog(window, QtCore, QtGui, QtWidgets)
    dlg.win.show()
    qapp.processEvents()
    try:
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        assert dlg.win.width() <= screen.width()
        assert dlg.win.height() <= screen.height()
        assert dlg.win.minimumWidth() <= screen.width()
        assert dlg.win.minimumHeight() <= screen.height()

        tight = QtCore.QSize(min(700, screen.width()), min(520, screen.height()))
        dlg.win.resize(tight)
        qapp.processEvents()
        _assert_dialog_unclipped(dlg, QtWidgets)

        # Wider than the tab bar used to clip the last tab's border. Scroll instead.
        dlg.win.resize(min(1080, screen.width()), min(840, screen.height()))
        qapp.processEvents()
        _assert_tabs_reachable(dlg, QtWidgets)

        field = dlg.win.findChild(QtWidgets.QSpinBox, "field_Perception_c11_max_agents")
        assert field is not None and field.width() >= 160
        original = int(PerceptionSettings.c11_max_agents)
        field.setValue(original + 5)
        dlg._apply_editors()
        assert PerceptionSettings.c11_max_agents == original + 5
        dlg._reset_one(PerceptionSettings)
        default = int(schema_of(PerceptionSettings).model_validate({}).c11_max_agents)
        assert PerceptionSettings.c11_max_agents == default
        assert field.value() == default
        execution_before = ExecutionSettings.c40_log_level
        dlg._reset_stack(exclude_execution=True)
        assert ExecutionSettings.c40_log_level == execution_before

        box = dlg.win.findChild(QtWidgets.QCheckBox, "field_Visualizer_p67_global_plan_view")
        assert box is not None and box.isChecked()
        box.setChecked(False)
        dlg._apply_editors()
        window.update_views()
        qapp.processEvents()
        assert window.global_plot.isHidden()
        box.setChecked(True)
        dlg._apply_editors()
        window.update_views()
        qapp.processEvents()
        assert window.global_plot.isVisible()

        repo = dlg.win.findChild(QtWidgets.QCheckBox, "editRepoConfigs")
        assert (repo is not None) == bool(ConfigPaths.can_edit_bundled())
        assert dlg.win.findChild(QtWidgets.QComboBox, "nextProfileCombo") is not None
        dlg.next_profile.addItem("audit-profile")
        dlg.next_profile.setCurrentText("audit-profile")
        assert window.setting.p60_next_profile.get() == "audit-profile"
        window.setting.p60_next_profile.set("default")

        warnings = []
        real_warning = QtWidgets.QMessageBox.warning
        QtWidgets.QMessageBox.warning = staticmethod(lambda *args, **kwargs: warnings.append(args))
        try:
            dlg._delete_profile()
            dlg._rename_profile()
        finally:
            QtWidgets.QMessageBox.warning = real_warning
        assert len(warnings) == 2
        assert not dlg._profile_buttons["profileDelete"].isEnabled()

        table = dlg.win.findChild(QtWidgets.QTableWidget, "shortcutTable")
        assert table is not None and table.rowCount() > 0
        replan = _row(table, "replan")
        table.item(replan, 3).setText("Z")
        assert dlg._store_shortcuts(quiet=True)
        assert PluginSettings.p60_shortcuts.get("replan") == "Z"
        help_label = window.win.findChild(QtWidgets.QLabel, "shortcutHelp_replan")
        assert help_label is not None and help_label.text() == "Z"
        other = 0 if replan != 0 else 1
        previous = table.item(other, 3).text()
        table.item(other, 3).setText("Z")
        assert dlg._store_shortcuts(quiet=True) is False
        assert PluginSettings.p60_shortcuts.get("replan") == "Z"
        table.item(other, 3).setText(previous)

        calls = []
        planning = window.stack_panels.planning
        planning._local_replan = lambda: calls.append("replan")
        dlg.tabs.setFocus()
        qapp.processEvents()
        _press(qapp, window, "r", QtCore.Qt.Key_R)
        assert calls == []
        _press(qapp, window, "Z", QtCore.Qt.Key_Z, QtCore.Qt.ShiftModifier)
        assert calls == ["replan"]
        dlg._reset_shortcuts()
        assert PluginSettings.p60_shortcuts == {}
        assert help_label.text() == "r"
        _press(qapp, window, "r", QtCore.Qt.Key_R)
        assert calls == ["replan", "replan"]

        table.selectRow(replan)
        dlg._begin_capture()
        _press(qapp, window, "r", QtCore.Qt.Key_R)
        assert table.item(replan, 3).text() == "r"
        dlg.win.reject()
        qapp.processEvents()
        calls.clear()
        _press(qapp, window, "r", QtCore.Qt.Key_R)
        assert calls == ["replan"]

        plugins = dlg.win.findChild(QtWidgets.QWidget, "settingsPluginsPage")
        assert plugins is not None
        assert plugins.findChild(QtWidgets.QListWidget, "communityPlugins") is not None
        assert plugins.findChild(QtWidgets.QPushButton, "addCommunityPlugin") is not None
        assert plugins.findChild(QtWidgets.QListWidget, "communityPlugins").count() >= 1

        recorded = []
        real_save = setting_utils.save_setting

        def _fake_save(setting, profile=None, **kwargs):
            recorded.append(type(setting).__name__)

        setting_utils.save_setting = _fake_save
        try:
            field.setValue(default)
            dlg._save()
        finally:
            setting_utils.save_setting = real_save
        assert any(name.startswith("PerceptionSettings") for name in recorded)
    finally:
        for cls, saved in saved_settings:
            _restore(cls, saved)
        PluginSettings.p60_shortcuts = saved_shortcuts
        window.setting.p60_next_profile.set("default")
        window.setting.p67_global_plan_view.set(saved_view)
        window.config_bar.refresh_shortcut_labels()
        window.update_views()
        dlg.win.close()


def test_panel_fold_collapses_and_restores(window, qapp):
    _show(window, qapp)
    control = window.stack_panels.control
    button = window.win.findChild(window.QtWidgets.QToolButton, "foldControl")
    assert button is not None and button.isVisible()
    opened = control.height()
    assert opened > 80
    window.setting.p67_control_folded.set(True)
    qapp.processEvents()
    try:
        assert control.height() < opened / 2
        assert control.height() <= 48
        assert button.text() == "▸"
        button.click()
        qapp.processEvents()
        assert window.setting.p67_control_folded.get() is False
        assert control.height() > 80
        assert button.text() == "▾"
    finally:
        window.setting.p67_control_folded.set(False)
        qapp.processEvents()


def _row(table, shortcut_id: str) -> int:
    from PyQt5 import QtCore

    for row in range(table.rowCount()):
        item = table.item(row, 3)
        if item is not None and item.data(QtCore.Qt.UserRole) == shortcut_id:
            return row
    raise AssertionError(shortcut_id)


def _press(qapp, window, text: str, key, modifiers=None) -> None:
    from PyQt5 import QtCore, QtGui, QtWidgets

    modifiers = QtCore.Qt.NoModifier if modifiers is None else modifiers
    event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, modifiers, text)
    QtWidgets.QApplication.sendEvent(window.win, event)
    qapp.processEvents()


def _assert_tabs_reachable(dlg, QtWidgets) -> None:
    bar = dlg.tabs.tabBar()
    for index in range(bar.count()):
        text = bar.tabText(index)
        assert bar.tabRect(index).width() + 1 >= bar.fontMetrics().horizontalAdvance(text), text
    content_right = max(bar.tabRect(i).right() for i in range(bar.count()))
    content_left = min(bar.tabRect(i).left() for i in range(bar.count()))
    overflow = content_right > bar.width() - 2 or content_left < 0
    if overflow:
        assert any(button.isVisible() for button in bar.findChildren(QtWidgets.QToolButton))


def _assert_dialog_unclipped(dlg, QtWidgets) -> None:
    from PyQt5 import QtCore

    bar = dlg.win.findChild(QtWidgets.QWidget, "profileBar")
    assert bar is not None
    scroll = bar.parentWidget().parentWidget()
    viewport = scroll.viewport()
    # Profile rows stay fully visible. A scrollbar must not cover them.
    assert bar.height() <= viewport.height() + 2, (bar.height(), viewport.height())
    if bar.width() > viewport.width() + 1:
        assert scroll.horizontalScrollBar().maximum() > 0
    geometries = []
    for child in bar.findChildren(QtWidgets.QWidget):
        if not child.isVisible() or child.objectName().startswith("qt_"):
            continue
        if not isinstance(child, (
            QtWidgets.QLabel, QtWidgets.QPushButton, QtWidgets.QCheckBox,
            QtWidgets.QLineEdit, QtWidgets.QComboBox,
        )):
            continue
        parent = child.parentWidget()
        assert child.geometry().right() <= parent.width() + 1, child.objectName()
        assert child.geometry().bottom() <= parent.height() + 1, child.objectName()
        assert _text_fits(child), (child.objectName(), child.text() if hasattr(child, "text") else "")
        origin = child.mapTo(bar, child.rect().topLeft())
        geometries.append(QtCore.QRect(origin, child.size()))
        shown = child.mapTo(viewport, child.rect().topLeft())
        assert viewport.rect().contains(QtCore.QRect(shown, child.size())), child.objectName()
    for i, first in enumerate(geometries):
        for second in geometries[i + 1 :]:
            overlap = first.intersected(second)
            assert not overlap.isValid() or overlap.width() <= 1 or overlap.height() <= 1

    box = dlg.win.findChild(QtWidgets.QDialogButtonBox)
    assert box is not None
    for button in box.buttons():
        origin = button.mapTo(dlg.win, button.rect().topLeft())
        rect = QtCore.QRect(origin, button.size())
        assert dlg.win.rect().contains(rect), (button.text(), rect, dlg.win.rect())
        assert _text_fits(button), button.text()

    for index in range(dlg.tabs.count()):
        dlg.tabs.setCurrentIndex(index)
        qapp_pump(dlg)
        page = dlg.tabs.currentWidget()
        scrolls = []
        if isinstance(page, QtWidgets.QScrollArea):
            scrolls.append(page)
        scrolls.extend(page.findChildren(QtWidgets.QScrollArea))
        for area in scrolls:
            if not area.isVisible():
                continue
            content = area.widget()
            assert content is not None
            view = area.viewport()
            if content.width() > view.width() + 1:
                assert area.horizontalScrollBar().maximum() > 0, dlg.tabs.tabText(index)
            if content.height() > view.height() + 1:
                assert area.verticalScrollBar().maximum() > 0, dlg.tabs.tabText(index)
            for label in content.findChildren(QtWidgets.QLabel):
                if not label.isVisible() or label.wordWrap() or not label.text():
                    continue
                assert _text_fits(label), (dlg.tabs.tabText(index), label.text(), label.width())
            for button in content.findChildren(QtWidgets.QPushButton):
                if button.isVisible() and button.text():
                    assert _text_fits(button), (dlg.tabs.tabText(index), button.text(), button.width())
    _assert_tabs_reachable(dlg, QtWidgets)


def qapp_pump(dlg) -> None:
    from PyQt5 import QtWidgets

    QtWidgets.QApplication.processEvents()
