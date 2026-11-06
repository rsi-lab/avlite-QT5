"""Main Qt5 visualizer window — feature parity target for the Tk app."""

from __future__ import annotations

import logging
import time
from typing import Optional

from avlite import __version__
from avlite.c10_perception.c11_perception_model import AgentState
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c62_factory import executor_factory, load_stack_settings
from avlite.c60_apps.c63_plugins import reload_lib
from avlite.c60_apps.c65_setting_utils import list_profiles, load_setting
from avlite.c60_apps.c68_paths import ConfigPaths
from avlite.c60_apps.c69_settings import AppSettings

from .setting_model import VisualizationSettings

log = logging.getLogger(__name__)


class VisualizerQtWindow:
    """Owns the QMainWindow, stack, plots, and UI refresh timer.

    Sim / planning work is driven by ``ExecutionPanel`` calling ``exec.step``
    (AsyncThreadedExecuter keeps heavy work off the UI thread). Plot redraws
    run on a dedicated QTimer at ``p60_ui_refresh_hz`` and only call
    ``PlotDataItem.setData`` — never Matplotlib.
    """

    hosting_plugin_name = "avlite-QT5"

    def __init__(self):
        from .qt_binding import configure_qt_env, import_pyqtgraph, import_qt

        configure_qt_env()
        QtCore, QtGui, QtWidgets = import_qt()
        self.QtCore = QtCore
        self.QtGui = QtGui
        self.QtWidgets = QtWidgets
        import_pyqtgraph()  # OpenGL on when available; disabled offscreen via env

        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle(f"AVLite Qt Visualizer v{__version__}")
        self.win.resize(1400, 1000)
        self.exec = None
        self.setting = VisualizationSettings()
        self.loaded_profile: Optional[str] = None
        self._ui_frames = 0
        self._ui_t0 = time.perf_counter()

        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        from .plot_widgets import GlobalPlotWidget, LocalPlotWidget

        self.plots_splitter = QtWidgets.QSplitter()
        self.plots_splitter.setChildrenCollapsible(False)
        self.local_plot = LocalPlotWidget(self)
        self.global_plot = GlobalPlotWidget(self)
        self.plots_splitter.addWidget(self.local_plot)
        self.plots_splitter.addWidget(self.global_plot)
        self.plots_splitter.setStretchFactor(0, 1)
        self.plots_splitter.setStretchFactor(1, 1)
        self.plots_splitter.setMinimumHeight(360)
        root_layout.addWidget(self.plots_splitter, 1)

        from .config_bar import ConfigBar

        self.config_bar = ConfigBar(self)
        self.shortcut_help = self.config_bar.shortcut_frame
        root_layout.addWidget(self.config_bar)
        root_layout.addWidget(self.shortcut_help)

        from .stack_panels import StackPanels

        self.stack_panels = StackPanels(self)
        root_layout.addWidget(self.stack_panels)

        from .log_panel import LogPanel

        self.log_panel = LogPanel(self)
        root_layout.addWidget(self.log_panel, 0)

        self._build_menu()
        self.win.closeEvent = self._on_close  # type: ignore[method-assign]
        self.win.keyPressEvent = self._key_press_event  # type: ignore[method-assign]
        self.win.keyReleaseEvent = self._key_release_event  # type: ignore[method-assign]
        self._install_key_filter()

        self._in_ui_update = False
        self._last_plot_t = 0.0
        self._ui_timer = QtCore.QTimer(self.win)
        self._ui_timer.timeout.connect(self._on_ui_timer)
        hz = max(5.0, float(self.setting.p60_ui_refresh_hz))
        self._ui_timer.start(int(1000 / hz))

        self.setting.profile_list = list_profiles(AppSettings)
        startup = ConfigPaths.startup_profile()
        if startup and startup in self.setting.profile_list:
            self.setting.c60_selected_profile.set(startup)

        self.load_settings()
        self.reload_stack(reload_code=False)
        self._apply_theme()
        self.update_views()
        self.log_panel._flush()
        self.log_panel._apply_level(str(self.setting.log_level.get() or "INFO"))
        self._attach_folds()

    def _attach_folds(self) -> None:
        from .panel_fold import attach_panel_fold

        setting = self.setting
        panels = self.stack_panels
        attach_panel_fold(self.config_bar, setting.p60_settings_folded, name="foldSettings")
        attach_panel_fold(self.shortcut_help, setting.p60_shortcuts_folded, name="foldShortcuts")
        attach_panel_fold(panels.perception, setting.p67_perception_folded, name="foldPerception")
        attach_panel_fold(panels.planning, setting.p67_plan_folded, name="foldPlanning")
        attach_panel_fold(panels.control, setting.p67_control_folded, name="foldControl")
        attach_panel_fold(panels.execution, setting.p67_execution_folded, name="foldExecution")
        attach_panel_fold(self.log_panel, setting.p68_log_folded, name="foldLog")

    def _build_menu(self) -> None:
        QtWidgets = self.QtWidgets
        menubar = self.win.menuBar()
        file_menu = menubar.addMenu("&File")
        act_settings = file_menu.addAction("Settings")
        act_settings.triggered.connect(self.config_bar.open_settings)
        act_plugins = file_menu.addAction("Community Plugins")
        act_plugins.triggered.connect(self.config_bar.open_plugins)
        file_menu.addSeparator()
        act_quit = file_menu.addAction("Exit")
        act_quit.triggered.connect(self.win.close)

        view_menu = menubar.addMenu("&View")
        act_global = view_menu.addAction("Toggle global plot")
        act_global.triggered.connect(
            lambda: (
                self.setting.p67_global_plan_view.set(not self.setting.p67_global_plan_view.get()),
                self.update_views(),
            )
        )
        act_local = view_menu.addAction("Toggle local plot")
        act_local.triggered.connect(
            lambda: (
                self.setting.p67_show_local_global_view.set(
                    not self.setting.p67_show_local_global_view.get()
                ),
                self.update_views(),
            )
        )
        act_dark = view_menu.addAction("Toggle dark mode")
        act_dark.triggered.connect(self._toggle_dark)

        help_menu = menubar.addMenu("&Help")
        act_update = help_menu.addAction("Update…")
        act_update.triggered.connect(self._on_check_update)
        act_about = help_menu.addAction("About AVLite")
        act_about.triggered.connect(self._show_about)

    def _show_about(self) -> None:
        self.QtWidgets.QMessageBox.about(
            self.win,
            "About AVLite",
            f"<h3>AVLite</h3><p>Version {__version__}</p>"
            "<p>Qt5 visualizer plugin (pyqtgraph).</p>",
        )

    def _toggle_dark(self) -> None:
        self.setting.p60_dark_mode.set(not self.setting.p60_dark_mode.get())
        self._apply_theme()

    def _apply_theme(self) -> None:
        dark = bool(self.setting.p60_dark_mode.get())
        if dark:
            self.setting.p60_bg_color = "#1e1e1e"
            self.setting.p60_fg_color = "#dddddd"
            self.win.setStyleSheet(
                "QWidget { background: #2b2b2b; color: #dddddd; }"
                "QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 6px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
                "QPlainTextEdit { background: #1a1a1a; color: #cfcfcf; }"
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #3a3a3a; }"
            )
        else:
            self.setting.p60_bg_color = "#f5f5f5"
            self.setting.p60_fg_color = "#222222"
            self.win.setStyleSheet("")
        self.local_plot.update_theme(self.setting.p60_bg_color, self.setting.p60_fg_color)
        self.global_plot.update_theme(self.setting.p60_bg_color, self.setting.p60_fg_color)

    def _open_profile_dialog(self) -> None:
        profiles = list_profiles(AppSettings) or ["default"]
        current = self.setting.c60_selected_profile.get()
        name, ok = self.QtWidgets.QInputDialog.getItem(
            self.win, "Profiles", "Load profile:", profiles,
            max(0, profiles.index(current) if current in profiles else 0), False,
        )
        if ok and name:
            self.load_settings(profile=name)
            self.reload_stack(reload_code=False)

    def show(self) -> None:
        self.win.show()

    def exec_app(self) -> int:
        self.show()
        return int(self.app.exec_())

    def _on_close(self, event) -> None:
        try:
            if getattr(self, "_keys", None) is not None:
                self.app.removeEventFilter(self._keys)
            self.stack_panels.execution.stop_exec()
            self.log_panel.shutdown()
            self._ui_timer.stop()
        finally:
            event.accept()

    def _install_key_filter(self) -> None:
        """Catch shortcuts before a focused child (the log view) swallows them."""
        owner = self
        QtCore = self.QtCore

        class _Keys(QtCore.QObject):
            def eventFilter(self, obj, event):  # noqa: N802
                et = event.type()
                if et == QtCore.QEvent.KeyPress and owner._on_key(event):
                    return True
                if et == QtCore.QEvent.KeyRelease and owner._on_key_release(event):
                    return True
                return False

        self._keys = _Keys(self.win)
        self.app.installEventFilter(self._keys)

    def _key_press_event(self, event) -> None:
        if not self._on_key(event):
            self.QtWidgets.QMainWindow.keyPressEvent(self.win, event)

    def _key_release_event(self, event) -> None:
        if not self._on_key_release(event):
            self.QtWidgets.QMainWindow.keyReleaseEvent(self.win, event)

    def _typing_focus(self) -> bool:
        w = self.app.focusWidget()
        return isinstance(w, (self.QtWidgets.QLineEdit, self.QtWidgets.QPlainTextEdit, self.QtWidgets.QSpinBox))

    def _drive_blocked(self, key: str) -> bool:
        from avlite.c30_control.c36_keyboard import keyboard_drive_owns

        ctrl = getattr(self.exec, "controller", None) if self.exec is not None else None
        return bool(keyboard_drive_owns(ctrl, key))

    def _on_key(self, event) -> bool:
        """Handle one key press. Letter case matches the Tk catalog (``S`` is Shift+S)."""
        key = event.key()
        text = event.text()
        mods = event.modifiers()
        Qt = self.QtCore.Qt
        ctrl = bool(mods & Qt.ControlModifier)
        shift = bool(mods & Qt.ShiftModifier)

        if self._typing_focus() and key != Qt.Key_Escape:
            return False
        if key == Qt.Key_Escape:
            self.win.setFocus()
            return True
        sequence = self._pressed_sequence(event)
        extra = self._override_action(sequence)
        if extra and self._run_shortcut(extra):
            return True
        # A rebound shortcut no longer answers on its original key.
        if sequence and self._default_suppressed(sequence):
            return True
        if text == "Q":
            self.win.close()
            return True
        if ctrl and key == Qt.Key_S and not shift:
            self.config_bar.save_settings()
            return True
        if text == "S" or (key == Qt.Key_S and shift and not ctrl):
            self.setting.p60_shortcut_mode.set(not self.setting.p60_shortcut_mode.get())
            self.update_shortcut_mode()
            return True
        if text == "F":
            self.switch_profile()
            return True
        if text == "R":
            self.reload_stack(reload_code=True)
            return True
        if text == "r":
            self.stack_panels.planning._local_replan()
            return True
        if text == "T":
            self.config_bar.open_settings()
            return True
        if text == "t":
            self.stack_panels.execution.reset_exec()
            return True
        if text == "E":
            self.log_panel.toggle_height()
            return True
        if text == "x":
            self.stack_panels.execution.toggle_exec()
            return True
        if text == "c":
            self.stack_panels.execution.step_exec()
            return True
        if text == "L":
            self.log_panel.clear()
            return True
        if text == "k":
            self.log_panel.scroll_lines(-1)
            return True
        if text == "j":
            self.log_panel.scroll_lines(1)
            return True
        if text == "g":
            self.log_panel.scroll_top()
            return True
        if text == "G":
            self.log_panel.scroll_bottom()
            return True
        if ctrl and key == Qt.Key_U:
            self.log_panel.scroll_half_page(-1)
            return True
        if ctrl and key == Qt.Key_D:
            self.log_panel.scroll_half_page(1)
            return True
        if key == Qt.Key_Up and not self._drive_blocked("up"):
            self.log_panel.scroll_lines(-1)
            return True
        if key == Qt.Key_Down and not self._drive_blocked("down"):
            self.log_panel.scroll_lines(1)
            return True
        if key in (Qt.Key_Plus, Qt.Key_Equal) and (text in ("+", "=") or not text):
            if ctrl:
                self.setting.p66_frenet_zoom = max(5.0, float(self.setting.p66_frenet_zoom) - 5.0)
            else:
                self.setting.p66_xy_zoom = max(5.0, float(self.setting.p66_xy_zoom) - 5.0)
            self.update_ui()
            return True
        if key == Qt.Key_Minus or text == "-":
            if ctrl:
                self.setting.p66_frenet_zoom = min(500.0, float(self.setting.p66_frenet_zoom) + 5.0)
            else:
                self.setting.p66_xy_zoom = min(500.0, float(self.setting.p66_xy_zoom) + 5.0)
            self.update_ui()
            return True
        if text == "w" and not self._drive_blocked("w"):
            self.stack_panels.control._drive(3, 0)
            return True
        if text == "s" and not self._drive_blocked("s"):
            self.stack_panels.control._drive(-3, 0)
            return True
        if text == "a" and not self._drive_blocked("a"):
            self.stack_panels.control._drive(0, 0.7)
            return True
        if text == "d" and not self._drive_blocked("d"):
            self.stack_panels.control._drive(0, -0.7)
            return True
        if text == "h":
            self.stack_panels.control._step()
            return True
        if text == "i":
            self.stack_panels.control._align()
            return True
        if text == "n" and self.exec and self.exec.local_planner:
            gt = self.exec.local_planner.global_trajectory
            if hasattr(gt, "step_wp"):
                gt.step_wp(1)
            self.update_ui()
            return True
        if text == "b" and self.exec and self.exec.local_planner:
            gt = self.exec.local_planner.global_trajectory
            if hasattr(gt, "step_wp"):
                gt.step_wp(-1)
            self.update_ui()
            return True
        return False

    def _pressed_sequence(self, event) -> str:
        Qt = self.QtCore.Qt
        key = event.key()
        text = event.text()
        ctrl = bool(event.modifiers() & Qt.ControlModifier)
        special = {
            Qt.Key_Up: "Up",
            Qt.Key_Down: "Down",
            Qt.Key_Escape: "Esc",
            Qt.Key_Plus: "+",
            Qt.Key_Equal: "+",
            Qt.Key_Minus: "-",
        }
        if ctrl:
            name = special.get(key)
            if name is None and Qt.Key_A <= key <= Qt.Key_Z:
                name = chr(key)
            return ("Ctrl+" + name) if name else ""
        if text and text.isprintable():
            return text
        return special.get(key, "")

    def _override_action(self, sequence: str) -> str:
        if not sequence:
            return ""
        from .settings import PluginSettings
        from .shortcuts import SHORTCUTS

        overrides = dict(getattr(PluginSettings, "p60_shortcuts", {}) or {})
        for shortcut in SHORTCUTS:
            bound = str(overrides.get(shortcut.id, "") or "")
            if bound and bound != shortcut.default and bound == sequence:
                return shortcut.id
        return ""

    def _default_suppressed(self, sequence: str) -> bool:
        """True when *sequence* is a catalog default that an override replaced."""
        from .settings import PluginSettings
        from .shortcuts import SHORTCUTS

        overrides = dict(getattr(PluginSettings, "p60_shortcuts", {}) or {})
        for shortcut in SHORTCUTS:
            bound = str(overrides.get(shortcut.id, "") or "")
            if bound and bound != shortcut.default and shortcut.default == sequence:
                return True
        return False

    def _run_shortcut(self, action: str) -> bool:
        if action == "quit":
            self.win.close()
            return True
        if action == "toggle_shortcuts":
            self.setting.p60_shortcut_mode.set(not self.setting.p60_shortcut_mode.get())
            self.update_shortcut_mode()
            return True
        if action == "switch_profile":
            self.switch_profile()
            return True
        if action == "reload":
            self.reload_stack(reload_code=True)
            return True
        if action == "open_settings":
            self.config_bar.open_settings()
            return True
        if action == "save_settings":
            self.config_bar.save_settings()
            return True
        if action == "focus":
            self.win.setFocus()
            return True
        if action == "replan":
            self.stack_panels.planning._local_replan()
            return True
        if action == "step_plan" and self.exec and self.exec.local_planner:
            gt = self.exec.local_planner.global_trajectory
            if hasattr(gt, "step_wp"):
                gt.step_wp(1)
            self.update_ui()
            return True
        if action == "step_plan_back" and self.exec and self.exec.local_planner:
            gt = self.exec.local_planner.global_trajectory
            if hasattr(gt, "step_wp"):
                gt.step_wp(-1)
            self.update_ui()
            return True
        if action == "zoom_in":
            self.setting.p66_xy_zoom = max(5.0, float(self.setting.p66_xy_zoom) - 5.0)
            self.update_ui()
            return True
        if action == "zoom_out":
            self.setting.p66_xy_zoom = min(500.0, float(self.setting.p66_xy_zoom) + 5.0)
            self.update_ui()
            return True
        if action == "zoom_in_frenet":
            self.setting.p66_frenet_zoom = max(5.0, float(self.setting.p66_frenet_zoom) - 5.0)
            self.update_ui()
            return True
        if action == "zoom_out_frenet":
            self.setting.p66_frenet_zoom = min(500.0, float(self.setting.p66_frenet_zoom) + 5.0)
            self.update_ui()
            return True
        if action == "toggle_exec":
            self.stack_panels.execution.toggle_exec()
            return True
        if action == "step_exec":
            self.stack_panels.execution.step_exec()
            return True
        if action == "reset_exec":
            self.stack_panels.execution.reset_exec()
            return True
        if action == "clear_log":
            self.log_panel.clear()
            return True
        if action == "toggle_log_height":
            self.log_panel.toggle_height()
            return True
        if action == "log_up":
            self.log_panel.scroll_lines(-1)
            return True
        if action == "log_down":
            self.log_panel.scroll_lines(1)
            return True
        if action == "log_top":
            self.log_panel.scroll_top()
            return True
        if action == "log_bottom":
            self.log_panel.scroll_bottom()
            return True
        if action == "log_page_up":
            self.log_panel.scroll_half_page(-1)
            return True
        if action == "log_page_down":
            self.log_panel.scroll_half_page(1)
            return True
        if action == "accelerate" and not self._drive_blocked("w"):
            self.stack_panels.control._drive(3, 0)
            return True
        if action == "decelerate" and not self._drive_blocked("s"):
            self.stack_panels.control._drive(-3, 0)
            return True
        if action == "steer_left" and not self._drive_blocked("a"):
            self.stack_panels.control._drive(0, 0.7)
            return True
        if action == "steer_right" and not self._drive_blocked("d"):
            self.stack_panels.control._drive(0, -0.7)
            return True
        if action == "step_control":
            self.stack_panels.control._step()
            return True
        if action == "align_control":
            self.stack_panels.control._align()
            return True
        return False

    def _on_key_release(self, event) -> bool:
        text = event.text()
        if text in ("a", "d") and not self._typing_focus() and not self._drive_blocked(text):
            self.stack_panels.control._drive(0, 0)
            return True
        return False

    def update_shortcut_mode(self) -> None:
        short = bool(self.setting.p60_shortcut_mode.get())
        self.shortcut_help.setVisible(short)
        self.stack_panels.setVisible(not short)

    def _on_check_update(self) -> None:
        from avlite.c60_apps.c66_app_update import AppUpdater

        try:
            latest = AppUpdater.latest()
            newer = AppUpdater.is_newer(latest)
        except Exception as exc:
            log.warning("Update check failed: %s", exc)
            self.QtWidgets.QMessageBox.warning(
                self.win, "Update", f"Could not check for updates.\n\n{exc}",
            )
            return
        if not newer:
            self.QtWidgets.QMessageBox.information(
                self.win, "Update", f"AVLite is up to date ({__version__}).",
            )
            return
        self.QtWidgets.QMessageBox.information(
            self.win, "Update", f"Update available: {__version__} → {latest}",
        )

    def update_views(self) -> None:
        show_g = bool(self.setting.p67_global_plan_view.get())
        show_l = bool(self.setting.local_plan_view_enabled())
        self.global_plot.setVisible(show_g)
        self.local_plot.setVisible(show_l)
        self.local_plot.update_visibility()
        hide_menu = bool(getattr(self.setting.schema, "p60_hide_menubar", False))
        self.win.menuBar().setVisible(not hide_menu)
        if hasattr(self, "config_bar"):
            self.config_bar.sync_from_settings()
        self.update_shortcut_mode()

    def local_plan_view_enabled(self) -> bool:
        return self.setting.local_plan_view_enabled()

    def _on_ui_timer(self) -> None:
        """Timer path. Plots are capped at 30 Hz; readouts still follow the timer."""
        self.update_ui(throttle=True)

    def update_ui(self, throttle: bool = False) -> None:
        if self.exec is None or self._in_ui_update:
            return
        self._in_ui_update = True
        try:
            self._update_ui_body(throttle)
        finally:
            self._in_ui_update = False

    def _update_ui_body(self, throttle: bool) -> None:
        t0 = time.perf_counter()
        # Tk caps plot redraws at ~30 Hz (p61 update_ui) and, while the executor
        # is running, only polls the UI at 20 Hz. Direct calls (tests, mouse)
        # still redraw immediately.
        do_plot = (not throttle) or (t0 - self._last_plot_t) >= (1.0 / 30.0)
        if do_plot:
            latency = 0.0
            if self.setting.p67_global_plan_view.get():
                latency += self.global_plot.redraw()
            if self.setting.local_plan_view_enabled():
                latency += self.local_plot.redraw()
            self._last_plot_t = t0
            self.setting.plot_latency_ms.set(f"{latency:7.1f}")

        ego = self.exec.ego_state
        # Fixed field widths: proportional digits used to change QLabel size hints
        # and grow the window whenever a value gained a character.
        self.setting.vehicle_state.set(
            f"Loc: ({ego.x:+11.2f}, {ego.y:+11.2f}), Vel: {ego.velocity:8.2f} "
            f"({ego.velocity * 3.6:8.2f} km/h), θ: {ego.theta:+7.1f}"
        )
        if self.exec.local_planner is not None:
            self.setting.current_wp.set(f"{int(self.exec.local_planner.global_trajectory.current_wp):6d}")
            self.setting.lap.set(f"{int(self.exec.local_planner.lap):6d}")
        if self.exec.controller is not None:
            self.setting.cte_vel.set(float(getattr(self.exec.controller, "cte_velocity", 0.0) or 0.0))
            self.setting.cte_steer.set(float(getattr(self.exec.controller, "cte_steer", 0.0) or 0.0))
            cmd = getattr(self.exec.controller, "cmd", None)
            if cmd is not None:
                self.setting.cmd_accel.set(float(getattr(cmd, "acceleration", 0.0) or 0.0))
                self.setting.cmd_steer.set(float(getattr(cmd, "steer", 0.0) or 0.0))
        self.setting.elapsed_real_time.set(f"{self.exec.elapsed_real_time:8.2f}")
        self.setting.elapsed_sim_time.set(f"{self.exec.elapsed_sim_time:8.2f}")
        self.setting.replan_fps.set(f"{self.exec.planner_fps:7.1f}")
        self.setting.control_fps.set(f"{self.exec.control_fps:7.1f}")
        self.setting.perception_fps.set(f"{self.exec.perception_fps:7.1f}")

        self._ui_frames += 1
        elapsed = time.perf_counter() - self._ui_t0
        if elapsed >= 0.5:
            self.setting.ui_fps.set(f"{self._ui_frames / elapsed:7.1f}")
            self._ui_frames = 0
            self._ui_t0 = time.perf_counter()
        if do_plot:
            log.debug("UI update %.2f ms (plot %.2f)", (time.perf_counter() - t0) * 1000, latency)

    def apply_global_plan(
        self, global_plan: GlobalPlan, ego_xy: Optional[tuple[float, float]] = None
    ) -> None:
        if self.exec is None:
            return
        if "apply_global_plan" in vars(type(self.exec)):
            self.exec.apply_global_plan(global_plan, ego_xy=ego_xy)
            return
        if global_plan is None or global_plan.trajectory is None or len(global_plan.trajectory.path_s) == 0:
            log.error("apply_global_plan: invalid plan")
            return
        ego_xy = ego_xy if ego_xy is not None else (self.exec.ego_state.x, self.exec.ego_state.y)
        if self.exec.local_planner:
            self.exec.local_planner.set_global_plan(global_plan, ego_xy=ego_xy)
        if self.exec.controller:
            self.exec.controller.set_trajectory_tracker(global_plan.trajectory)
            self.exec.controller.reset()

    def replan_global(self) -> None:
        if self.exec is None or not self.exec.global_planner:
            log.error("Global replan failed: no global planner.")
            return
        goal = self.exec.global_planner.goal_point
        if goal is not None:
            self.exec.global_planner.set_start_goal((self.exec.ego_state.x, self.exec.ego_state.y), goal)
        new_plan = self.exec.global_planner.plan(
            perception_model=self.exec.pm,
            sensors=self.exec.world.get_sensor_frame(),
        )
        if new_plan is None or new_plan.trajectory is None:
            log.error("Global replan failed: planner returned no valid plan.")
            return
        self.apply_global_plan(new_plan)
        self.update_ui()

    def teleport_ego(self, x, y, theta=None) -> None:
        self.exec.world.teleport_ego(x, y, theta)
        self.exec.pm.ego_vehicle.copy_from(self.exec.world.get_ego_state())

    def apply_world_control(self, cmd, dt) -> None:
        self.exec.world.control_ego_state(cmd=cmd, dt=dt)
        self.exec.pm.ego_vehicle.copy_from(self.exec.world.get_ego_state())

    def spawn_agent(self, agent_state: AgentState) -> None:
        if self.exec is None:
            return
        global_plan = self.exec.local_planner.global_plan if self.exec.local_planner else None
        self.exec.world.spawn_agent(agent_state, global_plan=global_plan)

    def _pin_self_as_community_plugin(self) -> None:
        """Keep this package listed so stack sync does not unload the running UI."""
        from pathlib import Path

        plugin_dir = str(Path(__file__).resolve().parent)
        plugins = dict(AppSettings.c62_community_plugins or {})
        plugins.setdefault(self.hosting_plugin_name, plugin_dir)
        AppSettings.c62_community_plugins = plugins

    def load_settings(self, only_stack: bool = False, profile: str | None = None) -> None:
        if profile:
            self.setting.c60_selected_profile.set(profile)
        else:
            profile = self.setting.c60_selected_profile.get()
        load_setting(AppSettings, profile=profile)
        self._pin_self_as_community_plugin()
        self.setting.sync_app_from_singleton()
        if not only_stack:
            load_setting(self.setting.schema, profile=profile)
            # Push plugin schema defaults into runtime vars where applicable
            from .settings import PluginSettings

            self.setting.p60_dark_mode.set(bool(PluginSettings.p60_dark_mode))
            self.setting.p60_ui_refresh_hz = float(PluginSettings.p60_ui_refresh_hz)
            self.setting.p66_xy_zoom = float(PluginSettings.p66_xy_zoom)
            self.setting.p66_frenet_zoom = float(PluginSettings.p66_frenet_zoom)
            self.setting.p66_global_zoom = float(PluginSettings.p66_global_zoom)
            self.setting.pull_plugin_schema()
        load_stack_settings(profile=profile)
        self._pin_self_as_community_plugin()
        self.setting.sync_stack_from_singletons()
        ConfigPaths.set_startup_profile(profile)
        self.loaded_profile = profile
        self.setting.profile_list = list_profiles(AppSettings)
        if hasattr(self, "config_bar"):
            self.config_bar.sync_from_settings()
        log.info("Loaded settings from profile: %s", profile)

    def refresh_pipeline(self) -> None:
        if self.exec is None:
            return
        from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline as PercPipe
        from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningPipeline as LocPipe

        if type(self.exec.perception).__name__ == "PerceptionPipeline":
            self.exec.perception = PercPipe(self.exec.pm)
        if self.exec.local_planner is not None and type(self.exec.local_planner).__name__ == "LocalPlanningPipeline":
            lp = self.exec.local_planner
            self.exec.local_planner = LocPipe(global_plan=lp.global_plan, env=self.exec.pm)
        try:
            self.exec.validate_stack()
        except ValueError as e:
            log.error("Pipeline refresh failed: %s", e)
        self.update_ui()

    def reload_stack(self, reload_code: bool = True) -> None:
        self.stack_panels.execution.stop_exec()
        error = None
        try:
            if reload_code:
                reload_lib(exclude_settings=True, reload_plugins=self.setting.c62_load_plugins.get())
            self.exec = executor_factory(
                executer_type=self.setting.executer_type.get(),
                bridge=self.setting.execution_bridge.get(),
                perception_strategy_name=self.setting.perception_type.get(),
                localization_strategy_name=self.setting.localization_type.get(),
                mapping_strategy_name=self.setting.mapping_type.get(),
                global_planner_strategy_name=self.setting.global_planner_type.get(),
                local_planner_strategy_name=self.setting.local_planner_type.get(),
                controller_strategy_name=self.setting.controller_type.get(),
                execution_task_names=self.setting.execution_task_names(),
                perception_dt=self.setting.perception_dt.get(),
                localization_dt=self.setting.localization_dt.get(),
                replan_dt=self.setting.replan_dt.get(),
                control_dt=self.setting.control_dt.get(),
                map_file=ExecutionSettings.c40_map,
                default_global_trajectory_file=self.setting.default_global_plan_file.get(),
                load_plugins=self.setting.c62_load_plugins.get(),
                async_combined_perception_planning=ExecutionSettings.c40_async_combined_perception_planning,
            )
        except Exception as e:
            error = e
            log.error("Error reloading stack: %s", e, exc_info=True)
        finally:
            self.local_plot.reset()
            self.global_plot.reset()
            self.stack_panels.refresh()
            self.update_views()
            if self.exec is not None:
                self.update_ui()
        if error is not None:
            self.QtWidgets.QMessageBox.critical(self.win, "Reload failed", str(error))

    def step_control(self) -> None:
        """One controller tick, then apply the command to the plant (Tk Control Step)."""
        if self.exec is None or self.exec.controller is None:
            return
        local_plan = self.exec.local_planner.get_local_plan() if self.exec.local_planner else None
        cmd = self.exec.controller.control(
            self.exec.ego_state,
            local_plan,
            control_dt=float(self.setting.sim_dt.get()),
            perception_model=self.exec.pm,
            sensors=self.exec.world.get_sensor_frame(),
        )
        self.apply_world_control(cmd, dt=float(self.setting.sim_dt.get()))
        self.update_ui()

    def align_control(self) -> None:
        """Snap ego onto the local planner location and reset the controller."""
        if self.exec is None or self.exec.controller is None or self.exec.local_planner is None:
            return
        x, y = self.exec.local_planner.location_xy
        self.teleport_ego(x, y)
        self.exec.controller.reset()
        self.update_ui()

    def drive(self, acceleration: float = 0.0, steer: float = 0.0) -> None:
        from avlite.c30_control.c31_control_model import ControlCommand

        if self.exec is None:
            return
        self.apply_world_control(
            ControlCommand(acceleration=acceleration, steer=steer),
            dt=float(self.setting.sim_dt.get()),
        )
        self.update_ui()

    def step_plan(self) -> None:
        if self.exec is None or self.exec.local_planner is None:
            return
        self.exec.local_planner.step_wp()
        gt = self.exec.local_planner.global_trajectory
        self.setting.current_wp.set(str(gt.next_wp - 1))
        self.update_ui()

    def step_plan_back(self) -> None:
        if self.exec is None or self.exec.local_planner is None:
            return
        wp = max(0, int(self.exec.local_planner.global_trajectory.current_wp) - 1)
        self.set_waypoint(wp)

    def set_waypoint(self, wp: int) -> None:
        if self.exec is None or self.exec.local_planner is None:
            return
        self.exec.local_planner.reset(wp=int(wp))
        self.setting.current_wp.set(str(int(wp)))
        self.update_ui()

    def align_plan(self) -> None:
        if self.exec is None or self.exec.local_planner is None:
            return
        self.exec.local_planner.step(self.exec.world.get_ego_state())
        self.update_ui()

    def local_replan(self) -> None:
        if self.exec is None or self.exec.local_planner is None:
            return
        self.exec.local_planner.replan(
            perception_model=self.exec.pm,
            sensors=self.exec.world.get_sensor_frame(),
        )
        self.update_ui()

    def save_global_plan(self, path: str | None = None) -> str | None:
        if self.exec is None or self.exec.global_planner is None:
            return None
        plan = self.exec.global_planner.global_plan
        if path is None:
            from avlite.c60_apps.c68_paths import DataPaths
            from datetime import datetime

            data_dir = DataPaths.user_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            path = str(data_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_global_plan.json")
        plan.to_file(path)
        log.info("Saved global plan to %s", path)
        return path

    def save_occupancy_map(self, path: str | None = None) -> str | None:
        from avlite.c10_perception.c11_perception_model import OccupancyMap

        if self.exec is None:
            return None
        om = getattr(self.exec.pm, "occupancy_map", None)
        if not isinstance(om, OccupancyMap):
            fallback = getattr(self.exec.pm, "map", None)
            om = fallback if isinstance(fallback, OccupancyMap) else None
        if om is None:
            log.info("No occupancy map to save")
            return None
        if path is None:
            from avlite.c60_apps.c68_paths import DataPaths
            from datetime import datetime

            data_dir = DataPaths.user_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            path = str(data_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_occupancy.json")
        om.to_file(path)
        return path

    def set_start_pose(self) -> None:
        from avlite.c60_apps.c65_setting_utils import save_setting

        if self.exec is None:
            return
        ego = self.exec.world.get_ego_state()
        stack = self.exec.ego_state
        ExecutionSettings.c40_start_pose = [ego.x, ego.y, ego.theta]
        world_v, stack_v = ego.velocity, stack.velocity
        ego.velocity = 0.0
        stack.velocity = 0.0
        ego.set_start()
        stack.set_start()
        ego.velocity = world_v
        stack.velocity = stack_v
        profile = self.setting.c60_selected_profile.get()
        save_setting(ExecutionSettings, profile=profile)
        log.info("Start pose saved to profile %r", profile)

    def set_default_map(self, path: str) -> None:
        self.setting.default_map_file.set(path)
        ExecutionSettings.c40_map = path
        self.reload_stack(reload_code=False)

    def set_default_global_plan(self, path: str) -> None:
        self.setting.default_global_plan_file.set(path)
        ExecutionSettings.c40_global_trajectory = path
        self.reload_stack(reload_code=False)

    def set_capability_filters(self, world: list[str] | None = None, stack: list[str] | None = None) -> None:
        if world is not None:
            ExecutionSettings.c41_world_capabilities = list(world)
        if stack is not None:
            ExecutionSettings.c41_world_stack_capabilities = list(stack)
        self.update_ui()

    def switch_profile(self) -> None:
        self.stack_panels.execution.stop_exec()
        nxt = self.setting.p60_next_profile.get() or "default"
        self.load_settings(profile=nxt)
        self.reload_stack(reload_code=False)
        self.update_ui()


def launch(*, offscreen: bool = False) -> int:
    """Create the visualizer and run the Qt event loop."""
    from .qt_binding import configure_qt_env

    configure_qt_env(offscreen=offscreen)
    window = VisualizerQtWindow()
    return window.exec_app()
