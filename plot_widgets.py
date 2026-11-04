"""High-performance pyqtgraph plot widgets (no Matplotlib).

Each frame updates PlotDataItem/ScatterPlotItem via ``setData`` — O(n) GPU/CPU
path instead of rebuilding a Matplotlib figure. Suitable for 60+ FPS UI refresh
while AsyncThreadedExecuter / SyncExecuter.step runs off or between UI ticks.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import numpy as np


if TYPE_CHECKING:
    from .main_window import VisualizerQtWindow

log = logging.getLogger(__name__)



def _closed_poly(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if corners is None or len(corners) == 0:
        return np.array([]), np.array([])
    xs = np.append(corners[:, 0], corners[0, 0])
    ys = np.append(corners[:, 1], corners[0, 1])
    return xs, ys


def _safe_xy(xs, ys) -> tuple[np.ndarray, np.ndarray]:
    if xs is None or ys is None:
        return np.array([]), np.array([])
    x = np.asarray(xs, dtype=float).ravel()
    y = np.asarray(ys, dtype=float).ravel()
    n = min(len(x), len(y))
    return x[:n], y[:n]


def _agent_list(pm) -> list:
    """Perception stores NPCs on ``agent_vehicles``. ``agents`` is not a field."""
    if pm is None:
        return []
    vehicles = getattr(pm, "agent_vehicles", None)
    if vehicles:
        return list(vehicles)
    return list(getattr(pm, "agents", []) or [])

_WIDGETS_READY = False


def _install_widgets() -> None:
    """Bind Qt classes on first use so community-plugin import does not init Qt."""
    global _WIDGETS_READY, QtCore, QtGui, QtWidgets, pg, GlobalPlotWidget, LocalPlotWidget
    if _WIDGETS_READY:
        return
    from .qt_binding import import_pyqtgraph, import_qt

    QtCore, QtGui, QtWidgets = import_qt()
    pg = import_pyqtgraph()

    _WIDGETS_READY = True

    class GlobalPlotWidget(pg.PlotWidget):
        """Global map / raceline / HD-map overview with ego marker."""

        def __init__(self, root: "VisualizerQtWindow", parent=None):
            super().__init__(parent=parent, background=root.setting.p60_bg_color)
            self.root = root
            self.setLabel("bottom", "x (m)")
            self.setLabel("left", "y (m)")
            self.showGrid(x=True, y=True, alpha=0.25)
            self.setAspectLocked(True)
            self._center_delta = np.array([0.0, 0.0])
            self._map_bounds: Optional[tuple[float, float, float, float]] = None
            self._drag_pan = False
            self._pan_anchor: Optional[tuple[float, float]] = None
            self._teleport_anchor: Optional[tuple[float, float]] = None
            self._orienting = False
            self._left_press_time = 0.0
            self._start_point: Optional[tuple[float, float]] = None
            self._hover_plan_xy: Optional[np.ndarray] = None
            self._hover_plan_v: Optional[np.ndarray] = None

            self.left_boundary = self.plotItem.plot(pen=pg.mkPen("#ff8c00", width=2))
            self.right_boundary = self.plotItem.plot(pen=pg.mkPen("#d2b48c", width=2))
            self.global_plan = self.plotItem.plot(pen=pg.mkPen("#4fc3f7", width=3))
            self.hdmap_lanes: list[pg.PlotDataItem] = []
            self.ego_box = self.plotItem.plot(pen=pg.mkPen("r", width=2))
            self.ego_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="r", symbolSize=10)
            self.past_path = self.plotItem.plot(pen=pg.mkPen("#66bb6a", width=1.5))
            self.start_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="b", symbolSize=14)
            self.goal_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="g", symbolSize=14)
            self.hover_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="w", symbolSize=8)
            self.hover_text = pg.TextItem(color="w", anchor=(0, 1))
            self.addItem(self.hover_text)
            self.hover_text.hide()
            self.orient_line = self.plotItem.plot(pen=pg.mkPen("#ffffff", width=1, style=QtCore.Qt.DashLine))

            from .plot_input import attach_plot_mouse

            attach_plot_mouse(self, self)

            self._last_draw_ms = 0.0
            self._fps = 0.0
            self._fps_t0 = time.perf_counter()
            self._fps_frames = 0

        def reset(self) -> None:
            self._map_bounds = None
            self._center_delta[:] = 0.0
            self._start_point = None
            self._drag_pan = False
            self._orienting = False
            self.orient_line.setData([], [])
            for item in self.hdmap_lanes:
                self.removeItem(item)
            self.hdmap_lanes.clear()
            for curve in (
                self.left_boundary, self.right_boundary, self.global_plan,
                self.ego_box, self.ego_marker, self.past_path,
                self.start_marker, self.goal_marker,
            ):
                curve.setData([], [])

        def update_theme(self, bg: str, fg: str) -> None:
            self.setBackground(bg)
            self.getAxis("bottom").setPen(fg)
            self.getAxis("left").setPen(fg)

        def _ctrl(self, modifiers) -> bool:
            return bool(modifiers & QtCore.Qt.ControlModifier)

        def _xy(self, pos) -> tuple[float, float]:
            return float(pos.x()), float(pos.y())

        def on_press(self, pos, button, modifiers) -> None:
            if self.root.exec is None:
                return
            x, y = self._xy(pos)
            self._status(f"Teleport {x:.1f},{y:.1f}")
            if button == QtCore.Qt.LeftButton:
                self._left_press_time = time.time()
                if self._ctrl(modifiers):
                    self._drag_pan = True
                    self._pan_anchor = (x, y)
                    self._orienting = False
                    self.orient_line.setData([], [])
                else:
                    self._drag_pan = False
                    self._teleport_anchor = (x, y)
                    self._orienting = False
                    self.orient_line.setData([], [])
            elif button == QtCore.Qt.RightButton:
                self._set_start_or_goal(x, y)

        def on_move(self, pos, buttons, modifiers) -> None:
            x, y = self._xy(pos)
            if not buttons:
                self._status(f"Teleport {x:.1f},{y:.1f}")
                self._on_mouse_moved(pos)
                return
            if self.root.exec is None:
                return
            if buttons & QtCore.Qt.LeftButton:
                # Ctrl at press, or Ctrl within 0.5s, pans. Otherwise the drag sets heading.
                late_pan = (
                    self._ctrl(modifiers)
                    and (not self._orienting or time.time() - self._left_press_time < 0.5)
                )
                if self._drag_pan or late_pan:
                    if self._pan_anchor is None or not self._drag_pan:
                        self._pan_anchor = (x, y)
                        self._drag_pan = True
                        self._orienting = False
                        self.orient_line.setData([], [])
                        return
                    factor = float(self.root.setting.p69_mouse_drag_slowdown_factor)
                    dx = -(x - self._pan_anchor[0]) * factor
                    dy = -(y - self._pan_anchor[1]) * factor
                    self._center_delta[0] += dx
                    self._center_delta[1] += dy
                    self._pan_anchor = (x, y)
                    self.redraw()
                    return
                if self._teleport_anchor is None:
                    return
                x0, y0 = self._teleport_anchor
                theta = float(np.arctan2(y - y0, x - x0))
                self._orienting = True
                self.orient_line.setData(
                    [x0, x0 + 4.0 * np.cos(theta)],
                    [y0, y0 + 4.0 * np.sin(theta)],
                )
                self.root.teleport_ego(x=x0, y=y0, theta=theta)
                self._step_local_planner()
                self.root.update_ui()

        def on_release(self, pos, button, modifiers) -> None:
            if button != QtCore.Qt.LeftButton or self.root.exec is None:
                self._drag_pan = False
                self._pan_anchor = None
                return
            if not self._drag_pan and not self._orienting and self._teleport_anchor is not None:
                x0, y0 = self._teleport_anchor
                self.root.teleport_ego(x=x0, y=y0)
                self._step_local_planner()
            self._drag_pan = False
            self._pan_anchor = None
            self._teleport_anchor = None
            self._orienting = False
            self.orient_line.setData([], [])
            if self.root.exec.controller is not None:
                self.root.exec.controller.reset()
            self.root.update_ui()

        def on_wheel(self, pos, delta: int) -> None:
            from .plot_input import zoom_step

            zoom = zoom_step(
                float(self.root.setting.p66_global_zoom), delta, lo=5.0, hi=5000.0,
            )
            bounds = self._map_bounds
            if bounds is not None and delta < 0:
                width = float(bounds[1] - bounds[0])
                height = float(bounds[3] - bounds[2])
                aspect = max(self.width() / max(self.height(), 1), 0.25)
                cap = min(width, height * aspect)
                if cap > 5.0:
                    zoom = min(zoom, cap)
            self.root.setting.p66_global_zoom = zoom
            if self.root.exec is not None and not self.root.setting.exec_running:
                self.redraw()

        def _set_start_or_goal(self, x: float, y: float) -> None:
            if self.root.exec is None or self.root.exec.global_planner is None:
                return
            if self._start_point is None:
                self._start_point = (x, y)
                self.start_marker.setData([x], [y])
                self.goal_marker.setData([], [])
                log.info("Set start: %s", self._start_point)
                return
            start = self._start_point
            goal = (x, y)
            self.start_marker.setData([start[0]], [start[1]])
            self.goal_marker.setData([goal[0]], [goal[1]])
            gp = self.root.exec.global_planner
            gp.set_start_goal(start_point=start, goal_point=goal)
            log.info("Set start: %s, goal: %s", start, goal)
            try:
                plan = gp.plan(
                    perception_model=self.root.exec.pm,
                    sensors=self.root.exec.world.get_sensor_frame(),
                )
            except Exception:
                log.exception("Global replan from right-click failed")
                self.root.update_ui()
                return
            plan = plan if plan is not None else getattr(gp, "global_plan", None)
            path = getattr(plan, "path", None)
            if path is not None and len(path) == 0:
                # Tk keeps the start so the next right-click tries another goal.
                log.warning("No global plan found. Please check the start and goal points.")
                return
            planner_type = str(self.root.setting.global_planner_type.get() or "")
            traj = getattr(plan, "trajectory", None) if plan is not None else None
            path_s = getattr(traj, "path_s", None) if traj is not None else None
            if "HDMap" in planner_type and path_s is not None and len(path_s) > 0:
                self.root.apply_global_plan(
                    plan, ego_xy=(self.root.exec.ego_state.x, self.root.exec.ego_state.y),
                )
            self._start_point = None
            self.root.update_ui()

        def _step_local_planner(self) -> None:
            planner = getattr(self.root.exec, "local_planner", None)
            if planner is None:
                return
            try:
                planner.step(state=self.root.exec.world.get_ego_state())
            except Exception:
                log.debug("local planner step after teleport failed", exc_info=True)

        def _status(self, text: str) -> None:
            status = getattr(self.root.setting, "perception_status_text", None)
            if status is not None:
                status.set(text)

        def _on_mouse_moved(self, pos):
            if self._hover_plan_xy is None or len(self._hover_plan_xy) == 0:
                self.hover_marker.setData([], [])
                self.hover_text.hide()
                return
            mouse = pos
            xy = self._hover_plan_xy
            d2 = (xy[:, 0] - mouse.x()) ** 2 + (xy[:, 1] - mouse.y()) ** 2
            i = int(np.argmin(d2))
            if d2[i] > (self.root.setting.p66_global_zoom * 0.05) ** 2:
                self.hover_marker.setData([], [])
                self.hover_text.hide()
                return
            x, y = xy[i]
            v = float(self._hover_plan_v[i]) if self._hover_plan_v is not None else 0.0
            self.hover_marker.setData([x], [y])
            self.hover_text.setText(f"{v:.1f} m/s ({v * 3.6:.0f} km/h)")
            self.hover_text.setPos(x, y)
            self.hover_text.show()

        def redraw(self) -> float:
            """Redraw from executer state. Returns draw latency in ms."""
            t0 = time.perf_counter()
            stack = self.root.exec
            if stack is None:
                return 0.0
            setting = self.root.setting
            ego = stack.ego_state

            # Map / plan geometry
            plan = None
            if stack.global_planner is not None:
                plan = getattr(stack.global_planner, "global_plan", None)
            if (plan is None or getattr(plan, "trajectory", None) is None) and stack.local_planner is not None:
                plan = getattr(stack.local_planner, "global_plan", None)

            if setting.p66_show_global_plan_boundaries.get() and plan is not None:
                self.left_boundary.setData(*_safe_xy(plan.left_boundary_x, plan.left_boundary_y))
                self.right_boundary.setData(*_safe_xy(plan.right_boundary_x, plan.right_boundary_y))
                xs = list(plan.left_boundary_x or []) + list(plan.right_boundary_x or [])
                ys = list(plan.left_boundary_y or []) + list(plan.right_boundary_y or [])
                if xs and ys:
                    self._map_bounds = (min(xs), max(xs), min(ys), max(ys))
            else:
                self.left_boundary.setData([], [])
                self.right_boundary.setData([], [])

            # HD map lanes (static-ish; redraw when empty)
            gp = stack.global_planner
            if gp is not None and hasattr(gp, "hdmap") and gp.hdmap is not None and not self.hdmap_lanes:
                for lane in gp.hdmap.lanes:
                    color = "#427b58" if lane.type == "driving" else (
                        "#af3a03" if lane.type == "shoulder" else "#777777"
                    )
                    alpha = 0.7 if lane.type == "driving" else 0.35
                    item = self.plotItem.plot(
                        lane.center_line[0], lane.center_line[1],
                        pen=pg.mkPen(pg.mkColor(color), width=1.5),
                    )
                    item.setOpacity(alpha)
                    self.hdmap_lanes.append(item)
                all_x = [c for lane in gp.hdmap.lanes for c in lane.center_line[0]]
                all_y = [c for lane in gp.hdmap.lanes for c in lane.center_line[1]]
                if all_x and all_y:
                    self._map_bounds = (min(all_x), max(all_x), min(all_y), max(all_y))

            if setting.p66_show_global_plan.get() and plan is not None and plan.trajectory is not None:
                traj = plan.trajectory
                xs, ys = _safe_xy(traj.path_x, traj.path_y)
                self.global_plan.setData(xs, ys)
                if len(xs):
                    self._hover_plan_xy = np.column_stack([xs, ys])
                    self._hover_plan_v = np.asarray(traj.velocity, dtype=float).ravel()[: len(xs)]
            else:
                self.global_plan.setData([], [])
                self._hover_plan_xy = None

            # Ego
            if setting.p66_show_state.get():
                bx, by = _closed_poly(ego.get_bb_corners())
                self.ego_box.setData(bx, by)
                self.ego_marker.setData([ego.x], [ego.y])
            else:
                self.ego_box.setData([], [])
                self.ego_marker.setData([], [])

            if setting.p66_show_past_locations.get() and stack.local_planner is not None:
                hist = getattr(stack.local_planner, "past_locations", None) or getattr(
                    stack, "past_locations", None
                )
                if hist:
                    pts = np.asarray(hist[-100:], dtype=float)
                    if pts.ndim == 2 and pts.shape[1] >= 2:
                        self.past_path.setData(pts[:, 0], pts[:, 1])
                    else:
                        self.past_path.setData([], [])
                else:
                    self.past_path.setData([], [])

            # View window
            zoom = float(setting.p66_global_zoom)
            cx = ego.x + self._center_delta[0]
            cy = ego.y + self._center_delta[1]
            if setting.p66_global_view_follow_planner.get() and stack.local_planner is not None:
                loc = getattr(stack.local_planner, "location_xy", None)
                if loc is not None:
                    cx, cy = float(loc[0]), float(loc[1])
            self.setXRange(cx - zoom, cx + zoom, padding=0)
            aspect = max(self.width() / max(self.height(), 1), 0.25)
            self.setYRange(cy - zoom / aspect, cy + zoom / aspect, padding=0)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._last_draw_ms = dt_ms
            self._fps_frames += 1
            elapsed = time.perf_counter() - self._fps_t0
            if elapsed >= 0.5:
                self._fps = self._fps_frames / elapsed
                self._fps_frames = 0
                self._fps_t0 = time.perf_counter()
            return dt_ms


    class LocalPlotWidget(QtWidgets.QWidget):
        """Dual-pane local XY + Frenet view."""

        def __init__(self, root: "VisualizerQtWindow", parent=None):
            super().__init__(parent)
            self.root = root
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            self.xy_plot = pg.PlotWidget(background=root.setting.p60_bg_color)
            self.frenet_plot = pg.PlotWidget(background=root.setting.p60_bg_color)
            for pw, xlabel, ylabel in (
                (self.xy_plot, "x (m)", "y (m)"),
                (self.frenet_plot, "s (m)", "d (m)"),
            ):
                pw.setLabel("bottom", xlabel)
                pw.setLabel("left", ylabel)
                pw.showGrid(x=True, y=True, alpha=0.25)
                pw.setAspectLocked(True)
            layout.addWidget(self.xy_plot, 1)
            layout.addWidget(self.frenet_plot, 1)

            self._init_curves()
            self._last_draw_ms = 0.0
            self._left_anchor: Optional[tuple[float, float]] = None
            self._right_anchor: Optional[tuple[float, float]] = None
            self._spawn_agent = None
            self._spawn_in_xy = True
            self._moved = False

            from .plot_input import attach_plot_mouse

            attach_plot_mouse(self.xy_plot, _PaneMouse(self, frenet=False))
            attach_plot_mouse(self.frenet_plot, _PaneMouse(self, frenet=True))

        def _init_curves(self) -> None:
            self.xy_left = self.xy_plot.plotItem.plot(pen=pg.mkPen("#ff8c00", width=2))
            self.xy_right = self.xy_plot.plotItem.plot(pen=pg.mkPen("#d2b48c", width=2))
            self.xy_ref = self.xy_plot.plotItem.plot(pen=pg.mkPen("#888888", width=2))
            self.xy_local = self.xy_plot.plotItem.plot(pen=pg.mkPen("#ef5350", width=3))
            self.xy_ego = self.xy_plot.plotItem.plot(pen=pg.mkPen("r", width=2))
            self.xy_agents = [
                self.xy_plot.plotItem.plot(pen=pg.mkPen("#1565c0", width=1.5)) for _ in range(12)
            ]
            self.xy_lidar = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(50, 255, 50, 200), pen=None)
            self.xy_plot.addItem(self.xy_lidar)
            self.xy_clusters = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(255, 255, 0, 220), pen=None)
            self.xy_plot.addItem(self.xy_clusters)
            self.xy_pred = self.xy_plot.plotItem.plot(
                pen=pg.mkPen("#ff9800", width=1.5, style=QtCore.Qt.DashLine)
            )
            self.xy_past = self.xy_plot.plotItem.plot(pen=pg.mkPen("#66bb6a", width=1.5))
            self.xy_lattice = self.xy_plot.plotItem.plot(
                pen=pg.mkPen("#90caf9", width=1), connect="finite"
            )

            self.sd_left = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#ff8c00", width=1))
            self.sd_right = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#d2b48c", width=1))
            self.sd_ref = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#888888", width=1))
            self.sd_local = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#ef5350", width=3))
            self.sd_ego = self.frenet_plot.plotItem.plot(pen=pg.mkPen("r", width=2))
            self.sd_agents = [
                self.frenet_plot.plotItem.plot(pen=pg.mkPen("#1565c0", width=1.5)) for _ in range(12)
            ]
            self.sd_lidar = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(50, 255, 50, 200), pen=None)
            self.frenet_plot.addItem(self.sd_lidar)
            ruler_pen = pg.mkPen("#00bcd4", width=1, style=QtCore.Qt.DashLine)
            self.xy_ruler = self.xy_plot.plotItem.plot(pen=ruler_pen)
            self.sd_ruler = self.frenet_plot.plotItem.plot(pen=ruler_pen)
            self.xy_ruler_text = pg.TextItem(color="#00bcd4", anchor=(0, 1))
            self.sd_ruler_text = pg.TextItem(color="#00bcd4", anchor=(0, 1))
            self.xy_plot.addItem(self.xy_ruler_text)
            self.frenet_plot.addItem(self.sd_ruler_text)
            self.xy_ruler_text.hide()
            self.sd_ruler_text.hide()

        def reset(self) -> None:
            for curve in (
                self.xy_left, self.xy_right, self.xy_ref, self.xy_local, self.xy_ego,
                self.xy_pred, self.xy_past, self.xy_lattice,
                self.sd_left, self.sd_right, self.sd_ref, self.sd_local, self.sd_ego,
            ):
                curve.setData([], [])
            for c in self.xy_agents + self.sd_agents:
                c.setData([], [])
            self.xy_lidar.setData([], [])
            self.xy_clusters.setData([], [])
            self.sd_lidar.setData([], [])
            self._hide_ruler()

        def update_theme(self, bg: str, fg: str) -> None:
            for pw in (self.xy_plot, self.frenet_plot):
                pw.setBackground(bg)
                pw.getAxis("bottom").setPen(fg)
                pw.getAxis("left").setPen(fg)

        def update_visibility(self) -> None:
            s = self.root.setting
            self.xy_plot.setVisible(bool(s.p67_show_local_global_view.get()))
            self.frenet_plot.setVisible(bool(s.p67_show_local_frenet_view.get()))

        def _status(self, text: str) -> None:
            status = getattr(self.root.setting, "perception_status_text", None)
            if status is not None:
                status.set(text)

        def _step_local(self) -> None:
            planner = getattr(self.root.exec, "local_planner", None)
            if planner is None:
                return
            try:
                planner.step(state=self.root.exec.world.get_ego_state())
            except Exception:
                log.debug("local planner step after teleport failed", exc_info=True)

        def handle_press(self, x: float, y: float, button, *, frenet: bool) -> None:
            if self.root.exec is None:
                return
            self._moved = False
            if button == QtCore.Qt.LeftButton:
                self._left_press(x, y, frenet=frenet)
            elif button == QtCore.Qt.RightButton:
                self._right_press(x, y, frenet=frenet)

        def handle_move(self, x: float, y: float, buttons, *, frenet: bool) -> None:
            if self.root.exec is None:
                return
            if not buttons:
                self._status(f"Spawn {x:.1f},{y:.1f}" if not frenet else f"Spawn S:{x:.1f},D:{y:.1f}")
                self._update_ruler(x, y, frenet=frenet)
                return
            self._hide_ruler()
            self._moved = True
            if buttons & QtCore.Qt.LeftButton and self._left_anchor is not None:
                self._left_drag(x, y, frenet=frenet)
            elif buttons & QtCore.Qt.RightButton and self._spawn_agent is not None:
                self._right_drag(x, y, frenet=frenet)

        def handle_release(self, button) -> None:
            if self.root.exec is None:
                return
            if button == QtCore.Qt.LeftButton and not self._moved and self._left_anchor is not None:
                # A click with no drag already teleported on press (Tk). Nothing else.
                pass
            if button in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
                if self.root.exec.controller is not None and button == QtCore.Qt.LeftButton:
                    self.root.exec.controller.reset()
                self._left_anchor = None
                self._right_anchor = None
                self._spawn_agent = None
                self.root.update_ui()

        def handle_wheel(self, delta: int, *, frenet: bool) -> None:
            from .plot_input import zoom_step

            if frenet:
                self.root.setting.p66_frenet_zoom = zoom_step(
                    float(self.root.setting.p66_frenet_zoom), delta, lo=5.0, hi=500.0,
                )
            else:
                self.root.setting.p66_xy_zoom = zoom_step(
                    float(self.root.setting.p66_xy_zoom), delta, lo=5.0, hi=500.0,
                )
            if self.root.exec is not None and not self.root.setting.exec_running:
                self.root.update_ui()

        def _left_press(self, x: float, y: float, *, frenet: bool) -> None:
            if frenet:
                traj = self.root.exec.local_planner.global_plan.trajectory
                wx, wy = traj.convert_sd_to_xy(x, y)
            else:
                wx, wy = x, y
            self._left_anchor = (wx, wy, x, y, frenet)
            self.root.teleport_ego(wx, wy)
            self._step_local()
            self.root.update_ui()
            self._status(f"Spawn {wx:.1f},{wy:.1f}")

        def _left_drag(self, x: float, y: float, *, frenet: bool) -> None:
            wx, wy, ax, ay, was_frenet = self._left_anchor
            if was_frenet != frenet:
                return
            if frenet:
                theta_sd = float(np.arctan2(y - ay, x - ax))
                traj = self.root.exec.local_planner.global_plan.trajectory
                _x, _y, theta = traj.convert_sd_orientation_to_xy_orientation(x, y, theta_sd)
                self.root.teleport_ego(wx, wy, theta)
            else:
                theta = float(np.arctan2(y - wy, x - wx))
                self.root.teleport_ego(wx, wy, theta)
            self._step_local()
            self.root.update_ui()

        def _right_press(self, x: float, y: float, *, frenet: bool) -> None:
            from avlite.c10_perception.c11_perception_model import AgentState

            ego = self.root.exec.ego_state
            if frenet:
                traj = self.root.exec.local_planner.global_plan.trajectory
                wx, wy = traj.convert_sd_to_xy(x, y)
                self._status(f"Spawn S:{x:.1f},D:{y:.1f}")
            else:
                wx, wy = x, y
                self._status(f"Spawn {x:.1f},{y:.1f}")
            agent = AgentState(x=float(wx), y=float(wy), theta=float(ego.theta), velocity=0)
            self.root.spawn_agent(agent)
            self._spawn_agent = agent
            self._right_anchor = (float(wx), float(wy), x, y, frenet)
            self._spawn_in_xy = not frenet
            self.root.update_ui()
            log.info("Spawned agent at (%.2f, %.2f)", wx, wy)

        def _right_drag(self, x: float, y: float, *, frenet: bool) -> None:
            wx, wy, ax, ay, was_frenet = self._right_anchor
            if was_frenet != frenet or self._spawn_agent is None:
                return
            if frenet:
                theta_sd = float(np.arctan2(y - ay, x - ax))
                traj = self.root.exec.local_planner.global_plan.trajectory
                _x, _y, theta = traj.convert_sd_orientation_to_xy_orientation(ax, ay, theta_sd)
            else:
                theta = float(np.arctan2(y - wy, x - wx))
            self._spawn_agent.theta = theta
            self.root.update_ui()

        def redraw(self) -> float:
            t0 = time.perf_counter()
            stack = self.root.exec
            if stack is None:
                return 0.0
            s = self.root.setting
            self.update_visibility()
            ego = stack.ego_state
            lp = stack.local_planner
            gt = getattr(lp, "global_trajectory", None) if lp is not None else None

            # Boundaries / reference
            plan = getattr(lp, "global_plan", None) if lp is not None else None
            if plan is not None and s.p66_show_race_boundary.get():
                self.xy_left.setData(*_safe_xy(plan.left_boundary_x, plan.left_boundary_y))
                self.xy_right.setData(*_safe_xy(plan.right_boundary_x, plan.right_boundary_y))
            else:
                self.xy_left.setData([], [])
                self.xy_right.setData([], [])

            if plan is not None and plan.trajectory is not None and s.p66_show_global_plan.get():
                self.xy_ref.setData(*_safe_xy(plan.trajectory.path_x, plan.trajectory.path_y))
                if gt is not None:
                    # Frenet reference: s along path, d≈0
                    sx = np.asarray(plan.trajectory.path_s, dtype=float)
                    self.sd_ref.setData(sx, np.zeros_like(sx))
                    if hasattr(plan, "left_boundary_d") and plan.left_boundary_d is not None:
                        sd_s = np.asarray(getattr(plan, "path_s", plan.trajectory.path_s), dtype=float)
                        n = min(len(sd_s), len(plan.left_boundary_d), len(plan.right_boundary_d))
                        self.sd_left.setData(sd_s[:n], np.asarray(plan.left_boundary_d[:n], dtype=float))
                        self.sd_right.setData(sd_s[:n], np.asarray(plan.right_boundary_d[:n], dtype=float))
            else:
                self.xy_ref.setData([], [])
                self.sd_ref.setData([], [])

            # Local plan
            if lp is not None and s.p66_show_local_plan.get():
                local_plan = getattr(lp, "local_plan", None)
                traj = getattr(local_plan, "trajectory", None) if local_plan is not None else None
                if traj is not None:
                    self.xy_local.setData(*_safe_xy(traj.path_x, traj.path_y))
                    if gt is not None:
                        try:
                            sd = gt.convert_xy_path_to_sd_path_np(
                                np.column_stack([traj.path_x, traj.path_y])
                            )
                            self.sd_local.setData(sd[:, 0], sd[:, 1])
                        except Exception:
                            self.sd_local.setData([], [])
                else:
                    self.xy_local.setData([], [])
                    self.sd_local.setData([], [])
            else:
                self.xy_local.setData([], [])
                self.sd_local.setData([], [])

            # Lattice
            if lp is not None and s.p66_show_local_lattice.get():
                edges = getattr(lp, "lattice_edges", None) or getattr(lp, "edges", None) or []
                xs: list[float] = []
                ys: list[float] = []
                for edge in edges[:200]:
                    pts = getattr(edge, "points_xy", None) or getattr(edge, "xy", None)
                    if pts is None:
                        continue
                    arr = np.asarray(pts, dtype=float)
                    if arr.ndim != 2 or arr.shape[0] < 2:
                        continue
                    xs.extend(arr[:, 0].tolist() + [np.nan])
                    ys.extend(arr[:, 1].tolist() + [np.nan])
                self.xy_lattice.setData(xs, ys)
            else:
                self.xy_lattice.setData([], [])

            # Ego + agents
            if s.p66_show_state.get():
                self.xy_ego.setData(*_closed_poly(ego.get_bb_corners()))
                if gt is not None:
                    try:
                        sd_c = gt.convert_xy_path_to_sd_path_np(ego.get_bb_corners())
                        self.sd_ego.setData(*_closed_poly(sd_c))
                    except Exception:
                        self.sd_ego.setData([], [])
                agents = _agent_list(stack.pm)[:12]
                if not agents and getattr(stack, "world", None) is not None:
                    agents = _agent_list(getattr(stack.world, "pm", None))[:12]
                for i, curve in enumerate(self.xy_agents):
                    if i < len(agents):
                        curve.setData(*_closed_poly(agents[i].get_bb_corners()))
                    else:
                        curve.setData([], [])
                if gt is not None:
                    for i, curve in enumerate(self.sd_agents):
                        if i < len(agents):
                            try:
                                sd_c = gt.convert_xy_path_to_sd_path_np(agents[i].get_bb_corners())
                                curve.setData(*_closed_poly(sd_c))
                            except Exception:
                                curve.setData([], [])
                        else:
                            curve.setData([], [])
            else:
                self.xy_ego.setData([], [])
                self.sd_ego.setData([], [])

            # LiDAR
            lidar_xy = None
            try:
                frame = stack.world.get_sensor_frame()
                lidar = getattr(frame, "lidar", None)
                pts = getattr(lidar, "points", None) if lidar is not None else None
                if pts is not None and len(pts):
                    arr = np.asarray(pts, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        lidar_xy = arr[:, :2]
            except Exception:
                lidar_xy = None

            if lidar_xy is not None and s.p66_show_lidar_global.get():
                self.xy_lidar.setData(lidar_xy[:, 0], lidar_xy[:, 1])
            else:
                self.xy_lidar.setData([], [])
            if lidar_xy is not None and s.p66_show_lidar_frenet.get() and gt is not None:
                try:
                    sd = gt.convert_xy_path_to_sd_path_np(lidar_xy)
                    self.sd_lidar.setData(sd[:, 0], sd[:, 1])
                except Exception:
                    self.sd_lidar.setData([], [])
            else:
                self.sd_lidar.setData([], [])

            # Predictions (first mode polylines flattened)
            if s.p67_show_prediction.get() and stack.pm is not None:
                pred_x: list[float] = []
                pred_y: list[float] = []
                for agent in _agent_list(stack.pm):
                    pred = getattr(agent, "prediction", None)
                    traj = None
                    if pred is not None:
                        if hasattr(pred, "trajectories") and pred.trajectories:
                            traj = pred.trajectories[0]
                        elif hasattr(pred, "trajectory"):
                            traj = pred.trajectory
                    if traj is None:
                        continue
                    xs = getattr(traj, "path_x", None)
                    ys = getattr(traj, "path_y", None)
                    if xs is None:
                        continue
                    pred_x.extend(list(xs) + [np.nan])
                    pred_y.extend(list(ys) + [np.nan])
                self.xy_pred.setData(pred_x, pred_y)
            else:
                self.xy_pred.setData([], [])

            # View windows
            zoom = float(s.p66_xy_zoom)
            cx, cy = ego.x, ego.y
            if s.p66_global_view_follow_planner.get() and lp is not None:
                loc = getattr(lp, "location_xy", None)
                if loc is not None:
                    cx, cy = float(loc[0]), float(loc[1])
            self.xy_plot.setXRange(cx - zoom, cx + zoom, padding=0)
            aspect = max(self.xy_plot.width() / max(self.xy_plot.height(), 1), 0.25)
            self.xy_plot.setYRange(cy - zoom / aspect, cy + zoom / aspect, padding=0)

            fzoom = float(s.p66_frenet_zoom)
            if gt is not None:
                try:
                    sd_ego = gt.convert_xy_to_sd(ego.x, ego.y)
                    cs, cd = float(sd_ego[0]), float(sd_ego[1])
                except Exception:
                    cs, cd = 0.0, 0.0
            else:
                cs, cd = 0.0, 0.0
            self.frenet_plot.setXRange(cs - fzoom, cs + fzoom, padding=0)
            self.frenet_plot.setYRange(cd - fzoom / 2, cd + fzoom / 2, padding=0)

            self._last_draw_ms = (time.perf_counter() - t0) * 1000.0
            return self._last_draw_ms

        def _hide_ruler(self) -> None:
            self.xy_ruler.setData([], [])
            self.sd_ruler.setData([], [])
            self.xy_ruler_text.hide()
            self.sd_ruler_text.hide()

        def _update_ruler(self, x: float, y: float, *, frenet: bool) -> None:
            """Paused distance from the ego front axle to the cursor, matching Tk."""
            if (
                self.root.exec is None
                or self.root.setting.exec_running
                or self._left_anchor is not None
                or self._spawn_agent is not None
            ):
                self._hide_ruler()
                return
            ego = self.root.exec.ego_state
            ctrl = self.root.exec.controller
            front = float(getattr(ctrl, "ego_distance_front_axle", 2.5) or 2.5)
            fx = float(ego.x) + front * float(np.cos(ego.theta))
            fy = float(ego.y) + front * float(np.sin(ego.theta))
            if not frenet:
                dist = float(np.hypot(x - fx, y - fy))
                self.xy_ruler.setData([fx, x], [fy, y])
                self.xy_ruler_text.setText(f"{dist:.1f} m")
                self.xy_ruler_text.setPos(x, y)
                self.xy_ruler_text.show()
                self.sd_ruler.setData([], [])
                self.sd_ruler_text.hide()
                return
            traj = getattr(getattr(self.root.exec, "local_planner", None), "global_trajectory", None)
            if traj is None:
                self._hide_ruler()
                return
            try:
                fs, fd = traj.convert_xy_to_sd(fx, fy)
                cx, cy = traj.convert_sd_to_xy(x, y)
            except Exception:
                self._hide_ruler()
                return
            dist = float(np.hypot(float(cx) - fx, float(cy) - fy))
            self.sd_ruler.setData([float(fs), x], [float(fd), y])
            self.sd_ruler_text.setText(f"{dist:.1f} m")
            self.sd_ruler_text.setPos(x, y)
            self.sd_ruler_text.show()
            self.xy_ruler.setData([], [])
            self.xy_ruler_text.hide()

    class _PaneMouse:
        def __init__(self, plot: "LocalPlotWidget", *, frenet: bool):
            self.plot = plot
            self.frenet = frenet

        def on_press(self, pos, button, modifiers) -> None:
            self.plot.handle_press(float(pos.x()), float(pos.y()), button, frenet=self.frenet)

        def on_move(self, pos, buttons, modifiers) -> None:
            self.plot.handle_move(float(pos.x()), float(pos.y()), buttons, frenet=self.frenet)

        def on_release(self, pos, button, modifiers) -> None:
            self.plot.handle_release(button)

        def on_wheel(self, pos, delta: int) -> None:
            self.plot.handle_wheel(delta, frenet=self.frenet)


def __getattr__(name: str):
    if name in {'GlobalPlotWidget', 'LocalPlotWidget'}:
        _install_widgets()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
