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


def _nan_join(paths: list[np.ndarray]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for path in paths:
        arr = np.asarray(path, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            continue
        xs.extend(arr[:, 0].tolist())
        xs.append(np.nan)
        ys.extend(arr[:, 1].tolist())
        ys.append(np.nan)
    return xs, ys


def _to_sd(gt, xy: np.ndarray) -> np.ndarray:
    if gt is None or xy is None or len(xy) == 0:
        return np.zeros((0, 2))
    try:
        sd = np.asarray(gt.convert_xy_path_to_sd_path_np(np.asarray(xy, dtype=float)[:, :2]), dtype=float)
    except Exception:
        return np.zeros((0, 2))
    if sd.ndim != 2 or sd.shape[1] < 2:
        return np.zeros((0, 2))
    return sd[:, :2]


def _segments_xy(segs) -> tuple[list[float], list[float]]:
    arr = np.asarray(segs, dtype=float) if segs is not None else np.zeros((0, 2, 2))
    if arr.ndim != 3 or arr.shape[-1] < 2 or arr.shape[-2] < 2:
        return [], []
    paths = [arr[i, :, :2] for i in range(arr.shape[0])]
    return _nan_join(paths)


def _track_segments(world, plan):
    segs = getattr(world, "boundary_segments", None) if world is not None else None
    if segs is not None and len(segs):
        return segs
    if plan is None:
        return None
    try:
        from avlite.c40_execution.c46_basic_sim import boundary_segments_from_global_plan

        return boundary_segments_from_global_plan(plan)
    except Exception:
        return None


def _cluster_xy(pm) -> np.ndarray | None:
    raw = getattr(pm, "detection_clusters", None) if pm is not None else None
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        parts = []
        for item in raw:
            arr = np.asarray(item, dtype=float)
            if arr.ndim == 1 and arr.size >= 2:
                parts.append(arr[:2].reshape(1, 2))
            elif arr.ndim == 2 and arr.shape[1] >= 2 and len(arr):
                parts.append(arr[:, :2])
        if not parts:
            return None
        return np.vstack(parts)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr):
        return arr[:, :2]
    return None


def _forecast_modes(pred) -> list[tuple[object, int, np.ndarray, float]]:
    """``(agent_id, mode, Nx2 path, weight)`` from Single/Multi/GP/GMM forecasts."""
    if pred is None:
        return []
    weights = getattr(pred, "weights", None) or {}
    store = getattr(pred, "trajectories", None)
    if store is None:
        store = getattr(pred, "means", None)
    if not isinstance(store, dict):
        return []
    out: list[tuple[object, int, np.ndarray, float]] = []
    for aid, modes in store.items():
        arr = np.asarray(modes, dtype=float)
        if arr.ndim == 2 and arr.shape[-1] >= 2:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        if arr.ndim != 3 or arr.shape[-1] < 2:
            continue
        w = np.asarray(weights.get(aid, []), dtype=float) if isinstance(weights, dict) else np.array([])
        for i in range(arr.shape[0]):
            weight = float(np.clip(w[i], 0.0, 1.0)) if i < len(w) else 1.0
            out.append((aid, i, arr[i, :, :2], weight))
    return out


def _ellipse_polygon(center: np.ndarray, cov: np.ndarray, percentile: float, n: int = 16) -> np.ndarray | None:
    if not 0.0 < percentile < 1.0:
        return None
    sym = np.asarray(cov, dtype=float)
    if sym.shape != (2, 2) or not np.all(np.isfinite(sym)):
        return None
    sym = 0.5 * (sym + sym.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, 0.0, None)
    if float(vals[-1]) <= 0.0:
        return None
    radii = np.sqrt(-2.0 * np.log(1.0 - percentile)) * np.sqrt(vals)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    local = np.column_stack((np.cos(theta), np.sin(theta))) * radii
    polygon = np.asarray(center, dtype=float) + local @ vecs.T
    return np.vstack((polygon, polygon[:1]))


def _step_covariance(pred, agent_id, mode_index: int, step: int) -> np.ndarray | None:
    joint = getattr(pred, "covariance", None)
    if isinstance(joint, dict) and agent_id in joint:
        mat = np.asarray(joint[agent_id])
        i = 2 * step
        if mat.ndim == 2 and mat.shape[0] >= i + 2 and mat.shape[1] >= i + 2:
            return mat[i:i + 2, i:i + 2]
    covs = getattr(pred, "covariances", None)
    if isinstance(covs, dict) and agent_id in covs:
        mat = np.asarray(covs[agent_id])
        if mat.ndim == 4 and mode_index < mat.shape[0] and step < mat.shape[1]:
            return mat[mode_index, step]
    return None


def _occupancy_grid(pm):
    om = getattr(pm, "occupancy_map", None) if pm is not None else None
    if om is None or getattr(om, "grid", None) is None:
        fallback = getattr(pm, "map", None) if pm is not None else None
        if fallback is not None and getattr(fallback, "grid", None) is not None and hasattr(fallback, "origin_x"):
            om = fallback
    if om is None or getattr(om, "grid", None) is None or np.size(om.grid) == 0:
        return None
    return om


def _flow_image(pred):
    """Max-over-horizon occupancy, matching the Tk occupancy-flow painter."""
    if pred is None:
        return None
    flow = getattr(pred, "occupancy_flow", None)
    if flow is None:
        return None
    origin = (
        float(getattr(pred, "origin_x", 0.0)),
        float(getattr(pred, "origin_y", 0.0)),
        float(getattr(pred, "resolution", 0.25) or 0.25),
    )
    if isinstance(flow, dict):
        series = [steps for steps in flow.values() if steps]
        if not series:
            return None
        n_steps = max(len(steps) for steps in series)
        acc = None
        for step in range(n_steps):
            union = None
            for steps in series:
                if step >= len(steps) or np.size(steps[step]) == 0:
                    continue
                grid = np.clip(np.asarray(steps[step], dtype=float), 0.0, 1.0)
                union = grid if union is None else 1.0 - (1.0 - union) * (1.0 - grid)
            if union is None:
                continue
            acc = union if acc is None else np.maximum(acc, union)
        if acc is None:
            return None
        return acc, origin
    grids = [np.asarray(g, dtype=float) for g in flow if np.size(g)]
    if not grids:
        return None
    return np.maximum.reduce(grids), origin


def _lattice_edges(lp) -> list:
    lattice = getattr(lp, "lattice", None)
    edges = getattr(lattice, "edges", None) if lattice is not None else None
    if not edges:
        edges = getattr(lp, "lattice_edges", None) or getattr(lp, "edges", None) or []
    return list(edges)


def _edge_xy(edge) -> np.ndarray:
    traj = getattr(edge, "local_trajectory", None) or getattr(edge, "trajectory", None)
    if traj is not None and getattr(traj, "path_x", None) is not None:
        x, y = _safe_xy(traj.path_x, traj.path_y)
        return np.column_stack([x, y]) if len(x) else np.zeros((0, 2))
    pts = getattr(edge, "points_xy", None) or getattr(edge, "xy", None)
    if pts is None:
        return np.zeros((0, 2))
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2))
    return arr[:, :2]


def _edge_sd(edge) -> np.ndarray:
    traj = getattr(edge, "local_trajectory", None)
    if traj is None:
        return np.zeros((0, 2))
    x, y = _safe_xy(getattr(traj, "path_s_from_parent", None), getattr(traj, "path_d_from_parent", None))
    return np.column_stack([x, y]) if len(x) else np.zeros((0, 2))


def _edge_kind(edge, lp) -> str:
    if getattr(edge, "collision", False) or getattr(edge, "boundary_violation", False):
        return "collision"
    feasible = True
    fn = getattr(lp, "_is_curvature_feasible", None)
    if callable(fn):
        try:
            feasible = bool(fn(edge))
        except Exception:
            feasible = True
    return "feasible" if feasible else "kinematic"


def _local_plan_paths(lp) -> list[np.ndarray]:
    paths: list[np.ndarray] = []
    if lp is None:
        return paths
    if getattr(lp, "selected_local_plan", None) is not None:
        edge = lp.selected_local_plan
        for _ in range(8):
            if edge is None:
                break
            xy = _edge_xy(edge)
            if len(xy):
                paths.append(xy)
            edge = getattr(edge, "selected_next_local_plan", None)
        if paths:
            return paths
    local_plan = getattr(lp, "local_plan", None)
    getter = getattr(lp, "get_local_plan", None)
    if callable(getter):
        try:
            got = getter()
            if got is not None:
                local_plan = got
        except Exception:
            pass
    if local_plan is None:
        return paths
    traj = local_plan.as_trajectory() if hasattr(local_plan, "as_trajectory") else getattr(local_plan, "trajectory", None)
    if traj is None or getattr(traj, "path_x", None) is None:
        return paths
    x, y = _safe_xy(traj.path_x, traj.path_y)
    if len(x):
        paths.append(np.column_stack([x, y]))
    return paths


def _velocity_colors(speed: np.ndarray, scale: str) -> list[tuple[int, int, int, int]]:
    v = np.asarray(speed, dtype=float)
    if scale == "absolute":
        t = np.clip(v / 30.0, 0.0, 1.0)
    else:
        finite = v[np.isfinite(v)]
        if len(finite) == 0 or float(np.max(finite) - np.min(finite)) < 1e-6:
            t = np.zeros_like(v)
        else:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    colors = []
    for u in t:
        if not np.isfinite(u):
            colors.append((80, 80, 80, 0))
            continue
        r = int(40 + 215 * u)
        g = int(80 + 140 * (1.0 - abs(u - 0.5) * 2.0))
        b = int(220 * (1.0 - u))
        colors.append((r, g, b, 230))
    return colors


def _pin_plot_chrome(plot) -> None:
    """Freeze axis pixels and turn off auto-range.

    pyqtgraph's AxisItem grows and shrinks with the current tick strings
    (``autoExpandTextSpace`` / ``autoReduceTextSpace``). That changes the
    ViewBox width by tens of pixels, which changes a locked aspect ratio and
    the PlotWidget size hint. The hint feeds the window minimum, so the whole
    window jumps when coordinates gain a digit. A fixed axis size breaks the loop.
    """
    if getattr(plot, "_avlite_chrome_pinned", False):
        return
    plot.hideButtons()
    plot.disableAutoRange()
    plot.setAspectLocked(False)
    for side in ("left", "bottom"):
        axis = plot.getAxis(side)
        axis.enableAutoSIPrefix(False)
        axis.setStyle(autoExpandTextSpace=False, autoReduceTextSpace=False)
        if side == "left":
            axis.setWidth(78)
        else:
            axis.setHeight(44)
    if plot.width() > 0:
        plot._avlite_chrome_pinned = True


def _set_equal_view(plot, cx: float, cy: float, half_x: float) -> None:
    """Pan to ``(cx, cy)`` at a constant scale. No auto-range.

    The pixel aspect is cached for the current widget size so a 1px axis
    tweak cannot change the zoom. Following the ego still updates the center.
    """
    _pin_plot_chrome(plot)
    vb = plot.getViewBox()
    widget_wh = (int(plot.width()), int(plot.height()))
    cached = getattr(plot, "_avlite_aspect", None)
    if cached is None or cached[0] != widget_wh:
        rect = vb.boundingRect()
        w = int(rect.width())
        h = int(rect.height())
        if w < 2 or h < 2:
            w = max(widget_wh[0], 1)
            h = max(widget_wh[1], 1)
        aspect = max(w / float(max(h, 1)), 0.25)
        plot._avlite_aspect = (widget_wh, aspect)
    else:
        aspect = cached[1]
    half_x = float(half_x)
    half_y = half_x / aspect
    plot.setRange(
        xRange=(float(cx) - half_x, float(cx) + half_x),
        yRange=(float(cy) - half_y, float(cy) + half_y),
        padding=0.0,
        disableAutoRange=True,
    )

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
            self.global_plan_speed = pg.ScatterPlotItem(size=7, pen=None, pxMode=True)
            self.addItem(self.global_plan_speed)
            self.hdmap_lanes: list[pg.PlotDataItem] = []
            self._lane_cache: list[tuple] = []
            self.hd_closest = self.plotItem.plot(pen=pg.mkPen("#ffeb3b", width=3))
            self.hd_text = pg.TextItem(color="#ffeb3b", anchor=(0, 1))
            self.addItem(self.hd_text)
            self.hd_text.hide()
            self.ego_box = self.plotItem.plot(pen=pg.mkPen("r", width=2))
            self.ego_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="r", symbolSize=10)
            self.heading_line = self.plotItem.plot(pen=pg.mkPen("#ef5350", width=2))
            self.planner_marker = self.plotItem.plot(
                pen=None, symbol="t", symbolBrush="#ce93d8", symbolSize=12,
            )
            self.planner_text = pg.TextItem("L", color="#ce93d8", anchor=(0.5, 1.4))
            self.addItem(self.planner_text)
            self.planner_text.hide()
            self.past_path = self.plotItem.plot(pen=pg.mkPen("#66bb6a", width=1.5))
            self.start_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="b", symbolSize=14)
            self.goal_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="g", symbolSize=14)
            self.start_text = pg.TextItem("S", color="#64b5f6", anchor=(0.5, 1.6))
            self.goal_text = pg.TextItem("G", color="#81c784", anchor=(0.5, 1.6))
            self.addItem(self.start_text)
            self.addItem(self.goal_text)
            self.start_text.hide()
            self.goal_text.hide()
            self.g_wp_current = self.plotItem.plot(
                pen=None, symbol="x", symbolPen=pg.mkPen("#ffffff", width=2), symbolSize=14,
            )
            self.g_wp_next = self.plotItem.plot(
                pen=None, symbol="+", symbolPen=pg.mkPen("#90caf9", width=2), symbolSize=14,
            )
            self.hover_marker = self.plotItem.plot(pen=None, symbol="o", symbolBrush="w", symbolSize=8)
            self.hover_text = pg.TextItem(color="w", anchor=(0, 1))
            self.addItem(self.hover_text)
            self.hover_text.hide()
            self.orient_line = self.plotItem.plot(pen=pg.mkPen("#ffffff", width=1, style=QtCore.Qt.DashLine))

            from .plot_input import attach_plot_mouse

            attach_plot_mouse(self, self)
            _pin_plot_chrome(self)

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
            self._lane_cache.clear()
            self.global_plan_speed.setData([], [])
            self.hd_closest.setData([], [])
            self.hd_text.hide()
            self.planner_text.hide()
            self.start_text.hide()
            self.goal_text.hide()
            for curve in (
                self.left_boundary, self.right_boundary, self.global_plan,
                self.ego_box, self.ego_marker, self.past_path,
                self.start_marker, self.goal_marker, self.heading_line,
                self.planner_marker, self.g_wp_current, self.g_wp_next,
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
            self._status(f"Teleport {x:+9.1f},{y:+9.1f}")
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
                self._status(f"Teleport {x:+9.1f},{y:+9.1f}")
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
                self.start_text.setPos(x, y)
                self.start_text.show()
                self.goal_text.hide()
                log.info("Set start: %s", self._start_point)
                return
            start = self._start_point
            goal = (x, y)
            self.start_marker.setData([start[0]], [start[1]])
            self.goal_marker.setData([goal[0]], [goal[1]])
            self.start_text.setPos(start[0], start[1])
            self.goal_text.setPos(goal[0], goal[1])
            self.start_text.show()
            self.goal_text.show()
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

        def _paint_plan_speed(self, xs, ys, vel) -> None:
            if vel is None or len(xs) == 0:
                self.global_plan_speed.setData([], [])
                return
            n = min(len(xs), len(vel))
            step = max(1, n // 400)
            scale = str(getattr(self.root.setting.schema, "p66_global_plan_velocity_scale", "relative"))
            colors = _velocity_colors(np.asarray(vel[:n])[::step], scale)
            brushes = [pg.mkBrush(*c) for c in colors]
            self.global_plan_speed.setData(xs[:n:step], ys[:n:step], brush=brushes)

        def _paint_waypoints(self, stack, plan) -> None:
            gt = None
            if stack.local_planner is not None:
                gt = getattr(stack.local_planner, "global_trajectory", None)
            if gt is None and plan is not None:
                gt = getattr(plan, "trajectory", None)
            if gt is None or not hasattr(gt, "current_wp"):
                self.g_wp_current.setData([], [])
                self.g_wp_next.setData([], [])
                return
            try:
                i = int(gt.current_wp)
                j = int(getattr(gt, "next_wp", i + 1))
                xs = np.asarray(gt.path_x, dtype=float)
                ys = np.asarray(gt.path_y, dtype=float)
            except Exception:
                self.g_wp_current.setData([], [])
                self.g_wp_next.setData([], [])
                return
            if 0 <= i < len(xs):
                self.g_wp_current.setData([float(xs[i])], [float(ys[i])])
            else:
                self.g_wp_current.setData([], [])
            if 0 <= j < len(xs):
                self.g_wp_next.setData([float(xs[j])], [float(ys[j])])
            else:
                self.g_wp_next.setData([], [])

        def _paint_heading(self, ego, setting) -> None:
            if not setting.p66_show_state.get():
                self.heading_line.setData([], [])
                return
            ctrl = getattr(self.root.exec, "controller", None)
            front = float(getattr(ctrl, "ego_distance_front_axle", 2.5) or 2.5)
            length = float(getattr(ego, "length", front + 1.0) or (front + 1.0))
            rear = max(length - front, 0.0)
            c, s = float(np.cos(ego.theta)), float(np.sin(ego.theta))
            self.heading_line.setData(
                [ego.x + front * c, ego.x - rear * c],
                [ego.y + front * s, ego.y - rear * s],
            )

        def _paint_planner_loc(self, stack, setting) -> None:
            lp = stack.local_planner
            loc = getattr(lp, "location_xy", None) if lp is not None else None
            if loc is None:
                self.planner_marker.setData([], [])
                self.planner_text.hide()
                return
            x, y = float(loc[0]), float(loc[1])
            self.planner_marker.setData([x], [y])
            self.planner_text.setPos(x, y)
            self.planner_text.show()

        def _update_hd_hover(self, x: float, y: float) -> None:
            if not self._lane_cache:
                self.hd_closest.setData([], [])
                self.hd_text.hide()
                return
            best = None
            best_d = (float(self.root.setting.p66_global_zoom) * 0.08) ** 2
            for xs, ys, lane_id, road_id, pred, succ in self._lane_cache:
                if len(xs) == 0:
                    continue
                step = max(1, len(xs) // 80)
                d2 = (xs[::step] - x) ** 2 + (ys[::step] - y) ** 2
                i = int(np.argmin(d2))
                if float(d2[i]) < best_d:
                    best_d = float(d2[i])
                    best = (xs, ys, lane_id, road_id, pred, succ)
            if best is None:
                self.hd_closest.setData([], [])
                self.hd_text.hide()
                return
            xs, ys, lane_id, road_id, pred, succ = best
            self.hd_closest.setData(xs, ys)
            pred_s = ",".join(str(p) for p in list(pred)[:4]) if pred else "—"
            succ_s = ",".join(str(p) for p in list(succ)[:4]) if succ else "—"
            self.hd_text.setText(f"lane {lane_id}  road {road_id}\npred {pred_s}\nsucc {succ_s}")
            self.hd_text.setPos(x, y)
            self.hd_text.show()

        def _on_mouse_moved(self, pos):
            self._update_hd_hover(float(pos.x()), float(pos.y()))
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
            v = float(self._hover_plan_v[i]) if self._hover_plan_v is not None and i < len(self._hover_plan_v) else 0.0
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
                self._lane_cache.clear()
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
                    xs = np.asarray(lane.center_line[0], dtype=float)
                    ys = np.asarray(lane.center_line[1], dtype=float)
                    self._lane_cache.append((
                        xs, ys,
                        getattr(lane, "id", ""),
                        getattr(lane, "road_id", getattr(lane, "road", "")),
                        getattr(lane, "predecessors", ()),
                        getattr(lane, "successors", ()),
                    ))
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
                    vel = getattr(traj, "velocity", None)
                    self._hover_plan_v = (
                        np.asarray(vel, dtype=float).ravel()[: len(xs)] if vel is not None else None
                    )
                    self._paint_plan_speed(xs, ys, self._hover_plan_v)
                else:
                    self.global_plan_speed.setData([], [])
            else:
                self.global_plan.setData([], [])
                self.global_plan_speed.setData([], [])
                self._hover_plan_xy = None

            self._paint_waypoints(stack, plan)
            self._paint_heading(ego, setting)
            self._paint_planner_loc(stack, setting)

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

            # View window. One explicit range; axis pixels stay fixed (see _pin_plot_chrome).
            zoom = float(setting.p66_global_zoom)
            cx = ego.x + self._center_delta[0]
            cy = ego.y + self._center_delta[1]
            if setting.p66_global_view_follow_planner.get() and stack.local_planner is not None:
                loc = getattr(stack.local_planner, "location_xy", None)
                if loc is not None:
                    cx, cy = float(loc[0]), float(loc[1])
            _set_equal_view(self, cx, cy, zoom)

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
            self.layer_point_counts: dict[str, int] = {}
            for pw, xlabel, ylabel in (
                (self.xy_plot, "x (m)", "y (m)"),
                (self.frenet_plot, "s (m)", "d (m)"),
            ):
                pw.setLabel("bottom", xlabel)
                pw.setLabel("left", ylabel)
                pw.showGrid(x=True, y=True, alpha=0.25)
                _pin_plot_chrome(pw)
            layout.addWidget(self.xy_plot, 1)
            layout.addWidget(self.frenet_plot, 1)
            self.legend = QtWidgets.QLabel(
                "orange/tan boundary · cyan plan · red ego · green past · "
                "lime lidar · yellow clusters · orange ahead / purple behind prediction · "
                "green feasible / blue kinematic / red collision lattice"
            )
            self.legend.setObjectName("plotLegend")
            self.legend.setWordWrap(True)
            layout.addWidget(self.legend)

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
                pen=pg.mkPen("#ff9800", width=1.5, style=QtCore.Qt.DashLine), connect="finite",
            )
            self.xy_pred_behind = self.xy_plot.plotItem.plot(
                pen=pg.mkPen("#7e57c2", width=1.5, style=QtCore.Qt.DashLine), connect="finite",
            )
            self.xy_ellipses = self.xy_plot.plotItem.plot(
                pen=pg.mkPen("#ff9800", width=1), connect="finite",
            )
            self.xy_past = self.xy_plot.plotItem.plot(pen=pg.mkPen("#66bb6a", width=1.5))
            self.xy_lattice = self.xy_plot.plotItem.plot(
                pen=pg.mkPen("#8ec07c", width=1), connect="finite",
            )
            self.xy_lattice_hit = self.xy_plot.plotItem.plot(
                pen=pg.mkPen("#b22222", width=1), connect="finite",
            )
            self.xy_lattice_kin = self.xy_plot.plotItem.plot(
                pen=pg.mkPen("#4169e1", width=1), connect="finite",
            )
            self.xy_track = self.xy_plot.plotItem.plot(
                pen=pg.mkPen("#bdbdbd", width=1), connect="finite",
            )
            self.xy_heading = self.xy_plot.plotItem.plot(pen=pg.mkPen("#ef5350", width=2))
            self.xy_wp = self.xy_plot.plotItem.plot(
                pen=None, symbol="x", symbolPen=pg.mkPen("#ffffff", width=2), symbolSize=12,
            )
            self.xy_wp_next = self.xy_plot.plotItem.plot(
                pen=None, symbol="+", symbolPen=pg.mkPen("#90caf9", width=2), symbolSize=12,
            )
            self.xy_lwp = self.xy_plot.plotItem.plot(
                pen=None, symbol="o", symbolBrush=None, symbolPen=pg.mkPen("#ef5350", width=2), symbolSize=10,
            )
            self.xy_lwp_next = self.xy_plot.plotItem.plot(
                pen=None, symbol="s", symbolBrush=None, symbolPen=pg.mkPen("#ef9a9a", width=2), symbolSize=9,
            )
            self.xy_occ = pg.ImageItem()
            self.xy_flow = pg.ImageItem()
            self.xy_plot.addItem(self.xy_occ)
            self.xy_plot.addItem(self.xy_flow)
            self.xy_occ.setZValue(-20)
            self.xy_flow.setZValue(-10)
            self.xy_occ.hide()
            self.xy_flow.hide()

            self.sd_left = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#ff8c00", width=1))
            self.sd_right = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#d2b48c", width=1))
            self.sd_ref = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#888888", width=1))
            self.sd_local = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#ef5350", width=3))
            self.sd_ego = self.frenet_plot.plotItem.plot(pen=pg.mkPen("r", width=2))
            self.sd_agents = [
                self.frenet_plot.plotItem.plot(pen=pg.mkPen("#1565c0", width=1.5)) for _ in range(12)
            ]
            self.sd_pred = self.frenet_plot.plotItem.plot(
                pen=pg.mkPen("#ff9800", width=1.5, style=QtCore.Qt.DashLine), connect="finite",
            )
            self.sd_pred_behind = self.frenet_plot.plotItem.plot(
                pen=pg.mkPen("#7e57c2", width=1.5, style=QtCore.Qt.DashLine), connect="finite",
            )
            self.sd_ellipses = self.frenet_plot.plotItem.plot(
                pen=pg.mkPen("#ff9800", width=1), connect="finite",
            )
            self.sd_lattice = self.frenet_plot.plotItem.plot(
                pen=pg.mkPen("#8ec07c", width=1), connect="finite",
            )
            self.sd_lattice_hit = self.frenet_plot.plotItem.plot(
                pen=pg.mkPen("#b22222", width=1), connect="finite",
            )
            self.sd_lattice_kin = self.frenet_plot.plotItem.plot(
                pen=pg.mkPen("#4169e1", width=1), connect="finite",
            )
            self.sd_track = self.frenet_plot.plotItem.plot(
                pen=pg.mkPen("#bdbdbd", width=1), connect="finite",
            )
            self.sd_wp = self.frenet_plot.plotItem.plot(
                pen=None, symbol="x", symbolPen=pg.mkPen("#ffffff", width=2), symbolSize=12,
            )
            self.sd_wp_next = self.frenet_plot.plotItem.plot(
                pen=None, symbol="+", symbolPen=pg.mkPen("#90caf9", width=2), symbolSize=12,
            )
            self.sd_lwp = self.frenet_plot.plotItem.plot(
                pen=None, symbol="o", symbolBrush=None, symbolPen=pg.mkPen("#ef5350", width=2), symbolSize=10,
            )
            self.sd_lwp_next = self.frenet_plot.plotItem.plot(
                pen=None, symbol="s", symbolBrush=None, symbolPen=pg.mkPen("#ef9a9a", width=2), symbolSize=9,
            )
            self.sd_clusters = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(255, 255, 0, 220), pen=None)
            self.frenet_plot.addItem(self.sd_clusters)
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
                self.xy_pred, self.xy_pred_behind, self.xy_ellipses, self.xy_past,
                self.xy_lattice, self.xy_lattice_hit, self.xy_lattice_kin,
                self.xy_track, self.xy_heading, self.xy_wp, self.xy_wp_next,
                self.xy_lwp, self.xy_lwp_next,
                self.sd_left, self.sd_right, self.sd_ref, self.sd_local, self.sd_ego,
                self.sd_pred, self.sd_pred_behind, self.sd_ellipses,
                self.sd_lattice, self.sd_lattice_hit, self.sd_lattice_kin,
                self.sd_track, self.sd_wp, self.sd_wp_next, self.sd_lwp, self.sd_lwp_next,
            ):
                curve.setData([], [])
            for c in self.xy_agents + self.sd_agents:
                c.setData([], [])
            self.xy_lidar.setData([], [])
            self.xy_clusters.setData([], [])
            self.sd_lidar.setData([], [])
            self.sd_clusters.setData([], [])
            self.xy_occ.hide()
            self.xy_flow.hide()
            self.layer_point_counts.clear()
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
            self.legend.setVisible(bool(s.p66_show_legend.get()))

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
                self._status(
                    f"Spawn S:{x:+9.1f},D:{y:+9.1f}" if frenet else f"Spawn {x:+9.1f},{y:+9.1f}"
                )
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
            self._status(f"Spawn {wx:+9.1f},{wy:+9.1f}")

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
                self._status(f"Spawn S:{x:+9.1f},D:{y:+9.1f}")
            else:
                wx, wy = x, y
                self._status(f"Spawn {x:+9.1f},{y:+9.1f}")
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

            self._paint_local_plan(lp, gt, s)
            self._paint_lattice(lp, gt, s)
            self._paint_track(stack, plan, gt, s)
            self._paint_ego(stack, ego, gt, s)
            self._paint_lidar(stack, gt, s)
            self._paint_clusters(stack, gt, s)
            self._paint_predictions(stack, ego, gt, s)
            self._paint_occupancy(stack, s)

            # View windows. Equal scale from the pinned view box, not auto-range.
            zoom = float(s.p66_xy_zoom)
            cx, cy = ego.x, ego.y
            if s.p66_global_view_follow_planner.get() and lp is not None:
                loc = getattr(lp, "location_xy", None)
                if loc is not None:
                    cx, cy = float(loc[0]), float(loc[1])
            _set_equal_view(self.xy_plot, cx, cy, zoom)

            fzoom = float(s.p66_frenet_zoom)
            if gt is not None:
                try:
                    sd_ego = gt.convert_xy_to_sd(ego.x, ego.y)
                    cs, cd = float(sd_ego[0]), float(sd_ego[1])
                except Exception:
                    cs, cd = 0.0, 0.0
            else:
                cs, cd = 0.0, 0.0
            _set_equal_view(self.frenet_plot, cs, cd, fzoom)

            self._last_draw_ms = (time.perf_counter() - t0) * 1000.0
            return self._last_draw_ms

        def _count(self, name: str, xs) -> None:
            self.layer_point_counts[name] = int(np.count_nonzero(np.isfinite(np.asarray(xs, dtype=float)))) if len(xs) else 0

        def _paint_local_plan(self, lp, gt, s) -> None:
            paths = _local_plan_paths(lp) if lp is not None and s.p66_show_local_plan.get() else []
            xs, ys = _nan_join(paths)
            self.xy_local.setData(xs, ys)
            self._count("local_plan_xy", xs)
            if gt is not None and paths:
                sd_paths = [_to_sd(gt, p) for p in paths]
                sx, sy = _nan_join(sd_paths)
                self.sd_local.setData(sx, sy)
            else:
                self.sd_local.setData([], [])
            self._paint_local_waypoints(lp, gt, s, paths)

        def _paint_local_waypoints(self, lp, gt, s, paths) -> None:
            self.xy_lwp.setData([], [])
            self.xy_lwp_next.setData([], [])
            self.sd_lwp.setData([], [])
            self.sd_lwp_next.setData([], [])
            self.xy_wp.setData([], [])
            self.xy_wp_next.setData([], [])
            self.sd_wp.setData([], [])
            self.sd_wp_next.setData([], [])
            if gt is not None and hasattr(gt, "current_wp") and getattr(gt, "path_x", None) is not None:
                try:
                    i = int(gt.current_wp)
                    j = int(getattr(gt, "next_wp", i + 1))
                    px = np.asarray(gt.path_x, dtype=float)
                    py = np.asarray(gt.path_y, dtype=float)
                    ps = np.asarray(getattr(gt, "path_s", px), dtype=float)
                    pd = np.asarray(getattr(gt, "path_d", np.zeros_like(px)), dtype=float)
                    if 0 <= i < len(px):
                        self.xy_wp.setData([float(px[i])], [float(py[i])])
                        self.sd_wp.setData([float(ps[i])], [float(pd[i])])
                    if 0 <= j < len(px):
                        self.xy_wp_next.setData([float(px[j])], [float(py[j])])
                        self.sd_wp_next.setData([float(ps[j])], [float(pd[j])])
                except Exception:
                    pass
            if not paths or lp is None:
                return
            local = None
            getter = getattr(lp, "get_local_plan", None)
            if callable(getter):
                try:
                    local = getter()
                except Exception:
                    local = None
            traj = None
            if local is not None and hasattr(local, "as_trajectory"):
                try:
                    traj = local.as_trajectory()
                except Exception:
                    traj = None
            if traj is None or not hasattr(traj, "get_current_xy"):
                return
            try:
                x, y = traj.get_current_xy()
                x2, y2 = traj.get_xy_by_waypoint(int(traj.next_wp))
                self.xy_lwp.setData([float(x)], [float(y)])
                self.xy_lwp_next.setData([float(x2)], [float(y2)])
                if gt is not None:
                    s0, d0 = gt.convert_xy_to_sd(float(x), float(y))
                    s1, d1 = gt.convert_xy_to_sd(float(x2), float(y2))
                    self.sd_lwp.setData([float(s0)], [float(d0)])
                    self.sd_lwp_next.setData([float(s1)], [float(d1)])
            except Exception:
                return

        def _paint_lattice(self, lp, gt, s) -> None:
            buckets = {"feasible": [], "collision": [], "kinematic": []}
            sd_buckets = {"feasible": [], "collision": [], "kinematic": []}
            if lp is not None and s.p66_show_local_lattice.get():
                for edge in _lattice_edges(lp)[:240]:
                    xy = _edge_xy(edge)
                    if len(xy) < 2:
                        continue
                    kind = _edge_kind(edge, lp)
                    buckets[kind].append(xy)
                    sd = _edge_sd(edge)
                    if len(sd) >= 2:
                        sd_buckets[kind].append(sd)
                    elif gt is not None:
                        sd_buckets[kind].append(_to_sd(gt, xy))
            mapping = (
                ("feasible", self.xy_lattice, self.sd_lattice),
                ("collision", self.xy_lattice_hit, self.sd_lattice_hit),
                ("kinematic", self.xy_lattice_kin, self.sd_lattice_kin),
            )
            for kind, xy_item, sd_item in mapping:
                xs, ys = _nan_join(buckets[kind])
                xy_item.setData(xs, ys)
                sx, sy = _nan_join(sd_buckets[kind])
                sd_item.setData(sx, sy)
                self.layer_point_counts[f"lattice_{kind}"] = int(np.count_nonzero(np.isfinite(xs))) if xs else 0
            self.layer_point_counts["lattice_xy"] = sum(
                self.layer_point_counts.get(f"lattice_{k}", 0) for k in buckets
            )

        def _paint_track(self, stack, plan, gt, s) -> None:
            segs = _track_segments(getattr(stack, "world", None), plan) if s.p66_show_race_boundary.get() else None
            xs, ys = _segments_xy(segs)
            self.xy_track.setData(xs, ys)
            self._count("track_xy", xs)
            if gt is None or not xs:
                self.sd_track.setData([], [])
                return
            arr = np.asarray(segs, dtype=float)
            sd_paths = []
            for i in range(arr.shape[0]):
                sd = _to_sd(gt, arr[i, :, :2])
                if len(sd) < 2:
                    continue
                if abs(float(sd[0, 0]) - float(sd[1, 0])) > 80.0:
                    continue
                sd_paths.append(sd)
            sx, sy = _nan_join(sd_paths)
            self.sd_track.setData(sx, sy)

        def _world_pm(self, stack):
            world = getattr(stack, "world", None)
            if world is None:
                return None
            try:
                from avlite.c40_execution.c41_world_bridge import is_world_stack_capability_enabled
                from avlite.c50_common.c51_capabilities import StackCapability

                if not is_world_stack_capability_enabled(StackCapability.DETECTION):
                    return None
            except Exception:
                return None
            getter = getattr(world, "get_ground_truth_perception_model", None)
            if not callable(getter):
                return None
            try:
                return getter()
            except Exception:
                return None

        def _paint_ego(self, stack, ego, gt, s) -> None:
            if not s.p66_show_state.get():
                self.xy_ego.setData([], [])
                self.sd_ego.setData([], [])
                self.xy_heading.setData([], [])
                for curve in self.xy_agents + self.sd_agents:
                    curve.setData([], [])
                return
            self.xy_ego.setData(*_closed_poly(ego.get_bb_corners()))
            ctrl = stack.controller
            front = float(getattr(ctrl, "ego_distance_front_axle", 2.5) or 2.5)
            length = float(getattr(ego, "length", front + 1.0) or (front + 1.0))
            rear = max(length - front, 0.0)
            c, sn = float(np.cos(ego.theta)), float(np.sin(ego.theta))
            self.xy_heading.setData(
                [ego.x + front * c, ego.x - rear * c],
                [ego.y + front * sn, ego.y - rear * sn],
            )
            if gt is not None:
                try:
                    sd_c = gt.convert_xy_path_to_sd_path_np(ego.get_bb_corners())
                    if len(sd_c) >= 2 and abs(float(sd_c[0, 0]) - float(sd_c[1, 0])) < 10:
                        self.sd_ego.setData(*_closed_poly(sd_c))
                    else:
                        self.sd_ego.setData([], [])
                except Exception:
                    self.sd_ego.setData([], [])
            agents = _agent_list(self._world_pm(stack) or stack.pm)[:12]
            for i, curve in enumerate(self.xy_agents):
                if i < len(agents):
                    curve.setData(*_closed_poly(agents[i].get_bb_corners()))
                else:
                    curve.setData([], [])
            if gt is None:
                return
            for i, curve in enumerate(self.sd_agents):
                if i < len(agents):
                    try:
                        sd_c = gt.convert_xy_path_to_sd_path_np(agents[i].get_bb_corners())
                        curve.setData(*_closed_poly(sd_c))
                    except Exception:
                        curve.setData([], [])
                else:
                    curve.setData([], [])

        def _lidar_allowed(self, stack) -> bool:
            try:
                from avlite.c40_execution.c41_world_bridge import is_world_capability_enabled
                from avlite.c50_common.c51_capabilities import WorldCapability

                return is_world_capability_enabled(WorldCapability.LIDAR_2D) or is_world_capability_enabled(
                    WorldCapability.LIDAR_3D
                )
            except Exception:
                return True

        def _paint_lidar(self, stack, gt, s) -> None:
            lidar_xy = None
            if self._lidar_allowed(stack):
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
                self.layer_point_counts["lidar_xy"] = int(len(lidar_xy))
            else:
                self.xy_lidar.setData([], [])
                self.layer_point_counts["lidar_xy"] = 0
            if lidar_xy is not None and s.p66_show_lidar_frenet.get() and gt is not None:
                sd = _to_sd(gt, lidar_xy)
                self.sd_lidar.setData(sd[:, 0], sd[:, 1]) if len(sd) else self.sd_lidar.setData([], [])
            else:
                self.sd_lidar.setData([], [])

        def _paint_clusters(self, stack, gt, s) -> None:
            xy = _cluster_xy(stack.pm) if s.p66_show_lidar_clusters.get() else None
            if xy is not None and len(xy):
                self.xy_clusters.setData(xy[:, 0], xy[:, 1])
                self.layer_point_counts["clusters_xy"] = int(len(xy))
                if gt is not None:
                    sd = _to_sd(gt, xy)
                    self.sd_clusters.setData(sd[:, 0], sd[:, 1]) if len(sd) else self.sd_clusters.setData([], [])
                else:
                    self.sd_clusters.setData([], [])
            else:
                self.xy_clusters.setData([], [])
                self.sd_clusters.setData([], [])
                self.layer_point_counts["clusters_xy"] = 0

        def _paint_predictions(self, stack, ego, gt, s) -> None:
            ahead_xy: list[np.ndarray] = []
            behind_xy: list[np.ndarray] = []
            ellipses: list[np.ndarray] = []
            pred = getattr(stack.pm, "prediction", None) if stack.pm is not None else None
            if s.p67_show_prediction.get() and pred is not None:
                agents = {getattr(a, "agent_id", i): a for i, a in enumerate(_agent_list(stack.pm))}
                heading = np.array([np.cos(float(ego.theta)), np.sin(float(ego.theta))])
                for aid, mode, path, weight in _forecast_modes(pred):
                    agent = agents.get(aid)
                    pose = np.array([[float(agent.x), float(agent.y)]]) if agent is not None else path[:1]
                    full = np.vstack([pose, path]) if len(path) else path
                    ahead = True
                    if agent is not None:
                        to_agent = np.array([float(agent.x) - float(ego.x), float(agent.y) - float(ego.y)])
                        ahead = float(np.dot(heading, to_agent)) >= 0.0
                    (ahead_xy if ahead else behind_xy).append(full)
                    if weight <= 0.0 or len(path) == 0:
                        continue
                    step = len(path) - 1
                    cov = _step_covariance(pred, aid, mode, step)
                    if cov is None:
                        continue
                    poly = _ellipse_polygon(path[step], cov, 0.5)
                    if poly is not None:
                        ellipses.append(poly)
            ax, ay = _nan_join(ahead_xy)
            bx, by = _nan_join(behind_xy)
            ex, ey = _nan_join(ellipses)
            self.xy_pred.setData(ax, ay)
            self.xy_pred_behind.setData(bx, by)
            self.xy_ellipses.setData(ex, ey)
            self._count("prediction_xy", ax + bx)
            self._count("prediction_ellipses", ex)
            if gt is not None:
                self.sd_pred.setData(*_nan_join([_to_sd(gt, p) for p in ahead_xy]))
                self.sd_pred_behind.setData(*_nan_join([_to_sd(gt, p) for p in behind_xy]))
                self.sd_ellipses.setData(*_nan_join([_to_sd(gt, p) for p in ellipses]))
            else:
                self.sd_pred.setData([], [])
                self.sd_pred_behind.setData([], [])
                self.sd_ellipses.setData([], [])

        def _set_image(self, item, grid, origin_x, origin_y, resolution, lut=None) -> None:
            image = np.asarray(grid, dtype=float)
            item.setImage(np.ascontiguousarray(image.T), autoLevels=False, levels=(0.0, 1.0))
            if lut is not None:
                item.setLookupTable(lut)
            h, w = image.shape
            item.setRect(QtCore.QRectF(float(origin_x), float(origin_y), w * resolution, h * resolution))
            item.show()

        def _paint_occupancy(self, stack, s) -> None:
            pm = stack.pm
            show_map = bool(s.p67_show_occupancy_map.get())
            om = _occupancy_grid(pm) if show_map else None
            if om is not None:
                self._set_image(self.xy_occ, om.grid, om.origin_x, om.origin_y, float(om.resolution or 0.25))
                self.layer_point_counts["occupancy"] = int(np.size(om.grid))
            else:
                self.xy_occ.hide()
                self.layer_point_counts["occupancy"] = 0
            flow = _flow_image(getattr(pm, "prediction", None)) if s.p67_show_prediction.get() else None
            if flow is not None:
                grid, (ox, oy, res) = flow
                lut = np.zeros((256, 4), dtype=np.ubyte)
                ramp = np.arange(256, dtype=np.ubyte)
                lut[:, 0] = ramp
                lut[:, 1] = (ramp // 3)
                lut[:, 2] = 255 - ramp
                lut[:, 3] = np.where(ramp > 8, 160, 0).astype(np.ubyte)
                self._set_image(self.xy_flow, grid, ox, oy, res, lut)
                self.layer_point_counts["occupancy_flow"] = int(np.count_nonzero(grid > 0.02))
            else:
                self.xy_flow.hide()
                self.layer_point_counts["occupancy_flow"] = 0

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
