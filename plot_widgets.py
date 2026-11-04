"""pyqtgraph plot widgets (no Matplotlib).

Dynamic marks use ``setData``. Track, boundaries, and the global plan are
cached and clipped to the view. The window timer caps plot redraws at 30 Hz.
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


def _nan_join(paths: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Join polylines with a NaN break. Returns arrays so ``setData`` skips a Python copy."""
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    for path in paths:
        arr = np.asarray(path, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            continue
        n = int(arr.shape[0])
        xs = np.empty(n + 1, dtype=float)
        ys = np.empty(n + 1, dtype=float)
        xs[:n] = arr[:, 0]
        ys[:n] = arr[:, 1]
        xs[n] = np.nan
        ys[n] = np.nan
        chunks_x.append(xs)
        chunks_y.append(ys)
    if not chunks_x:
        return np.array([]), np.array([])
    return np.concatenate(chunks_x), np.concatenate(chunks_y)


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


def _segments_xy(segs) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(segs, dtype=float) if segs is not None else np.zeros((0, 2, 2))
    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[-1] < 2 or arr.shape[-2] < 2:
        return np.array([]), np.array([])
    count = int(arr.shape[0])
    xs = np.empty(count * 3, dtype=float)
    ys = np.empty(count * 3, dtype=float)
    xs[0::3] = arr[:, 0, 0]
    xs[1::3] = arr[:, 1, 0]
    xs[2::3] = np.nan
    ys[0::3] = arr[:, 0, 1]
    ys[1::3] = arr[:, 1, 1]
    ys[2::3] = np.nan
    return xs, ys


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
    u = np.asarray(t, dtype=float)
    r = (40 + 215 * u).astype(np.int32)
    g = (80 + 140 * (1.0 - np.abs(u - 0.5) * 2.0)).astype(np.int32)
    b = (220 * (1.0 - u)).astype(np.int32)
    colors = np.column_stack([r, g, b, np.full(u.shape, 230, dtype=np.int32)])
    colors[~np.isfinite(u)] = (80, 80, 80, 0)
    return [tuple(int(c) for c in row) for row in colors]


def _paths_token(paths) -> tuple:
    if not paths:
        return (0,)
    count = 0
    acc = 0.0
    for path in paths:
        arr = np.asarray(path, dtype=float)
        if arr.size == 0:
            continue
        count += int(arr.shape[0]) if arr.ndim == 2 else int(arr.size)
        flat = arr.reshape(-1)
        acc += float(flat[0]) + float(flat[flat.size // 2]) + float(flat[-1])
    return (len(paths), count, round(acc, 3))


def _plan_arrays(plan) -> dict:
    """Copy plan polylines once. Callers cache the result by plan identity."""
    traj = getattr(plan, "trajectory", None)

    def xy(xs, ys):
        return _safe_xy(xs, ys)

    lbx, lby = xy(getattr(plan, "left_boundary_x", None), getattr(plan, "left_boundary_y", None))
    rbx, rby = xy(getattr(plan, "right_boundary_x", None), getattr(plan, "right_boundary_y", None))
    if traj is not None:
        refx, refy = xy(getattr(traj, "path_x", None), getattr(traj, "path_y", None))
        refs = np.asarray(getattr(traj, "path_s", []) or [], dtype=float).ravel()
        vel = np.asarray(getattr(traj, "velocity", []) or [], dtype=float).ravel()
    else:
        refx = refy = refs = vel = np.array([])
    left_d = getattr(plan, "left_boundary_d", None)
    right_d = getattr(plan, "right_boundary_d", None)
    n = min(int(refs.size), len(left_d) if left_d is not None else 0, len(right_d) if right_d is not None else 0)
    if n:
        sd_s = refs[:n]
        sd_l = np.asarray(left_d[:n], dtype=float)
        sd_r = np.asarray(right_d[:n], dtype=float)
    else:
        sd_s = sd_l = sd_r = np.array([])
    bounds = None
    if lbx.size and rbx.size:
        bounds = (
            float(min(lbx.min(), rbx.min())),
            float(max(lbx.max(), rbx.max())),
            float(min(lby.min(), rby.min())),
            float(max(lby.max(), rby.max())),
        )
    elif refx.size:
        bounds = (float(refx.min()), float(refx.max()), float(refy.min()), float(refy.max()))
    return {
        "lbx": lbx, "lby": lby, "rbx": rbx, "rby": rby,
        "refx": refx, "refy": refy, "refs": refs, "vel": vel,
        "sd_s": sd_s, "sd_l": sd_l, "sd_r": sd_r,
        "bounds": bounds, "traj": traj,
    }


def _window_token(window, quantum: float = 1.0):
    """Quantize a view window so a moving ego does not reupload static geometry every tick."""
    if window is None:
        return None
    q = float(quantum)
    return tuple(int(round(float(v) / q)) for v in window)


def _decimate_polyline(x: np.ndarray, y: np.ndarray, window, margin_frac: float = 0.15):
    """Keep the vertices near ``window``, with NaN breaks between disjoint runs."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(int(x.size), int(y.size))
    x, y = x[:n], y[:n]
    if window is None or n == 0:
        return x, y
    x0, x1, y0, y1 = window
    mx = (x1 - x0) * margin_frac
    my = (y1 - y0) * margin_frac
    inside = (x >= x0 - mx) & (x <= x1 + mx) & (y >= y0 - my) & (y <= y1 + my)
    if not inside.any():
        return x[:0], y[:0]
    idx = np.flatnonzero(inside)
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    xs_parts: list[np.ndarray] = []
    ys_parts: list[np.ndarray] = []
    nan = np.array([np.nan])
    for k, run in enumerate(runs):
        a = max(int(run[0]) - 1, 0)
        b = min(int(run[-1]) + 2, n)
        if k:
            xs_parts.append(nan)
            ys_parts.append(nan)
        xs_parts.append(x[a:b])
        ys_parts.append(y[a:b])
    return np.concatenate(xs_parts), np.concatenate(ys_parts)


def _clip_segments(segs, window, margin_frac: float = 0.15) -> np.ndarray:
    arr = np.asarray(segs, dtype=float)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.zeros((0, 2, 2))
    if window is None:
        return arr
    x0, x1, y0, y1 = window
    mid = arr.mean(axis=1)
    mx = (x1 - x0) * margin_frac
    my = (y1 - y0) * margin_frac
    mask = (
        (mid[:, 0] >= x0 - mx) & (mid[:, 0] <= x1 + mx)
        & (mid[:, 1] >= y0 - my) & (mid[:, 1] <= y1 + my)
    )
    return arr[mask]


def _push_hidden(cache: dict, item, key: str) -> None:
    if cache.get(key) == ("off",):
        return
    cache[key] = ("off",)
    item.setData([], [])


def _push_windowed(cache: dict, item, key: str, x, y, window, quantum: float = 1.0) -> None:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(int(x.size), int(y.size))
    x, y = x[:n], y[:n]
    if n == 0:
        src = (0,)
    else:
        src = (n, round(float(x[0]), 3), round(float(y[0]), 3), round(float(x[-1]), 3), round(float(y[-1]), 3))
    token = (src, _window_token(window, quantum))
    if cache.get(key) == token:
        return
    cache[key] = token
    xs, ys = _decimate_polyline(x, y, window)
    item.setData(xs, ys)


def _series_token(x: np.ndarray, y: np.ndarray, extra=()):
    n = min(int(x.size), int(y.size))
    if n == 0:
        return (0, extra)
    finite = np.flatnonzero(np.isfinite(x[:n]) & np.isfinite(y[:n]))
    if finite.size == 0:
        return (0, extra)
    i0 = int(finite[0])
    i1 = int(finite[finite.size // 2])
    i2 = int(finite[-1])
    step = max(1, int(finite.size) // 8)
    sample = finite[::step]
    return (
        n,
        round(float(x[i0]), 3), round(float(y[i0]), 3),
        round(float(x[i1]), 3), round(float(y[i1]), 3),
        round(float(x[i2]), 3), round(float(y[i2]), 3),
        round(float(np.sum(x[sample])), 2),
        round(float(np.sum(y[sample])), 2),
        extra,
    )


def _set_curve(cache: dict, item, key: str, x, y, extra=()) -> bool:
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = min(int(xa.size), int(ya.size))
    xa, ya = xa[:n], ya[:n]
    token = _series_token(xa, ya, extra)
    if cache.get(key) == token:
        return False
    cache[key] = token
    item.setData(xa, ya)
    return True


def _pin_plot_chrome(plot) -> None:
    """Freeze axis pixels and turn off auto-range.

    pyqtgraph's AxisItem grows and shrinks with the current tick strings
    (``autoExpandTextSpace`` / ``autoReduceTextSpace``). That changes the
    ViewBox width by tens of pixels, which changes a locked aspect ratio and
    the PlotWidget size hint. The hint feeds the window minimum, so the whole
    window jumps when coordinates gain a digit. A fixed axis size breaks the loop.
    """
    if hasattr(plot, "setAntialiasing"):
        plot.setAntialiasing(False)
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


def _plot_aspect(plot) -> float:
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
        return aspect
    return cached[1]


def _view_window(plot, cx: float, cy: float, half_x: float) -> tuple[float, float, float, float]:
    aspect = _plot_aspect(plot)
    half_x = float(half_x)
    half_y = half_x / aspect
    return (float(cx) - half_x, float(cx) + half_x, float(cy) - half_y, float(cy) + half_y)


def _set_equal_view(plot, cx: float, cy: float, half_x: float) -> tuple[float, float, float, float]:
    """Pan to ``(cx, cy)`` at a constant scale. No auto-range.

    The pixel aspect is cached for the current widget size so a 1px axis
    tweak cannot change the zoom. The range upload is skipped when the camera
    has not moved by a centimeter.
    """
    window = _view_window(plot, cx, cy, half_x)
    token = tuple(round(v, 2) for v in window)
    if getattr(plot, "_avlite_range", None) != token:
        plot._avlite_range = token
        x0, x1, y0, y1 = window
        plot.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.0, disableAutoRange=True)
    return window

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

            self.left_boundary = self.plotItem.plot(pen=pg.mkPen("#ff8c00", width=2), connect="finite")
            self.right_boundary = self.plotItem.plot(pen=pg.mkPen("#d2b48c", width=2), connect="finite")
            self.global_plan = self.plotItem.plot(pen=pg.mkPen("#4fc3f7", width=3), connect="finite")
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
            self._curve_cache: dict = {}
            self._geom_token = None
            self._geom = None

        def reset(self) -> None:
            self._map_bounds = None
            self._center_delta[:] = 0.0
            self._start_point = None
            self._drag_pan = False
            self._orienting = False
            self._curve_cache.clear()
            self._geom_token = None
            self._geom = None
            self._avlite_range = None
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
            xs = np.asarray(xs, dtype=float).ravel()
            ys = np.asarray(ys, dtype=float).ravel()
            vel_a = np.asarray([] if vel is None else vel, dtype=float).ravel()
            n = min(int(xs.size), int(ys.size), int(vel_a.size)) if vel_a.size else 0
            if n == 0:
                _push_hidden(self._curve_cache, self.global_plan_speed, "speed")
                return
            step = max(1, n // 400)
            scale = str(getattr(self.root.setting.schema, "p66_global_plan_velocity_scale", "relative"))
            sample = vel_a[:n:max(1, n // 8)]
            token = (n, step, scale, round(float(np.nansum(sample)), 2))
            if self._curve_cache.get("speed") == token:
                return
            self._curve_cache["speed"] = token
            colors = _velocity_colors(vel_a[:n:step], scale)
            brushes = [pg.mkBrush(*c) for c in colors]
            self.global_plan_speed.setData(xs[:n:step], ys[:n:step], brush=brushes)

        def _geom_for(self, plan):
            traj = getattr(plan, "trajectory", None) if plan is not None else None
            size = len(traj.path_x) if traj is not None and getattr(traj, "path_x", None) is not None else 0
            token = (id(plan), id(traj), size)
            if token == self._geom_token and self._geom is not None:
                return self._geom
            geom = _plan_arrays(plan) if plan is not None else None
            self._geom_token = token
            self._geom = geom
            if geom is not None and geom["refx"].size:
                self._hover_plan_xy = np.column_stack([geom["refx"], geom["refy"]])
                vel = geom["vel"]
                self._hover_plan_v = vel[: geom["refx"].size] if vel.size else None
            else:
                self._hover_plan_xy = None
                self._hover_plan_v = None
            if geom is not None and geom["bounds"] is not None:
                self._map_bounds = geom["bounds"]
            return geom

        def _paint_waypoints(self, stack, plan) -> None:
            gt = None
            if stack.local_planner is not None:
                gt = getattr(stack.local_planner, "global_trajectory", None)
            if gt is None and plan is not None:
                gt = getattr(plan, "trajectory", None)
            if gt is None or not hasattr(gt, "current_wp"):
                _push_hidden(self._curve_cache, self.g_wp_current, "gwp")
                _push_hidden(self._curve_cache, self.g_wp_next, "gwn")
                return
            try:
                i = int(gt.current_wp)
                j = int(getattr(gt, "next_wp", i + 1))
                xs = np.asarray(gt.path_x, dtype=float)
                ys = np.asarray(gt.path_y, dtype=float)
            except Exception:
                _push_hidden(self._curve_cache, self.g_wp_current, "gwp")
                _push_hidden(self._curve_cache, self.g_wp_next, "gwn")
                return
            if 0 <= i < len(xs):
                _set_curve(self._curve_cache, self.g_wp_current, "gwp", [xs[i]], [ys[i]])
            else:
                _push_hidden(self._curve_cache, self.g_wp_current, "gwp")
            if 0 <= j < len(xs):
                _set_curve(self._curve_cache, self.g_wp_next, "gwn", [xs[j]], [ys[j]])
            else:
                _push_hidden(self._curve_cache, self.g_wp_next, "gwn")

        def _paint_heading(self, ego, setting) -> None:
            if not setting.p66_show_state.get():
                _push_hidden(self._curve_cache, self.heading_line, "hdg")
                return
            ctrl = getattr(self.root.exec, "controller", None)
            front = float(getattr(ctrl, "ego_distance_front_axle", 2.5) or 2.5)
            length = float(getattr(ego, "length", front + 1.0) or (front + 1.0))
            rear = max(length - front, 0.0)
            c, s = float(np.cos(ego.theta)), float(np.sin(ego.theta))
            _set_curve(
                self._curve_cache, self.heading_line, "hdg",
                [ego.x + front * c, ego.x - rear * c],
                [ego.y + front * s, ego.y - rear * s],
            )

        def _paint_planner_loc(self, stack, setting) -> None:
            lp = stack.local_planner
            loc = getattr(lp, "location_xy", None) if lp is not None else None
            if loc is None:
                _push_hidden(self._curve_cache, self.planner_marker, "ploc")
                self.planner_text.hide()
                return
            x, y = float(loc[0]), float(loc[1])
            _set_curve(self._curve_cache, self.planner_marker, "ploc", [x], [y])
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

            zoom = float(setting.p66_global_zoom)
            cx = ego.x + self._center_delta[0]
            cy = ego.y + self._center_delta[1]
            if setting.p66_global_view_follow_planner.get() and stack.local_planner is not None:
                loc = getattr(stack.local_planner, "location_xy", None)
                if loc is not None:
                    cx, cy = float(loc[0]), float(loc[1])
            window = _view_window(self, cx, cy, zoom)
            geom = self._geom_for(plan) if plan is not None else None

            if setting.p66_show_global_plan_boundaries.get() and geom is not None:
                _push_windowed(self._curve_cache, self.left_boundary, "lb", geom["lbx"], geom["lby"], window)
                _push_windowed(self._curve_cache, self.right_boundary, "rb", geom["rbx"], geom["rby"], window)
            else:
                _push_hidden(self._curve_cache, self.left_boundary, "lb")
                _push_hidden(self._curve_cache, self.right_boundary, "rb")

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

            if setting.p66_show_global_plan.get() and geom is not None and geom["traj"] is not None:
                _push_windowed(self._curve_cache, self.global_plan, "gp", geom["refx"], geom["refy"], window)
                self._paint_plan_speed(geom["refx"], geom["refy"], geom["vel"])
            else:
                _push_hidden(self._curve_cache, self.global_plan, "gp")
                _push_hidden(self._curve_cache, self.global_plan_speed, "speed")
                self._hover_plan_xy = None
                self._hover_plan_v = None

            self._paint_waypoints(stack, plan)
            self._paint_heading(ego, setting)
            self._paint_planner_loc(stack, setting)

            pose = (round(float(ego.x), 3), round(float(ego.y), 3), round(float(ego.theta), 4))
            if setting.p66_show_state.get():
                bx, by = _closed_poly(ego.get_bb_corners())
                _set_curve(self._curve_cache, self.ego_box, "ego", bx, by, pose)
                _set_curve(self._curve_cache, self.ego_marker, "ego_m", [ego.x], [ego.y], pose)
            else:
                _push_hidden(self._curve_cache, self.ego_box, "ego")
                _push_hidden(self._curve_cache, self.ego_marker, "ego_m")

            if setting.p66_show_past_locations.get() and stack.local_planner is not None:
                hist = getattr(stack.local_planner, "past_locations", None) or getattr(
                    stack, "past_locations", None
                )
                if hist:
                    pts = np.asarray(hist[-100:], dtype=float)
                    if pts.ndim == 2 and pts.shape[1] >= 2:
                        _set_curve(self._curve_cache, self.past_path, "past", pts[:, 0], pts[:, 1])
                    else:
                        _push_hidden(self._curve_cache, self.past_path, "past")
                else:
                    _push_hidden(self._curve_cache, self.past_path, "past")
            else:
                _push_hidden(self._curve_cache, self.past_path, "past")

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
            self._curve_cache: dict = {}
            self._track_cache = None
            self._lidar_cache = None
            self._geom_token = None
            self._geom = None
            self._vis_flags = None
            self._xy_window = None
            self._sd_window = None
            self._xy_on = True
            self._frenet_on = True
            self._flow_lut_cache = None
            self._left_anchor: Optional[tuple[float, float]] = None
            self._right_anchor: Optional[tuple[float, float]] = None
            self._spawn_agent = None
            self._spawn_in_xy = True
            self._moved = False

            from .plot_input import attach_plot_mouse

            attach_plot_mouse(self.xy_plot, _PaneMouse(self, frenet=False))
            attach_plot_mouse(self.frenet_plot, _PaneMouse(self, frenet=True))

        def _init_curves(self) -> None:
            self.xy_left = self.xy_plot.plotItem.plot(pen=pg.mkPen("#ff8c00", width=2), connect="finite")
            self.xy_right = self.xy_plot.plotItem.plot(pen=pg.mkPen("#d2b48c", width=2), connect="finite")
            self.xy_ref = self.xy_plot.plotItem.plot(pen=pg.mkPen("#888888", width=2), connect="finite")
            self.xy_local = self.xy_plot.plotItem.plot(pen=pg.mkPen("#ef5350", width=3), connect="finite")
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

            self.sd_left = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#ff8c00", width=1), connect="finite")
            self.sd_right = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#d2b48c", width=1), connect="finite")
            self.sd_ref = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#888888", width=1), connect="finite")
            self.sd_local = self.frenet_plot.plotItem.plot(pen=pg.mkPen("#ef5350", width=3), connect="finite")
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
            self._curve_cache.clear()
            self._track_cache = None
            self._lidar_cache = None
            self._geom_token = None
            self._geom = None
            self._vis_flags = None
            self.xy_plot._avlite_range = None
            self.frenet_plot._avlite_range = None
            self._hide_ruler()

        def update_theme(self, bg: str, fg: str) -> None:
            for pw in (self.xy_plot, self.frenet_plot):
                pw.setBackground(bg)
                pw.getAxis("bottom").setPen(fg)
                pw.getAxis("left").setPen(fg)

        def update_visibility(self) -> None:
            s = self.root.setting
            flags = (
                bool(s.p67_show_local_global_view.get()),
                bool(s.p67_show_local_frenet_view.get()),
                bool(s.p66_show_legend.get()),
            )
            if self._vis_flags == flags:
                return
            self._vis_flags = flags
            self.xy_plot.setVisible(flags[0])
            self.frenet_plot.setVisible(flags[1])
            self.legend.setVisible(flags[2])

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
            self._xy_on = bool(s.p67_show_local_global_view.get())
            self._frenet_on = bool(s.p67_show_local_frenet_view.get())
            ego = stack.ego_state
            lp = stack.local_planner
            gt = getattr(lp, "global_trajectory", None) if lp is not None else None
            plan = getattr(lp, "global_plan", None) if lp is not None else None

            zoom = float(s.p66_xy_zoom)
            cx, cy = float(ego.x), float(ego.y)
            if s.p66_global_view_follow_planner.get() and lp is not None:
                loc = getattr(lp, "location_xy", None)
                if loc is not None:
                    cx, cy = float(loc[0]), float(loc[1])
            self._xy_window = _view_window(self.xy_plot, cx, cy, zoom) if self._xy_on else None

            fzoom = float(s.p66_frenet_zoom)
            cs, cd = 0.0, 0.0
            if self._frenet_on and gt is not None:
                try:
                    sd_ego = gt.convert_xy_to_sd(ego.x, ego.y)
                    cs, cd = float(sd_ego[0]), float(sd_ego[1])
                except Exception:
                    cs, cd = 0.0, 0.0
            self._sd_window = _view_window(self.frenet_plot, cs, cd, fzoom) if self._frenet_on else None

            self._paint_static_plan(plan, gt, s)
            self._paint_local_plan(lp, gt, s)
            self._paint_lattice(lp, gt, s)
            self._paint_track(stack, plan, gt, s)
            self._paint_ego(stack, ego, gt, s)
            self._paint_lidar(stack, gt, s)
            self._paint_clusters(stack, gt, s)
            self._paint_predictions(stack, ego, gt, s)
            self._paint_occupancy(stack, s)

            if self._xy_on:
                _set_equal_view(self.xy_plot, cx, cy, zoom)
            if self._frenet_on:
                _set_equal_view(self.frenet_plot, cs, cd, fzoom)

            self._last_draw_ms = (time.perf_counter() - t0) * 1000.0
            return self._last_draw_ms

        def _count(self, name: str, xs) -> None:
            arr = np.asarray(xs, dtype=float).ravel() if xs is not None else np.array([])
            self.layer_point_counts[name] = int(np.count_nonzero(np.isfinite(arr))) if arr.size else 0

        def _remember_plan(self, plan):
            traj = getattr(plan, "trajectory", None) if plan is not None else None
            size = len(traj.path_x) if traj is not None and getattr(traj, "path_x", None) is not None else 0
            token = (id(plan), id(traj), size)
            if token == self._geom_token and self._geom is not None:
                return self._geom
            self._geom_token = token
            self._geom = _plan_arrays(plan) if plan is not None else None
            return self._geom

        def _paint_static_plan(self, plan, gt, s) -> None:
            show_b = plan is not None and bool(s.p66_show_race_boundary.get())
            show_g = (
                plan is not None
                and getattr(plan, "trajectory", None) is not None
                and bool(s.p66_show_global_plan.get())
            )
            if not show_b and not show_g:
                for key, item in (
                    ("ll", self.xy_left), ("lr", self.xy_right), ("lref", self.xy_ref),
                    ("sdl", self.sd_left), ("sdr", self.sd_right), ("sdref", self.sd_ref),
                ):
                    _push_hidden(self._curve_cache, item, key)
                return
            geom = self._remember_plan(plan)
            if show_b and self._xy_on:
                _push_windowed(self._curve_cache, self.xy_left, "ll", geom["lbx"], geom["lby"], self._xy_window)
                _push_windowed(self._curve_cache, self.xy_right, "lr", geom["rbx"], geom["rby"], self._xy_window)
            else:
                _push_hidden(self._curve_cache, self.xy_left, "ll")
                _push_hidden(self._curve_cache, self.xy_right, "lr")
            if show_g and self._xy_on:
                _push_windowed(self._curve_cache, self.xy_ref, "lref", geom["refx"], geom["refy"], self._xy_window)
            else:
                _push_hidden(self._curve_cache, self.xy_ref, "lref")
            if show_g and self._frenet_on and gt is not None and geom["refs"].size:
                _push_windowed(
                    self._curve_cache, self.sd_ref, "sdref",
                    geom["refs"], np.zeros(geom["refs"].shape, dtype=float), self._sd_window,
                )
                if geom["sd_s"].size:
                    _push_windowed(self._curve_cache, self.sd_left, "sdl", geom["sd_s"], geom["sd_l"], self._sd_window)
                    _push_windowed(self._curve_cache, self.sd_right, "sdr", geom["sd_s"], geom["sd_r"], self._sd_window)
                else:
                    _push_hidden(self._curve_cache, self.sd_left, "sdl")
                    _push_hidden(self._curve_cache, self.sd_right, "sdr")
            else:
                _push_hidden(self._curve_cache, self.sd_ref, "sdref")
                _push_hidden(self._curve_cache, self.sd_left, "sdl")
                _push_hidden(self._curve_cache, self.sd_right, "sdr")

        def _paint_local_plan(self, lp, gt, s) -> None:
            show = lp is not None and bool(s.p66_show_local_plan.get())
            paths = _local_plan_paths(lp) if show else []
            token = (show, self._xy_on, self._frenet_on, id(gt) if gt is not None else 0, _paths_token(paths))
            xs, ys = _nan_join(paths) if paths else (np.array([]), np.array([]))
            if self._curve_cache.get("local") != token:
                self._curve_cache["local"] = token
                if self._xy_on:
                    self.xy_local.setData(xs, ys)
                else:
                    self.xy_local.setData([], [])
                if self._frenet_on and gt is not None and paths:
                    self.sd_local.setData(*_nan_join([_to_sd(gt, p) for p in paths]))
                else:
                    self.sd_local.setData([], [])
            self._count("local_plan_xy", xs if self._xy_on else [])
            self._paint_local_waypoints(lp, gt, s, paths)

        def _paint_local_waypoints(self, lp, gt, s, paths) -> None:
            if gt is not None and hasattr(gt, "current_wp") and getattr(gt, "path_x", None) is not None:
                try:
                    i = int(gt.current_wp)
                    j = int(getattr(gt, "next_wp", i + 1))
                    px = np.asarray(gt.path_x, dtype=float)
                    py = np.asarray(gt.path_y, dtype=float)
                    ps = np.asarray(getattr(gt, "path_s", px), dtype=float)
                    pd = np.asarray(getattr(gt, "path_d", np.zeros_like(px)), dtype=float)
                    if self._xy_on and 0 <= i < len(px):
                        _set_curve(self._curve_cache, self.xy_wp, "xwp", [px[i]], [py[i]])
                    else:
                        _push_hidden(self._curve_cache, self.xy_wp, "xwp")
                    if self._frenet_on and 0 <= i < len(ps):
                        _set_curve(self._curve_cache, self.sd_wp, "swp", [ps[i]], [pd[i]])
                    else:
                        _push_hidden(self._curve_cache, self.sd_wp, "swp")
                    if self._xy_on and 0 <= j < len(px):
                        _set_curve(self._curve_cache, self.xy_wp_next, "xwn", [px[j]], [py[j]])
                    else:
                        _push_hidden(self._curve_cache, self.xy_wp_next, "xwn")
                    if self._frenet_on and 0 <= j < len(ps):
                        _set_curve(self._curve_cache, self.sd_wp_next, "swn", [ps[j]], [pd[j]])
                    else:
                        _push_hidden(self._curve_cache, self.sd_wp_next, "swn")
                except Exception:
                    for key, item in (
                        ("xwp", self.xy_wp), ("xwn", self.xy_wp_next),
                        ("swp", self.sd_wp), ("swn", self.sd_wp_next),
                    ):
                        _push_hidden(self._curve_cache, item, key)
            else:
                for key, item in (
                    ("xwp", self.xy_wp), ("xwn", self.xy_wp_next),
                    ("swp", self.sd_wp), ("swn", self.sd_wp_next),
                ):
                    _push_hidden(self._curve_cache, item, key)
            if not paths or lp is None:
                for key, item in (
                    ("xlwp", self.xy_lwp), ("xlwn", self.xy_lwp_next),
                    ("slwp", self.sd_lwp), ("slwn", self.sd_lwp_next),
                ):
                    _push_hidden(self._curve_cache, item, key)
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
                token = (
                    round(float(x), 3), round(float(y), 3),
                    round(float(x2), 3), round(float(y2), 3),
                    self._frenet_on, id(gt) if gt is not None else 0,
                )
                if self._curve_cache.get("lwp") == token:
                    return
                self._curve_cache["lwp"] = token
                if self._xy_on:
                    self.xy_lwp.setData([float(x)], [float(y)])
                    self.xy_lwp_next.setData([float(x2)], [float(y2)])
                if self._frenet_on and gt is not None:
                    s0, d0 = gt.convert_xy_to_sd(float(x), float(y))
                    s1, d1 = gt.convert_xy_to_sd(float(x2), float(y2))
                    self.sd_lwp.setData([float(s0)], [float(d0)])
                    self.sd_lwp_next.setData([float(s1)], [float(d1)])
                else:
                    self.sd_lwp.setData([], [])
                    self.sd_lwp_next.setData([], [])
            except Exception:
                return

        def _paint_lattice(self, lp, gt, s) -> None:
            show = lp is not None and bool(s.p66_show_local_lattice.get())
            edges = _lattice_edges(lp)[:240] if show else []
            if edges:
                xy0 = _edge_xy(edges[0])
                xy1 = _edge_xy(edges[-1])
                token = (
                    len(edges),
                    int(xy0.shape[0]) if getattr(xy0, "ndim", 0) == 2 else 0,
                    round(float(xy0.flat[0]), 3) if xy0.size else 0.0,
                    round(float(xy0.flat[-1]), 3) if xy0.size else 0.0,
                    round(float(xy1.flat[0]), 3) if xy1.size else 0.0,
                    round(float(xy1.flat[-1]), 3) if xy1.size else 0.0,
                    self._xy_on, self._frenet_on, id(gt) if gt is not None else 0,
                )
            else:
                token = (0, self._xy_on, self._frenet_on)
            if self._curve_cache.get("lattice") == token:
                return
            self._curve_cache["lattice"] = token
            buckets = {"feasible": [], "collision": [], "kinematic": []}
            sd_buckets = {"feasible": [], "collision": [], "kinematic": []}
            for edge in edges:
                xy = _edge_xy(edge)
                if len(xy) < 2:
                    continue
                kind = _edge_kind(edge, lp)
                buckets[kind].append(xy)
                if not self._frenet_on:
                    continue
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
                self.layer_point_counts[f"lattice_{kind}"] = int(np.count_nonzero(np.isfinite(xs))) if xs.size else 0
            self.layer_point_counts["lattice_xy"] = sum(
                self.layer_point_counts.get(f"lattice_{k}", 0) for k in buckets
            )

        def _cached_track(self, stack, plan, gt):
            segs = _track_segments(getattr(stack, "world", None), plan)
            arr = np.asarray(segs, dtype=float) if segs is not None else np.zeros((0, 2, 2))
            if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[-1] < 2 or arr.shape[-2] < 2:
                self._track_cache = None
                return None
            gt_id = id(gt) if gt is not None else 0
            key = (gt_id, arr.shape, float(arr.flat[0]), float(arr.flat[-1]))
            cached = self._track_cache
            if cached is not None and cached[0] == key:
                return cached
            sd_segs = None
            if gt is not None:
                pts = np.ascontiguousarray(arr.reshape(-1, arr.shape[-1])[:, :2])
                sd = _to_sd(gt, pts)
                if sd.shape[0] == pts.shape[0] and sd.size:
                    sd_segs = sd.reshape(-1, 2, 2)
                    s_diff = np.abs(sd_segs[:, 1, 0] - sd_segs[:, 0, 0])
                    med = float(np.median(s_diff)) if s_diff.size else 0.0
                    sd_segs = sd_segs[s_diff < (med * 20.0 + 50.0)]
            self._track_cache = (key, arr, sd_segs)
            return self._track_cache

        def _paint_track(self, stack, plan, gt, s) -> None:
            if not s.p66_show_race_boundary.get():
                _push_hidden(self._curve_cache, self.xy_track, "track_xy")
                _push_hidden(self._curve_cache, self.sd_track, "track_sd")
                self.layer_point_counts["track_xy"] = 0
                return
            cached = self._cached_track(stack, plan, gt)
            if cached is None:
                _push_hidden(self._curve_cache, self.xy_track, "track_xy")
                _push_hidden(self._curve_cache, self.sd_track, "track_sd")
                self.layer_point_counts["track_xy"] = 0
                return
            key, arr, sd_segs = cached
            if self._xy_on:
                wtok = _window_token(self._xy_window)
                token = ("xy", key, wtok)
                if self._curve_cache.get("track_xy") != token:
                    self._curve_cache["track_xy"] = token
                    xs, ys = _segments_xy(_clip_segments(arr, self._xy_window))
                    self.xy_track.setData(xs, ys)
                    self._count("track_xy", xs)
            else:
                _push_hidden(self._curve_cache, self.xy_track, "track_xy")
                self.layer_point_counts["track_xy"] = 0
            if self._frenet_on and sd_segs is not None and len(sd_segs):
                token = ("sd", key, _window_token(self._sd_window))
                if self._curve_cache.get("track_sd") != token:
                    self._curve_cache["track_sd"] = token
                    self.sd_track.setData(*_segments_xy(_clip_segments(sd_segs, self._sd_window)))
            else:
                _push_hidden(self._curve_cache, self.sd_track, "track_sd")

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
                if self._curve_cache.get("ego_off"):
                    return
                self._curve_cache["ego_off"] = True
                self._curve_cache.pop("ego_pose", None)
                self._curve_cache.pop("agents", None)
                self.xy_ego.setData([], [])
                self.sd_ego.setData([], [])
                self.xy_heading.setData([], [])
                for curve in self.xy_agents + self.sd_agents:
                    curve.setData([], [])
                return
            self._curve_cache["ego_off"] = False
            pose = (
                round(float(ego.x), 3), round(float(ego.y), 3), round(float(ego.theta), 4),
                self._xy_on, self._frenet_on, id(gt) if gt is not None else 0,
            )
            if self._curve_cache.get("ego_pose") != pose:
                self._curve_cache["ego_pose"] = pose
                if self._xy_on:
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
                else:
                    self.xy_ego.setData([], [])
                    self.xy_heading.setData([], [])
                if self._frenet_on and gt is not None:
                    try:
                        sd_c = gt.convert_xy_path_to_sd_path_np(ego.get_bb_corners())
                        if len(sd_c) >= 2 and abs(float(sd_c[0, 0]) - float(sd_c[1, 0])) < 10:
                            self.sd_ego.setData(*_closed_poly(sd_c))
                        else:
                            self.sd_ego.setData([], [])
                    except Exception:
                        self.sd_ego.setData([], [])
                else:
                    self.sd_ego.setData([], [])
            agents = _agent_list(self._world_pm(stack) or stack.pm)[:12]
            atoken = (
                self._xy_on, self._frenet_on, id(gt) if gt is not None else 0,
                tuple(
                    (round(float(a.x), 2), round(float(a.y), 2), round(float(a.theta), 3))
                    for a in agents
                ),
            )
            if self._curve_cache.get("agents") == atoken:
                return
            self._curve_cache["agents"] = atoken
            for i, curve in enumerate(self.xy_agents):
                if self._xy_on and i < len(agents):
                    curve.setData(*_closed_poly(agents[i].get_bb_corners()))
                else:
                    curve.setData([], [])
            for i, curve in enumerate(self.sd_agents):
                if self._frenet_on and gt is not None and i < len(agents):
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

        def _fetch_lidar(self, stack):
            try:
                frame = stack.world.get_sensor_frame()
                lidar = getattr(frame, "lidar", None)
                pts = getattr(lidar, "points", None) if lidar is not None else None
                if pts is None or len(pts) == 0:
                    return None
                arr = np.asarray(pts, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return np.array(arr[:, :2], copy=True)
            except Exception:
                return None
            return None

        def _paint_lidar(self, stack, gt, s) -> None:
            show_xy = self._xy_on and bool(s.p66_show_lidar_global.get())
            show_sd = self._frenet_on and bool(s.p66_show_lidar_frenet.get())
            if (not show_xy and not show_sd) or not self._lidar_allowed(stack):
                if not self._curve_cache.get("lidar_off"):
                    self.xy_lidar.setData([], [])
                    self.sd_lidar.setData([], [])
                    self.layer_point_counts["lidar_xy"] = 0
                    self._curve_cache["lidar_off"] = True
                    self._curve_cache.pop("lidar_xy", None)
                    self._curve_cache.pop("lidar_sd", None)
                return
            self._curve_cache["lidar_off"] = False
            ego = stack.ego_state
            pose = (round(float(ego.x), 2), round(float(ego.y), 2), round(float(ego.theta), 3))
            now = time.perf_counter()
            cached = self._lidar_cache
            if cached is not None and (cached[0] == pose or now - cached[2] < 0.05):
                lidar_xy = cached[1]
            else:
                lidar_xy = self._fetch_lidar(stack)
                self._lidar_cache = (pose, lidar_xy, now)
            if show_xy and lidar_xy is not None and len(lidar_xy):
                if self._curve_cache.get("lidar_xy") != id(lidar_xy):
                    self.xy_lidar.setData(lidar_xy[:, 0], lidar_xy[:, 1])
                    self._curve_cache["lidar_xy"] = id(lidar_xy)
                self.layer_point_counts["lidar_xy"] = int(len(lidar_xy))
            else:
                if self._curve_cache.get("lidar_xy") != 0:
                    self.xy_lidar.setData([], [])
                    self._curve_cache["lidar_xy"] = 0
                self.layer_point_counts["lidar_xy"] = 0
            if show_sd and lidar_xy is not None and len(lidar_xy) and gt is not None:
                if self._curve_cache.get("lidar_sd") != id(lidar_xy):
                    sd = _to_sd(gt, lidar_xy)
                    if len(sd):
                        self.sd_lidar.setData(sd[:, 0], sd[:, 1])
                    else:
                        self.sd_lidar.setData([], [])
                    self._curve_cache["lidar_sd"] = id(lidar_xy)
            elif self._curve_cache.get("lidar_sd") != 0:
                self.sd_lidar.setData([], [])
                self._curve_cache["lidar_sd"] = 0

        def _paint_clusters(self, stack, gt, s) -> None:
            xy = _cluster_xy(stack.pm) if s.p66_show_lidar_clusters.get() and self._xy_on else None
            if xy is None or len(xy) == 0:
                if self._curve_cache.get("clu") != 0:
                    self.xy_clusters.setData([], [])
                    self.sd_clusters.setData([], [])
                    self._curve_cache["clu"] = 0
                    self.layer_point_counts["clusters_xy"] = 0
                return
            flat = xy.reshape(-1)
            token = (
                xy.shape,
                round(float(flat[::max(1, flat.size // 8)].sum()), 3),
                self._frenet_on,
                id(gt) if gt is not None else 0,
            )
            self.layer_point_counts["clusters_xy"] = int(len(xy))
            if self._curve_cache.get("clu") == token:
                return
            self._curve_cache["clu"] = token
            self.xy_clusters.setData(xy[:, 0], xy[:, 1])
            if self._frenet_on and gt is not None:
                sd = _to_sd(gt, xy)
                if len(sd):
                    self.sd_clusters.setData(sd[:, 0], sd[:, 1])
                else:
                    self.sd_clusters.setData([], [])
            else:
                self.sd_clusters.setData([], [])

        def _paint_predictions(self, stack, ego, gt, s) -> None:
            pred = getattr(stack.pm, "prediction", None) if stack.pm is not None else None
            show = bool(s.p67_show_prediction.get()) and pred is not None
            modes = _forecast_modes(pred) if show else []
            token = (
                show, self._xy_on, self._frenet_on, id(gt) if gt is not None else 0,
                id(pred) if pred is not None else 0, _paths_token([m[2] for m in modes]),
            )
            if self._curve_cache.get("pred") == token:
                return
            self._curve_cache["pred"] = token
            ahead_xy: list[np.ndarray] = []
            behind_xy: list[np.ndarray] = []
            ellipses: list[np.ndarray] = []
            if modes:
                agents = {getattr(a, "agent_id", i): a for i, a in enumerate(_agent_list(stack.pm))}
                heading = np.array([np.cos(float(ego.theta)), np.sin(float(ego.theta))])
                for aid, mode, path, weight in modes:
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
            if self._xy_on:
                self.xy_pred.setData(ax, ay)
                self.xy_pred_behind.setData(bx, by)
                self.xy_ellipses.setData(ex, ey)
            else:
                self.xy_pred.setData([], [])
                self.xy_pred_behind.setData([], [])
                self.xy_ellipses.setData([], [])
            counted = np.concatenate([ax, bx]) if ax.size or bx.size else ax
            self._count("prediction_xy", counted if self._xy_on else [])
            self._count("prediction_ellipses", ex if self._xy_on else [])
            if self._frenet_on and gt is not None:
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

        def _flow_sample(self, pred):
            if pred is None:
                return None
            raw = getattr(pred, "occupancy_flow", None)
            if raw is None:
                return None
            grid = None
            if isinstance(raw, dict):
                for steps in raw.values():
                    if steps:
                        grid = np.asarray(steps[-1], dtype=float)
                        break
            else:
                try:
                    if len(raw):
                        grid = np.asarray(raw[-1], dtype=float)
                except TypeError:
                    return None
            if grid is None or grid.size == 0:
                return None
            flat = grid.reshape(-1)
            return (grid.shape, round(float(flat[::max(1, flat.size // 8)].sum()), 4))

        def _flow_lut(self):
            if self._flow_lut_cache is not None:
                return self._flow_lut_cache
            lut = np.zeros((256, 4), dtype=np.ubyte)
            ramp = np.arange(256, dtype=np.ubyte)
            lut[:, 0] = ramp
            lut[:, 1] = ramp // 3
            lut[:, 2] = 255 - ramp
            lut[:, 3] = np.where(ramp > 8, 160, 0).astype(np.ubyte)
            self._flow_lut_cache = lut
            return lut

        def _paint_occupancy(self, stack, s) -> None:
            pm = stack.pm
            show_map = bool(s.p67_show_occupancy_map.get()) and self._xy_on
            om = _occupancy_grid(pm) if show_map else None
            if om is not None:
                grid = np.asarray(om.grid)
                flat = grid.reshape(-1)
                token = (
                    grid.shape,
                    round(float(flat[::max(1, flat.size // 16)].sum()), 5),
                    round(float(om.origin_x), 3),
                    round(float(om.origin_y), 3),
                    float(om.resolution or 0.25),
                )
                if self._curve_cache.get("occ") != token:
                    self._curve_cache["occ"] = token
                    self._set_image(
                        self.xy_occ, om.grid, om.origin_x, om.origin_y, float(om.resolution or 0.25),
                    )
                self.layer_point_counts["occupancy"] = int(grid.size)
            else:
                if self._curve_cache.get("occ") != 0:
                    self.xy_occ.hide()
                    self._curve_cache["occ"] = 0
                self.layer_point_counts["occupancy"] = 0
            if not (s.p67_show_prediction.get() and self._xy_on):
                if self._curve_cache.get("flow") != 0:
                    self.xy_flow.hide()
                    self._curve_cache["flow"] = 0
                    self.layer_point_counts["occupancy_flow"] = 0
                return
            pred = getattr(pm, "prediction", None)
            sample = self._flow_sample(pred)
            if sample is None:
                if self._curve_cache.get("flow") != 0:
                    self.xy_flow.hide()
                    self._curve_cache["flow"] = 0
                    self.layer_point_counts["occupancy_flow"] = 0
                return
            if self._curve_cache.get("flow") == sample:
                return
            flow = _flow_image(pred)
            if flow is None:
                self.xy_flow.hide()
                self._curve_cache["flow"] = sample
                self.layer_point_counts["occupancy_flow"] = 0
                return
            grid, (ox, oy, res) = flow
            self._curve_cache["flow"] = sample
            self._set_image(self.xy_flow, grid, ox, oy, res, self._flow_lut())
            self.layer_point_counts["occupancy_flow"] = int(np.count_nonzero(grid > 0.02))

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
