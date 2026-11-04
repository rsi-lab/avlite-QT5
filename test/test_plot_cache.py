"""Static track geometry is converted to Frenet once, then reused."""

from __future__ import annotations

import numpy as np


def test_track_frenet_is_not_reconverted_every_frame(window):
    gt = window.exec.local_planner.global_trajectory
    calls: list[int] = []
    original = gt.convert_xy_path_to_sd_path_np

    def wrapped(points):
        arr = np.asarray(points)
        calls.append(int(arr.shape[0]) if getattr(arr, "ndim", 1) == 2 else 1)
        return original(points)

    gt.convert_xy_path_to_sd_path_np = wrapped
    try:
        window.update_ui()
        after_first = len(calls)
        large = sum(1 for n in calls if n > 80)
        window.update_ui()
        assert len(calls) - after_first < 20
        assert sum(1 for n in calls if n > 80) == large
        before = len(calls)
        ego = window.exec.ego_state
        window.teleport_ego(float(ego.x) + 0.4, float(ego.y) - 0.3, float(ego.theta))
        window.update_ui()
        assert len(calls) - before < 20
        assert window.local_plot.layer_point_counts.get("track_xy", 0) > 0
    finally:
        gt.convert_xy_path_to_sd_path_np = original
