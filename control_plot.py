"""Control plot: speed gauge, steering wheel, and a short g-g trail.

Qt stays lazy. AVLite imports every plugin module at startup, and importing
PyQt there switches Matplotlib onto the Qt backend.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import VisualizerQtWindow

_G = 9.80665
_READY = False


def _kmh_major_step(max_kmh: float) -> int:
    """Major tick that lands nearest 6–8 labels and on the dial maximum."""
    best = 100
    best_rank = None
    for step in (10, 20, 30, 50, 60, 100):
        count = int(max_kmh // step) + 1
        on_max = abs(max_kmh - step * round(max_kmh / step)) < 0.05
        rank = (0 if 6 <= count <= 8 else 1, 0 if on_max else 1, abs(count - 7))
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = step
    return best


def _colors(widget) -> dict:
    from .qt_binding import import_qt

    _QtCore, QtGui, _QtWidgets = import_qt()
    dark = widget.palette().color(QtGui.QPalette.Window).lightness() < 140
    if dark:
        return {
            "face": QtGui.QColor("#1e1e1e"),
            "track": QtGui.QColor("#5a5a5a"),
            "rim": QtGui.QColor("#8d8d8d"),
            "ink": QtGui.QColor("#dddddd"),
            "muted": QtGui.QColor("#8a8a8a"),
            "accent": QtGui.QColor("#6abf69"),
            "throttle": QtGui.QColor("#e0963a"),
            "neg": QtGui.QColor("#e07a7a"),
            "faint": QtGui.QColor("#888888"),
            "now": QtGui.QColor("#3d8bfd"),
        }
    return {
        "face": QtGui.QColor("#f7f7f7"),
        "track": QtGui.QColor("#c4c4c4"),
        "rim": QtGui.QColor("#555555"),
        "ink": QtGui.QColor("#222222"),
        "muted": QtGui.QColor("#666666"),
        "accent": QtGui.QColor("#2e7d32"),
        "throttle": QtGui.QColor("#d65d0e"),
        "neg": QtGui.QColor("#c62828"),
        "faint": QtGui.QColor("#888888"),
        "now": QtGui.QColor("#3d8bfd"),
    }


def _courier(point_size: int):
    from .qt_binding import import_qt

    _QtCore, QtGui, _QtWidgets = import_qt()
    font = QtGui.QFont("Courier")
    font.setStyleHint(QtGui.QFont.TypeWriter)
    font.setFixedPitch(True)
    font.setPointSize(max(6, int(point_size)))
    return font


def _caption_heights(dial: int) -> tuple[int, int]:
    """Pixel height under the speed dial (km/h plus m/s) and under the wheel."""
    from .qt_binding import import_qt

    _QtCore, QtGui, _QtWidgets = import_qt()
    scale = max(int(dial), 1) / 96.0
    kmh = QtGui.QFontMetrics(_courier(round(11 * scale))).height()
    mps = QtGui.QFontMetrics(_courier(max(7, round(9 * scale)))).height()
    steer = QtGui.QFontMetrics(_courier(round(11 * scale))).height()
    return kmh + mps + 4, steer + 4


def _install() -> None:
    global _READY, QtCore, QtGui, QtWidgets, ControlPlot
    if _READY:
        return
    from .qt_binding import import_qt

    QtCore, QtGui, QtWidgets = import_qt()
    _READY = True

    class SpeedGauge(QtWidgets.QWidget):
        """Speed needle, acceleration arc, km/h, and a smaller m/s line."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.speed = 0.0
            self.accel = 0.0
            self.max_mps = 100.0
            self.accel_min = -20.0
            self.accel_max = 10.0
            self.dial_px = 96
            self.setObjectName("speedGauge")

        def set(self, speed_mps: float, accel: float) -> None:
            self.speed = float(speed_mps)
            self.accel = float(accel)
            self.update()

        def caption(self) -> str:
            return f"{self.speed * 3.6:.1f} km/h"

        def caption_mps(self) -> str:
            return f"{self.speed:.1f} m/s"

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            colors = _colors(self)
            dial = max(1, self.dial_px)
            scale = dial / 96.0
            cx = self.width() / 2.0
            cy = dial / 2.0 + 2.0 * scale
            radius = dial / 2.0 - 4.0 * scale
            painter.setPen(QtGui.QPen(colors["rim"], max(1.0, 2.0 * scale)))
            painter.setBrush(colors["face"])
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

            max_kmh = self.max_mps * 3.6
            major = _kmh_major_step(max_kmh)
            minor = major / 2.0
            ticks = int(round(max_kmh / minor))
            if ticks * minor > max_kmh + 0.05:
                ticks -= 1
            tick_font = _courier(round(7 * scale))
            painter.setFont(tick_font)
            metrics = QtGui.QFontMetrics(tick_font)
            inner_for_labels = radius
            for i in range(ticks + 1):
                if i % 2:
                    continue
                text = str(int(round(i * minor)))
                deg = 210.0 - 240.0 * ((i * minor) / max_kmh)
                rad = math.radians(deg)
                half_w = metrics.horizontalAdvance(text) / 2.0 + 1.0
                half_h = metrics.height() / 2.0
                extent = half_w * abs(math.cos(rad)) + half_h * abs(math.sin(rad))
                inner_for_labels = min(inner_for_labels, radius - 14.0 * scale - extent)
            arc_r = max(inner_for_labels - 6.0 * scale, 0.42 * radius)

            def at(r, deg):
                rad = math.radians(deg)
                return cx + r * math.cos(rad), cy - r * math.sin(rad)

            for i in range(ticks + 1):
                kmh = i * minor
                deg = 210.0 - 240.0 * (kmh / max_kmh)
                is_major = i % 2 == 0
                outer = radius - 3.0 * scale
                inner = radius - (12.0 if is_major else 9.0) * scale
                x0, y0 = at(inner, deg)
                x1, y1 = at(outer, deg)
                painter.setPen(QtGui.QPen(colors["ink"] if is_major else colors["muted"], max(1.0, (2.0 if is_major else 1.0) * scale)))
                painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
                if is_major:
                    label_r = inner - 2.0 * scale - (
                        metrics.horizontalAdvance(str(int(round(kmh)))) / 2.0 * abs(math.cos(math.radians(deg)))
                        + metrics.height() / 2.0 * abs(math.sin(math.radians(deg)))
                    )
                    lx, ly = at(max(8.0, label_r), deg)
                    painter.setPen(colors["ink"])
                    painter.drawText(QtCore.QRectF(lx - 24, ly - 10, 48, 20), QtCore.Qt.AlignCenter, str(int(round(kmh))))

            def arc(radius_px, frac, color, width):
                frac = max(0.0, min(1.0, frac))
                if frac <= 0.004:
                    return
                rect = QtCore.QRectF(cx - radius_px, cy - radius_px, radius_px * 2.0, radius_px * 2.0)
                pen = QtGui.QPen(color, width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
                painter.setPen(pen)
                painter.drawArc(rect, 210 * 16, int(-240 * 16 * frac))

            painter.setBrush(QtCore.Qt.NoBrush)
            arc(arc_r, 1.0, colors["track"], max(2.0, 3.0 * scale))
            shown = min(max(self.speed, 0.0), self.max_mps) / self.max_mps
            arc(arc_r, shown, colors["accent"], max(3.0, 8.0 * scale))
            if self.accel >= 0.0:
                accel_frac = self.accel / max(self.accel_max, 1e-3)
                accel_color = colors["throttle"]
            else:
                accel_frac = -self.accel / max(abs(self.accel_min), 1e-3)
                accel_color = colors["neg"]
            arc(arc_r * 0.7, accel_frac, accel_color, max(2.0, 3.0 * scale))

            tip_x, tip_y = at(arc_r - scale, 210.0 - 240.0 * shown)
            painter.setPen(QtGui.QPen(colors["ink"], max(1.5, 2.0 * scale), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(tip_x, tip_y))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(colors["neg"] if self.speed < 0.0 else colors["accent"])
            painter.drawEllipse(QtCore.QPointF(cx, cy), 3.5 * scale, 3.5 * scale)

            caption_fill = colors["neg"] if self.speed < 0.0 else colors["ink"]
            kmh_font = _courier(round(11 * scale))
            mps_font = _courier(max(7, round(9 * scale)))
            kmh_h = QtGui.QFontMetrics(kmh_font).height()
            painter.setFont(kmh_font)
            painter.setPen(caption_fill)
            painter.drawText(
                QtCore.QRectF(0, dial, self.width(), kmh_h),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                self.caption(),
            )
            painter.setFont(mps_font)
            painter.drawText(
                QtCore.QRectF(0, dial + kmh_h, self.width(), self.height() - dial - kmh_h),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
                self.caption_mps(),
            )

    class SteerWheel(QtWidgets.QWidget):
        """Wheel whose top mark rotates with the steer command. Positive is left."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.steer = 0.0
            self.limit = 0.7
            self.dial_px = 96
            self.setObjectName("steerWheel")

        def set(self, steer: float) -> None:
            self.steer = float(steer)
            self.update()

        def caption(self) -> str:
            return f"{self.steer:+.2f} rad"

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            colors = _colors(self)
            dial = max(1, self.dial_px)
            scale = dial / 96.0
            cx = self.width() / 2.0
            cy = dial / 2.0 + 6.0 * scale
            radius = dial / 2.0 - 18.0 * scale
            painter.setPen(QtGui.QPen(colors["rim"], max(2.0, 5.0 * scale)))
            painter.setBrush(colors["face"])
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

            def at(r, deg):
                rad = math.radians(deg)
                return cx + r * math.cos(rad), cy - r * math.sin(rad)

            tick_font = _courier(round(7 * scale))
            painter.setFont(tick_font)
            metrics = QtGui.QFontMetrics(tick_font)
            for deg, label in ((90.0, "0"), (210.0, "+"), (-30.0, "−")):
                x0, y0 = at(radius - 10.0 * scale, deg)
                x1, y1 = at(radius - 3.0 * scale, deg)
                painter.setPen(QtGui.QPen(colors["muted"], max(1.0, 2.0 * scale)))
                painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
                rad = math.radians(deg)
                extent = metrics.horizontalAdvance(label) / 2.0 + metrics.height() / 2.0
                lx, ly = at(radius - 12.0 * scale - extent, deg)
                painter.setPen(colors["ink"])
                painter.drawText(QtCore.QRectF(lx - 16, ly - 10, 32, 20), QtCore.Qt.AlignCenter, label)

            frac = max(-1.0, min(1.0, self.steer / max(self.limit, 1e-3)))
            rotation = frac * 120.0
            inner = radius * 0.22
            outer = radius - 8.0 * scale
            painter.setPen(QtGui.QPen(colors["ink"], max(1.5, 2.0 * scale), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            for spoke in range(3):
                deg = 90.0 + rotation + spoke * 120.0
                x0, y0 = at(inner, deg)
                x1, y1 = at(outer, deg)
                painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
            x0, y0 = at(inner + 4.0 * scale, 90.0 + rotation)
            x1, y1 = at(outer, 90.0 + rotation)
            painter.setPen(QtGui.QPen(colors["accent"], max(3.0, 6.0 * scale), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
            painter.setPen(QtGui.QPen(colors["rim"], max(1.0, scale)))
            painter.setBrush(colors["track"])
            painter.drawEllipse(QtCore.QPointF(cx, cy), inner, inner)
            tip_y = cy - radius - 2.0 * scale
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(colors["muted"])
            painter.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(cx - 3.5 * scale, tip_y - 6.0 * scale),
                QtCore.QPointF(cx + 3.5 * scale, tip_y - 6.0 * scale),
                QtCore.QPointF(cx, tip_y),
            ]))

            painter.setFont(_courier(round(11 * scale)))
            painter.setPen(colors["ink"])
            painter.drawText(
                QtCore.QRectF(0, dial, self.width(), self.height() - dial),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                self.caption(),
            )

    class GGDiagram(QtWidgets.QWidget):
        """Three rings and a short trail of lateral and longitudinal acceleration."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.trail: list[tuple[float, float]] = []
            self.limit = 20.0
            self.setObjectName("ggDiagram")

        def show_trail(self, trail: list[tuple[float, float]], limit: float) -> None:
            self.trail = trail
            self.limit = max(float(limit), 1e-3)
            self.update()

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            colors = _colors(self)
            painter.fillRect(self.rect(), colors["face"])
            font = _courier(8)
            painter.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            side_m = metrics.horizontalAdvance("-0.0") + 12
            top_m = metrics.height() + 4
            radius = min(self.width() - 2 * side_m, self.height() - top_m) / 2.0 - 4.0
            if radius < 12:
                return
            cx = self.width() / 2.0
            cy = (self.height() + top_m) / 2.0
            outer_g = max(1.0, float(math.ceil(self.limit / _G)))
            slant = 2 ** -0.5
            roomy = radius / 3.0 * slant >= metrics.height() + 1
            painter.setBrush(QtCore.Qt.NoBrush)
            for step in range(1, 4):
                ring = radius * step / 3.0
                painter.setPen(QtGui.QPen(colors["ink"], 1))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(QtCore.QPointF(cx, cy), ring, ring)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(colors["ink"])
                for dx, dy in ((0.0, -ring), (ring, 0.0), (0.0, ring), (-ring, 0.0)):
                    painter.drawEllipse(QtCore.QPointF(cx + dx, cy + dy), 2.0, 2.0)
                if roomy or step == 3:
                    text = f"{outer_g * step / 3.0:.1f}"
                    tx = cx - ring * slant
                    ty = cy - ring * slant
                    painter.setPen(colors["faint"])
                    painter.drawText(QtCore.QRectF(tx - 18, ty - 8, 36, 16), QtCore.Qt.AlignCenter, text)
            painter.setPen(colors["faint"])
            painter.drawText(QtCore.QRectF(cx - 16, cy - 8, 32, 16), QtCore.Qt.AlignCenter, "0.0")

            if self.trail:
                lateral, longitudinal = self.trail[-1]
                long_text = f"{longitudinal / _G:+.1f}"
                lat_text = f"{lateral / _G:+.1f}"
            else:
                long_text = lat_text = "0.0"
            painter.setPen(colors["ink"])
            painter.drawText(
                QtCore.QRectF(cx - 28, cy - radius - metrics.height() - 2, 56, metrics.height()),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
                long_text,
            )
            painter.drawText(
                QtCore.QRectF(cx + radius + 6, cy - metrics.height() / 2.0, side_m, metrics.height()),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                lat_text,
            )
            span = outer_g * _G
            if len(self.trail) >= 2:
                path = QtGui.QPainterPath()
                x0 = cx + self.trail[0][0] / span * radius
                y0 = cy - self.trail[0][1] / span * radius
                path.moveTo(x0, y0)
                for lateral, longitudinal in self.trail[1:]:
                    path.lineTo(cx + lateral / span * radius, cy - longitudinal / span * radius)
                painter.setPen(QtGui.QPen(colors["ink"], 1.2))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawPath(path)
            if self.trail:
                lateral, longitudinal = self.trail[-1]
                x = cx + lateral / span * radius
                y = cy - longitudinal / span * radius
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(colors["now"])
                painter.drawEllipse(QtCore.QPointF(x, y), 4.0, 4.0)

    class ControlPlot(QtWidgets.QWidget):
        """One pane: speed, steer, and g-g, sharing a circle size."""

        def __init__(self, root: "VisualizerQtWindow"):
            super().__init__()
            self.root = root
            self.setObjectName("controlPlot")
            self.speed_gauge = SpeedGauge(self)
            self.steer_wheel = SteerWheel(self)
            self.gg = GGDiagram(self)
            self.trail: list[tuple[float, float]] = []
            self.previous: tuple[float, float, float] | None = None
            self.dial = 0
            self.placed_size = (0, 0)
            self.setMinimumSize(140, 140)
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        def reset(self) -> None:
            self.trail.clear()
            self.previous = None
            self.gg.show_trail(self.trail, self.gg.limit)

        def update_theme(self, _bg: str, _fg: str) -> None:
            for child in (self.speed_gauge, self.steer_wheel, self.gg):
                child.update()

        def resizeEvent(self, event) -> None:  # noqa: N802
            self.place()
            super().resizeEvent(event)

        def place(self) -> None:
            margin = 8
            inner_w = self.width() - 2 * margin
            inner_h = self.height() - 2 * margin
            if inner_w < 48 or inner_h < 48:
                return
            side_m = 36
            top_m = 18

            def largest(allow_wrap: bool) -> int:
                found = 1
                lo, hi = 1, max(1, min(inner_w, inner_h))
                while lo <= hi:
                    mid = (lo + hi) // 2
                    speed_cap, steer_cap = _caption_heights(mid)
                    sizes = (
                        (max(mid, 72), mid + speed_cap),
                        (max(mid, 72), mid + steer_cap),
                        (mid + 2 * side_m, mid + top_m),
                    )
                    used = column_h = column_w = 0
                    open_column = False
                    fits = True
                    for w, h in sizes:
                        if w > inner_w or h > inner_h:
                            fits = False
                            break
                        if open_column and column_h + h > inner_h:
                            if not allow_wrap:
                                fits = False
                                break
                            used += column_w
                            column_h = column_w = 0
                            open_column = False
                        open_column = True
                        column_w = max(column_w, w)
                        column_h += h
                    if fits:
                        fits = used + column_w <= inner_w
                    if fits:
                        found = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return found

            stacked, wrapped = largest(False), largest(True)
            wanted = wrapped if wrapped > 1.5 * stacked else stacked
            if wanted < 24:
                return
            size = (self.width(), self.height())
            if self.placed_size == size and self.speed_gauge.width() > 0:
                return
            self.placed_size = size
            self.dial = wanted
            speed_cap, steer_cap = _caption_heights(wanted)
            self.speed_gauge.dial_px = wanted
            self.steer_wheel.dial_px = wanted
            gg_w, gg_h = wanted + 2 * side_m, wanted + top_m
            items = (
                (self.speed_gauge, max(wanted, 72), wanted + speed_cap),
                (self.steer_wheel, max(wanted, 72), wanted + steer_cap),
                (self.gg, gg_w, gg_h),
            )
            columns: list[list] = []
            column: list = []
            y = col_w = 0
            for widget, w, h in items:
                if column and y + h > inner_h:
                    columns.append((column, col_w, y))
                    column = []
                    y = col_w = 0
                column.append((widget, w, h, y))
                col_w = max(col_w, w)
                y += h
            if column:
                columns.append((column, col_w, y))
            used_w = sum(width for _column, width, _height in columns)
            x = max(0, (self.width() - used_w) // 2) if len(columns) == 1 else margin
            for column, width, height in columns:
                y0 = max(0, (self.height() - height) // 2) if len(columns) == 1 else margin
                for widget, w, h, top in column:
                    widget.setGeometry(x + max(0, (width - w) // 2), y0 + top, w, h)
                x += width

        def redraw(self) -> float:
            t0 = time.perf_counter()
            stack = self.root.exec
            if stack is None or stack.ego_state is None:
                return 0.0
            ego = stack.ego_state
            now = float(stack.elapsed_sim_time)
            if self.previous is not None:
                then, speed, heading = self.previous
                dt = now - then
                if 0.0 < dt <= 1.0:
                    turn = (ego.theta - heading + math.pi) % (2.0 * math.pi) - math.pi
                    self.trail.append((ego.velocity * turn / dt, (ego.velocity - speed) / dt))
                    del self.trail[:-80]
            self.previous = (now, float(ego.velocity), float(ego.theta))
            self.place()
            from avlite.c30_control.c39_settings import ControlSettings

            controller = getattr(stack, "controller", None)
            cmd = getattr(controller, "cmd", None) if controller is not None else None
            if cmd is None:
                cmd = getattr(self.root, "applied_cmd", None)
            self.speed_gauge.accel_min = float(ControlSettings.c32_ego_min_acceleration)
            self.speed_gauge.accel_max = float(ControlSettings.c32_ego_max_acceleration)
            self.speed_gauge.set(float(ego.velocity), 0.0 if cmd is None else float(getattr(cmd, "acceleration", 0.0) or 0.0))
            self.steer_wheel.limit = max(
                abs(float(ControlSettings.c32_ego_min_steering)),
                abs(float(ControlSettings.c32_ego_max_steering)),
                1e-3,
            )
            self.steer_wheel.set(0.0 if cmd is None else float(getattr(cmd, "steer", 0.0) or 0.0))
            limit = max(
                abs(float(ControlSettings.c32_ego_min_acceleration)),
                abs(float(ControlSettings.c32_ego_max_acceleration)),
                1e-3,
            )
            self.gg.show_trail(self.trail, limit)
            return (time.perf_counter() - t0) * 1000.0


def __getattr__(name: str):
    if name == "ControlPlot":
        _install()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
