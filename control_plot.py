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
    # A fixed floor would overflow a small pane. The dial fit refuses a size that does not clear.
    font.setPointSize(max(1, int(point_size)))
    return font


def _pane_point(dial: int, base: float) -> int:
    """Tk's point size on a 96 px dial, grown and shrunk with the pane."""
    return max(1, int(round(base * max(int(dial), 1) / 96.0)))


def _inset(dial: int) -> float:
    """Tick and spoke gaps were authored on Tk's 148 px dial."""
    return max(int(dial), 1) / 148.0


def _at(cx: float, cy: float, radius: float, deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    return cx + radius * math.cos(rad), cy - radius * math.sin(rad)


def _label_inside(metrics, text: str, deg: float, outer: float) -> tuple[float, float]:
    """Center radius, and how far the glyph reaches, so the ink stays inside ``outer``."""
    half_w = metrics.horizontalAdvance(text) / 2.0 + 1.0
    half_h = metrics.height() / 2.0
    rad = math.radians(deg)
    extent = half_w * abs(math.cos(rad)) + half_h * abs(math.sin(rad))
    return outer - extent, extent


def _draw_centered(painter, font, text: str, x: float, y: float):
    """Draw ``text`` on its baseline, centered on ``(x, y)``, and return the ink rectangle.

    A fixed box clips a font that has scaled up with the pane, and a box larger than
    the glyph covers the arc even when the letters themselves would fit.
    """
    from .qt_binding import import_qt

    QtCore, QtGui, _QtWidgets = import_qt()
    painter.setFont(font)
    metrics = QtGui.QFontMetrics(font)
    width = float(metrics.horizontalAdvance(text))
    height = float(metrics.height())
    baseline = y - height / 2.0 + metrics.ascent()
    painter.drawText(QtCore.QPointF(x - width / 2.0, baseline), text)
    return QtCore.QRectF(x - width / 2.0, y - height / 2.0, width, height)


def _rects_clear(rects, gap: float) -> bool:
    for index, left in enumerate(rects):
        grown = left.adjusted(-gap, -gap, gap, gap)
        for right in rects[index + 1 :]:
            if grown.intersects(right):
                return False
    return True


def _inside(rect, width: float, height: float, margin: float = 0.5) -> bool:
    return (
        rect.left() >= margin
        and rect.top() >= margin
        and rect.right() <= width - margin
        and rect.bottom() <= height - margin
    )


def _gg_margins(dial: int) -> tuple[int, int]:
    """Side and top bands for the live g readout. Measured, so the text is not clipped."""
    from .qt_binding import import_qt

    _QtCore, QtGui, _QtWidgets = import_qt()
    metrics = QtGui.QFontMetrics(_courier(_pane_point(dial, 8)))
    return metrics.horizontalAdvance("-00.0") + 12, metrics.height() + 4


def _gauge_chrome(dial: int) -> tuple[int, int, int, int, int, int]:
    """Speed width, speed caption height, steer width, steer caption height, g-g side, g-g top."""
    from .qt_binding import import_qt

    _QtCore, QtGui, _QtWidgets = import_qt()
    kmh = QtGui.QFontMetrics(_courier(_pane_point(dial, 11)))
    mps = QtGui.QFontMetrics(_courier(_pane_point(dial, 9)))
    steer = QtGui.QFontMetrics(_courier(_pane_point(dial, 11)))
    speed_w = max(kmh.horizontalAdvance("-999.9 km/h"), mps.horizontalAdvance("-999.9 m/s")) + 8
    steer_w = steer.horizontalAdvance("-0.70 rad") + 8
    side, top = _gg_margins(dial)
    return speed_w, kmh.height() + mps.height() + 4, steer_w, steer.height() + 4, side, top


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
            self.label_rects: list = []
            self.arc_center = (0.0, 0.0)
            self.arc_outer = 0.0
            self.setObjectName("speedGauge")

        def set(self, speed_mps: float, accel: float) -> None:
            self.speed = float(speed_mps)
            self.accel = float(accel)
            self.update()

        def caption(self) -> str:
            return f"{self.speed * 3.6:.1f} km/h"

        def caption_mps(self) -> str:
            return f"{self.speed:.1f} m/s"

        def _label_plan(self, cx: float, cy: float, radius: float, k: float, max_kmh: float):
            """Largest tick font that stays inside the ticks and leaves a ring inside the numbers."""
            dial = max(1, self.dial_px)
            major = _kmh_major_step(max_kmh)
            minor = major / 2.0
            count = int(round(max_kmh / minor)) if minor else 0
            if count * minor > max_kmh + 0.05:
                count -= 1
            marks = []
            for index in range(count + 1):
                kmh = index * minor
                deg = 210.0 - 240.0 * (kmh / max_kmh if max_kmh else 0.0)
                marks.append((str(int(round(kmh))), deg, index % 2 == 0))
            desired = max(2.0, min(12.0, 8.0 * math.sqrt(dial / 96.0)))
            fallback = None
            for point in range(_pane_point(dial, 7), 0, -1):
                font = _courier(point)
                metrics = QtGui.QFontMetrics(font)
                rects = []
                inner = radius
                failed = False
                for text, deg, is_major in marks:
                    if not is_major:
                        continue
                    label_r, extent = _label_inside(metrics, text, deg, radius - 14.0 * k)
                    if label_r < 4.0:
                        failed = True
                        break
                    lx, ly = _at(cx, cy, label_r, deg)
                    rect = QtCore.QRectF(
                        lx - metrics.horizontalAdvance(text) / 2.0,
                        ly - metrics.height() / 2.0,
                        float(metrics.horizontalAdvance(text)),
                        float(metrics.height()),
                    )
                    limit = dial - max(2.0, k)
                    if rect.top() < 1 or rect.bottom() > limit or not _inside(rect, self.width(), self.height()):
                        failed = True
                        break
                    inner = min(inner, label_r - extent)
                    rects.append((text, deg, rect))
                gap = max(2.0, metrics.height() * 0.12)
                if failed or not _rects_clear([rect for _text, _deg, rect in rects], gap):
                    continue
                room = inner - gap
                stroke = min(desired, max(1.5, room * 0.5))
                arc_r = room - stroke / 2.0
                if arc_r < 1.0:
                    continue
                plan = (font, rects, arc_r, stroke, marks)
                if fallback is None:
                    fallback = plan
                if arc_r >= 0.22 * radius:
                    return plan
            if fallback is not None:
                return fallback
            return _courier(1), [], max(8.0, 0.45 * radius), 2.0, marks

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
            colors = _colors(self)
            dial = max(1, self.dial_px)
            k = _inset(dial)
            cx = self.width() / 2.0
            cy = dial / 2.0 + k
            radius = max(4.0, dial / 2.0 - 4.0 * k)
            max_kmh = max(self.max_mps, 1e-3) * 3.6
            font, labels, arc_r, stroke, marks = self._label_plan(cx, cy, radius, k, max_kmh)
            self.arc_center = (cx, cy)
            self.arc_outer = arc_r + stroke / 2.0

            painter.setPen(QtGui.QPen(colors["rim"], max(1.0, 2.0 * k)))
            painter.setBrush(colors["face"])
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)
            for text, deg, is_major in marks:
                outer = radius - 3.0 * k
                inner = radius - (12.0 if is_major else 9.0) * k
                x0, y0 = _at(cx, cy, inner, deg)
                x1, y1 = _at(cx, cy, outer, deg)
                painter.setPen(QtGui.QPen(colors["ink"] if is_major else colors["muted"], max(1.0, (2.0 if is_major else 1.0) * k)))
                painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

            def arc(radius_px, frac, color, width):
                frac = max(0.0, min(1.0, frac))
                if frac <= 0.004:
                    return
                rect = QtCore.QRectF(cx - radius_px, cy - radius_px, radius_px * 2.0, radius_px * 2.0)
                pen = QtGui.QPen(color, width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawArc(rect, 210 * 16, int(-240 * 16 * frac))

            arc(arc_r, 1.0, colors["track"], max(1.5, stroke * 0.4))
            shown = min(max(self.speed, 0.0), self.max_mps) / max(self.max_mps, 1e-3)
            arc(arc_r, shown, colors["accent"], stroke)
            if self.accel >= 0.0:
                accel_frac = self.accel / max(self.accel_max, 1e-3)
                accel_color = colors["throttle"]
            else:
                accel_frac = -self.accel / max(abs(self.accel_min), 1e-3)
                accel_color = colors["neg"]
            accel_w = max(1.5, stroke * 0.4)
            arc(max(4.0, arc_r * 0.7), accel_frac, accel_color, accel_w)

            tip_x, tip_y = _at(cx, cy, max(0.0, arc_r - k), 210.0 - 240.0 * shown)
            painter.setPen(QtGui.QPen(colors["ink"], max(1.5, 2.0 * k), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(tip_x, tip_y))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(colors["neg"] if self.speed < 0.0 else colors["accent"])
            painter.drawEllipse(QtCore.QPointF(cx, cy), max(2.0, 3.5 * k), max(2.0, 3.5 * k))

            # Numbers are drawn after the arc and the needle so a thick stroke cannot cover them.
            painter.setPen(colors["ink"])
            drawn = []
            for text, deg, _rect in labels:
                label_r, _extent = _label_inside(QtGui.QFontMetrics(font), text, deg, radius - 14.0 * k)
                lx, ly = _at(cx, cy, label_r, deg)
                drawn.append(_draw_centered(painter, font, text, lx, ly))

            caption_fill = colors["neg"] if self.speed < 0.0 else colors["ink"]
            kmh_font = _courier(_pane_point(dial, 11))
            mps_font = _courier(_pane_point(dial, 9))
            kmh_h = QtGui.QFontMetrics(kmh_font).height()
            mps_h = QtGui.QFontMetrics(mps_font).height()
            painter.setPen(caption_fill)
            drawn.append(_draw_centered(painter, kmh_font, self.caption(), cx, dial + kmh_h / 2.0))
            drawn.append(_draw_centered(painter, mps_font, self.caption_mps(), cx, dial + kmh_h + mps_h / 2.0))
            self.label_rects = drawn

    class SteerWheel(QtWidgets.QWidget):
        """Wheel whose top mark rotates with the steer command. Positive is left."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.steer = 0.0
            self.limit = 0.7
            self.dial_px = 96
            self.label_rects: list = []
            self.spoke_outer = 0.0
            self.label_inner = 0.0
            self.setObjectName("steerWheel")

        def set(self, steer: float) -> None:
            self.steer = float(steer)
            self.update()

        def caption(self) -> str:
            return f"{self.steer:+.2f} rad"

        def _label_plan(self, cx: float, cy: float, radius: float, k: float):
            """Largest 0 / + / − font that stays off the spokes and inside the rim."""
            dial = max(1, self.dial_px)
            names = ((90.0, "0"), (210.0, "+"), (-30.0, "−"))
            hub_r = radius * 0.22
            accent_w = max(2.0, 6.0 * k)
            for point in range(_pane_point(dial, 7), 0, -1):
                font = _courier(point)
                metrics = QtGui.QFontMetrics(font)
                placed = []
                inner_edge = radius
                failed = False
                for deg, text in names:
                    label_r, extent = _label_inside(metrics, text, deg, radius - 12.0 * k)
                    edge = label_r - extent
                    if edge < hub_r + accent_w / 2.0 + 2.0 * k:
                        failed = True
                        break
                    lx, ly = _at(cx, cy, label_r, deg)
                    rect = QtCore.QRectF(
                        lx - metrics.horizontalAdvance(text) / 2.0,
                        ly - metrics.height() / 2.0,
                        float(metrics.horizontalAdvance(text)),
                        float(metrics.height()),
                    )
                    if rect.bottom() > dial - max(2.0, k) or not _inside(rect, self.width(), self.height()):
                        failed = True
                        break
                    inner_edge = min(inner_edge, edge)
                    placed.append((text, deg, rect))
                gap = max(2.0, metrics.height() * 0.1)
                if failed or not _rects_clear([rect for _text, _deg, rect in placed], gap):
                    continue
                # Tk stops the spokes well short of the letters. A scaled font can sit closer, so stop early.
                spoke_outer = min(radius - 34.0 * k, inner_edge - accent_w / 2.0 - max(1.5, 2.0 * k))
                if spoke_outer <= hub_r + 2.0 * k:
                    continue
                return font, placed, spoke_outer, inner_edge, accent_w, hub_r
            return _courier(1), [], hub_r + 4.0, radius, accent_w, hub_r

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
            colors = _colors(self)
            dial = max(1, self.dial_px)
            k = _inset(dial)
            cx = self.width() / 2.0
            cy = dial / 2.0 + 6.0 * k
            radius = max(8.0, dial / 2.0 - 18.0 * k)
            font, labels, spoke_outer, label_inner, accent_w, hub_r = self._label_plan(cx, cy, radius, k)
            self.spoke_outer = spoke_outer
            self.label_inner = label_inner

            painter.setPen(QtGui.QPen(colors["rim"], max(2.0, 5.0 * k)))
            painter.setBrush(colors["face"])
            painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)
            for _text, deg, _rect in labels:
                x0, y0 = _at(cx, cy, radius - 10.0 * k, deg)
                x1, y1 = _at(cx, cy, radius - 3.0 * k, deg)
                painter.setPen(QtGui.QPen(colors["muted"], max(1.0, 2.0 * k)))
                painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

            frac = max(-1.0, min(1.0, self.steer / max(self.limit, 1e-3)))
            rotation = frac * 120.0
            painter.setPen(QtGui.QPen(colors["ink"], max(1.5, 2.0 * k), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            for spoke in range(3):
                deg = 90.0 + rotation + spoke * 120.0
                x0, y0 = _at(cx, cy, hub_r, deg)
                x1, y1 = _at(cx, cy, spoke_outer, deg)
                painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
            x0, y0 = _at(cx, cy, hub_r + 4.0 * k, 90.0 + rotation)
            x1, y1 = _at(cx, cy, spoke_outer, 90.0 + rotation)
            painter.setPen(QtGui.QPen(colors["accent"], accent_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
            painter.setPen(QtGui.QPen(colors["rim"], max(1.0, k)))
            painter.setBrush(colors["track"])
            painter.drawEllipse(QtCore.QPointF(cx, cy), hub_r, hub_r)
            tip_y = cy - radius - 2.0 * k
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(colors["muted"])
            painter.drawPolygon(QtGui.QPolygonF([
                QtCore.QPointF(cx - 3.5 * k, tip_y - 6.0 * k),
                QtCore.QPointF(cx + 3.5 * k, tip_y - 6.0 * k),
                QtCore.QPointF(cx, tip_y),
            ]))

            painter.setPen(colors["ink"])
            drawn = []
            for text, deg, _rect in labels:
                label_r, _extent = _label_inside(QtGui.QFontMetrics(font), text, deg, radius - 12.0 * k)
                lx, ly = _at(cx, cy, label_r, deg)
                drawn.append(_draw_centered(painter, font, text, lx, ly))
            caption_font = _courier(_pane_point(dial, 11))
            caption_h = float(QtGui.QFontMetrics(caption_font).height())
            band = max(caption_h, float(self.height() - dial))
            drawn.append(_draw_centered(painter, caption_font, self.caption(), cx, dial + band / 2.0))
            self.label_rects = drawn

    class GGDiagram(QtWidgets.QWidget):
        """Three rings and a short trail of lateral and longitudinal acceleration."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.trail: list[tuple[float, float]] = []
            self.limit = 20.0
            self.dial_px = 96
            self.label_rects: list = []
            self.setObjectName("ggDiagram")

        def show_trail(self, trail: list[tuple[float, float]], limit: float) -> None:
            self.trail = trail
            self.limit = max(float(limit), 1e-3)
            self.update()

        def _readout(self) -> tuple[str, str]:
            if self.trail:
                lateral, longitudinal = self.trail[-1]
                return f"{longitudinal / _G:+.1f}", f"{lateral / _G:+.1f}"
            return "0.0", "0.0"

        def _label_plan(self, cx: float, cy: float, radius: float, outer_g: float):
            """Ring numbers and the two live readouts, scaled down until none of them meet."""
            dial = max(1, self.dial_px)
            slant = 2 ** -0.5
            long_text, lat_text = self._readout()
            width = float(self.width())
            height = float(self.height())

            def attempt(point: int, rings: bool):
                font = _courier(point)
                metrics = QtGui.QFontMetrics(font)
                roomy = radius / 3.0 * slant >= metrics.height() + 1
                if rings and not roomy:
                    return None
                steps = (1, 2, 3) if rings else ()
                items = []
                for step in steps:
                    ring = radius * step / 3.0
                    text = f"{outer_g * step / 3.0:.1f}"
                    rect = QtCore.QRectF(
                        cx - ring * slant - metrics.horizontalAdvance(text) / 2.0,
                        cy - ring * slant - metrics.height() / 2.0,
                        float(metrics.horizontalAdvance(text)),
                        float(metrics.height()),
                    )
                    items.append((text, rect, True))
                center = QtCore.QRectF(
                    cx - metrics.horizontalAdvance("0.0") / 2.0,
                    cy - metrics.height() / 2.0,
                    float(metrics.horizontalAdvance("0.0")),
                    float(metrics.height()),
                )
                items.append(("0.0", center, True))
                line = float(metrics.height())
                long_w = float(metrics.horizontalAdvance(long_text))
                lat_w = float(metrics.horizontalAdvance(lat_text))
                long_rect = QtCore.QRectF(cx - long_w / 2.0, cy - radius - 3.0 - line, long_w, line)
                lat_rect = QtCore.QRectF(cx + radius + 10.0, cy - line / 2.0, lat_w, line)
                items.append((long_text, long_rect, False))
                items.append((lat_text, lat_rect, False))
                rects = [rect for _text, rect, _faint in items]
                if any(not _inside(rect, width, height) for rect in rects):
                    return None
                if not _rects_clear(rects, 1.0):
                    return None
                return font, items

            target = _pane_point(dial, 8)
            for point in range(target, 0, -1):
                plan = attempt(point, rings=True)
                if plan is not None:
                    return plan
            for point in range(target, 0, -1):
                plan = attempt(point, rings=False)
                if plan is not None:
                    return plan
            font = _courier(1)
            return font, []

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
            colors = _colors(self)
            painter.fillRect(self.rect(), colors["face"])
            if self.width() < 8 or self.height() < 8:
                self.label_rects = []
                return
            dial = max(1, self.dial_px)
            side_m, top_m = _gg_margins(dial)
            radius = min(self.width() - 2 * side_m, self.height() - top_m) / 2.0 - 4.0
            cx = self.width() / 2.0
            cy = (self.height() + top_m) / 2.0
            outer_g = max(1.0, float(math.ceil(self.limit / _G)))
            if radius < 8.0:
                long_text, lat_text = self._readout()
                font = _courier(_pane_point(dial, 8))
                painter.setPen(colors["ink"])
                line = float(QtGui.QFontMetrics(font).height())
                self.label_rects = [
                    _draw_centered(painter, font, long_text, cx, line / 2.0 + 1.0),
                    _draw_centered(painter, font, lat_text, cx, line * 1.5 + 2.0),
                ]
                return
            font, items = self._label_plan(cx, cy, radius, outer_g)
            dot = max(1.5, min(2.5, 2.0 * dial / 96.0))
            painter.setBrush(QtCore.Qt.NoBrush)
            for step in range(1, 4):
                ring = radius * step / 3.0
                painter.setPen(QtGui.QPen(colors["ink"], 1))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(QtCore.QPointF(cx, cy), ring, ring)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(colors["ink"])
                for dx, dy in ((0.0, -ring), (ring, 0.0), (0.0, ring), (-ring, 0.0)):
                    painter.drawEllipse(QtCore.QPointF(cx + dx, cy + dy), dot, dot)
            span = outer_g * _G
            if len(self.trail) >= 2:
                path = QtGui.QPainterPath()
                path.moveTo(cx + self.trail[0][0] / span * radius, cy - self.trail[0][1] / span * radius)
                for lateral, longitudinal in self.trail[1:]:
                    path.lineTo(cx + lateral / span * radius, cy - longitudinal / span * radius)
                painter.setPen(QtGui.QPen(colors["ink"], max(1.0, min(2.0, dial / 96.0))))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawPath(path)
            if self.trail:
                lateral, longitudinal = self.trail[-1]
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(colors["now"])
                painter.drawEllipse(
                    QtCore.QPointF(cx + lateral / span * radius, cy - longitudinal / span * radius),
                    min(6.0, max(3.0, 4.0 * dial / 96.0)),
                    min(6.0, max(3.0, 4.0 * dial / 96.0)),
                )
            # A plate breaks the ring and the trail behind each number. The stroke used to run through the glyph.
            drawn = []
            for text, rect, faint in items:
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(colors["face"])
                painter.drawRect(rect.adjusted(-1.0, 0.0, 1.0, 0.0))
                painter.setPen(colors["faint"] if faint else colors["ink"])
                drawn.append(_draw_centered(painter, font, text, rect.center().x(), rect.center().y()))
            self.label_rects = drawn

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

            def largest(allow_wrap: bool) -> int:
                found = 1
                lo, hi = 1, max(1, min(inner_w, inner_h))
                while lo <= hi:
                    mid = (lo + hi) // 2
                    speed_w, speed_h, steer_w, steer_h, side_m, top_m = _gauge_chrome(mid)
                    sizes = (
                        (max(mid, speed_w), mid + speed_h),
                        (max(mid, steer_w), mid + steer_h),
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
            speed_w, speed_h, steer_w, steer_h, side_m, top_m = _gauge_chrome(wanted)
            self.speed_gauge.dial_px = wanted
            self.steer_wheel.dial_px = wanted
            self.gg.dial_px = wanted
            items = (
                (self.speed_gauge, max(wanted, speed_w), wanted + speed_h),
                (self.steer_wheel, max(wanted, steer_w), wanted + steer_h),
                (self.gg, wanted + 2 * side_m, wanted + top_m),
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
