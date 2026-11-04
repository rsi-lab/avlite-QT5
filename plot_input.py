"""Plot mouse and wheel input that bypasses pyqtgraph's context menu.

Importing this module does not import Qt. ``attach_plot_mouse`` does, and only
when a plot is actually created.
"""

from __future__ import annotations


def attach_plot_mouse(plot, handler) -> None:
    """Send press, move, release, and wheel events to *handler* in view coordinates.

    The pyqtgraph ViewBox menu is turned off. Events are taken on both the
    ``PlotWidget`` and its viewport so a real click and ``QTest.mouseClick``
    on either target are handled once and never open the default menu.
    """
    from .qt_binding import import_qt

    QtCore, _QtGui, _QtWidgets = import_qt()

    plot.plotItem.setMenuEnabled(False, enableViewBoxMenu=False)
    plot.getViewBox().setMenuEnabled(False)

    class _Filter(QtCore.QObject):
        def eventFilter(self, obj, ev):  # noqa: N802
            et = ev.type()
            if et == QtCore.QEvent.MouseButtonPress:
                handler.on_press(_view_pos(plot, obj, ev), ev.button(), ev.modifiers())
                return True
            if et == QtCore.QEvent.MouseMove:
                handler.on_move(_view_pos(plot, obj, ev), ev.buttons(), ev.modifiers())
                return True
            if et == QtCore.QEvent.MouseButtonRelease:
                handler.on_release(_view_pos(plot, obj, ev), ev.button(), ev.modifiers())
                return True
            if et == QtCore.QEvent.Wheel:
                delta = ev.angleDelta().y() if hasattr(ev, "angleDelta") else ev.delta()
                handler.on_wheel(_view_pos(plot, obj, ev), int(delta))
                return True
            return False

    filt = _Filter(plot)
    plot.installEventFilter(filt)
    plot.viewport().installEventFilter(filt)
    plot._avlite_mouse_filter = filt


def _view_pos(plot, obj, ev):
    if obj is plot.viewport():
        scene = plot.mapToScene(ev.pos())
    else:
        scene = plot.mapToScene(plot.viewport().mapFrom(plot, ev.pos()))
    return plot.getViewBox().mapSceneToView(scene)


def zoom_step(current: float, wheel_delta: int, *, lo: float, hi: float, step: float = 10.0) -> float:
    """Match the Tk visualizer: wheel up decreases the half-width (zoom in)."""
    if wheel_delta > 0:
        return float(max(lo, current - step))
    if wheel_delta < 0:
        return float(min(hi, current + step))
    return float(current)
