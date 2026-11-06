"""Title-bar fold button for a group box. Qt is imported only when attached."""

from __future__ import annotations


def attach_panel_fold(group, var, *, name: str) -> None:
    """Fold *group* down to its title when *var* is true.

    The button sits on the title row and is not part of the layout, so an
    unfolded panel keeps the size it had before the button was added.
    """
    from .qt_binding import import_qt

    QtCore, _QtGui, QtWidgets = import_qt()
    button = QtWidgets.QToolButton(group)
    button.setObjectName(name)
    button.setAutoRaise(True)
    button.setFixedSize(18, 18)
    button.setFocusPolicy(QtCore.Qt.NoFocus)
    button.setToolTip("Fold panel")

    def _place() -> None:
        button.move(max(0, group.width() - button.width() - 8), 1)
        button.raise_()

    def _walk(layout, visible: bool) -> None:
        if layout is None:
            return
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget()
            if widget is not None and widget is not button:
                widget.setVisible(visible)
            child = item.layout()
            if child is not None:
                _walk(child, visible)

    def _apply() -> None:
        folded = bool(var.get())
        button.setText("▸" if folded else "▾")
        _walk(group.layout(), not folded)
        if folded:
            group.setMaximumHeight(max(24, group.fontMetrics().height() + 14))
        else:
            group.setMaximumHeight(16777215)
        _place()

    class _FollowResize(QtCore.QObject):
        def eventFilter(self, obj, event):  # noqa: N802
            if event.type() == QtCore.QEvent.Resize:
                _place()
            return False

    follow = _FollowResize(group)
    group.installEventFilter(follow)
    group._fold_follow = follow  # keep the filter alive
    button.clicked.connect(lambda: var.set(not bool(var.get())))
    var.bind(_apply)
    _apply()
