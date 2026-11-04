"""Capability contract dialogs. Qt is imported only when a dialog opens."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_MET = "#6abf69"
_UNMET = "#e06c75"
_UNUSED = "#888888"
_REDUNDANT = "#d19a66"
_CONSUMED = "#6abf69"


def _caps(obj, name: str) -> set:
    val = getattr(obj, name, None)
    if isinstance(val, property) or val is None:
        return set()
    try:
        return set(val)
    except TypeError:
        return set()


def _mentioned(requirements: set) -> set:
    names = set()
    for req in requirements:
        caps = getattr(req, "capabilities", None)
        if caps:
            names.update(caps)
        else:
            names.add(req)
    return names


def show_contract_dialog(parent, *, title: str, target, executer, bridge: bool = False) -> None:
    """Color-coded world / stack contract, matching the Tk popup sections."""
    from .qt_binding import import_qt

    QtGui, QtWidgets = import_qt()[1:]
    from avlite.c40_execution.c41_world_bridge import (
        is_world_capability_enabled,
        is_world_stack_capability_enabled,
    )
    from avlite.c50_common.c51_capabilities import combine_stack_requirements

    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle(title or "(none)")
    dlg.setObjectName("contractDialog")
    dlg.resize(420, 480)
    layout = QtWidgets.QVBoxLayout(dlg)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    body = QtWidgets.QWidget()
    form = QtWidgets.QVBoxLayout(body)

    def heading(text: str) -> None:
        label = QtWidgets.QLabel(text)
        label.setStyleSheet("font-weight: 600;")
        form.addWidget(label)

    def row(text: str, color: str) -> None:
        label = QtWidgets.QLabel("  " + text)
        label.setStyleSheet(f"color: {color};")
        form.addWidget(label)

    if target is None:
        row("(none)", _UNUSED)
        form.addStretch(1)
        scroll.setWidget(body)
        layout.addWidget(scroll)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dlg.accept)
        layout.addWidget(close)
        dlg.exec_()
        return

    world_caps = _caps(target, "world_capabilities")
    world_reqs = _caps(target, "world_requirements")
    stack_reqs = _caps(target, "stack_requirements")
    provided = _caps(target, "stack_capabilities") or _caps(target, "capabilities")
    available: set = set()
    if executer is not None and hasattr(executer, "available_stack_capabilities"):
        try:
            available = set(executer.available_stack_capabilities())
        except Exception:
            log.debug("available_stack_capabilities failed", exc_info=True)

    modules = []
    if executer is not None:
        for attr in ("perception", "localization", "global_planner", "local_planner", "controller"):
            mod = getattr(executer, attr, None)
            if mod is not None and mod is not target:
                modules.append(mod)
        world = getattr(executer, "world", None)
        if world is not None and world is not target:
            modules.append(world)
    try:
        consumed = _mentioned(combine_stack_requirements(modules, soft=True))
    except Exception:
        consumed = set()
    redundant: set = set()
    for mod in modules:
        redundant |= _caps(mod, "stack_capabilities") or _caps(mod, "capabilities")

    if bridge:
        heading("World capabilities")
        if not world_caps:
            row("(none)", _UNUSED)
        for cap in sorted(world_caps, key=lambda c: c.name):
            on = executer is None or is_world_capability_enabled(cap)
            row(cap.name, _MET if on else _UNUSED)
    else:
        heading("World requirements")
        if not world_reqs:
            row("(none)", _UNUSED)
        for cap in sorted(world_reqs, key=lambda c: getattr(c, "name", str(c))):
            present = cap in world_caps or (
                executer is not None
                and getattr(getattr(executer, "world", None), "world_capabilities", None)
                and cap in set(executer.world.world_capabilities)
            )
            label = getattr(cap, "name", str(cap))
            row(label, _MET if present else _UNMET)

    heading("Stack requirements")
    if not stack_reqs:
        row("(none)", _UNUSED)
    for req in sorted(stack_reqs, key=lambda c: getattr(c, "name", str(c))):
        members = set(getattr(req, "capabilities", None) or {req})
        present = bool(members & available) or bool(members & provided)
        label = getattr(req, "name", str(req))
        row(label, _MET if present else _UNMET)

    heading("Stack capabilities")
    if not provided:
        row("(none)", _UNUSED)
    for cap in sorted(provided, key=lambda c: c.name):
        if bridge:
            enabled = executer is None or is_world_stack_capability_enabled(cap)
            if not enabled:
                color = _UNUSED
            elif cap in redundant:
                color = _REDUNDANT
            elif cap in consumed:
                color = _CONSUMED
            else:
                color = _UNUSED
        elif cap in redundant:
            color = _REDUNDANT
        elif cap in consumed:
            color = _CONSUMED
        else:
            color = _UNUSED
        row(cap.name, color)

    form.addStretch(1)
    scroll.setWidget(body)
    layout.addWidget(scroll)
    close = QtWidgets.QPushButton("Close")
    close.clicked.connect(dlg.accept)
    layout.addWidget(close)
    dlg.exec_()
