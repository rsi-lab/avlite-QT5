"""Keyboard shortcut catalog (mirrors the Tk visualizer defaults).

Display strings match ``p64_shortcuts.display_sequence``: a bare uppercase
letter is the shifted key (``S`` is Shift+S), a bare lowercase letter is
unshifted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shortcut:
    id: str
    group: str
    label: str
    default: str


SHORTCUTS: tuple[Shortcut, ...] = (
    Shortcut("quit", "App", "Quit", "Q"),
    Shortcut("toggle_shortcuts", "App", "Toggle shortcut panel", "S"),
    Shortcut("switch_profile", "App", "Switch profile", "F"),
    Shortcut("reload", "App", "Reload stack", "R"),
    Shortcut("open_settings", "App", "Open settings", "T"),
    Shortcut("save_settings", "App", "Save settings", "Ctrl+S"),
    Shortcut("focus", "App", "Focus window", "Esc"),
    Shortcut("step_plan", "Plan", "Step plan", "n"),
    Shortcut("step_plan_back", "Plan", "Step plan back", "b"),
    Shortcut("replan", "Plan", "Replan", "r"),
    Shortcut("zoom_in", "Plan", "Zoom in", "+"),
    Shortcut("zoom_out", "Plan", "Zoom out", "-"),
    Shortcut("zoom_in_frenet", "Plan", "Zoom in Frenet", "Ctrl++"),
    Shortcut("zoom_out_frenet", "Plan", "Zoom out Frenet", "Ctrl+-"),
    Shortcut("step_control", "Control", "Control step", "h"),
    Shortcut("align_control", "Control", "Align control", "i"),
    Shortcut("accelerate", "Control", "Accelerate", "w"),
    Shortcut("steer_left", "Control", "Steer left", "a"),
    Shortcut("steer_right", "Control", "Steer right", "d"),
    Shortcut("decelerate", "Control", "Decelerate", "s"),
    Shortcut("step_exec", "Execute", "Step execution", "c"),
    Shortcut("reset_exec", "Execute", "Reset execution", "t"),
    Shortcut("toggle_exec", "Execute", "Toggle execution", "x"),
    Shortcut("log_up", "Log", "Scroll log up", "k"),
    Shortcut("log_down", "Log", "Scroll log down", "j"),
    Shortcut("log_page_up", "Log", "Log half page up", "Ctrl+U"),
    Shortcut("log_page_down", "Log", "Log half page down", "Ctrl+D"),
    Shortcut("log_top", "Log", "Log top", "g"),
    Shortcut("log_bottom", "Log", "Log bottom", "G"),
    Shortcut("log_up_arrow", "Log", "Scroll log up (arrow)", "Up"),
    Shortcut("log_down_arrow", "Log", "Scroll log down (arrow)", "Down"),
    Shortcut("toggle_log_height", "Log", "Expand or collapse log", "E"),
    Shortcut("clear_log", "Log", "Clear log", "L"),
)

SHORTCUT_BY_ID = {s.id: s for s in SHORTCUTS}
