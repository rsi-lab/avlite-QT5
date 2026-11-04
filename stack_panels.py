"""Perception / Planning / Control / Execution panels for the Qt visualizer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from avlite.c10_perception.c12_perception_strategy import PerceptionPipeline
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningPipeline
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c40_execution.c43_task_strategy import TaskStrategy

from .setting_model import Var

if TYPE_CHECKING:
    from .main_window import VisualizerQtWindow

log = logging.getLogger(__name__)


def pin_readout(label, sample: str) -> None:
    """Lock a numeric label to one monospace width so digit changes cannot reflow."""
    from .qt_binding import import_qt

    QtCore, QtGui, QtWidgets = import_qt()
    font = QtGui.QFont("monospace")
    font.setStyleHint(QtGui.QFont.TypeWriter)
    font.setFixedPitch(True)
    font.setPointSize(10)
    label.setFont(font)
    metrics = QtGui.QFontMetrics(font)
    advance = metrics.horizontalAdvance(sample) if hasattr(metrics, "horizontalAdvance") else metrics.width(sample)
    label.setFixedWidth(int(advance) + 8)
    label.setFixedHeight(metrics.height() + 4)
    label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)


def pin_fluid_readout(label) -> None:
    """Monospace line that fills its cell and cannot change the layout's minimum size."""
    from .qt_binding import import_qt

    QtCore, QtGui, QtWidgets = import_qt()
    font = QtGui.QFont("monospace")
    font.setStyleHint(QtGui.QFont.TypeWriter)
    font.setFixedPitch(True)
    font.setPointSize(10)
    label.setFont(font)
    label.setWordWrap(False)
    metrics = QtGui.QFontMetrics(font)
    label.setFixedHeight(metrics.height() + 4)
    label.setMinimumWidth(0)
    label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
    label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)


def _combo(parent, var: Var, choices: list[str], on_change: Callable[[], None] | None = None) -> QtWidgets.QComboBox:
    box = QtWidgets.QComboBox(parent)
    box.addItems([str(c) for c in choices])
    current = str(var.get() or "")
    idx = box.findText(current)
    box.setCurrentIndex(idx if idx >= 0 else 0)

    def _sync() -> None:
        text = box.currentText()
        if text != str(var.get() or ""):
            var.set(text)
            if on_change:
                on_change()

    box.currentTextChanged.connect(lambda _t: _sync())
    var.bind(lambda: _set_combo(box, var))
    return box


def _set_combo(box: QtWidgets.QComboBox, var: Var) -> None:
    text = str(var.get() or "")
    idx = box.findText(text)
    if idx >= 0 and box.currentIndex() != idx:
        box.blockSignals(True)
        box.setCurrentIndex(idx)
        box.blockSignals(False)


def _check(parent, text: str, var: Var, on_change: Callable[[], None] | None = None) -> QtWidgets.QCheckBox:
    box = QtWidgets.QCheckBox(text, parent)
    box.setChecked(bool(var.get()))

    def _sync(state: int) -> None:
        var.set(bool(state))
        if on_change:
            on_change()

    box.stateChanged.connect(_sync)
    var.bind(lambda: box.setChecked(bool(var.get())) if box.isChecked() != bool(var.get()) else None)
    return box


def _info_button(parent, combo, registry, *, bridge: bool = False):
    """ⓘ button that opens the Tk-style capability contract popup."""
    from .qt_binding import import_qt

    _QtCore, _QtGui, QtWidgets = import_qt()
    btn = QtWidgets.QPushButton("ⓘ")
    btn.setFixedWidth(28)
    btn.setObjectName("contractInfo")
    btn.setToolTip("World requirements, stack requirements and capabilities")

    def _show() -> None:
        from .contracts import show_contract_dialog

        name = combo.currentText()
        cls = (registry or {}).get(name)
        executer = parent.root.exec
        target = cls
        if executer is not None and cls is not None:
            live_attrs = ("world",) if bridge else (
                "perception", "localization", "mapping", "global_planner", "local_planner", "controller",
            )
            for attr in live_attrs:
                live = getattr(executer, attr, None)
                if live is not None and type(live).__name__ == getattr(cls, "__name__", ""):
                    target = live
                    break
        show_contract_dialog(
            parent, title=name or "(none)", target=target, executer=executer, bridge=bridge,
        )

    btn.clicked.connect(_show)
    return btn


def _dbl_spin(parent, var: Var, lo=0.001, hi=10.0, step=0.01) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox(parent)
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(3)
    spin.setValue(float(var.get()))
    spin.valueChanged.connect(lambda v: var.set(float(v)))
    var.bind(lambda: spin.setValue(float(var.get())) if abs(spin.value() - float(var.get())) > 1e-9 else None)
    return spin

_PANELS_READY = False


def _install_panels() -> None:
    """Bind Qt classes on first use so community-plugin import does not init Qt."""
    global _PANELS_READY, QtCore, QtGui, QtWidgets, StackPanels, PerceptionPanel, PlanningPanel, ControlPanel, ExecutionPanel
    if _PANELS_READY:
        return
    from .qt_binding import import_qt

    QtCore, QtGui, QtWidgets = import_qt()

    _PANELS_READY = True

    class StackPanels(QtWidgets.QWidget):
        def __init__(self, root: "VisualizerQtWindow"):
            super().__init__()
            self.root = root
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)

            row = QtWidgets.QHBoxLayout()
            self.perception = PerceptionPanel(root)
            self.planning = PlanningPanel(root)
            self.control = ControlPanel(root)
            row.addWidget(self.perception, 1)
            row.addWidget(self.planning, 1)
            row.addWidget(self.control, 1)
            layout.addLayout(row)

            layout.addWidget(self.perception.state_lbl)

            self.execution = ExecutionPanel(root)
            layout.addWidget(self.execution)

        def refresh(self) -> None:
            self.perception.refresh()
            self.planning.refresh()
            self.control.refresh()
            self.execution.refresh()


    class PerceptionPanel(QtWidgets.QGroupBox):
        def __init__(self, root: "VisualizerQtWindow"):
            super().__init__("Perception")
            self.root = root
            grid = QtWidgets.QGridLayout(self)
            choices = root.setting.registry_choices()
            grid.addWidget(QtWidgets.QLabel("Strategy"), 0, 0)
            self.perc = _combo(
                self, root.setting.perception_type, choices["perception"],
                on_change=lambda: root.reload_stack(reload_code=False),
            )
            grid.addWidget(self.perc, 0, 1)
            from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy

            grid.addWidget(_info_button(self, self.perc, PerceptionStrategy.registry), 0, 2)
            grid.addWidget(QtWidgets.QLabel("Detect"), 1, 0)
            self.det = _combo(self, root.setting.detection_strategy_type, choices["detection"], root.refresh_pipeline)
            grid.addWidget(self.det, 1, 1)
            grid.addWidget(QtWidgets.QLabel("Track"), 2, 0)
            self.trk = _combo(self, root.setting.tracking_strategy_type, choices["tracking"], root.refresh_pipeline)
            grid.addWidget(self.trk, 2, 1)
            grid.addWidget(QtWidgets.QLabel("Predict"), 3, 0)
            self.prd = _combo(self, root.setting.prediction_strategy_type, choices["prediction"], root.refresh_pipeline)
            grid.addWidget(self.prd, 3, 1)
            grid.addWidget(QtWidgets.QLabel("Map"), 4, 0)
            self.mapping = _combo(
                self, root.setting.mapping_type, choices["mapping"],
                on_change=lambda: root.reload_stack(reload_code=False),
            )
            grid.addWidget(self.mapping, 4, 1)
            grid.addWidget(QtWidgets.QLabel("Loc"), 5, 0)
            self.loc = _combo(
                self, root.setting.localization_type, choices["localization"],
                on_change=lambda: root.reload_stack(reload_code=False),
            )
            grid.addWidget(self.loc, 5, 1)
            # Full-width placement happens in StackPanels. Keeping it in this
            # column made the monospace line clip, and letting it grow moved the panel.
            self.state_lbl = QtWidgets.QLabel()
            self.state_lbl.setObjectName("egoState")
            self.state_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            pin_fluid_readout(self.state_lbl)
            root.setting.vehicle_state.bind(lambda: self.state_lbl.setText(str(root.setting.vehicle_state.get())))
            self._update_pipeline_enabled()

        def _update_pipeline_enabled(self) -> None:
            pipe = self.root.setting.perception_type.get() == PerceptionPipeline.__name__
            for w in (self.det, self.trk, self.prd):
                w.setEnabled(pipe)

        def refresh(self) -> None:
            choices = self.root.setting.registry_choices()
            for box, key in (
                (self.perc, "perception"), (self.det, "detection"), (self.trk, "tracking"),
                (self.prd, "prediction"), (self.mapping, "mapping"), (self.loc, "localization"),
            ):
                cur = box.currentText()
                box.blockSignals(True)
                box.clear()
                box.addItems(choices[key])
                idx = box.findText(cur)
                box.setCurrentIndex(idx if idx >= 0 else 0)
                box.blockSignals(False)
            self._update_pipeline_enabled()


    class PlanningPanel(QtWidgets.QGroupBox):
        def __init__(self, root: "VisualizerQtWindow"):
            super().__init__("Planning")
            self.root = root
            grid = QtWidgets.QGridLayout(self)
            choices = root.setting.registry_choices()
            grid.addWidget(QtWidgets.QLabel("Global"), 0, 0)
            self.gplan = _combo(
                self, root.setting.global_planner_type, choices["global_planner"],
                on_change=lambda: root.reload_stack(reload_code=False),
            )
            grid.addWidget(self.gplan, 0, 1)
            from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy

            grid.addWidget(_info_button(self, self.gplan, GlobalPlannerStrategy.registry), 0, 3)
            btn_greplan = QtWidgets.QPushButton("Global Replan")
            btn_greplan.clicked.connect(root.replan_global)
            grid.addWidget(btn_greplan, 0, 2)
            grid.addWidget(QtWidgets.QLabel("Local"), 1, 0)
            self.lplan = _combo(
                self, root.setting.local_planner_type, choices["local_planner"],
                on_change=lambda: root.reload_stack(reload_code=False),
            )
            grid.addWidget(self.lplan, 1, 1)
            btn_lreplan = QtWidgets.QPushButton("Local Replan")
            btn_lreplan.clicked.connect(self._local_replan)
            grid.addWidget(btn_lreplan, 1, 2)
            wp_row = QtWidgets.QHBoxLayout()
            for label, slot in (
                ("WP+", lambda: root.step_plan()),
                ("WP-", lambda: root.step_plan_back()),
                ("Align", lambda: root.align_plan()),
                ("Save plan", lambda: root.save_global_plan()),
            ):
                b = QtWidgets.QPushButton(label)
                b.clicked.connect(slot)
                wp_row.addWidget(b)
            grid.addLayout(wp_row, 7, 0, 1, 3)
            grid.addWidget(QtWidgets.QLabel("Behavior"), 2, 0)
            self.beh = _combo(self, root.setting.behavioral_strategy_type, [""] + [], root.refresh_pipeline)
            grid.addWidget(self.beh, 2, 1)
            grid.addWidget(QtWidgets.QLabel("Path"), 3, 0)
            self.path = _combo(self, root.setting.path_strategy_type, [""], root.refresh_pipeline)
            grid.addWidget(self.path, 3, 1)
            grid.addWidget(QtWidgets.QLabel("Speed"), 4, 0)
            self.vel = _combo(self, root.setting.velocity_strategy_type, [""], root.refresh_pipeline)
            grid.addWidget(self.vel, 4, 1)
            self.prob = _check(self, "Probabilistic collision", root.setting.probabilistic_collision_checking)
            grid.addWidget(self.prob, 5, 0, 1, 2)
            save_occ = QtWidgets.QPushButton("Save occupancy")
            save_occ.clicked.connect(lambda: root.save_occupancy_map())
            grid.addWidget(save_occ, 5, 2)
            self.wp_lbl = QtWidgets.QLabel("WP:      0   Lap:      0")
            self.wp_lbl.setObjectName("wpLap")
            pin_readout(self.wp_lbl, "WP: 999999   Lap: 999999")
            grid.addWidget(self.wp_lbl, 6, 0, 1, 3)
            root.setting.current_wp.bind(self._update_wp)
            root.setting.lap.bind(self._update_wp)
            self.refresh()

        def _update_wp(self) -> None:
            wp = str(self.root.setting.current_wp.get()).strip()
            lap = str(self.root.setting.lap.get()).strip()
            self.wp_lbl.setText(f"WP: {wp:>6}   Lap: {lap:>6}")

        def _local_replan(self) -> None:
            if self.root.exec is None or self.root.exec.local_planner is None:
                return
            self.root.exec.local_planner.replan(
                perception_model=self.root.exec.pm,
                sensors=self.root.exec.world.get_sensor_frame(),
            )
            self.root.update_ui()

        def refresh(self) -> None:
            from avlite.c20_planning.c23_local_planning_strategy import (
                LocalBehavioralPlanningStrategy,
                LocalPathPlanningStrategy,
                LocalVelocityPlanningStrategy,
            )

            choices = self.root.setting.registry_choices()
            for box, items in (
                (self.gplan, choices["global_planner"]),
                (self.lplan, choices["local_planner"]),
                (self.beh, [""] + list(LocalBehavioralPlanningStrategy.registry.keys())),
                (self.path, [""] + list(LocalPathPlanningStrategy.registry.keys())),
                (self.vel, [""] + list(LocalVelocityPlanningStrategy.registry.keys())),
            ):
                cur = box.currentText()
                box.blockSignals(True)
                box.clear()
                box.addItems(items)
                idx = box.findText(cur)
                box.setCurrentIndex(idx if idx >= 0 else 0)
                box.blockSignals(False)
            pipe = self.root.setting.local_planner_type.get() == LocalPlanningPipeline.__name__
            for w in (self.beh, self.path, self.vel):
                w.setEnabled(pipe)


    class ValueGauge(QtWidgets.QWidget):
        """Horizontal meter matching the Tk ValueGauge (value inside min/max)."""

        def __init__(self, lo: float, hi: float, parent=None):
            super().__init__(parent)
            self._lo = float(lo)
            self._hi = float(hi)
            self._value = 0.0
            self.setFixedHeight(10)
            self.setMinimumWidth(48)
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        def setValue(self, value: float) -> None:
            self._value = float(value)
            self.update()

        def paintEvent(self, _event) -> None:  # noqa: N802
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtGui.QColor("#2b2b2b"))
            span = self._hi - self._lo
            if span == 0:
                return
            frac = max(0.0, min(1.0, (self._value - self._lo) / span))
            painter.fillRect(0, 0, int(self.width() * frac), self.height(), QtGui.QColor("#6abf69"))

    class ControlPanel(QtWidgets.QGroupBox):
        def __init__(self, root: "VisualizerQtWindow"):
            super().__init__("Control")
            self.root = root
            grid = QtWidgets.QGridLayout(self)
            choices = root.setting.registry_choices()
            grid.addWidget(QtWidgets.QLabel("Controller"), 0, 0)
            self.ctrl = _combo(
                self, root.setting.controller_type, choices["controller"],
                on_change=lambda: root.reload_stack(reload_code=False),
            )
            grid.addWidget(self.ctrl, 0, 1, 1, 2)
            from avlite.c30_control.c32_control_strategy import ControlStrategy

            grid.addWidget(_info_button(self, self.ctrl, ControlStrategy.registry), 0, 3)
            self.gauges = {
                "vel_cte": QtWidgets.QLabel("Vel CTE:    +0.00"),
                "pos_cte": QtWidgets.QLabel("Pos CTE:    +0.00"),
                "accel": QtWidgets.QLabel("Accel:      +0.00"),
                "steer": QtWidgets.QLabel("Steer:      +0.00"),
            }
            pin_readout(self.gauges["vel_cte"], "Vel CTE: -9999.99")
            pin_readout(self.gauges["pos_cte"], "Pos CTE: -9999.99")
            pin_readout(self.gauges["accel"], "Accel:   -9999.99")
            pin_readout(self.gauges["steer"], "Steer:   -9999.99")
            for name, gauge in self.gauges.items():
                gauge.setObjectName(f"gauge_{name}")
            grid.addWidget(self.gauges["vel_cte"], 1, 0)
            grid.addWidget(self.gauges["pos_cte"], 1, 1)
            grid.addWidget(self.gauges["accel"], 2, 0)
            grid.addWidget(self.gauges["steer"], 2, 1)
            btn_row = QtWidgets.QHBoxLayout()
            for label, slot in (
                ("Step", self._step),
                ("Align", self._align),
                ("W", lambda: self._drive(3, 0)),
                ("A", lambda: self._drive(0, 0.7)),
                ("S", lambda: self._drive(-3, 0)),
                ("D", lambda: self._drive(0, -0.7)),
            ):
                b = QtWidgets.QPushButton(label)
                b.clicked.connect(slot)
                btn_row.addWidget(b)
            grid.addLayout(btn_row, 3, 0, 1, 3)
            self.value_gauges = {}
            from avlite.c30_control.c39_settings import ControlSettings

            amin = float(getattr(ControlSettings, "c32_ego_min_acceleration", -20.0))
            smax = float(getattr(ControlSettings, "c32_ego_max_steering", 0.7))
            ranges = {
                "vel_cte": (-10.0, 10.0),
                "pos_cte": (-5.0, 5.0),
                "accel": (amin, abs(amin)),
                "steer": (-smax, smax),
            }
            for key, (row, col) in (
                ("vel_cte", (4, 0)), ("pos_cte", (4, 1)), ("accel", (5, 0)), ("steer", (5, 1)),
            ):
                bar = ValueGauge(*ranges[key])
                bar.setObjectName(f"valueGauge_{key}")
                self.value_gauges[key] = bar
                grid.addWidget(bar, row, col)
            root.setting.cte_vel.bind(self._update_gauges)
            root.setting.cte_steer.bind(self._update_gauges)
            root.setting.cmd_accel.bind(self._update_gauges)
            root.setting.cmd_steer.bind(self._update_gauges)

        def _update_gauges(self) -> None:
            s = self.root.setting
            self.gauges["vel_cte"].setText(f"Vel CTE: {float(s.cte_vel.get()):+9.2f}")
            self.gauges["pos_cte"].setText(f"Pos CTE: {float(s.cte_steer.get()):+9.2f}")
            self.gauges["accel"].setText(f"Accel:   {float(s.cmd_accel.get()):+9.2f}")
            self.gauges["steer"].setText(f"Steer:   {float(s.cmd_steer.get()):+9.2f}")
            if hasattr(self, "value_gauges"):
                self.value_gauges["vel_cte"].setValue(float(s.cte_vel.get()))
                self.value_gauges["pos_cte"].setValue(float(s.cte_steer.get()))
                self.value_gauges["accel"].setValue(float(s.cmd_accel.get()))
                self.value_gauges["steer"].setValue(float(s.cmd_steer.get()))

        def _drive(self, accel: float, steer: float) -> None:
            if self.root.exec is None:
                return
            self.root.apply_world_control(
                ControlCommand(acceleration=accel, steer=steer),
                dt=float(self.root.setting.sim_dt.get()),
            )
            self.root.update_ui()

        def _step(self) -> None:
            self.root.step_control()

        def _align(self) -> None:
            self.root.align_control()

        def refresh(self) -> None:
            choices = self.root.setting.registry_choices()
            cur = self.ctrl.currentText()
            self.ctrl.blockSignals(True)
            self.ctrl.clear()
            self.ctrl.addItems(choices["controller"])
            idx = self.ctrl.findText(cur)
            self.ctrl.setCurrentIndex(idx if idx >= 0 else 0)
            self.ctrl.blockSignals(False)


    class ExecutionPanel(QtWidgets.QGroupBox):
        def __init__(self, root: "VisualizerQtWindow"):
            super().__init__("Execution")
            self.root = root
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._exec_tick)
            v = QtWidgets.QVBoxLayout(self)

            top = QtWidgets.QHBoxLayout()
            choices = root.setting.registry_choices()
            top.addWidget(QtWidgets.QLabel("Executer"))
            self.executer = _combo(
                self, root.setting.executer_type, choices["executer"],
                on_change=lambda: root.reload_stack(reload_code=False),
            )
            top.addWidget(self.executer)
            top.addWidget(QtWidgets.QLabel("Bridge"))
            self.bridge = _combo(
                self, root.setting.execution_bridge, choices["bridge"],
                on_change=lambda: root.reload_stack(reload_code=False),
            )
            top.addWidget(self.bridge)
            from avlite.c40_execution.c41_world_bridge import WorldBridge

            top.addWidget(_info_button(self, self.bridge, WorldBridge.registry, bridge=True))
            top.addWidget(_check(self, "Perception", root.setting.exec_perceive))
            top.addWidget(_check(self, "Planning", root.setting.exec_plan))
            top.addWidget(_check(self, "Control", root.setting.exec_control))
            top.addWidget(_check(self, "Tasks", root.setting.exec_tasks))
            v.addLayout(top)

            dts = QtWidgets.QHBoxLayout()
            for label, pace_var, dt_var in (
                ("Perception Δt", root.setting.pace_perception, root.setting.perception_dt),
                ("Replan Δt", root.setting.pace_replan, root.setting.replan_dt),
                ("Control Δt", root.setting.pace_control, root.setting.control_dt),
                ("Sim Δt", root.setting.pace_sim, root.setting.sim_dt),
            ):
                dts.addWidget(_check(self, "", pace_var))
                dts.addWidget(QtWidgets.QLabel(label))
                dts.addWidget(_dbl_spin(self, dt_var))
            v.addLayout(dts)

            tasks = QtWidgets.QHBoxLayout()
            tasks.addWidget(QtWidgets.QLabel("Tasks"))
            self.tasks_edit = QtWidgets.QLineEdit(str(root.setting.execution_tasks.get()))
            self.tasks_edit.editingFinished.connect(
                lambda: root.setting.execution_tasks.set(self.tasks_edit.text())
            )
            root.setting.execution_tasks.bind(
                lambda: self.tasks_edit.setText(str(root.setting.execution_tasks.get()))
            )
            tasks.addWidget(self.tasks_edit, 1)
            add_btn = QtWidgets.QPushButton("+")
            add_btn.clicked.connect(self._add_task)
            tasks.addWidget(add_btn)
            v.addLayout(tasks)

            btns = QtWidgets.QHBoxLayout()
            self.start_btn = QtWidgets.QPushButton("Start")
            self.start_btn.clicked.connect(self.toggle_exec)
            btns.addWidget(self.start_btn)
            for label, slot in (
                ("Stop", self.stop_exec),
                ("Step", self.step_exec),
                ("Reset", self.reset_exec),
            ):
                b = QtWidgets.QPushButton(label)
                b.clicked.connect(slot)
                btns.addWidget(b)
            save_start = QtWidgets.QPushButton("Save Start")
            save_start.clicked.connect(root.set_start_pose)
            btns.addWidget(save_start)
            v.addLayout(btns)
            self.cap_label = QtWidgets.QLabel("caps: —")
            self.cap_label.setObjectName("capLabel")
            pin_fluid_readout(self.cap_label)
            v.addWidget(self.cap_label)
            self.cap_box = QtWidgets.QWidget()
            self.cap_box.setObjectName("capabilityPanel")
            self.cap_box.setMaximumHeight(36)
            cap_row = QtWidgets.QHBoxLayout(self.cap_box)
            cap_row.setContentsMargins(0, 0, 0, 0)
            self.world_cap_host = QtWidgets.QWidget()
            self.world_cap_host.setObjectName("worldCapabilities")
            self.world_cap_layout = QtWidgets.QHBoxLayout(self.world_cap_host)
            self.world_cap_layout.setContentsMargins(0, 0, 0, 0)
            self.stack_cap_host = QtWidgets.QWidget()
            self.stack_cap_host.setObjectName("stackCapabilities")
            self.stack_cap_layout = QtWidgets.QHBoxLayout(self.stack_cap_host)
            self.stack_cap_layout.setContentsMargins(0, 0, 0, 0)
            cap_row.addWidget(self.world_cap_host, 1)
            cap_row.addWidget(self.stack_cap_host, 1)
            v.addWidget(self.cap_box)
            self._cap_key = None
            self._world_checks = []
            self._stack_checks = []

            files = QtWidgets.QHBoxLayout()
            files.addWidget(QtWidgets.QLabel("Map"))
            self.map_edit = QtWidgets.QLineEdit(str(root.setting.default_map_file.get() or ""))
            self.map_edit.setObjectName("mapFile")
            self.map_edit.editingFinished.connect(
                lambda: root.setting.default_map_file.set(self.map_edit.text())
            )
            files.addWidget(self.map_edit, 1)
            map_btn = QtWidgets.QPushButton("…")
            map_btn.setFixedWidth(28)
            map_btn.clicked.connect(lambda: self._pick_file(self.map_edit, root.setting.default_map_file))
            files.addWidget(map_btn)
            files.addWidget(QtWidgets.QLabel("Plan"))
            self.plan_edit = QtWidgets.QLineEdit(str(root.setting.default_global_plan_file.get() or ""))
            self.plan_edit.setObjectName("planFile")
            self.plan_edit.editingFinished.connect(
                lambda: root.setting.default_global_plan_file.set(self.plan_edit.text())
            )
            files.addWidget(self.plan_edit, 1)
            plan_btn = QtWidgets.QPushButton("…")
            plan_btn.setFixedWidth(28)
            plan_btn.clicked.connect(
                lambda: self._pick_file(self.plan_edit, root.setting.default_global_plan_file)
            )
            files.addWidget(plan_btn)
            v.addLayout(files)

            self.task_host = QtWidgets.QWidget()
            self.task_host.setObjectName("taskChips")
            self.task_host.setMaximumHeight(32)
            self.task_row = QtWidgets.QHBoxLayout(self.task_host)
            self.task_row.setContentsMargins(0, 0, 0, 0)
            v.addWidget(self.task_host)

            stats = QtWidgets.QHBoxLayout()
            self.stats_lbl = QtWidgets.QLabel("real=    0.00  sim=    0.00  perc=    0.0  plan=    0.0  ctrl=    0.0  ui=    0.0 Hz  plot=    0.0 ms")
            self.stats_lbl.setObjectName("execStats")
            pin_fluid_readout(self.stats_lbl)
            stats.addWidget(self.stats_lbl, 1)
            v.addLayout(stats)
            for var in (
                root.setting.elapsed_real_time, root.setting.elapsed_sim_time,
                root.setting.perception_fps, root.setting.replan_fps, root.setting.control_fps,
                root.setting.ui_fps, root.setting.plot_latency_ms,
            ):
                var.bind(self._update_stats)

        def _update_stats(self) -> None:
            s = self.root.setting
            self.stats_lbl.setText(
                f"real={s.elapsed_real_time.get()}  sim={s.elapsed_sim_time.get()}  "
                f"perc={s.perception_fps.get()}  plan={s.replan_fps.get()}  ctrl={s.control_fps.get()}  "
                f"ui={s.ui_fps.get()} Hz  plot={s.plot_latency_ms.get()} ms"
            )

        def _pick_file(self, edit, var) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", edit.text())
            if not path:
                return
            edit.setText(path)
            var.set(path)

        def _rebuild_task_chips(self) -> None:
            while self.task_row.count():
                item = self.task_row.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            for name in self.root.setting.execution_task_names():
                chip = QtWidgets.QPushButton(name)
                chip.setObjectName(f"taskChip_{name}")
                chip.clicked.connect(lambda _checked=False, n=name: self._remove_task(n))
                self.task_row.addWidget(chip)
            self.task_row.addStretch(1)

        def _remove_task(self, name: str) -> None:
            names = [n for n in self.root.setting.execution_task_names() if n != name]
            self.root.setting.execution_tasks.set(",".join(names))
            self.root.reload_stack(reload_code=False)

        def _add_task(self) -> None:
            selected = set(self.root.setting.execution_task_names())
            available = sorted(n for n in TaskStrategy.registry if n not in selected)
            if not available:
                QtWidgets.QMessageBox.information(self, "Tasks", "All registered tasks are already selected.")
                return
            name, ok = QtWidgets.QInputDialog.getItem(self, "Add Task", "TaskStrategy:", available, 0, False)
            if ok and name:
                names = self.root.setting.execution_task_names() + [name]
                self.root.setting.execution_tasks.set(",".join(names))
                self.root.reload_stack(reload_code=False)

        def toggle_exec(self) -> None:
            if self.root.setting.exec_running:
                self.stop_exec()
                return
            self.root.setting.exec_running = True
            self.start_btn.setEnabled(False)
            # Prefer executer-reported poll delay; fall back to ~1ms busy poll.
            delay_s = getattr(self.root.exec, "ui_poll_delay", None) if self.root.exec else None
            interval_ms = int((delay_s if delay_s is not None else 0.001) * 1000)
            self._timer.start(max(1, interval_ms))
            self.root.update_ui()

        def _exec_tick(self) -> None:
            if not self.root.setting.exec_running or self.root.exec is None:
                return
            s = self.root.setting
            self.root.exec.step(
                control_dt=float(s.control_dt.get()),
                replan_dt=float(s.replan_dt.get()),
                perception_dt=float(s.perception_dt.get()),
                sim_dt=float(s.sim_dt.get()),
                call_replan=bool(s.exec_plan.get()),
                call_control=bool(s.exec_control.get()),
                call_perceive=bool(s.exec_perceive.get()),
                call_tasks=bool(s.exec_tasks.get()),
                pace_perception=bool(s.pace_perception.get()),
                pace_replan=bool(s.pace_replan.get()),
                pace_control=bool(s.pace_control.get()),
                pace_sim=bool(s.pace_sim.get()),
            )
            # UI redraw is owned by the window's refresh timer (decoupled from sim).
            delay_s = getattr(self.root.exec, "ui_poll_delay", None)
            if delay_s is not None:
                self._timer.setInterval(max(1, int(delay_s * 1000)))

        def stop_exec(self) -> None:
            self.root.setting.exec_running = False
            self._timer.stop()
            if self.root.exec is not None:
                self.root.exec.stop()
            self.start_btn.setEnabled(True)
            self.root.update_ui()

        def step_exec(self) -> None:
            if self.root.exec is None:
                return
            s = self.root.setting
            self.root.exec.step(
                control_dt=float(s.control_dt.get()),
                replan_dt=float(s.replan_dt.get()),
                perception_dt=float(s.perception_dt.get()),
                sim_dt=float(s.sim_dt.get()),
                call_replan=bool(s.exec_plan.get()),
                call_control=bool(s.exec_control.get()),
                call_perceive=bool(s.exec_perceive.get()),
                call_tasks=bool(s.exec_tasks.get()),
                pace_perception=bool(s.pace_perception.get()),
                pace_replan=bool(s.pace_replan.get()),
                pace_control=bool(s.pace_control.get()),
                pace_sim=bool(s.pace_sim.get()),
            )
            self.root.update_ui()

        def reset_exec(self) -> None:
            if self.root.exec is not None:
                self.root.exec.reset()
            self.root.update_ui()

        def refresh(self) -> None:
            choices = self.root.setting.registry_choices()
            for box, key in ((self.executer, "executer"), (self.bridge, "bridge")):
                cur = box.currentText()
                box.blockSignals(True)
                box.clear()
                box.addItems(choices[key])
                idx = box.findText(cur)
                box.setCurrentIndex(idx if idx >= 0 else 0)
                box.blockSignals(False)
            self._refresh_caps()
            self._rebuild_task_chips()

        def _clear_layout(self, layout) -> None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        def _refresh_caps(self) -> None:
            from avlite.c40_execution.c41_world_bridge import (
                is_world_capability_enabled,
                is_world_stack_capability_enabled,
            )
            from avlite.c40_execution.c49_settings import ExecutionSettings

            world = getattr(self.root.exec, "world", None) if self.root.exec else None
            if world is None:
                self.cap_label.setText("caps: —")
                return
            skip = {"AGENT_SPAWN", "AGENT_CONTROL", "AGENT_SENSING"}
            sensors = [c for c in (getattr(world, "world_capabilities", ()) or ()) if c.name not in skip]
            gt = list(getattr(world, "stack_capabilities", ()) or ())
            self.cap_label.setText(
                "world: "
                + (", ".join(c.name for c in sensors) or "—")
                + " | stack: "
                + (", ".join(c.name for c in gt) or "—")
            )
            key = (tuple(c.name for c in sensors), tuple(c.name for c in gt))
            if key == self._cap_key:
                for box in self._world_checks:
                    box.blockSignals(True)
                    box.setChecked(is_world_capability_enabled(box.property("cap_obj")))
                    box.blockSignals(False)
                for box in self._stack_checks:
                    box.blockSignals(True)
                    box.setChecked(is_world_stack_capability_enabled(box.property("cap_obj")))
                    box.blockSignals(False)
                return
            self._cap_key = key
            self._clear_layout(self.world_cap_layout)
            self._clear_layout(self.stack_cap_layout)
            self._world_checks = []
            self._stack_checks = []
            self.world_cap_layout.addWidget(QtWidgets.QLabel("World"))
            for cap in sensors:
                box = QtWidgets.QCheckBox(cap.name)
                box.setObjectName(f"worldCap_{cap.name}")
                box.setProperty("cap_obj", cap)
                box.setChecked(is_world_capability_enabled(cap))
                box.stateChanged.connect(lambda _state: self._write_caps())
                self.world_cap_layout.addWidget(box)
                self._world_checks.append(box)
            self.stack_cap_layout.addWidget(QtWidgets.QLabel("GT"))
            for cap in gt:
                box = QtWidgets.QCheckBox(cap.name)
                box.setObjectName(f"stackCap_{cap.name}")
                box.setProperty("cap_obj", cap)
                box.setChecked(is_world_stack_capability_enabled(cap))
                box.stateChanged.connect(lambda _state: self._write_caps())
                self.stack_cap_layout.addWidget(box)
                self._stack_checks.append(box)

        def _write_caps(self) -> None:
            from avlite.c40_execution.c49_settings import ExecutionSettings

            ExecutionSettings.c41_world_capabilities = [
                box.property("cap_obj").name for box in self._world_checks if box.isChecked()
            ]
            ExecutionSettings.c41_world_stack_capabilities = [
                box.property("cap_obj").name for box in self._stack_checks if box.isChecked()
            ]
            self.root.update_ui()


def __getattr__(name: str):
    if name in {'StackPanels', 'PerceptionPanel', 'PlanningPanel', 'ControlPanel', 'ExecutionPanel'}:
        _install_panels()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
