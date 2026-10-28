"""The 19 Dynamixel XL330 servos, simulated by BAM (Rhoban's better-actuator-models) as identified on the bench."""
from __future__ import annotations

import copy
from collections import deque

import numpy as np

from .scene import JOINTS


class Servos:
    """The XL330 servos as identified on the bench (BAM model m6): position control by the firmware (P gain kp_fw),
    PWM voltage, DC motor with back-EMF, current limit, load-dependent friction; a supply voltage that drops under
    load; a command latency; torque on / off per servo."""

    def __init__(self, model, data, kp_fw=400.0, vin=7.5, vin_drop_resistance=0.0366, vin_min=6.0, max_current=1.75,
                 latency_steps=5):
        from bam.model import load_model, _resolve_json_path
        from bam.mujoco import MujocoController
        bam_model = load_model(_resolve_json_path(None, "xl330", "m6"))
        bam_model.actuator.kp = kp_fw
        bam_model.actuator.vin = vin
        bam_model.actuator.max_current = max_current          # the firmware's current limit
        # supply + wires resistance (ohm): the voltage drops by R * sum(|torque|) / kt
        self.ctrl = MujocoController(bam_model, JOINTS, model, data, vin_drop_resistance=vin_drop_resistance,
                                     vin_min=vin_min)
        self.latency = deque(maxlen=max(1, latency_steps + 1))
        self.kt = bam_model.kt.value
        self.data = data
        self.torque = np.ones(len(JOINTS), bool)              # torque enabled, per servo (order JOINTS)

    @property
    def kp(self) -> float:
        """The firmware P gain of every servo (the robot writes it to all of them)."""
        return float(self.ctrl.model.actuator.kp)

    @kp.setter
    def kp(self, value: float):
        self.ctrl.model.actuator.kp = float(value)

    def reset(self, qpos):
        """Start from rest at ``qpos`` (the full MuJoCo qpos), with an empty latency queue, every torque on."""
        self.ctrl.reset()
        self.torque[:] = True
        self.ctrl.q_target = np.asarray(qpos)[self.ctrl.qpos_indexes].copy()     # hold where the joints are
        self.latency.clear()

    def set_target(self, target):
        """Send joint targets (rad); the firmware gets the one sent ``latency_steps`` physics steps ago."""
        self.latency.append(np.asarray(target, float).copy())
        self.ctrl.q_target = self.latency[0].copy()

    def update(self):
        """Set the motor torques for the next physics step (none for a servo whose torque is off: it goes limp, only
        its friction left)."""
        self.ctrl.update()
        if not self.torque.all():
            self.data.ctrl[: len(JOINTS)][~self.torque] = 0.0

    def currents(self) -> np.ndarray:
        """Motor currents (A), what the real servos report."""
        return np.abs(self.data.ctrl[: len(JOINTS)]) / self.kt

    def get_state(self, model) -> dict:
        """The servos' memory: the commands on their way, the current target, BAM's internal state (its control law,
        the last control, the PWM duty cycle that drops the supply), the gain, the torque, and the joint friction
        BAM writes into ``model`` at every step (it depends on the load)."""
        c, act = self.ctrl, self.ctrl.model.actuator
        return {"latency": [t.copy() for t in self.latency], "q_target": c.q_target.copy(), "last_ts": c.last_ts,
                "torque": self.torque.copy(), "actuator_state": copy.deepcopy(c.actuator_state),
                "control": copy.deepcopy(c.control), "duty_cycle": copy.deepcopy(getattr(act, "duty_cycle", None)),
                "kp": act.kp, "frictionloss": model.dof_frictionloss.copy(), "damping": model.dof_damping.copy()}

    def set_state(self, model, state: dict):
        c = self.ctrl
        self.latency.clear()
        self.latency.extend(t.copy() for t in state["latency"])
        c.q_target, c.last_ts = state["q_target"].copy(), state["last_ts"]
        self.torque[:] = state["torque"]
        c.actuator_state, c.control = copy.deepcopy(state["actuator_state"]), copy.deepcopy(state["control"])
        if state["duty_cycle"] is not None:
            c.model.actuator.duty_cycle = copy.deepcopy(state["duty_cycle"])
        c.model.actuator.kp = state["kp"]
        model.dof_frictionloss[:] = state["frictionloss"]
        model.dof_damping[:] = state["damping"]
