# SPDX-License-Identifier: GPL-3.0-or-later
"""Run a move, the same file as on the real robot (a file of microban's ``src/moves/``, branch dev_v2).

A move is a class (no base class; its name ends with Move by convention) with ``step(obs) -> targets``, called every
20 ms with ``sim.observation()``. ``targets``: the 19 joint targets (rad, order ``JOINTS``), or ``{joint: angle}`` for
some joints (the others keep their last target); ``None``: the move is over. Optionally:

* ``start(obs)``: called when it starts; ``stopping(obs)``: called every 20 ms once it is over or stopped, returns
  targets until it returns None (a smooth stop);
* ``kp``: the servos' gain while it runs (default: the simulator's, ``kp_fw``; the robot's program: 400);
* ``options = {name: value}`` (int / float / bool), read as ``self.options[name]``, edited live in the robot's
  dashboard; ``option_docs``; ``sliders = {name: (min, max, start)}``, read as ``self.slider[name]``; ``keys`` and
  ``on_key(key, down)``, ``self.pressed`` (the keys held); ``self.velocity`` (the sticks / arrows: vx, vy, vtheta);
* ``description``, ``variants`` ("classic", "climber"), ``start_pose`` ("current" or "neutral": on the robot only).
* ``self.robot``: the servos one by one, for the moves that need more (:class:`SimRobot`, as on the robot):
  ``set_torque(on, joints)``, ``set_kp(kp)``.

In the simulator nobody edits options and sliders, no key is pressed and the velocity is zero: :func:`run_move`
gives them their defaults.
"""
import importlib.util
import inspect
import re
import sys
import traceback
from pathlib import Path

import numpy as np

from .scene import JOINTS

STOPPING_MAX_S = 10.0             # a stopping() that never returns None is cut after this


def _give_defaults(move) -> None:
    """What the robot's program sets on a move at its launch: its options (kept from one run to the next, {} when it
    has none), its sliders back to their start values, no key held, no velocity asked."""
    if "options" not in vars(move):
        move.options = dict(getattr(move, "options", None) or {})
    move.slider = {n: float(v[2]) for n, v in (getattr(move, "sliders", None) or {}).items()}
    move.pressed = set()
    move.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
    move.robot = SimRobot(move.robot.sim if isinstance(getattr(move, "robot", None), SimRobot) else None)


class SimRobot:
    """``self.robot`` of a move in the simulator, as on the robot (microban's moves/move.py Robot): ``set_kp(kp)`` (every
    joint at once: the simulated servos share one gain), ``set_torque(on, joints=None)`` (a joint goes limp at once;
    back on after this step's targets, held where it is). ``state`` is None (the robot's raw measurements); ``sim`` is
    the simulator."""

    def __init__(self, sim) -> None:
        self.sim, self.state = sim, None
        self.to_enable: set[int] = set()

    @staticmethod
    def _index(joints) -> list[int]:
        names = list(JOINTS) if joints is None else [joints] if isinstance(joints, str) else list(joints)
        unknown = set(names) - set(JOINTS)
        if unknown:
            raise ValueError(f"unknown joints {sorted(unknown)} (JOINTS)")
        return [JOINTS.index(n) for n in names]

    def set_kp(self, kp, joints=None) -> None:
        if joints is not None and sorted(self._index(joints)) != list(range(len(JOINTS))):
            raise NotImplementedError("the simulated servos share one gain: set_kp(kp) for every joint (per joint on "
                                      "the robot only)")
        self.sim.servos.kp = float(kp)

    def set_torque(self, on: bool, joints=None) -> None:
        idx = self._index(joints)
        if on:
            self.to_enable |= {i for i in idx if not self.sim.servos.torque[i]}
        else:
            self.sim.servos.torque[idx] = False
            self.to_enable -= set(idx)


def _step(sim, robot: SimRobot, last: np.ndarray, obs: dict) -> dict:
    """One step with these targets; a joint turned back on this step is held where it is, then powered."""
    if robot.to_enable:
        idx = sorted(robot.to_enable)
        last[idx] = obs["q"][idx]
    obs = sim.step(last)
    if robot.to_enable:
        sim.servos.torque[sorted(robot.to_enable)] = True
        robot.to_enable.clear()
    return obs


def _apply(targets, last: np.ndarray) -> np.ndarray:
    """The move's targets over the last ones: 19 angles, or a dict of some joints."""
    if isinstance(targets, dict):
        out = last.copy()
        for name, angle in targets.items():
            out[JOINTS.index(name)] = angle
        return out
    targets = np.asarray(targets, float)
    if targets.shape != (len(JOINTS),):
        raise ValueError(f"step must return {len(JOINTS)} targets (order JOINTS) or a dict, got shape {targets.shape}")
    return targets


def run_move(move, sim, seconds: float, on_step=None) -> dict:
    """Run ``move`` on ``sim`` (a :class:`microban_sim.MicrobanSim`, or the hackathon's ClimbingSim) for at most
    ``seconds`` of simulated time, from where the robot is (as on the robot when the torque goes on there), then its
    ``stopping`` if it has one. ``on_step(sim, obs)``, if given, is called after every step (to watch, log or stop:
    return True to stop). Returns the last observation; the servos' gain is the simulator's again after."""
    _give_defaults(move)
    move.robot.sim = sim
    kp = sim.servos.kp
    if getattr(move, "kp", None) is not None:
        sim.servos.kp = float(move.kp)
    sim.set_torque(True)
    obs = sim.observation()
    last = obs["q"].copy()
    if callable(getattr(move, "start", None)):
        move.start(obs)
    try:
        for _ in range(int(round(seconds / sim.control_dt))):
            targets = move.step(obs)
            if targets is None:
                break
            last = _apply(targets, last)
            obs = _step(sim, move.robot, last, obs)
            if on_step is not None and on_step(sim, obs):
                break
        stopping = getattr(move, "stopping", None)
        for _ in range(int(round(STOPPING_MAX_S / sim.control_dt)) if callable(stopping) else 0):
            targets = stopping(obs)
            if targets is None:
                break
            last = _apply(targets, last)
            obs = _step(sim, move.robot, last, obs)
    finally:
        sim.servos.kp = kp
        sim.set_torque(True)                       # the joints it left limp: back on, as on the robot
    return obs


def _move_name(cls) -> str:
    name = getattr(cls, "name", None)
    if isinstance(name, str) and name:
        return name
    base = cls.__name__
    if base.endswith("Move") and len(base) > 4:
        base = base[:-4]
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", base).lower()


def _is_move(cls) -> bool:
    step = getattr(cls, "step", None)
    if not callable(step) or inspect.isabstract(cls):
        return False
    try:
        params = list(inspect.signature(step).parameters.values())
    except (TypeError, ValueError):
        return False
    return len(params) == 2 and all(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)


def load_moves(path) -> dict:
    """Import the Python file ``path`` (a move file, as in microban's ``src/moves/``) and return ``{name: move}``, one
    instance of every class defined in that file with a ``step(self, obs)`` method (and not abstract). ``name``: the
    class's ``name`` attribute if it sets one, else the class name without its trailing "Move", in snake_case
    (``WaveMove`` -> ``"wave"``, ``IKSquatMove`` -> ``"ik_squat"``). As on the robot: the file's folder is on the path
    (it imports its own files by name), a file that fails to import gives ``{}`` and a class that fails to build is
    skipped, each with its error printed."""
    path = Path(path).resolve()
    if str(path.parent) not in sys.path:
        sys.path.append(str(path.parent))
    module_name = "microban_moves_" + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - as the robot: the file is skipped, the error shown
        print(f"load_moves: {path.name} skipped (it does not load):\n{traceback.format_exc()}")
        return {}
    moves = {}
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module_name or not _is_move(cls):
            continue
        try:
            moves[_move_name(cls)] = cls()
        except Exception as e:  # noqa: BLE001 - a broken move must not hide the others
            print(f"load_moves: {cls.__name__} in {path.name} skipped ({type(e).__name__}: {e})")
    return moves
