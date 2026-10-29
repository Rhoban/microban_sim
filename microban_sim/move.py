# SPDX-License-Identifier: GPL-3.0-or-later
"""Run a move, the same file as on the real robot (a file of microban's ``src/moves/``, branch dev_v2).

A simple move is a class (no base class needed) with ``step(obs) -> targets`` (the 19 joint targets, rad, order
``JOINTS``; None keeps the last ones), called every 20 ms with ``sim.observation()``, and optionally:

* ``reset(obs)``: called when it starts;
* ``kp_fw``: the servos' gain while it runs (125 by default, as the robot runs learned policies);
* ``options = {name: value}`` (int / float / bool): variables the robot's dashboard edits live, read as
  ``self.options[name]``; ``option_docs = {name: "..."}``;
* ``sliders = {name: (min, max, start)}``: shown while it runs, read as ``self.slider[name]``;
* ``keys = "..."``: the keys it reads (``self.pressed``: the set of held keys; ``on_key(key, down)``);
* ``description``, ``variants`` (the robot versions it is for: "classic", "climber").

In the simulator nobody edits options, sliders or keys: :func:`run_move` gives them their defaults.
"""
import importlib.util
import inspect
import re
import sys
from pathlib import Path

KP_MOVE = 125.0


def _give_defaults(move) -> None:
    """What the robot's dashboard sets on a move, at its defaults: ``options``, ``slider``, ``pressed``."""
    cls_options = getattr(type(move), "options", None)
    if isinstance(cls_options, dict) and "options" not in vars(move):
        move.options = dict(cls_options)
    if not hasattr(move, "slider"):
        move.slider = {n: float(v[2]) for n, v in (getattr(move, "sliders", None) or {}).items()}
    if not hasattr(move, "pressed"):
        move.pressed = set()


def run_move(move, sim, seconds: float, on_step=None) -> dict:
    """Run ``move`` on ``sim`` (a :class:`microban_sim.MicrobanSim`, or the hackathon's ClimbingSim) for
    ``seconds`` of simulated time, from where the robot is (as on the robot, where the torque goes on where the robot
    is when it starts). Its options and sliders keep their defaults, no key is pressed. ``on_step(sim, obs)``, if
    given, is called after every step (to watch, log or stop: return True to stop). Returns the last observation."""
    _give_defaults(move)
    sim.servos.kp = float(getattr(move, "kp_fw", KP_MOVE))
    sim.set_torque(True)
    obs = sim.observation()
    if hasattr(move, "reset"):
        move.reset(obs)
    for _ in range(int(round(seconds / sim.control_dt))):
        obs = sim.step(move.step(obs))
        if on_step is not None and on_step(sim, obs):
            break
    return obs


run_controller = run_move
"""The old name of :func:`run_move` (a move used to be called a Controller)."""


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
    (``WaveMove`` -> ``"wave"``). A class that fails to build is skipped, with a message."""
    path = Path(path).resolve()
    module_name = "microban_moves_" + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    moves = {}
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module_name or not _is_move(cls):
            continue
        try:
            moves[_move_name(cls)] = cls()
        except Exception as e:  # noqa: BLE001 - a broken move must not hide the others
            print(f"load_moves: {cls.__name__} in {path.name} skipped ({type(e).__name__}: {e})")
    return moves
