"""The robot standing, then turning its head, in MuJoCo's viewer (close the window to stop).

    uv run python examples/stand.py [classic|climber]
"""
import math
import sys

from microban_sim import JOINTS, MicrobanSim

sim = MicrobanSim(sys.argv[1] if len(sys.argv) > 1 else "classic")
obs = sim.reset()
target = obs["q"].copy()
head = JOINTS.index("head")
viewer = sim.view()                                       # real time
while viewer.is_running():
    target[head] = 0.5 * math.sin(obs["time"])
    obs = sim.step(target)
