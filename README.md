# microban_sim

MuJoCo simulator of the [Microban](https://github.com/Rhoban/microban) robot, for any project: the robot from
[microban_model](https://github.com/Rhoban/microban_model) (the classic version, or the climbing one), its 19
Dynamixel XL330 servos simulated by [BAM](https://github.com/Rhoban/bam) as identified on the bench, its sensors as
the robot reads them, and a fast step API for planning and learning.

```python
from microban_sim import MicrobanSim, JOINTS

sim = MicrobanSim("classic")             # or "climber": the climbing forearms and feet
obs = sim.reset()                        # standing on the floor, in the walking robot's standing pose
for t in range(500):                     # 500 x 20 ms = 10 s of simulated time, in about 0.5 s
    target = obs["q"]                    # your move: 19 joint angles (rad), order JOINTS
    obs = sim.step(target)
```

Install: `uv add git+https://github.com/Rhoban/microban_sim` (Python 3.12+), or clone it and `uv sync`.

## Two ways to use it

* **Fast**, in Python: `sim.step(target)` simulates 20 ms as fast as it computes (about 1 ms on a laptop), no window
  (`sim.view()` opens MuJoCo's viewer on it). For planning, learning, evaluating: save and restore a state, many
  processes in parallel.
* **As on the robot**: the robot's own program (its scheduler, moves, keyboard, gamepad, live dashboard) runs on this
  simulator, in real time: `make sim` in the [microban](https://github.com/Rhoban/microban) repository (branch
  `dev_v2`). What you test there is what `make deploy` sends to the robot.

## The API

| | |
|---|---|
| `MicrobanSim(variant, spec=None, control_dt=0.02, latency_ms=10, kp_fw=400, vin=7.5, ideal_sensors=False, seed=None)` | `spec`: a scene of your own (`build_spec(variant)` with more things in it, then pass it) |
| `sim.reset(trunk_pos, trunk_quat, joints)` | standing on the floor (the walking robot's standing pose), or a pose of yours |
| `obs = sim.step(target)` | 19 joint targets (rad, order `JOINTS`, None: keep the last), 20 ms of simulation |
| `sim.observation()` | what the robot measures, with its delays: `q`, `dq`, `gyro`, `accel`, `gravity` (trunk frame), `quat`, `current`, `time` |
| `sim.imu_raw()` | the IMU as its chip reads it (its own frame), for code that converts it itself (the robot's program) |
| `sim.set_torque(on)`, `sim.servos.kp` | the servos' Torque Enable (all, or 19) and firmware gain |
| `sim.trunk_pose()`, `sim.site_positions(sites)` | ground truth (not measured on the robot): for debugging, rewards, evaluation |
| `sim.site_positions_at(q, sites, ...)`, `sim.joint_angles_for({site: xyz}, trunk_pos, trunk_quat)` | forward and inverse kinematics, the simulation untouched |
| `state = sim.get_state()`, `sim.set_state(state)` | try a motion and come back: the same steps give exactly the same motion |
| `sim.render(camera)`, `sim.view()` | an image; MuJoCo's viewer on the running simulation |
| `run_move(move, sim, seconds)` | a move: a class with `step(obs) -> targets` (19 angles, or `{joint: angle}`; `None`: over), the same file as on the robot (a file of microban's `src/moves/`); then its `stopping`; its options and sliders at their defaults, its `kp` while it runs |
| `load_moves(path)` | `{name: move}`, every move class of a Python file (`WaveMove` -> `"wave"`) |

## What is simulated

* **The robot**: microban_model's MJCF (the one the robot's code uses), classic or climbing version, every joint
  zero at the motor's zero.
* **The servos**: BAM model m6 of the XL330: firmware position control (P gain `kp_fw`, 400 = the factory value;
  the robot runs learned policies at 125), PWM voltage, DC motor with back-EMF, 1.75 A current limit, load-dependent
  friction, a supply voltage that drops under load, torque on / off.
* **The timing**, measured on the robot: a target acts 10 ms after it is sent (the robot's program reads its
  sensors, computes, then writes); the joint velocities arrive one 20 ms step late, the gyro and the accelerometer
  3, the orientation 4 (`ideal_sensors=True`: none).
* Physics at 500 Hz (2 ms steps).

Built on it: the hackathon's climbing simulator ([rock-climbing-robots/simulator](https://github.com/rock-climbing-robots/simulator)),
the robot with its wall, holds, markers and cameras.

## Layout

```
microban_sim/
  scene.py        build_spec(variant): the robot on a floor, as an MjSpec to extend
  servos.py       Servos: the XL330 simulated by BAM
  sim.py          MicrobanSim: step, observation (with the robot's delays), kinematics, state, render, view
  move.py         run_move, load_moves
  poses/          the standing pose
examples/         stand.py
```

## License

GPL-3.0-or-later (LICENSE), as the software of [microban](https://github.com/Rhoban/microban).
