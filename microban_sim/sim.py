"""The simulator: physics at 500 Hz, the 19 XL330 servos simulated by BAM, the robot's sensors as it reads them.

    from microban_sim import MicrobanSim
    sim = MicrobanSim("classic")           # or "climber": the climbing forearms and feet
    obs = sim.reset()                      # standing on the floor, in the walking robot's standing pose
    for _ in range(500):
        obs = sim.step(target)             # 19 joint targets (rad, order JOINTS), one call = one 20 ms control step

What the real robot measures is in the observation (``sim.observation()``, also returned by ``step``), with the
robot's sensor delays: joint positions and velocities, the IMU (in the trunk frame), the motor currents.  The rest
(``trunk_pose``, ``site_positions``) is ground truth, for debugging and evaluation only.  ``imu_raw()`` gives the IMU
as the chip reads it (its own frame), for code that does the robot's conversions itself (the robot's program).

For planning: ``site_positions_at(q, sites, trunk_pos, trunk_quat)`` (forward kinematics), ``joint_angles_for(targets,
trunk_pos, trunk_quat)`` (inverse kinematics), both without touching the simulation, and ``get_state()`` /
``set_state(state)`` (try a motion, then come back).  ``view()`` opens MuJoCo's viewer on the running simulation.
"""
from __future__ import annotations

import copy
import json
import pathlib
import time
from collections import deque

import mujoco
import numpy as np

from .scene import JOINTS, build_spec
from .servos import Servos

POSES = pathlib.Path(__file__).parent / "poses"
#: the walking robot's standing pose (mjlab_microban HOME_FRAME), the pose the real robot stands in by itself
STAND = json.loads((POSES / "stand.json").read_text())
#: the robot's sensor delays, in control steps of 20 ms (as the robot's program reads them)
ROBOT_DELAYS = {"q": 0, "dq": 1, "current": 0, "gyro": 3, "accel": 3, "quat": 4}
LATENCY_MS = 10.0             # from the state read to the servos acting on the target: measured on the robot (10-29)


class MicrobanSim:
    """The Microban robot (classic or climbing version) on a floor.

    variant: "classic" (walking hands and feet) or "climber" (climbing forearms and feet); spec: a scene of one's own
        (``build_spec(variant)`` with more things in it) instead of the robot on the floor.
    control_dt: duration of one :meth:`step` (s); the physics runs at the scene's timestep (2 ms).
    latency_ms: from a target sent to the servos acting on it (10 ms: the robot's program reads its sensors, computes,
        then writes; measured on the robot).
    kp_fw: the servos' firmware P gain: 400, the XL330's factory value, by default; the robot runs its learned
        policies at 125.
    vin: the servos' supply voltage (V).
    ideal_sensors: True gives every measurement of the current step (no sensor delay), for debugging.
    seed: seed of ``self.rng`` (numpy random generator), for your own randomisation.

    Attributes you may use: ``joint_range`` (19 x 2, min and max angle of each joint, rad, order ``JOINTS``),
    ``control_dt`` (s), ``servos`` (gain, torque), ``rng``, ``model`` / ``data`` (the plain MuJoCo objects).
    """

    def __init__(self, variant: str = "classic", spec: mujoco.MjSpec | None = None, control_dt: float = 0.02,
                 latency_ms: float = LATENCY_MS, kp_fw: float = 400.0, vin: float = 7.5, ideal_sensors: bool = False,
                 seed: int | None = None):
        self.variant = variant
        self.spec = spec if spec is not None else build_spec(variant)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.n_sub = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.control_dt = self.n_sub * self.model.opt.timestep
        self.servos = Servos(self.model, self.data, kp_fw=kp_fw, vin=vin,
                             latency_steps=int(round(latency_ms / 1000 / self.model.opt.timestep)))
        self.delays = {k: 0 for k in ROBOT_DELAYS} if ideal_sensors else dict(ROBOT_DELAYS)
        self.rng = np.random.default_rng(seed)
        m = self.model
        self._qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINTS])
        self._vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINTS])
        self.joint_range = np.array([m.jnt_range[m.joint(n).id] for n in JOINTS])
        self._renderers = {}                  # one per image size
        self._fk_data = mujoco.MjData(m)       # scratch data for the kinematics
        R = np.zeros(9)                        # the IMU chip's orientation in the trunk (fixed)
        mujoco.mju_quat2Mat(R, m.site_quat[m.site("imu").id])
        self._R_trunk_imu = R.reshape(3, 3)
        self._history = deque(maxlen=max(self.delays.values()) + 1)    # the last readings, for the sensor delays
        self._viewer, self._realtime, self._last_sync = None, True, 0.0

    # ---- state ---------------------------------------------------------------------------------------
    def reset(self, trunk_pos=None, trunk_quat=None, joints: dict | None = None):
        """Robot standing on the floor at the origin, facing +x, in the standing pose of Rhoban's walking robot (the
        pose the real robot stands in by itself: arms along the body, hips pitched -10 deg, hip and ankle rolls +-5
        deg), its soles on the floor; ``trunk_pos`` / ``trunk_quat`` (world frame) and ``joints`` ({name: angle})
        change it.  At kp_fw 400 the robot stands still; with soft servos (kp_fw 125) it falls unless balanced
        actively."""
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[3:7] = STAND["trunk_quat"] if trunk_quat is None else trunk_quat
        j = dict(STAND["joints"]); j.update(joints or {})
        for n, a in zip(JOINTS, self._qadr):
            d.qpos[a] = j.get(n, 0.0)
        if trunk_pos is None:
            d.qpos[0:3] = [0.0, 0.0, 0.0]
            mujoco.mj_kinematics(self.model, d)
            d.qpos[2] = -self._lowest_point()                  # the soles on the floor
        else:
            d.qpos[0:3] = trunk_pos
        return self._start()

    def _start(self):
        """After the pose is set in ``data.qpos``: the servos hold it, the sensors start from it."""
        mujoco.mj_forward(self.model, self.data)
        self.servos.reset(self.data.qpos)
        self.servos.set_target(self.joint_positions())
        self._history.clear()
        self._history.append(self._read())
        return self.observation()

    def _lowest_point(self) -> float:
        """Height of the robot's lowest point (its colliding geoms' bounding boxes), at the current kinematics."""
        m, d = self.model, self.data
        z = np.inf
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0 or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            R = d.geom_xmat[g].reshape(3, 3)
            z = min(z, d.geom_xpos[g, 2] + R[2] @ c - np.abs(R[2]) @ h)
        return float(z)

    def step(self, target=None):
        """One control step: send ``target`` (19 joint angles, rad, in the order of ``JOINTS``; None: keep the last
        one) and simulate ``control_dt`` seconds, the servos simulated by BAM at every physics step.  Returns the
        observation (:meth:`observation`)."""
        if target is not None:
            target = np.asarray(target, float)
            assert target.shape == (len(JOINTS),), f"target: {len(JOINTS)} joint angles (order JOINTS)"
        for _ in range(self.n_sub):
            if target is not None:
                self.servos.set_target(target)
            self.servos.update()
            mujoco.mj_step(self.model, self.data)
        self._history.append(self._read())
        if self._viewer is not None:
            self._sync_viewer()
        return self.observation()

    def set_torque(self, on=True):
        """Servos' torque on or off, as on the real robot (Dynamixel Torque Enable): ``on`` is one bool for every
        servo, or 19 (order ``JOINTS``).  A servo whose torque is off goes limp (only its friction left)."""
        self.servos.torque[:] = np.broadcast_to(np.asarray(on, bool), self.servos.torque.shape)

    # ---- what the real robot measures ------------------------------------------------------------------
    def _read(self) -> dict:
        """Every sensor now (no delay), the IMU in its chip's frame."""
        s = lambda n: self.data.sensor(n).data.copy()
        return {"q": self.data.qpos[self._qadr].copy(), "dq": self.data.qvel[self._vadr].copy(),
                "current": self.servos.currents(), "gyro": s("imu_ang_vel"), "accel": s("imu_accel"),
                "quat": s("orientation")}

    def _delayed(self, key: str):
        h = self._history
        return h[max(0, len(h) - 1 - self.delays[key])][key]

    def imu_raw(self) -> dict:
        """The IMU as its chip reads it, in the chip's own frame (as the robot's program gets it), with the robot's
        delays: ``gyro`` (rad/s), ``accel`` (m/s^2, includes gravity: about +9.81 upwards at rest), ``quat`` (the
        chip's orientation, w x y z, world frame)."""
        return {k: self._delayed(k).copy() for k in ("gyro", "accel", "quat")}

    def joint_positions(self) -> np.ndarray:
        """The 19 joint angles now (rad, order ``JOINTS``), no delay."""
        return self.data.qpos[self._qadr].copy()

    def joint_velocities(self) -> np.ndarray:
        """The 19 joint velocities now (rad/s, order ``JOINTS``), no delay."""
        return self.data.qvel[self._vadr].copy()

    def currents(self) -> np.ndarray:
        """The 19 motor currents now (A, order ``JOINTS``)."""
        return self.servos.currents()

    def observation(self) -> dict:
        """What the real robot measures, with its sensor delays: ``q``, ``dq`` (joints), ``gyro`` (rad/s), ``accel``
        (m/s^2), ``gravity`` (unit vector of gravity, pointing down), all three in the TRUNK frame, ``quat`` (the
        trunk's orientation, w x y z, world frame), ``current`` (motors), plus the simulation ``time`` (s)."""
        imu = self.imu_raw()
        R_world_imu = np.zeros(9)
        mujoco.mju_quat2Mat(R_world_imu, imu["quat"])
        R_world_trunk = R_world_imu.reshape(3, 3) @ self._R_trunk_imu.T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_world_trunk.flatten())
        return {"q": self._delayed("q").copy(), "dq": self._delayed("dq").copy(),
                "gyro": self._R_trunk_imu @ imu["gyro"], "accel": self._R_trunk_imu @ imu["accel"],
                "gravity": R_world_trunk.T @ np.array([0.0, 0.0, -1.0]), "quat": quat,
                "current": self._delayed("current").copy(), "time": self.data.time}

    # ---- ground truth (not measured on the real robot: for debugging, rewards, evaluation) ------------
    def trunk_pose(self):
        """World position and orientation quaternion (w, x, y, z) of the trunk."""
        return self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy()

    def site_positions(self, sites) -> dict:
        """World positions of MuJoCo sites of the robot (name -> xyz), now."""
        return {s: self.data.site(s).xpos.copy() for s in sites}

    def site_positions_at(self, q, sites, trunk_pos=None, trunk_quat=None) -> dict:
        """Forward kinematics: world positions of ``sites`` (name -> xyz) for the joint angles ``q`` (19, order
        ``JOINTS``) and a trunk pose (default: the current one).  The simulation is not touched."""
        d = self._fk_data
        d.qpos[:] = self.data.qpos
        if trunk_pos is not None:
            d.qpos[0:3] = trunk_pos
        if trunk_quat is not None:
            d.qpos[3:7] = trunk_quat
        d.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return {s: d.site(s).xpos.copy() for s in sites}

    def joint_angles_for(self, targets: dict, trunk_pos, trunk_quat, q_start=None, iters: int = 100,
                         tol: float = 1e-4):
        """Inverse kinematics: joint angles (19, order ``JOINTS``) that put each site of ``targets`` (site name ->
        world xyz; the joints not needed keep their angles) where asked, the trunk at ``trunk_pos`` / ``trunk_quat``.
        Damped least squares from ``q_start`` (default: the current angles), joints kept within their ranges.  Returns
        ``(q, error)``, error = the largest distance left (m); check it: a target out of reach stays off."""
        m, d = self.model, self._fk_data
        low, high = self.joint_range.T
        d.qpos[:] = self.data.qpos
        d.qpos[0:3], d.qpos[3:7] = trunk_pos, trunk_quat
        d.qpos[self._qadr] = self.joint_positions() if q_start is None else q_start
        sites = [m.site(s).id for s in targets]
        goal = np.concatenate([np.asarray(targets[s], float) for s in targets])
        jac = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = goal - np.concatenate([d.site_xpos[s] for s in sites])
            if np.abs(err).max() < tol:
                break
            J = []
            for s in sites:
                mujoco.mj_jacSite(m, d, jac, None, s)
                J.append(jac[:, self._vadr].copy())             # the 19 joints only: the trunk stays put
            J = np.vstack(J)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(len(err)), err)
            d.qpos[self._qadr] = np.clip(d.qpos[self._qadr] + dq, low, high)
        mujoco.mj_kinematics(m, d)
        err = goal - np.concatenate([d.site_xpos[s] for s in sites])
        return d.qpos[self._qadr].copy(), float(np.linalg.norm(err.reshape(-1, 3), axis=1).max())

    # ---- save / restore (planning: try a motion, then come back) ------------------------------------
    def get_state(self) -> dict:
        """A copy of everything that evolves: the MuJoCo data, the servos' memory and the sensors' delays."""
        data = mujoco.MjData(self.model)
        mujoco.mj_copyData(data, self.model, self.data)
        return {"data": data, "servos": self.servos.get_state(self.model), "sensors": copy.deepcopy(self._history)}

    def set_state(self, state: dict):
        """Go back to a state from :meth:`get_state` (the same steps then give exactly the same motion)."""
        mujoco.mj_copyData(self.data, self.model, state["data"])
        self.servos.set_state(self.model, state["servos"])
        self._history = copy.deepcopy(state["sensors"])

    # ---- images and viewer --------------------------------------------------------------------------
    def render(self, camera="side", width=960, height=720) -> np.ndarray:
        """RGB image (H, W, 3), uint8, from a camera of the scene ("side" sees the robot from its right)."""
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        r = self._renderers[key]
        r.update_scene(self.data, camera=camera)
        return r.render()

    def view(self, realtime: bool = True, key_callback=None):
        """Open MuJoCo's viewer on the simulation: every :meth:`step` then shows in it, paced to real time when
        ``realtime`` (else as fast as it computes).  Returns the viewer (``viewer.is_running()``: not closed)."""
        import mujoco.viewer
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback)
        self._realtime, self._last_sync = realtime, time.perf_counter()
        return self._viewer

    def _sync_viewer(self):
        if self._realtime:
            wait = self._last_sync + self.control_dt - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        self._last_sync = time.perf_counter()
        self._viewer.sync()

    def close(self):
        """Free the renderers (their OpenGL contexts) and close the viewer."""
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
