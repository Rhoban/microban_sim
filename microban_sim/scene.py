# SPDX-License-Identifier: GPL-3.0-or-later
"""The MuJoCo scene: the robot from microban_model (classic or climbing version) on a floor, with lights.

:class:`microban_sim.MicrobanSim` builds it for you; :func:`build_spec` gives it as a ``mujoco.MjSpec`` to add your own
things (a wall, objects, cameras) before compiling it, as the hackathon's simulator does.
"""
from __future__ import annotations

import microban_model
import mujoco
import numpy as np

#: joints driven by servos, in the order of the robot's actuators (microban_model.JOINTS; the head is a servo too)
JOINTS = list(microban_model.JOINTS)
#: the robot's versions: walking hands and feet, or the climbing forearms and feet
VARIANTS = ("classic", "climber")


def build_spec(variant: str = "classic", floor: bool = True, floor_friction: float = 1.0,
               timestep: float = 0.002) -> mujoco.MjSpec:
    """The robot (``variant``: "classic" or "climber") on a floor, as a ``mujoco.MjSpec``: add things to it, then
    ``.compile()`` it.  The servos are torque-controlled (BAM simulates them, see :class:`microban_sim.Servos`), so
    the model's own actuator limits, joint damping and friction are cleared."""
    if variant not in VARIANTS:
        raise ValueError(f"variant: one of {VARIANTS}")
    spec = mujoco.MjSpec.from_file(str(microban_model.mjcf(climb=variant == "climber")))
    spec.modelname = f"microban_{variant}"
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.ccd_iterations = 12
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1920
    spec.visual.quality.offsamples = 4
    spec.compiler.texturedir = ""
    for act in spec.actuators:
        act.set_to_motor()
        act.gear = [1.0, 0, 0, 0, 0, 0]
        act.forcelimited = False
        act.ctrllimited = False
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.damping = np.zeros((3, 1)) if np.ndim(j.damping) else 0.0
            j.frictionloss = 0.0

    w = spec.worldbody
    w.add_light(name="sun", pos=[0.4, -1.5, 2.5], dir=[0.0, 0.6, -1.0],
                type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True)
    w.add_light(name="fill", pos=[0.4, -2.0, 1.0], dir=[0.0, 1.0, 0.0], diffuse=[0.4, 0.4, 0.4], castshadow=False)
    if floor:
        spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.55, 0.55, 0.55], rgb2=[0.5, 0.5, 0.5], width=512, height=512)
        spec.add_material(name="floor", texrepeat=[8, 8], reflectance=0.05).textures[
            mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.05], material="floor",
                   friction=[floor_friction, 0.005, 0.0001])
    w.add_camera(name="side", pos=[0.0, -0.9, 0.35], xyaxes=[1, 0, 0, 0, 0.25, 1])
    return spec
