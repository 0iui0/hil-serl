"""Shared SpaceMouse-to-robot axis mapping.

Single source of truth for direction sign and axis swap conventions.
Both cr5af_server teleop and CR5AFEnv step() use this so only one place
needs to change when calibrating directions.
"""


def map_spacemouse_to_delta(action_6d, trans_scale: float, rot_scale: float):
    """Map raw SpaceMouse 6D action to robot translation/rotation deltas.

    Raw SpaceMouse axes:
      [0]=tx, [1]=ty, [2]=tz, [3]=pitch, [4]=roll, [5]=yaw

    Mapped robot delta:
      xyz:  [ tx,        ty,       -tz        ] * trans_scale
      rot:  [ -roll,     pitch,    -yaw       ] * rot_scale
              (rx=-roll, ry=pitch)

    Returns (xyz, rot) as lists of floats.
    """
    tx, ty, tz, pitch, roll, yaw = action_6d[:6]
    xyz = [
        tx * trans_scale,
        ty * trans_scale,
        -tz * trans_scale,
    ]
    rot = [
        -roll * rot_scale,
        pitch * rot_scale,
        -yaw * rot_scale,
    ]
    return xyz, rot
