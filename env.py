from robosuite.models import MujocoWorldBase
from robosuite.models.arenas import PegsArena
from robosuite.models.arenas import TableArena
from robosuite.models.arenas import CylinderArena
import mujoco
import mujoco.viewer


"""world = MujocoWorldBase()
mujoco_arena = PegsArena(
    table_full_size=(0.8, 0.8, 0.05),
    table_offset=(0, 0, 0.8)
)
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

model = world.get_model(mode="mujoco")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()"""


world = MujocoWorldBase()
mujoco_arena = CylinderArena(table_full_size=(0.8, 0.8, 0.05),
    table_offset=(0, 0, 0.8))
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

model = world.get_model(mode="mujoco")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

"""import robosuite
print("robosuite module:", robosuite)

print("\nSearch paths containing 'robosuite':")
import sys, os
for p in sys.path:
    if p and os.path.exists(os.path.join(p, "robosuite")):
        print(" -", os.path.join(p, "robosuite"))

import robosuite
print(robosuite.__file__)"""