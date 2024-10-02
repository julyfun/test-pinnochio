# get quat from rpy
import numpy as np
from scipy.spatial.transform import Rotation as R

from pino_ik_solver import PinoIkSolver, XyzQuat

quat = R.from_euler('xyz', [0.1, 0.2, 0.3]).as_quat()

svr = PinoIkSolver('[sample]', it_max=5)

svr.print_joints()

# for i in (
#     svr.solve_ik(
#         # move
#         'shoulder1_joint',
#         'wrist2_joint',
#         # target
#         ('universe',),
#         ('wrist2_joint',),
#         (np.array([1., 0., 1.]), quat),
#         None
# )):
#     print(i)


q = svr.solve_ik(
         # move
         'shoulder1_joint',
         'wrist2_joint',
         # target
         ('universe',),
         ('wrist2_joint',),
         (np.array([1., 0., 1.]), quat),
         None
 )[0]
