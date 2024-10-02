# get quat from rpy
import numpy as np
from scipy.spatial.transform import Rotation as R

from pino_ik_solver import PinoIkSolver, XyzQuat

quat = R.from_euler('xyz', [0.1, 0.2, 0.3]).as_quat()

svr = PinoIkSolver('[sample]', it_max=100)
svr.print_frames()

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


sol = svr.solve_ik(
         # move
         'shoulder1_joint',
         'wrist1_joint',
         # target
         ('universe',),
         ('wrist2_joint',),
         (np.array([1., 0., 1.]), quat),
         None
)
q = sol[0]
np.set_printoptions(suppress=True, precision=3)
print(sol[3])

xyz_quat = svr.get_tcp(q, ('universe',), ('wrist2_joint',))
print(xyz_quat[0], R.from_quat(xyz_quat[1]).as_euler('xyz'))
