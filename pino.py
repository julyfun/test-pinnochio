from __future__ import print_function

import numpy as np
from numpy.linalg import norm, solve
from scipy.spatial.transform import Rotation as R

import pinocchio
from pinocchio.visualize import MeshcatVisualizer

model = pinocchio.buildSampleModelManipulator()
data = model.createData()

JOINT_ID = 6
# 期望的 (世界 <- 关节6)
rot = R.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
j1_j6des = pinocchio.SE3(rot, np.array([1., 0., 1.]))

q = pinocchio.neutral(model)
eps = 1e-4
IT_MAX = 100
DT = 1e-1
damp = 1e-12

i = 0
first_err: np.array
while True:
    print('---')
    print(f'q: {q}')
    pinocchio.forwardKinematics(model, data, q)
    w_j1 = data.oMi[1]
    w_j6des = w_j1 * j1_j6des
    w_j6real = data.oMi[6]
    j6des_j6real = w_j6des.inverse() * w_j6real
    err = pinocchio.log(j6des_j6real).vector
    if i == 0:
        first_err = err
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    # 应该是是 se3 的雅可比吧？
    J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
    J_reduced = J[:, 1:]
    # 广义逆矩阵 + damp
    # 关节速度 v such that 沿着该方向前进 1s 可以到达期望位置
    v_reduced = - J_reduced.T.dot(solve(J_reduced.dot(J_reduced.T) + damp * np.eye(6), err))
    v = np.zeros(model.nv)
    v[1:] = v_reduced
    # v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    # q += v * DT
    q = pinocchio.integrate(model, q, v*DT)
    if not i % 10:
        print('%d: error = %s' % (i, err.T))
        # relative error for each in vector
        print('relative error = %s' % (err.T / first_err.T))
    i += 1

if success:
    print("Convergence achieved!")
else:
    print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

print('result: %s' % q.flatten().tolist())
print('final error: %s' % err.T)
print('relative error = %s' % (err.T / first_err.T))


w_j1 = data.oMi[1]
w_j6des = w_j1 * j1_j6des
w_j6real = data.oMi[6]
j1_j6real = w_j1.inverse() * w_j6real
print('j1_j6real: %s' % j1_j6real)
print('j1_j6des: %s' % j1_j6des)
# show rpy
print('j1_j6des rpy: %s' % R.from_matrix(j1_j6des.rotation).as_euler('xyz', degrees=True))
print('j1_j6real rpy: %s' % R.from_matrix(j1_j6real.rotation).as_euler('xyz', degrees=True))


# output all joint names and their angles
print(model.names[0])
for joint_id in range(1, model.njoints):
    print(joint_id)
    joint_name = model.names[joint_id]
    joint_angle = q[model.joints[joint_id].idx_q]
    print(f"Joint {joint_name}: angle = {joint_angle}")
