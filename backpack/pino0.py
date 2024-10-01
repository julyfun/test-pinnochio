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
oMdes = pinocchio.SE3(rot, np.array([1., 0., 1.]))

q = pinocchio.neutral(model)
eps = 1e-4
IT_MAX = 2
DT = 1e-1
damp = 1e-12

i = 0
while True:
    pinocchio.forwardKinematics(model, data, q)
    # 相当于计算 oMdes.inverse() * data.oMi[JOINT_ID]
    # dMi = T_(期6 <- 实6)
    dMi = oMdes.inverse() * data.oMi[JOINT_ID]
    print(f'dMi: {dMi}')
    # 给定两个 SE3，其差为 SE3，该差必有对应的 se3 向量
    # 该向量的指数映射结果为一矩阵，其可平滑表示两个 SE3 的差异
    # 例如矩阵为 A，则 A^0.1 约等于 0.1 倍旋转
    # https://blog.csdn.net/shao918516/article/details/116604377
    err = pinocchio.log(dMi).vector
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

print("first", first_err)
print("final", err)
if success:
    print("Convergence achieved!")
else:
    print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

print('\nresult: %s' % q.flatten().tolist())
print('\nfinal error: %s' % err.T)

# output all joint names and their angles
print(model.names[0])
for joint_id in range(1, model.njoints):
    joint_name = model.names[joint_id]
    joint_angle = q[model.joints[joint_id].idx_q]
    print(f"Joint {joint_name}: angle = {joint_angle}")
