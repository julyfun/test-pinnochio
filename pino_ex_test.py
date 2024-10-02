from __future__ import print_function

import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm, solve

import pinocchio

model = pinocchio.buildSampleModelManipulator()
data  = model.createData()

JOINT_ID = 6
# w<-j6 des

quat = R.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
oMdes = pinocchio.SE3(quat, np.array([1., 0., 1.]))

q      = pinocchio.neutral(model)
eps    = 1e-4
IT_MAX = 5
DT     = 1e-1
damp   = 1e-12

i=0
while True:
    print(f'q: {q}')
    pinocchio.forwardKinematics(model,data,q)
    # j6des<-w * w<-j6real
    # = j6des<-j6real
    dMi = oMdes.actInv(data.oMi[JOINT_ID])
    err = pinocchio.log(dMi).vector
    # print(f'err: {err}')
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pinocchio.computeJointJacobian(model,data,q,JOINT_ID)
    # print(f'J: {J}')
    v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pinocchio.integrate(model,q,v*DT)
    i += 1

if success:
    print("Convergence achieved!")
else:
    print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

print('\nresult: %s' % q.flatten().tolist())
print('\nfinal error: %s' % err.T)
