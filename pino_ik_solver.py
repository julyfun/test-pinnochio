from __future__ import print_function

import numpy as np
from numpy.linalg import norm, solve
from scipy.spatial.transform import Rotation as R
from typing import Tuple

XyzQuat = Tuple[np.ndarray, np.ndarray]

import pinocchio

def xyz_quat_to_big_se3(xyz_quat: XyzQuat):
    xyz, quat = xyz_quat
    return pinocchio.SE3(
        R.from_quat(quat).as_matrix(),
        xyz
    )

class PinoIkSolver:
    def __init__(
        self,
        urdf: str,
        eps=1e-3,
        it_max=1e3,
        dt=0.1,
        damp=1e-10
    ):
        if urdf == '[sample]':
            self.model = pinocchio.buildSampleModelManipulator()
        else:
            self.model = pinocchio.buildModelFromUrdf(urdf)
        self.data = self.model.createData()

        self.eps = eps
        self.it_max = it_max
        self.dt = dt
        self.damp = damp

    def get_chain_indices(self, base_j: str, end_j: str):
        base_j_id = self.model.getJointId(base_j)
        end_j_id = self.model.getJointId(end_j)
        cur = end_j_id
        chain_jid = [cur]
        while cur != base_j_id:
            cur = self.model.parents[cur]
            chain_jid.append(cur)
        chain_jid.reverse()
        # chain_q_indices = [self.model.joints[i].idx_q for i in chain]
        chain_qid = [self.model.joints[i].idx_q for i in chain_jid]
        return chain_jid, chain_qid

    def solve_ik(self, base_j: str, end_j: str, target: XyzQuat, ref_q: np.array=None):
        if ref_q is None:
            ref_q = pinocchio.neutral(self.model)
        q = ref_q
        base_j_id = self.model.getJointId(base_j)
        end_j_id = self.model.getJointId(end_j)
        chain_jid, chain_qid = self.get_chain_indices(base_j, end_j)

        first_err = np.zeros(6)
        i = 0
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            w4base_j = self.data.oMi[base_j_id]
            w4end_j_des = xyz_quat_to_big_se3(target)
            w4end_j_real = self.data.oMi[end_j_id]
            end_j_des4end_j_real = w4end_j_des.inverse() * w4end_j_real
            err = pinocchio.log(end_j_des4end_j_real).vector
            if i == 0:
                first_err = err
            if norm(err) < self.eps:
                success = True
                break
            if i >= self.it_max:
                success = False
                break
            jac = pinocchio.computeJointJacobian(self.model, self.data, q, end_j_id)
            jac_reduced = jac[:, chain_qid]
            v_reduced = -jac_reduced.T @ solve(jac_reduced @ jac_reduced.T + self.damp * np.eye(6), err)
            v = np.zeros(self.model.nv)
            v[chain_qid] = v_reduced
            q = pinocchio.integrate(self.model, q, v * self.dt)
            i += 1

        return q.flatten().tolist(), success, err, err / first_err
