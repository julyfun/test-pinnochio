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
        damp=1e-12
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

        # for frame in self.model.frames:
        #     print(f"Frame name: {frame.name}, Frame type: {frame.type}")
    def print_frames(self):
        for frame in self.model.frames:
            print(f"Frame name: {frame.name}, Frame type: {frame.type}")

    def print_joints(self):
        for joint in self.model.names:
            print(f"Joint name: {joint}")

    def get_joint_chain_idx_jq(self, base_j: str, end_j: str):
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

    def get_tcp(self, q: np.array, tcp_name: tuple):
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        tcp_id = self.model.getFrameId(*tcp_name)
        tcp_pose = self.data.oMf[tcp_id]
        return tcp_pose.translation, R.from_matrix(tcp_pose.rotation).as_quat()

    def solve_ik(
        self,
        move_base_j: str,
        move_end_j: str,
        target_base_f: tuple,
        target_end_f: tuple,
        target: XyzQuat,
        ref_q: np.array=None
    ):
        """
        move: THJ5 -> THJ1 (5 in total)
        target: palm -> thtip
        """
        if ref_q is None:
            ref_q = pinocchio.neutral(self.model)
        q = ref_q
        target_base_f_id = self.model.getFrameId(*target_base_f)
        # target_end_f_id = self.model.getFrameId(target_end_f)
        # Specify the type of target_end_f is Joint
        target_end_f_id = self.model.getFrameId(*target_end_f)
        move_chain_jid, move_chain_qid = self.get_joint_chain_idx_jq(move_base_j, move_end_j)

        first_err = np.zeros(6)
        i = 0
        while True:
            print(f'q: {q}')
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacements(self.model, self.data)
            # `4` means `from`

            # dMi = target.actInv(self.data.oMi[JOINT_ID])
            # print(f'dMi: {dMi}')

            w4tar_base_f = self.data.oMf[target_base_f_id]
            tar_base_f4tar_end_f_des = xyz_quat_to_big_se3(target)
            w4tar_end_f_des = w4tar_base_f * tar_base_f4tar_end_f_des
            w4tar_end_f_real = self.data.oMf[target_end_f_id]
            # should be desired from real
            target_end_f_des4target_end_f_real = w4tar_end_f_des.inverse() * w4tar_end_f_real
            # print('target_end_f_des4target_end_f_real: ', target_end_f_des4target_end_f_real)
            err = pinocchio.log(target_end_f_des4target_end_f_real).vector
            if i == 0:
                first_err = err
            if norm(err) < self.eps:
                success = True
                break
            if i >= self.it_max:
                success = False
                break
            # d w4tar_end_f_real / d q
            # ??
            jac = pinocchio.computeFrameJacobian(self.model, self.data, q, target_end_f_id)
            jac_reduced = jac[:, move_chain_qid]
            # v such that: J v = -err
            v_reduced = -jac_reduced.T @ solve(jac_reduced @ jac_reduced.T + self.damp * np.eye(6), err)
            v = np.zeros(self.model.nv)
            v[move_chain_qid] = v_reduced
            # print(v)
            q = pinocchio.integrate(self.model, q, v * self.dt)
            i += 1

        return q.flatten().tolist(), success, err, err / (first_err + self.damp)
