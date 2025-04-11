import os
import time

import numpy as np
import pinocchio
import torch
from numpy.linalg import norm, solve
from torch import Tensor


class IKSolver:
    def __init__(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        urdf_path = cwd + "/../../data/urdf/franka.urdf"
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.JOINT_ID = 7

        self.eps = 1e-4
        self.IT_MAX = 1000
        self.DT = 1e-1
        self.damp = 1e-12

    def solve(self, position: Tensor, orientation: Tensor):
        oMdes = pinocchio.SE3(orientation.cpu().numpy(), position.cpu().numpy())
        q = pinocchio.neutral(self.model)

        i = 0
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[self.JOINT_ID].actInv(oMdes)
            err = pinocchio.log(iMd).vector  # in joint frame
            if norm(err) < self.eps:
                success = True
                break
            if i >= self.IT_MAX:
                success = False
                break
            J = pinocchio.computeJointJacobian(
                self.model, self.data, q, self.JOINT_ID
            )  # in joint frame
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * self.DT)
            i += 1

        if success:
            return Tensor(q.flatten())
        else:
            raise RuntimeError("IK failed to converge")


if __name__ == "__main__":
    # fmt: off
    test_data ={
        (0.4, 0, 0.2): Tensor([-2.2671e-02, 7.6667e-02, 2.1580e-02, -2.7259e00, -4.9747e-03, 2.8028e00, 7.8894e-01, 9.8029e-05, 1.0173e-04]),
        (0.6, 0, 0.6): Tensor([-1.6977, -2.267e-02, 1.3992e-01, 1.8799e-02, -1.4575e00, -2.6461e-03, 1.5982e00, 7.8706e-01, 9.3350e-05, 1.0834e-04]),
        (0.4, 0, 0.6): Tensor([-1.7655, -2.267e-02, -4.9210e-01, 1.3407e-02, -2.1302e00, 6.3447e-03, 1.6397e00, 7.7912e-01, 9.3380e-05, 1.0829e-04]),
        (0.6, 0, 0.2): Tensor([-2.9749, -2.267e-02, 4.9310e-01, 2.9753e-02, -2.0311e00, -2.4329e-02, 2.5243e00, 8.0171e-01, 9.6156e-05, 1.0438e-04]),
    }
    # fmt: on

    ik_solver = IKSolver()

    for pos, expected_q in test_data.items():
        pos_tensor = Tensor(pos)
        q = ik_solver.solve(pos_tensor, torch.diag(torch.tensor([1, -1, -1])))
        assert torch.allclose(q, expected_q, atol=1e-4), (
            f"Test failed for position {pos}, \n expected {expected_q}, \n got {q}"
        )
        print(f"Test passed for position {pos}")
