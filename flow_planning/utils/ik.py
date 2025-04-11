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
    ik_solver = IKSolver()
    start_time = time.time()
    posiiton = Tensor([0.5, 0.0, 0.5])
    orientation = torch.eye(3)
    q = ik_solver.solve(posiiton, orientation)
    end_time = time.time()
    print("IK solution:", q)
    print(f"IK solved in {end_time - start_time:.4f} seconds")
