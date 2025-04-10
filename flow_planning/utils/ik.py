import os
import time

import numpy as np
import pinocchio
from numpy.linalg import norm, solve


class IKSolver:
    def __init__(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        urdf_path = cwd + "/../../data/urdf/franka.urdf"
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.JOINT_ID = 7
        self.oMdes = pinocchio.SE3(np.eye(3), np.array([0.70, 0.0, 0.2]))

        self.eps = 1e-4
        self.IT_MAX = 1000
        self.DT = 1e-1
        self.damp = 1e-12

    def solve(self):
        i = 0
        q = pinocchio.neutral(self.model)
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[self.JOINT_ID].actInv(self.oMdes)
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
            return q.flatten().tolist()
        else:
            raise RuntimeError("IK failed to converge")


if __name__ == "__main__":
    ik_solver = IKSolver()
    start_time = time.time()
    q = ik_solver.solve()
    end_time = time.time()
    print("IK solution:", q)
    print(f"IK solved in {end_time - start_time:.4f} seconds")
