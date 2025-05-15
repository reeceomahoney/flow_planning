import math
import time

import pytorch_kinematics as pk
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chain = pk.build_serial_chain_from_urdf(
    open("data/urdf/panda.urdf", mode="rb").read(), "panda_hand"
).to(device=device)

M = 1000
pos_x = (0.35, 0.65)
pos_y = (-0.2, 0.2)
pos_z = (0.15, 0.5)

pos = torch.zeros((M, 3), device=device)
pos[:, 0] = torch.rand((M,), device=device) * (pos_x[1] - pos_x[0]) + pos_x[0]
pos[:, 1] = torch.rand((M,), device=device) * (pos_y[1] - pos_y[0]) + pos_y[0]
pos[:, 2] = torch.rand((M,), device=device) * (pos_z[1] - pos_z[0]) + pos_z[0]
rot = torch.tensor([[0.0, math.pi, 0.0]], device=device).expand(M, -1)

goal_tf = pk.Transform3d(pos=pos, rot=rot, device=str(device))

lim = torch.tensor(chain.get_joint_limits(), device=device)

init_pos = torch.tensor(
    [[0.0000, -0.5690, 0.0000, -2.8100, 0.0000, 3.0370, 0.7410]],
    device="cuda:0",
)

ik = pk.PseudoInverseIK(
    chain,
    max_iterations=30,
    retry_configs=init_pos,
    joint_limits=lim.T,
    early_stopping_any_converged=True,
    early_stopping_no_improvement="all",
    debug=False,
    lr=0.2,
)

start = time.time()
sol = ik.solve(goal_tf)
end = time.time()

print("IK took %f seconds" % (end - start))
print("IK converged number: %d / %d" % (sol.converged.sum(), sol.converged.numel()))
print("IK took %d iterations" % sol.iterations)
print("IK solved %d / %d goals" % (sol.converged_any.sum(), M))
