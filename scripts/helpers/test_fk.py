from isaaclab.app import AppLauncher

AppLauncher({"headless": True})

import pytorch_kinematics as pk
import torch

import isaaclab.utils.math as math_utils

urdf_chain = pk.build_serial_chain_from_urdf(
    open("data/franka_panda/panda.urdf", mode="rb").read(), "panda_hand"
)
urdf_chain = urdf_chain.to(device="cuda")

# (0.5, 0.3, 0.2)
pos = torch.tensor([1.2836e-01, 2.8420e-01, 4.3287e-01, -2.0772e00, -1.6051e-01, 2.3218e00, 1.4306e00]).to("cuda")  # fmt: off
# (0.5, -0.3, 0.2)
pos = torch.tensor([-1.6615e-01, 2.7841e-01, -3.8028e-01, -2.0778e00, 1.3647e-01, 2.3238e00, 1.4746e-01]).to("cuda")  # fmt: off


th = urdf_chain.forward_kinematics(pos)
m = th.get_matrix()
pos = m[:, :3, 3]
rot = pk.matrix_to_quaternion(m[:, :3, :3])

pos_offset = torch.tensor([[0, 0, 0.107]]).to("cuda")
rot_offset = torch.tensor([[1, 0, 0, 0]]).to("cuda")

ee_pos, ee_quat = math_utils.combine_frame_transforms(pos, rot, pos_offset, rot_offset)

print(ee_pos)
print(torch.norm(ee_pos - torch.tensor([0.5, -0.3, 0.2]).to("cuda")))
