import copy
import os

import torch


def export_policy_as_jit(policy: object, path: str, filename="policy.pt"):
    policy_exporter = PolicyExporter(policy)
    policy_exporter.export(path, filename)


class PolicyExporter(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        policy.env = None
        self.policy = copy.deepcopy(policy)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor):
        return self.policy.act({"obs": obs, "goal": goal})["action"]

    @torch.jit.export
    def reset(self):
        pass

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
