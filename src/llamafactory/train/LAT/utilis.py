from torch.nn import Module
from typing import Any, Callable, List, Tuple, Union
import torch
# import pdb
# pdb.set_trace()


class GDAdversary(torch.nn.Module):
    
    def __init__(self, dim, epsilon, attack_mask, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.epsilon = epsilon

        if dtype:
            self.attack = torch.nn.Parameter(torch.zeros(
                attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device, dtype=dtype))
        else:
            self.attack = torch.nn.Parameter(torch.zeros(
                attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device))
        self.clip_attack()
        self.attack_mask = attack_mask
    
    def forward(self, x):
        if x.shape[1] == 1 and self.attack.shape[1] != 1:  # generation mode (perturbation already applied)
            return x
        else:

            if self.device is None or self.device != x.device:
                with torch.no_grad():
                    self.device = x.device
                    self.attack.data = self.attack.data.to(self.device)
                    self.attack_mask = self.attack_mask.to(self.device)

            perturbed_acts = x[self.attack_mask[:, :x.shape[1]]] + self.attack[self.attack_mask[:, :x.shape[1]]].to(x.dtype)
            x[self.attack_mask[:, :x.shape[1]]] = perturbed_acts

            return x
    
    def clip_attack(self):
        with torch.no_grad():
            # clip attack norm to eps
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)

            norms = torch.norm(self.attack, dim=-1)

class AdversaryWrapper(Module):
    def __init__(self, module: Module, adversary: Any):
        super().__init__()
        self.module = module
        self.adversary = adversary

    def forward(self, *inputs, **kwargs):
        outputs = self.module(*inputs, **kwargs)
        return self.adversary(outputs)
    


def add_hooks(model, layers, input_mask):
    adversaries = []
    hooks = []

    for layer_i in layers:
        layer = f'module.language_model.base_model.layers.{layer_i}'
        adversary = GDAdversary(
                dim=4096,
                device=model.device,
                epsilon=0.1, 
                attack_mask=input_mask,
                dtype = torch.bfloat16         
        )
        parent = model.get_submodule(layer)
        submodule = parent.get_submodule('mlp')
        wrapped_module = AdversaryWrapper(submodule, adversary)
        adversaries.append(adversary)
        hooks.append(wrapped_module)
        setattr(parent, 'mlp', wrapped_module)
    
    return adversaries, hooks

