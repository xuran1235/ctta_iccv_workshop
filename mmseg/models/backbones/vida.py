import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
import torch.nn.functional as F

import torch.nn as nn

class VidaInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, r2 = 64):
        super().__init__()

        self.linearVida = nn.Linear(in_features, out_features, bias)
        self.Vida_down = nn.Linear(in_features, r, bias=False)
        self.Vida_up = nn.Linear(r, out_features, bias=False)
        self.Vida_down2 = nn.Linear(in_features, r2, bias=False)
        self.Vida_up2 = nn.Linear(r2, out_features, bias=False)
        self.scale = 1.0

        nn.init.normal_(self.Vida_down.weight, std=1 / r**2)
        nn.init.zeros_(self.Vida_up.weight)

        nn.init.normal_(self.Vida_down2.weight, std=1 / r**2)
        nn.init.zeros_(self.Vida_up2.weight)

    def forward(self, input):
        return self.linearVida(input) + self.Vida_up(self.Vida_down(input)) * self.scale + self.Vida_up2(self.Vida_down2(input)) * self.scale



def inject_trainable_Vida(
    model: nn.Module,
    target_replace_module: List[str] = ["CrossAttention", "Attention"],
    r: int = 4,
    r2: int = 16,
):
    """
    inject Vida into model, and returns Vida parameter groups.
    """

    require_grad_params = []
    names = []

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = VidaInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        r2,
                    )
                    _tmp.linearVida.weight = weight
                    if bias is not None:
                        _tmp.linearVida.bias = bias

                    # switch the module
                    _module._modules[name] = _tmp

                    require_grad_params.extend(
                        list(_module._modules[name].Vida_up.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].Vida_down.parameters())
                    )
                    _module._modules[name].Vida_up.weight.requires_grad = True
                    _module._modules[name].Vida_down.weight.requires_grad = True

                    require_grad_params.extend(
                        list(_module._modules[name].Vida_up2.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].Vida_down2.parameters())
                    )
                    _module._modules[name].Vida_up2.weight.requires_grad = True
                    _module._modules[name].Vida_down2.weight.requires_grad = True                    
                    names.append(name)

    return require_grad_params, names