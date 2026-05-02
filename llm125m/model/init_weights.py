import math
import torch.nn as nn


def init_weights(model, config):
    n_layers = config.n_layers
    init_std = config.init_std

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)

    scaling = 1.0 / math.sqrt(2 * n_layers)
    for block in model.blocks:
        nn.init.normal_(block.attn.wo.weight, mean=0.0, std=init_std * scaling)
        nn.init.normal_(block.ffn.w_down.weight, mean=0.0, std=init_std * scaling)
