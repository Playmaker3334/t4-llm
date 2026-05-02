import torch.nn as nn
from torch.optim import AdamW

from llm125m.training.muon import Muon


def build_optimizers(model, train_config):
    matrix_param_ids = set()
    matrix_params = []

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if module.weight.ndim == 2:
                if id(module.weight) not in matrix_param_ids:
                    matrix_params.append(module.weight)
                    matrix_param_ids.add(id(module.weight))

    other_params = []
    for p in model.parameters():
        if id(p) not in matrix_param_ids:
            other_params.append(p)

    muon = Muon(
        matrix_params,
        lr=train_config.muon_lr,
        momentum=train_config.muon_momentum,
        weight_decay=train_config.weight_decay_matrices,
        ns_steps=train_config.muon_ns_steps,
        scale_factor=train_config.muon_scale_factor,
    )

    adamw = AdamW(
        other_params,
        lr=train_config.adamw_lr,
        betas=train_config.adamw_betas,
        eps=train_config.adamw_eps,
        weight_decay=train_config.weight_decay_other,
    )

    return muon, adamw, matrix_params, other_params
