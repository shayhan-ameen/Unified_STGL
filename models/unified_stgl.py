"""The unified spatio-temporal graph learning network.

    raw input --input encoder--> X in R^{T x N x F'}
    (X, A)    --ST-DEGaD------> A_D            (where to propagate)
    (X, A_D)  --ST-TAGCN------> Z_TAM          (what to emphasise)
    Z_TAM     --GAP, dropout, FC--> class logits

``forward`` returns the logits together with a dictionary of intermediate
quantities (A_D, ST-DA maps, modulation weights) used for visualisation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .graphs import build_graph, graph_name_for_task, normalize_adjacency
from .input_encoders import build_input_encoder
from .st_degad import STDEGaD
from .st_tagcn import STTAGCN


class UnifiedSTGL(nn.Module):
    """ST-DEGaD + ST-TAGCN classifier over a predefined graph prior.

    Parameters
    ----------
    encoder : task-specific input encoder producing ``[B, T, N, F']``.
    adjacency : binary predefined adjacency ``[N, N]`` (numpy or tensor).
    num_classes : number of target classes.
    cfg_model : the ``model`` section of the configuration.
    length : number of temporal positions T (needed only by the static-modulation ablation).
    """

    def __init__(self, encoder: nn.Module, adjacency, num_classes: int, cfg_model, length: int) -> None:
        super().__init__()
        adjacency = torch.as_tensor(np.asarray(adjacency), dtype=torch.float32)
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("adjacency_norm", normalize_adjacency(adjacency))
        num_nodes = adjacency.shape[0]
        in_dim = cfg_model.in_dim
        sg = cfg_model.sglayer
        self.use_stdegad = bool(cfg_model.use_stdegad)
        self.use_sttagcn = bool(cfg_model.use_sttagcn)

        self.encoder = encoder
        if self.use_stdegad:
            st, ce, dec = cfg_model.structural, cfg_model.contextual, cfg_model.decoder
            self.st_degad = STDEGaD(
                in_channels=in_dim,
                structural_channels=list(st.channels),
                contextual_dim=ce.dim,
                temporal_kernel=st.temporal_kernel,
                attention=st.attention,
                reduction=st.reduction,
                attention_dim=ce.attention_dim,
                hydra_d_state=ce.hydra.d_state,
                hydra_d_conv=ce.hydra.d_conv,
                hydra_expand=ce.hydra.expand,
                hydra_headdim=ce.hydra.headdim,
                radius=dec.radius,
                rel_dim=dec.rel_dim,
                gate_hidden=dec.gate_hidden,
                pair_hidden=dec.pair_hidden,
                sg_reduction=sg.reduction,
                sg_min_width=sg.min_width,
                use_checkpoint=bool(dec.get("checkpoint", False)),
            )
        else:
            self.st_degad = None

        tg = cfg_model.tagcn
        self.st_tagcn = STTAGCN(
            in_channels=in_dim,
            channels=list(tg.channels),
            temporal_kernel=tg.temporal_kernel,
            mod_radius=tg.mod_radius,
            transition_measure=tg.transition_measure,
            modulation=tg.modulation,
            cos_eps=float(tg.cos_eps),
            sg_reduction=sg.reduction,
            sg_min_width=sg.min_width,
            length=length,
            nodes=num_nodes,
            transition_adaptive=self.use_sttagcn,
        )
        self.dropout = nn.Dropout(float(cfg_model.dropout))
        self.classifier = nn.Linear(self.st_tagcn.out_channels, num_classes)

    def forward(
        self, inputs: torch.Tensor, prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``inputs``: task-specific tensor with leading ``[B, T, N, ...]`` axes.

        ``prior``: optional ``[B, T, N]`` input-level transition scores used by
        the ``flow_mag`` / ``landmark_vel`` ablations.
        """
        x = self.encoder(inputs)                                 # [B, T, N, F']
        aux: Dict[str, torch.Tensor] = {}
        if self.st_degad is not None:
            adj_dynamic, degad_aux = self.st_degad(x, self.adjacency_norm)
            aux.update(degad_aux)
        else:
            adj_dynamic = self.adjacency_norm.unsqueeze(0).expand(x.shape[0], -1, -1)
        aux["adjacency_dynamic"] = adj_dynamic
        z, tagcn_aux = self.st_tagcn(x, adj_dynamic, prior)
        aux.update(tagcn_aux)
        pooled = z.mean(dim=(1, 2))                              # global average pooling over T and N
        logits = self.classifier(self.dropout(pooled))
        return logits, aux


def build_model(cfg) -> UnifiedSTGL:
    """Instantiate the network described by a loaded configuration."""
    task = cfg.data.task
    adjacency = build_graph(graph_name_for_task(task, cfg.data.dataset))
    if adjacency.shape[0] != cfg.data.num_nodes:
        raise ValueError(f"Graph has {adjacency.shape[0]} nodes but data.num_nodes = {cfg.data.num_nodes}")
    length = sequence_length(cfg)
    encoder = build_input_encoder(cfg)
    return UnifiedSTGL(encoder, adjacency, cfg.data.num_classes, cfg.model, length)


def sequence_length(cfg) -> int:
    """Number of temporal positions T for the configured task."""
    task = cfg.data.task
    if task == "fer":
        return int(cfg.data.num_keyframes)
    if task == "esd":
        return int(cfg.data.num_windows)
    if task == "har":
        return int(cfg.data.num_frames)
    raise ValueError(f"Unknown task '{task}'")
