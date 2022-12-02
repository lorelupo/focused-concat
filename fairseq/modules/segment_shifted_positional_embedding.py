import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from .learned_positional_embedding import LearnedPositionalEmbedding
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

logger = logging.getLogger(__name__)

def shift_positions(shift, positions, nsents, seg_labels, padding_idx, incremental_state=None):
    if incremental_state is not None:
        # shifts = torch.zeros(positions.shape).to(dtype=positions.dtype)
        mask = positions.ne(padding_idx).int().T.to(device=seg_labels.device)
        shifts = (nsents-seg_labels.T.view(-1))*shift*mask
        if shifts[0]>shifts[1]:
            # in the decoder, the first token is the <END> and it needs no shift
            shifts[:] = 0
    else:
        mask = positions.ne(padding_idx).int()
        if torch.is_tensor(shift):
            shifts = (nsents.repeat(seg_labels.shape[1],1).T-seg_labels)*shift[:,None]*mask
        else:
            shifts = (nsents.repeat(seg_labels.shape[1],1).T-seg_labels)*shift*mask
        if shifts[0,0]>shifts[0,1]:
            # in the decoder, the first token is the <END> and it needs no shift
            shifts[:,0] = 0
    return positions.to(seg_labels.device) + shifts

class SegmentShiftedSinusoidalPositionalEmbedding(SinusoidalPositionalEmbedding):

    def forward(
        self,
        batch,
        nsents: Tensor,
        seg_labels: Tensor,
        shift: Tensor,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None
    ):
        """batch is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(batch)
        bsz, seq_len = bspair[0], bspair[1]
        max_shift = max(shift) if torch.is_tensor(shift) else shift
        max_pos = self.padding_idx + 1 + seq_len + max(nsents)*max_shift
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            positions = pos.expand(bsz) + self.padding_idx
            # shift positions after every <SEP>
            shifted_positions = shift_positions(
                shift, positions, nsents, seg_labels, self.padding_idx, incremental_state=incremental_state
            )
            return self.weights.index_select(0,shifted_positions).view(bsz,1,-1)    
        
        5# calculate positions
        positions = utils.make_positions(
            batch, self.padding_idx
        )
        # shift positions after every <SEP>
        shifted_positions = shift_positions(
            shift, positions, nsents, seg_labels, self.padding_idx
        )
        return (
            self.weights.index_select(0, shifted_positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

class SegmentShiftedLearnedPositionalEmbedding(LearnedPositionalEmbedding):

    def forward(
        self,
        batch: Tensor,
        nsents: Tensor,
        seg_labels: Tensor,
        shift: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """batch is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                positions = torch.zeros(
                    (1, 1), device=batch.device, dtype=batch.dtype
                ).fill_(int(self.padding_idx + batch.size(1)))
            else:
                positions = utils.make_positions(
                    batch, self.padding_idx
                )
        
        shifted_positions = shift_positions(
            shift, positions, nsents, seg_labels, self.padding_idx, incremental_state=None
        )
        
        return F.embedding(
            shifted_positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

def SegmentShiftedPositionalEmbedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        learned: bool = False,
):

    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = SegmentShiftedLearnedPositionalEmbedding(
            num_embeddings, embedding_dim, padding_idx
        )
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SegmentShiftedSinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, init_size=num_embeddings + padding_idx + 1,
        )
    return m
