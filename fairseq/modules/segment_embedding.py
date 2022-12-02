import logging
from typing import Dict, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from .positional_embedding import PositionalEmbedding

logger = logging.getLogger(__name__)

def SegmentEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: Optional[bool] = False,
    onehot: Optional[bool] = False,
    ):
    if learned:
        if onehot:
            raise ValueError(
                "learned segment embeddings can't be also one-hot. \
                    The two options are mutually exclusive."
            ) 
        logger.info("using learned segment embeddings")
        m = nn.Embedding(
            num_embeddings=num_embeddings + padding_idx + 1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
    elif onehot:
        logger.info("using one-hot segment embeddings")
        m = OneHotSegmentEmbedding(
            num_embeddings=num_embeddings + padding_idx + 1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
    else:
        logger.info("using sinusoidal segment embeddings")
        m = SinusoidalSegmentEmbedding(
            init_size=num_embeddings + padding_idx + 1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
    return m

class OneHotSegmentEmbedding(nn.Module):
    """This module produces one-hot positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = OneHotSegmentEmbedding.get_embedding(
            num_embeddings, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        # weights are a simply a diagonal matrix of ones
        emb = torch.eye(num_embeddings-padding_idx-1, embedding_dim, dtype=torch.float)
        # add pad embedding
        emb = torch.cat([torch.zeros(padding_idx+1, embedding_dim), emb], dim=0)

        return emb

    def forward(
        self,
        indices
    ):
        """Indices are expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(indices)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + indices.max()
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = OneHotSegmentEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, indices.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, indices.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

class SinusoidalSegmentEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalSegmentEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        indices
    ):
        """Indices are expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(indices)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + indices.max()
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalSegmentEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, indices.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, indices.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

class PositionSegmentEmbedding(nn.Module):

    def __init__(
        self,
        embedding_dim,
        pse_seg_dim,
        num_pos,
        num_seg,
        lrn_pos,
        lrn_seg,
        onehot_seg,
        pos_padding_idx,
        seg_padding_idx,
        max_norm: Optional[float] = None,
        norm_type: float = 2.,
        scale_grad_by_freq: bool = False,
    ):
        super().__init__()

        logger.info("using position-segment embeddings")
        self.num_pos = num_pos
        self.num_seg = num_seg
        self.seg_padding_idx = seg_padding_idx
        self.onnx_trace = False
        # attributes for F.embedding, taken from nn.Embedding
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.lrn_pos = lrn_pos
        self.lrn_seg = lrn_seg
        # needed to move embeddings to GPU
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

        # instantiate position embedding
        self.pe = PositionalEmbedding(
            num_embeddings=num_pos,
            embedding_dim=embedding_dim-pse_seg_dim,
            padding_idx=pos_padding_idx,
            learned=lrn_pos
            )
        # instantiate segment embedding
        self.se = SegmentEmbedding(
            num_embeddings=num_seg,
            embedding_dim=pse_seg_dim,
            padding_idx=seg_padding_idx,
            learned=lrn_seg,
            onehot=onehot_seg,
            )
        # extract embedding weights
        self.w_pe = self.pe.weight[pos_padding_idx+1:] if lrn_pos else self.pe.weights[pos_padding_idx+1:] # no pad emb
        self.w_se = self.se.weight[seg_padding_idx+1:] if lrn_seg else self.se.weights[seg_padding_idx+1:] # no pad emb
        # repeat position and segment embs to obtain all possible combinations
        # self.w_pe_rep = self.w_pe.unsqueeze(0).expand(self.w_se.shape[0],-1,-1).reshape(-1,self.w_pe.shape[1])
        self.w_pe_rep = self.w_pe.repeat(self.w_se.shape[0],1)
        # backward-compatible equivalent of
        # self.w_se.repeat_interleave(self.w_pe.shape[0],0) :
        l = []
        for w in self.w_se: 
            l.append(w.repeat(self.w_pe.shape[0],1))
        self.w_se_rep = torch.cat(l,dim=0) 
        # fusion embeddings in a single matrix with all combinations
        self._weight = torch.cat((self.w_pe_rep,self.w_se_rep),dim=1)
        # add padding embedding
        self.weight = torch.cat((
            torch.zeros((pos_padding_idx+1, self._weight.shape[1])),
            self._weight
        ))
        self.padding_idx = pos_padding_idx
        self.num_embeddings = num_seg * num_pos + self.padding_idx + 1

        # define max positions
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings
        
    def forward(
        self,
        batch: Tensor,
        seg_labels: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        # needed to move embeddings to GPU
        self.weight = self.weight.to(self._float_tensor)

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = torch.zeros(
                    (1, 1), device=batch.device, dtype=batch.dtype
                ).fill_(int(self.padding_idx + batch.size(1)))
            else:
                positions = utils.make_positions(
                    batch, self.padding_idx, onnx_trace=self.onnx_trace
                )

        idx = torch.where(
            positions==1,
            positions,
            positions+self.num_pos*(seg_labels-(self.seg_padding_idx+1))
        )

        emb = F.embedding(
            idx,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
        )
            
        return emb
