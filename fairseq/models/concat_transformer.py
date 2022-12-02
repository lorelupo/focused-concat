# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, NamedTuple

import torch
from torch import Tensor

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    base_architecture,
    transformer_test,
    transformer_voita_fairseq,
    transformer_vaswani_wmt_en_fr
)
from fairseq.modules.segment_embedding import (
    SegmentEmbedding,
    PositionSegmentEmbedding,
)
from fairseq.modules.segment_shifted_positional_embedding import (
    SegmentShiftedPositionalEmbedding
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_SEG_PADDING_IDX = 0

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("nsents", Optional[Tensor]),  # B x 1
        ("po_segment_labels", Optional[Tensor]),  # B x T
    ],
)

@register_model('concat_transformer')
class ConcatTransformer(TransformerModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--segment-shifted-positions',
            action='store_true',
            help='shift positions of --position-shift (below) after the end of '
                'a segment of the concatenated input sequence, i.e., after a <SEP> token'
        )
        parser.add_argument(
            '--position-shift',
            type=int,
            default=0,
            help='if --segment-shifted-positions, shift positions '
                 'of --position-shift after the end of a segment of '
                 'the concatenated input sequence, i.e., after a <SEP> token.'
                 'If position_shift=-1, then shift is equal to the average length'
                 'of the sentences belonging to the concatenated source sequence.'
        )
        parser.add_argument(
            '--use-segment-emb',
            action='store_true',
            help='use segment embeddings, sinusoidal by default'
        )
        parser.add_argument(
            '--lrn-segment-emb',
            action='store_true',
            help='learn segment embeddings'
        )
        parser.add_argument(
            '--onehot-segment-emb',
            action='store_true',
            help='learn segment embeddings'
        )
        parser.add_argument(
            '--pse-segment-dim',
            type=int,
            default=0,
            help='fuse position and segment embeddings in a single vector'
                '(pse) of size embedding_dim. Segment indices are encoded'
                'in the last pse_segment_dim dimensions of such vector,'
                'while the remaining dimensions are dedicated to positions'
        )
        parser.add_argument(
            '--persistent-positions',
            action='store_true',
            help='add position embeddings to the input of every layer'
                ' instead of the first layer only'
        )
        parser.add_argument(
            '--persistent-segment-emb',
            action='store_true',
            help='add segment embeddings to the input of every layer'
                ' instead of the first layer only'
        )
        parser.add_argument(
            '--persistent-pse',
            action='store_true',
            help='add pse to the input of every layer'
                ' instead of the first layer only'
        )
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        if getattr(args, "seg_padding_idx", None) is None:
            args.seg_padding_idx = DEFAULT_SEG_PADDING_IDX

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        # build token embeddings
        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError(
                    "--share-all-embeddings requires a joined dictionary"
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )  

        # build (position-)segment embeddings
        if args.pse_segment_dim > 0:
            if not args.need_seg_label:
                raise ValueError(
                    "--pse-segment-dim>0 requires --need-seg-label to be True"
                    " in the doc2doc_translation task"
                )
            if not args.use_segment_emb:
                raise ValueError(
                    "--pse-segment-dim >0 requires --use-segment-emb"
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "PSE with different encoder-decoder options not available"
                ) # TODO(lo)
            if args.encoder_learned_pos != args.decoder_learned_pos:
                raise ValueError(
                    "PSE with different encoder-decoder options not available"
                ) # TODO(lo)
            if args.max_source_positions != args.max_target_positions:
                raise ValueError(
                    "PSE with different encoder-decoder options not available"
                ) # TODO(lo)
            # standalone position embeddings are not neede
            # args.no_token_positional_embeddings = True
            encoder_embed_segments = cls.build_position_segment_embedding(
                args,
                embedding_dim=args.encoder_embed_dim,
                num_pos=args.max_source_positions,
                lrn_pos=args.encoder_learned_pos,
                pos_padding_idx=encoder_embed_tokens.padding_idx,
                seg_padding_idx=args.seg_padding_idx
            )
        else:
            if args.use_segment_emb and not args.need_seg_label:
                raise ValueError(
                    "the use of segment embeddings requires --need-seg-label"
                    "to be True in the doc2doc_translation task"
                )
            encoder_embed_segments = cls.build_segment_embedding(
                args,
                embedding_dim=args.encoder_embed_dim,
                padding_idx=args.seg_padding_idx
            ) if args.use_segment_emb else None
        # segment embeddings are always shared between source and target
        decoder_embed_segments = encoder_embed_segments          

        # build main blocks
        encoder = cls.build_encoder(
            args, src_dict, encoder_embed_tokens, encoder_embed_segments
        )
        decoder = cls.build_decoder(
            args, tgt_dict, decoder_embed_tokens, decoder_embed_segments
        )
        return cls(args, encoder, decoder)
    
    @classmethod
    def build_position_segment_embedding(
        cls,
        args,
        embedding_dim,
        num_pos,
        lrn_pos,
        pos_padding_idx,
        seg_padding_idx
    ):
        return PositionSegmentEmbedding(
            embedding_dim=embedding_dim,
            pse_seg_dim=args.pse_segment_dim,
            num_pos=num_pos,
            num_seg=args.num_sent,
            lrn_pos=lrn_pos,
            lrn_seg=getattr(args, "lrn_segment_emb", False),
            onehot_seg=getattr(args, "onehot_segment_emb", False),
            pos_padding_idx=pos_padding_idx,
            seg_padding_idx=seg_padding_idx,
        )

    @classmethod
    def build_segment_embedding(cls, args, embedding_dim, padding_idx):
        return SegmentEmbedding(
                num_embeddings=args.num_sent,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                learned=getattr(args, "lrn_segment_emb", False),
                onehot=getattr(args, "onehot_segment_emb", False),
                )
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, embed_segments):
        return ConcatTransformerEncoder(
            args,
            src_dict,
            embed_tokens,
            embed_segments
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, embed_segments):
        return ConcatTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            embed_segments,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(
        self,
        src_tokens,
        nsents,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        src_segment_labels: Optional[Tensor] = None,
        po_segment_labels: Optional[Tensor] = None,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):

        encoder_out = self.encoder(
            src_tokens,
            nsents,
            return_all_hiddens=return_all_hiddens,
            src_segment_labels=src_segment_labels,
        )

        decoder_out = self.decoder(
            prev_output_tokens,
            po_segment_labels=po_segment_labels,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

class ConcatTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens, embed_segments):
        super().__init__(args, dictionary, embed_tokens)
        self.analyze_self_attn = bool(args.need_encoder_self_attn)
        self.num_sent = args.num_sent
        if args.pse_segment_dim > 0:
            # embed position and segment information in the same vector
            self.embed_pse = embed_segments
            self.persistent_pse = getattr(args, "persistent_pse", False)
            self.embed_segments = None
            self.persistent_segment_emb = False
            self.persistent_positions = False
        else:
            # embed position and segment information in different vectors 
            self.embed_pse = None
            self.persistent_pse = False
            self.embed_segments = embed_segments
            self.persistent_segment_emb = getattr(args, "persistent_segment_emb", False) if self.embed_segments is not None else False
            self.persistent_positions = getattr(args, "persistent_positions", False) if self.embed_positions is not None else False
            self.segment_shifted_positions = getattr(args, "segment_shifted_positions", False)
            self.position_shift = getattr(args, "position_shift", 0)
            if self.segment_shifted_positions:
                self.embed_positions = SegmentShiftedPositionalEmbedding(
                    num_embeddings=args.max_source_positions,
                    embedding_dim=embed_tokens.embedding_dim,
                    padding_idx=embed_tokens.padding_idx,
                    learned=args.encoder_learned_pos,
                )
            # logging relevant info
            if self.position_shift > 0:
                logger.info("shifting positions of %s units after every <SEP>", self.position_shift)
            elif self.position_shift == -1:
                logger.info("shifting positions of average sentence length after every <SEP>")
        
        # logging relevant info
        if self.persistent_pse:
            logger.info("pse are persistent througout encoder layers")
        if self.persistent_segment_emb:
            logger.info("segment embeddings are persistent througout encoder layers")
        if self.persistent_positions:
            logger.info("positions are persistent througout encoder layers")

    def forward(
        self,
        src_tokens,
        nsents,
        return_all_hiddens: bool = False,
        src_segment_labels: Optional[Tensor] = None,
        po_segment_labels: Optional[Tensor] = None,
        ):

        # embed tokens
        x = encoder_embedding = self.embed_scale * self.embed_tokens(src_tokens)

        if self.embed_pse is not None:
            # embed segments and positions in a single vector
            pse = self.embed_pse(src_tokens, src_segment_labels)
            x += pse
        else:
            # embed positions
            if self.embed_positions is not None:
                if self.segment_shifted_positions:
                    if self.position_shift == -1:
                        # shift = the average length of sentences belonging to the concatenated sequence
                        shift = (src_tokens.ne(self.padding_idx).sum(dim=1) / nsents).ceil().to(dtype=nsents.dtype)
                    else:
                        shift = self.position_shift
                    positions = self.embed_positions(
                        src_tokens,
                        nsents,
                        src_segment_labels,
                        shift
                    )
                else:
                    positions = self.embed_positions(src_tokens)
                x += positions
            # embed segments
            if self.embed_segments is not None:
                segment_emb = self.embed_segments(src_segment_labels)
                x += segment_emb

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        if self.persistent_pse:
            pse = pse.transpose(0, 1)
        else:
            if self.persistent_positions:
                positions = positions.transpose(0, 1)
            if self.persistent_segment_emb:
                segment_emb = segment_emb.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        # encoder layers
        encoder_states = [] if return_all_hiddens else None
        attns = [] if self.analyze_self_attn else None
        attn: Optional[Tensor] = None
        for idx, layer in enumerate(self.layers):
            if self.persistent_pse and idx > 0:
                # we add position-segment embeddings to the input of each layer
                x += pse
            else:
                if self.persistent_positions and idx > 0:
                    # we add positions to the input of each layer
                    x += positions
                if self.persistent_segment_emb and idx > 0:
                    # we add segment embeddings to the input of each layer
                    x += segment_emb

            if self.analyze_self_attn:
                x, attn = layer(x, encoder_padding_mask, need_attn=self.analyze_self_attn)
                attns.append(attn)
            else:
                x = layer(x, encoder_padding_mask, need_attn=self.analyze_self_attn)

            if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.analyze_self_attn:
            def analyze_attn(attns, segment_labels, src_tokens, layer="all", debug=False):
                if layer == "all":
                    # average attention over all layers
                    attn = torch.stack(attns,dim=0).mean(0)
                else:
                    # last layer attention
                    attn = attns[int(layer)]
                if debug:
                    # calculate metrics on uniform attention distribution
                    attn = (attn!=0)/(attn!=0).sum(2).unsqueeze(1)
                # compute mask
                mask_pad = (segment_labels!=0).to(dtype=int)
                # mask out attention from padding as query
                attn = attn * mask_pad.unsqueeze(2)
                # entropy
                token_entr = torch.special.entr(attn).sum(2)
                avg_sent_entr = token_entr.sum(1)/(token_entr!=0).sum(1)
                # compute context mask
                mask_ctx = (segment_labels==1).to(dtype=int)
                # only retain attention weights of current queries
                attn_curr = attn * mask_ctx.unsqueeze(2)
                # for each query, only retain the sum of the attention weights to current keys,
                # then average over each current query in the batch
                attn_curr_to_curr = (attn_curr * mask_ctx.unsqueeze(1)).sum(2)
                avg_attn_curr_to_curr = attn_curr_to_curr.sum(1)/(attn_curr_to_curr!=0).sum(1)
                return avg_attn_curr_to_curr, avg_sent_entr

            # analyze attention distribution at each layer
            for layer in ["all", 0, 1, 2, 3, 4, 5]:
                attn_curr_to_curr, avg_sent_entr = analyze_attn(attns, src_segment_labels, src_tokens, layer=layer)
                # log results
                for a,e in zip(attn_curr_to_curr,avg_sent_entr):
                    logger.info(f"Encoder {layer} layer cur2cur attn: {a.item()}")
                    logger.info(f"Encoder {layer} layer avg attn entropy: {e.item()}")

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,
            nsents=nsents,
            po_segment_labels=po_segment_labels
        )


    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):

        encoder_padding_mask: Optional[Tensor
                                      ] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out if encoder_out.encoder_out is None else
            encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask if encoder_padding_mask is None else
            encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding if encoder_embedding is None else
            encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        nsents = encoder_out.nsents
        if nsents is not None:
            nsents = nsents.index_select(0, new_order)

        po_segment_labels = encoder_out.po_segment_labels
        if po_segment_labels is not None:
            po_segment_labels = po_segment_labels.index_select(0, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            nsents=nsents,
            po_segment_labels=po_segment_labels,
        )

class ConcatTransformerDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, embed_segments, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.analyze_self_attn = bool(args.need_decoder_self_attn)
        self.analyze_cross_attn = bool(args.need_cross_attn)
        self.num_sent = args.num_sent
        if args.pse_segment_dim > 0:
            # embed position and segment information in the same vector
            self.embed_pse = embed_segments
            self.persistent_pse = getattr(args, "persistent_pse", False)
            self.embed_segments = None
            self.persistent_segment_emb = False
            self.persistent_positions = False
        else:
            # embed position and segment information in different vectors 
            self.embed_pse = None
            self.persistent_pse = False
            self.embed_segments = embed_segments
            self.persistent_segment_emb = getattr(args, "persistent_segment_emb", False) if self.embed_segments is not None else False
            self.persistent_positions = getattr(args, "persistent_positions", False) if self.embed_positions is not None else False
            self.segment_shifted_positions = getattr(args, "segment_shifted_positions", False)
            self.position_shift = getattr(args, "position_shift", 0)
            if self.segment_shifted_positions:
                self.embed_positions = SegmentShiftedPositionalEmbedding(
                    num_embeddings=args.max_target_positions,
                    embedding_dim=embed_tokens.embedding_dim,
                    padding_idx=embed_tokens.padding_idx,
                    learned=args.decoder_learned_pos,
                )
        
        if self.persistent_pse:
            logger.info("pse are persistent througout decoder layers")
        if self.persistent_segment_emb:
            logger.info("segment embeddings are persistent througout decoder layers")
        if self.persistent_positions:
            logger.info("positions are persistent througout decoder layers")

    def forward(
        self,
        prev_output_tokens,
        po_segment_labels: Optional[Tensor] = None,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_all_hiddens: bool = False,
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            po_segment_labels=po_segment_labels,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        po_segment_labels: Optional[Tensor] = None,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):

        if self.embed_pse is not None:
            # embed segments and positions in a single vector
            if po_segment_labels is None:
                po_segment_labels = label_segments(
                    batch=prev_output_tokens,
                    eos_idx=self.dictionary.eos_index,
                    end_idx=self.dictionary.index('<END>'),
                    nsents=encoder_out.nsents
                )
            pse = self.embed_pse(
                prev_output_tokens,
                po_segment_labels,
                incremental_state=incremental_state
            )
        else:
            # embed positions
            if self.embed_positions is not None:
                if self.segment_shifted_positions:
                    if po_segment_labels is None:
                        po_segment_labels = label_segments(
                            batch=prev_output_tokens,
                            eos_idx=self.dictionary.eos_index,
                            end_idx=self.dictionary.index('<END>'),
                            nsents=encoder_out.nsents
                        )
                    if self.position_shift == -1:
                        # shift = the average length of sentences belonging to the concatenated sequence
                        shift = (encoder_out.src_tokens.ne(self.padding_idx).sum(dim=1) / encoder_out.nsents).ceil().to(dtype=encoder_out.nsents.dtype)
                    else:
                        shift = self.position_shift
                    positions = self.embed_positions(
                        prev_output_tokens,
                        encoder_out.nsents,
                        po_segment_labels,
                        shift,
                        incremental_state=incremental_state
                    )
                else:
                    positions = self.embed_positions(
                        prev_output_tokens, incremental_state=incremental_state
                    )
            # embed segments
            if self.embed_segments is not None:
                if po_segment_labels is None:
                    po_segment_labels = label_segments(
                        batch=prev_output_tokens,
                        eos_idx=self.dictionary.eos_index,
                        end_idx=self.dictionary.index('<END>'),
                        nsents=encoder_out.nsents
                    )
                segment_emb = self.embed_segments(po_segment_labels)

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        # embed tokens
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if self.embed_pse is not None:
            # add position-segment embeddings
            x += pse
        else:
            # add position embeddings
            if self.embed_positions is not None:
                x += positions
            # embed segments
            if self.embed_segments is not None:
                # embed segments
                x += segment_emb

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        if self.persistent_pse:
            pse = pse.transpose(0, 1)
        else:
            if self.persistent_positions:
                positions = positions.transpose(0, 1)
            if self.persistent_segment_emb:
                segment_emb = segment_emb.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attns = [] if (self.analyze_self_attn or self.analyze_cross_attn) else None
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            if self.persistent_pse and idx > 0:
                # we add position-segment embeddings to the input of each layer
                x += pse
            else:
                if self.persistent_positions and idx > 0:
                    # we add positions to the input of each layer
                    x += positions
                if self.persistent_segment_emb and idx > 0:
                    # we add segment embeddings to the input of each layer
                    x += segment_emb
            if self.analyze_self_attn or self.analyze_cross_attn:
                x, layer_attn, _ = layer(
                    x,
                    encoder_out.encoder_out if encoder_out is not None else None,
                    encoder_out.encoder_padding_mask
                    if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)) or self.analyze_cross_attn,
                    need_self_attn=self.analyze_self_attn
                )
                attns.append(layer_attn)
            else:
                x, layer_attn, _ = layer(
                    x,
                    encoder_out.encoder_out if encoder_out is not None else None,
                    encoder_out.encoder_padding_mask
                    if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
            inner_states.append(x)
        
        if attns is not None:
            def analyze_attn(attns, segment_labels, tokens, layer="all", debug=False):
                if layer == "all":
                    # average attention over all layers
                    attn = torch.stack(attns,dim=0).mean(0)
                else:
                    # last layer attention
                    attn = attns[int(layer)]
                if debug:
                    # calculate metrics on uniform attention distribution
                    attn = (attn!=0)/(attn!=0).sum(2).unsqueeze(1)
                # set to zero attention of the END token on itself in the decoder
                attn[attn==1]=0
                # compute mask
                mask_pad = (segment_labels!=0).to(dtype=int)
                # mask out attention from padding as query
                attn = attn * mask_pad.unsqueeze(2)
                # entropy
                token_entr = torch.special.entr(attn).sum(2)
                avg_sent_entr = token_entr.sum(1)/(token_entr!=0).sum(1)
                if attn.shape[1]!=attn.shape[2]:
                    # we don't calculate curr2curr attn for cross attention
                    # sometimes this condition does not apply to cross attn
                    # for target and source have same length. It doesnt matter
                    return None, avg_sent_entr
                else:
                    # compute context mask
                    mask_ctx = (segment_labels==1).to(dtype=int)
                    # only retain attention weights of current queries
                    attn_curr = attn * mask_ctx.unsqueeze(2)
                    # for each query, only retain the sum of the attention weights to current keys,
                    # then average over each current query in the batch
                    attn_curr_to_curr = (attn_curr * mask_ctx.unsqueeze(1)).sum(2)
                    avg_attn_curr_to_curr = attn_curr_to_curr.sum(1)/(attn_curr_to_curr!=0).sum(1)
                    return avg_attn_curr_to_curr, avg_sent_entr

            # analyze attention distribution at each layer
            for layer in ["all", 0, 1, 2, 3, 4, 5]:
                attn_curr_to_curr, avg_sent_entr = analyze_attn(attns, po_segment_labels, prev_output_tokens, layer=layer)
                # log results
                if attn_curr_to_curr is not None:
                    for a,e in zip(attn_curr_to_curr,avg_sent_entr):
                        logger.info(f"Decoder {layer} layer cur2cur attn: {a.item()}")
                        logger.info(f"Decoder {layer} layer avg attn entropy: {e.item()}")
                else:
                    for e in avg_sent_entr:
                        logger.info(f"Decoder {layer} layer avg attn entropy: {e.item()}")


            # if layer_attn is not None and idx == alignment_layer:
            #     attn = layer_attn.float().to(x)

        # if attn is not None:
        #     if alignment_heads is not None:
        #         attn = attn[:alignment_heads]

        #     # average probabilities over heads
        #     attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

@torch.no_grad()
def label_segments(
    batch: Tensor,
    eos_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    nsents:  Optional[Tensor] = None,
):
    segment_labels = batch.new_zeros(batch[:, -1:].shape)
    eoss = (batch == eos_idx)
    for i, doc in enumerate(batch):
        # for each document, retrieve the position of eos tokens
        doc_breaks_idx = eoss[i].nonzero()
        # the current segment label is the last one minus
        # the number of segments already decoded
        n_doc_breaks = doc_breaks_idx.shape[0]
        if n_doc_breaks > 0:
            if doc_breaks_idx[-1] == doc.shape[0]-1:
                # the previous decoded token was eos,
                # therefore its label is still the same of the previous
                segment_labels[i] = nsents[i] - n_doc_breaks + 1 if (nsents[i] - n_doc_breaks >= 0) else 1
            else:
                segment_labels[i] = nsents[i] - n_doc_breaks if (nsents[i] - n_doc_breaks >= 1) else 1
        else:
            # no eos has been yet decoded
            if doc[-1] == end_idx:
                # only the first token is decoded
                # (the <END> of the previous sentence)
                segment_labels[i] = 1
            else:
                segment_labels[i] = nsents[i]

    return segment_labels
    
###############################################################################

@register_model_architecture("concat_transformer", "concat_test")
def concat_test(args):
    # concat args
    args.use_segment_emb = getattr(args, "use_segment_emb", True)
    args.lrn_segment_emb = getattr(args, "lrn_segment_emb", False)
    args.onehot_segment_emb = getattr(args, "onehot_segment_emb", False)
    args.persistent_positions = getattr(args, "persistent_positions", False)
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", False)
    args.persistent_pse = getattr(args, "persistent_pse", False)
    args.segment_shifted_positions = getattr(args, "segment_shifted_positions", True)
    # transformer args
    transformer_test(args)

# EN-RU #######################################################################

@register_model_architecture("concat_transformer", "concat_voita_fairseq")
def concat_voita_fairseq(args):
    # concat args
    args.use_segment_emb = getattr(args, "use_segment_emb", False)
    args.lrn_segment_emb = getattr(args, "lrn_segment_emb", False)
    args.onehot_segment_emb = getattr(args, "onehot_segment_emb", False)
    args.persistent_positions = getattr(args, "persistent_positions", False)
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", False)
    # transformer args
    transformer_voita_fairseq(args)

@register_model_architecture("concat_transformer", "concat_segshift")
def concat_segshift(args):
    # concat args
    args.segment_shifted_positions = getattr(args, "segment_shifted_positions", True)
    # transformer args
    concat_voita_fairseq(args)

@register_model_architecture("concat_transformer", "concat_segshift_persistent")
def concat_segshift_persistent(args):
    # concat args
    args.persistent_positions = getattr(args, "persistent_positions", True)
    # transformer args
    concat_segshift(args)

@register_model_architecture("concat_transformer", "concat_use_seg")
def concat_use_seg(args):
    # concat args
    args.use_segment_emb = getattr(args, "use_segment_emb", True)
    args.lrn_segment_emb = getattr(args, "lrn_segment_emb", False)
    args.onehot_segment_emb = getattr(args, "onehot_segment_emb", False)
    args.persistent_positions = getattr(args, "persistent_positions", False)
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", False)
    # other args
    transformer_voita_fairseq(args)

@register_model_architecture("concat_transformer", "concat_lrn_seg")
def concat_lrn_seg(args):
    # concat args
    args.use_segment_emb = getattr(args, "use_segment_emb", True)
    args.lrn_segment_emb = getattr(args, "lrn_segment_emb", True)
    args.onehot_segment_emb = getattr(args, "onehot_segment_emb", False)
    args.persistent_positions = getattr(args, "persistent_positions", False)
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", False)
    # other args
    transformer_voita_fairseq(args)

@register_model_architecture("concat_transformer", "concat_onehot_seg")
def concat_onehot_seg(args):
    # concat args
    args.use_segment_emb = getattr(args, "use_segment_emb", True)
    args.lrn_segment_emb = getattr(args, "lrn_segment_emb", False)
    args.onehot_segment_emb = getattr(args, "onehot_segment_emb", True)
    args.persistent_positions = getattr(args, "persistent_positions", False)
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", False)
    # other args
    transformer_voita_fairseq(args)

@register_model_architecture("concat_transformer", "concat_use_seg_asbase")
def concat_use_seg_asbase(args):
    concat_use_seg(args)

@register_model_architecture("concat_transformer", "concat_lrn_seg_asbase")
def concat_lrn_seg_asbase(args):
    concat_lrn_seg(args)

@register_model_architecture("concat_transformer", "concat_persistent_pos")
def concat_persistent_pos(args):
    args.persistent_positions = getattr(args, "persistent_positions", True)
    concat_voita_fairseq(args)

@register_model_architecture("concat_transformer", "concat_use_seg_persistent")
def concat_use_seg_persistent(args):
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", True)
    concat_use_seg(args)

@register_model_architecture("concat_transformer", "concat_lrn_seg_persistent")
def concat_lrn_seg_persistent(args):
    # concat args
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", True)
    concat_lrn_seg(args)

@register_model_architecture("concat_transformer", "concat_onehot_seg_persistent")
def concat_onehot_seg_persistent(args):
    # concat args
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", True)
    concat_onehot_seg(args)

@register_model_architecture("concat_transformer", "concat_use_pse_persistent")
def concat_use_pse_persistent(args):
    # concat args
    args.persistent_pse = getattr(args, "persistent_pse", True)
    concat_use_seg(args)

@register_model_architecture("concat_transformer", "concat_lrn_pse_persistent")
def concat_lrn_pse_persistent(args):
    # concat args
    args.persistent_pse = getattr(args, "persistent_pse", True)
    concat_lrn_seg(args)

@register_model_architecture("concat_transformer", "concat_onehot_pse_persistent")
def concat_onehot_pse_persistent(args):
    # concat args
    args.persistent_pse = getattr(args, "persistent_pse", True)
    concat_onehot_seg(args)

# EN-FR/DE #######################################################################

@register_model_architecture("concat_transformer", "concat_vaswani_wmt_en_fr")
def concat_vaswani_wmt_en_fr(args):
    # concat args
    args.use_segment_emb = getattr(args, "use_segment_emb", False)
    args.lrn_segment_emb = getattr(args, "lrn_segment_emb", False)
    args.onehot_segment_emb = getattr(args, "onehot_segment_emb", False)
    args.persistent_positions = getattr(args, "persistent_positions", False)
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", False)
    # other args
    transformer_vaswani_wmt_en_fr(args)

@register_model_architecture("concat_transformer", "concat_segshift_vaswani_wmt_en_fr")
def concat_segshift_vaswani_wmt_en_fr(args):
    # concat args
    args.segment_shifted_positions = getattr(args, "segment_shifted_positions", True)
    # transformer args
    concat_vaswani_wmt_en_fr(args)

@register_model_architecture("concat_transformer", "concat_use_seg_vaswani_wmt_en_fr")
def concat_use_seg_vaswani_wmt_en_fr(args):
    # concat args
    args.use_segment_emb = getattr(args, "use_segment_emb", True)
    # other args
    concat_vaswani_wmt_en_fr(args)

@register_model_architecture("concat_transformer", "concat_lrn_seg_vaswani_wmt_en_fr")
def concat_lrn_seg_vaswani_wmt_en_fr(args):
    # concat args
    args.use_segment_emb = getattr(args, "use_segment_emb", True)
    args.lrn_segment_emb = getattr(args, "lrn_segment_emb", True)
    # other args
    concat_vaswani_wmt_en_fr(args)

@register_model_architecture("concat_transformer", "concat_segshift_persistent_vaswani_wmt_en_fr")
def concat_segshift_persistent_vaswani_wmt_en_fr(args):
    # concat args
    args.persistent_positions = getattr(args, "persistent_positions", True)
    # transformer args
    concat_segshift_vaswani_wmt_en_fr(args)

@register_model_architecture("concat_transformer", "concat_use_seg_persistent_vaswani_wmt_en_fr")
def concat_use_seg_persistent_vaswani_wmt_en_fr(args):
    # concat args
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", True)
    # other args
    concat_use_seg_vaswani_wmt_en_fr(args)

@register_model_architecture("concat_transformer", "concat_lrn_seg_persistent_vaswani_wmt_en_fr")
def concat_lrn_seg_persistent_vaswani_wmt_en_fr(args):
    # concat args
    args.persistent_segment_emb = getattr(args, "persistent_segment_emb", True)
    # other args
    concat_lrn_seg_vaswani_wmt_en_fr(args)

@register_model_architecture("concat_transformer", "concat_use_pse_persistent_vaswani_wmt_en_fr")
def concat_use_pse_persistent_vaswani_wmt_en_fr(args):
    # concat args
    args.persistent_pse = getattr(args, "persistent_pse", True)
    concat_use_seg_vaswani_wmt_en_fr(args)

@register_model_architecture("concat_transformer", "concat_lrn_pse_persistent_vaswani_wmt_en_fr")
def concat_lrn_pse_persistent_vaswani_wmt_en_fr(args):
    # concat args
    args.persistent_pse = getattr(args, "persistent_pse", True)
    concat_lrn_seg_vaswani_wmt_en_fr(args)