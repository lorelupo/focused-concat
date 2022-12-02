from typing import Optional
import bisect
import logging
import numpy as np
import torch

from fairseq.data import (
    data_utils
)
from . import FairseqDataset

logger = logging.getLogger(__name__)

def collate(
            samples,
            num_sent,
            pad_idx,
            end_idx,
            left_pad_source=True,
            left_pad_target=False,
            input_feeding=True,
            id_to_net_input=False,
            pad_to_length=None,
            pad_to_multiple=1,
            need_seg_label=False
        ):
            
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            end_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    id = torch.LongTensor([s['id'] for s in samples])

    src_tokens = merge('source',
                       left_pad=left_pad_source,
                       pad_to_length=pad_to_length['source']
                       if pad_to_length is not None else None)
    
    src_lengths = torch.LongTensor(
        [s['source'].ne(pad_idx).long().sum() for s in samples])

    nsents = torch.LongTensor([s['nsents'] for s in samples])

    # sort by descending source length
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    nsents = nsents.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge(
            'target',
            left_pad=left_pad_target,
            pad_to_length=pad_to_length['target']
            if pad_to_length is not None else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([
            s['target'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get('prev_output_tokens', None) is not None:
            prev_output_tokens = merge('prev_output_tokens',
                                       left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length['target']
                if pad_to_length is not None else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        'id': id,
        'ndocs': len(samples), # not accessed so far
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'nsents': nsents,
        },
        'target': target,
    }

    if id_to_net_input:
        batch['net_input']['id'] = id
        batch['net_input']['sort_order'] = sort_order

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens.index_select(0, sort_order)

    if need_seg_label:
        src_sents_len = data_utils.collate_tokens(
                [s['src_sents_len'] for s in samples],
                pad_idx=0,
                left_pad=left_pad_source,
                pad_to_length=pad_to_length,
                pad_to_multiple=pad_to_multiple,
            ).index_select(0, sort_order)

        batch['net_input']['src_segment_labels'] = label_segments(
            batch=src_tokens,
            sents_len=src_sents_len,
            left_pad=True,
            padding_idx=pad_idx,
        )

        if prev_output_tokens is not None:
            tgt_sents_len = data_utils.collate_tokens(
                    [s['tgt_sents_len'] for s in samples],
                    pad_idx=0,
                    left_pad=left_pad_source,
                    pad_to_length=pad_to_length,
                    pad_to_multiple=pad_to_multiple,
                )
            # batch['tgt_sents_len'] = tgt_sents_len.index_select(0, sort_order)
            tgt_sents_len[:,-1] -= 1
            po_sents_len = tgt_sents_len.index_select(0, sort_order)

            batch['net_input']['po_segment_labels'] = label_segments(
                batch=batch['net_input']['prev_output_tokens'],
                sents_len=po_sents_len,
                left_pad=False,
                padding_idx=pad_idx,
            )

    return batch


def label_segments(
    batch: torch.Tensor,
    sents_len: torch.Tensor,
    left_pad: Optional[bool] = False,
    padding_idx: Optional[int] = None,
):
    """
    Assign a segment label to each non-padding token of each documment
    in the batch. The last (current) sentence takes label 1,
    while the others are assigned incremental labels. Paddings take label 0.

    Args:
        batch: the batch of documents of shape `batch, tgt_len`
        eos_idx: the index of the 'end of sentence' special token

    """

    segment_labels = batch.new_zeros(batch.shape)
    mask = batch.ne(padding_idx).int()
    if left_pad:
        for i in range(batch.shape[0]):
            right=sents_len[i,-1]
            for l, len in enumerate(reversed(sents_len[i,:-1])):
                left = right + len
                segment_labels[i,-left:-right] = l+1
                right = left
    else:
        for i in range(batch.shape[0]):
            right=(mask[i]==0).sum()
            for l, len in enumerate(reversed(sents_len[i])):
                left = right + len
                segment_labels[i,-left:-right] = l
                right = left

    segment_labels = (segment_labels + 1) * mask

    return segment_labels


class Sent2DocDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets. 
    Takes a pair of sentence-level (src and tgt) datasets and builds document pairs.
    Document indices are computed and stored according to mode and 'num_tok' or 'num_sent'.
    Documents are built on the fly to avoid memory consumption.
    An <END> token marks the end of a document by default. 

    Args:
        src (torch.utils.data.Dataset): (sentence by sentence) source dataset to strip into docs
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset): (sentence by sentence) target dataset to strip into docs
        tgt_sizes (List[int]): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary): target vocabulary
        doc_heads (List[int]): ID of the first sentence for each document in the dataset

        mode (str, optional): Mode used for breaking docs. Values can be one of:
            - TODO: 'none': create equally sized contiguous docs up to 'num_tok'
            -'block':       create contiguous source docs up to 'num_tok'
                            target docs are created as needed to pair the source
            -'n2n_block':   create contiguous source docs with up to 'num_sent'
                            target docs are created as needed to pair the source
            -'slide_n2one': create source docs by concatenating 
                            up to 'num_sent' sentences: the current sentence
                            + 'num_sent'-1 past sentences, if available.
                            Source documents slide 1 sentence to the right,
                            the target is the current sentence only
            -'slide_n2n':   create sliding source and target docs by concatenating
                            up to 'num_sent' sentences (if available)
                            i.e docs slide 1 sentence to the right on both source
                            and target side
            -'slide_block': create sliding source docs up to 'num_tok'
                            i.e docs slide 1 sentence to the right
                            and as many sentences to the left such that
                            doc sizes are always as close as possible to num_tok.
                            Target docs are created as needed to pair the source
            - 'complete':   return whole docs  
            IMPORTANT NOTES:  
                * docs always contain complete sentences EXCEPT on 'none' mode
                * document boundaries are never crossed
        num_tok (int):   max length of documents in number of tokens
                            used by 'none', 'block' and 'slide_block' modes
                            (default: 1000)
        num_sent (int, optional): max length of documents in number of sentences
                            used by 'n2n_block', 'slide_n2one' and 'slide_n2n' modes
                            (default: None)
        add_doc_markup (bool, optional): if set, adds document markup (<BEG>, <BRK>, <CNT>, <END>)
            to each document (i.e. element) in the dataset 
            (default: False)

        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
    """

    def __init__(
        self, split,
        src, src_sizes, src_dict,
        tgt, tgt_sizes, tgt_dict,
        doc_heads,
        mode=None,
        num_tok=1000,
        num_sent=None, 
        add_doc_markup=False,
        left_pad_source=True,
        left_pad_target=False,
        pad_to_multiple=1,
        shuffle=True,
        current_tgt_only=False,
        input_feeding=True,
        need_seg_label=False,
    ):
        assert len(src) == len(tgt), "Source and target must contain the same number of examples"
        if mode == 'slide_block' and num_tok < max(src_sizes):
            raise ValueError('minimum num_tok for \'slide_block\' mode is {}'.format(max(src_sizes)))

        self.split = split
        self.src = src
        self.src_sizes = src_sizes
        self.tgt = tgt
        self.tgt_sizes = tgt_sizes
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.doc_heads = doc_heads
        self.mode = mode
        self.num_tok = num_tok
        self.num_sent = num_sent
        self.add_doc_markup = add_doc_markup
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.pad_to_multiple = pad_to_multiple
        self.shuffle = shuffle
        self.current_tgt_only = current_tgt_only
        self.input_feeding = input_feeding
        self.need_seg_label = need_seg_label

        self.doc_indices = [] # [[int, int]] start, end indices for each doc
        self.sents_per_doc = [] # [int] num of sents in each doc
        self.is_doc_split = [] # [bool] whether is a doc part or a whole doc

        # keep only realistic head indices (i.e. within corpus boundaries)
        corpus_size = len(src_sizes)
        # find index of the first out of boud head
        oob_index = bisect.bisect_left(self.doc_heads, corpus_size)
        # delete out of bound heads
        self.doc_heads = self.doc_heads[:oob_index]
        # add fake head at the end, useful for the following
        self.doc_heads.append(corpus_size)

        # make doc magic happen
        if self.mode == 'slide_n2one' or self.mode == 'slide_n2n' or self.mode == 'slide_block' \
            or self.mode is None:
            self.slice_slide_indices()
        else:
            self.slice_block_indices()

        self.doc_src_sizes, self.doc_tgt_sizes = self.get_sizes()
        self.doc_src_sizes = np.array(self.doc_src_sizes)
        self.doc_tgt_sizes = np.array(self.doc_tgt_sizes)
        self.sizes = np.vstack((self.src_sizes, self.tgt_sizes)).T

        # Print the sizes of the examples
        # from collections import Counter
        # np.set_printoptions(threshold=np.inf)
        # ord_doc_src_sizes = self.doc_src_sizes[np.argsort(self.doc_src_sizes)]
        # ord_doc_src_sizes = sorted(Counter(ord_doc_src_sizes).items())
        # logger.info('doc src sizes counts:{}'.format(ord_doc_src_sizes))

    def __getitem__(self, index):
        start, end = self.doc_indices[index][0], self.doc_indices[index][1]

        src_item = torch.cat([self.src[i] for i in range(start, end)])
        #######################################################################
        # src_item = []
        # segment_labels = []
        # for n, i in enumerate(range(start,end)):
        #     src_item.append(self.src[i])
        #     segment_labels.append([end-start-n]*(len(self.src[i]))) 
        # src_item = torch.cat(src_item)
        # segment_labels = torch.tensor(segment_labels)
        
        #######################################################################
        src_sents_len = torch.tensor([len(self.src[i]) for i in range(start, end)])
        if self.current_tgt_only:
            tgt_item = self.tgt[index]
            tgt_sents_len = None
        else:
            tgt_item = torch.cat([self.tgt[i] for i in range(start, end)])
            tgt_sents_len = torch.tensor([len(self.tgt[i]) for i in range(start, end)])

        # remove last eos token from doc and replace it by custom END token
        src_item = src_item[:-1] if src_item[-1] == self.src_dict.eos() else src_item
        tgt_item = tgt_item[:-1] if tgt_item[-1] == self.tgt_dict.eos() else tgt_item
        src_item = self.append_token_doc(src_item, token='<END>', dic=self.src_dict)
        tgt_item = self.append_token_doc(tgt_item, token='<END>', dic=self.tgt_dict)

        nsents = self.sents_per_doc[index]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'nsents': nsents,
            'src_sents_len': src_sents_len,
            'tgt_sents_len': tgt_sents_len
        }

        return example
    
    def __len__(self):
        return len(self.doc_indices)

    def slice_slide_indices(self):
        for h_idx, head in enumerate(self.doc_heads[:-1]):
            next_head = self.doc_heads[h_idx+1]

            if self.mode == 'slide_n2one' or self.mode == 'slide_n2n':
                for idx in range(head, next_head):
                    if idx - (self.num_sent-1) >= head:
                        self.doc_indices.append([idx-(self.num_sent-1), idx+1])
                        self.sents_per_doc.append(self.num_sent)
                    else:
                        self.doc_indices.append([head, idx+1])
                        self.sents_per_doc.append(idx-head+1)
        
            if self.mode == 'slide_block' or self.mode is None:
                for idx in range(head, next_head):
                    if idx == head:
                        self.doc_indices.append([idx, idx+1])
                        self.sents_per_doc.append(1)
                    else:
                        csum = 0
                        reached_num_tok = False
                        for i in range(idx, head-1, -1):
                            csum = csum + self.src_sizes[i]
                            if csum > self.num_tok:
                                reached_num_tok = True
                                break
                        if reached_num_tok:
                            self.doc_indices.append([i+1, idx+1])
                            self.sents_per_doc.append(idx-i+1)
                        else:
                            self.doc_indices.append([head, idx+1])
                            self.sents_per_doc.append(idx-head+1)

    def slice_block_indices(self):
        # build documents using doc heads
        for h_idx, head in enumerate(self.doc_heads[:-1]):
            next_head = self.doc_heads[h_idx+1]

            stop_sample, from_split = self._slice_block_indices(
                head, next_head
            )
            while from_split:
                stop_sample, from_split = self._slice_block_indices(
                    stop_sample, next_head, from_split
                )
    
    def _slice_block_indices(self, head, next_head, from_split=False):
        if self.mode == 'block':
            src_sizes_csum = self.src_sizes[head:next_head].cumsum()
            
            n_samples = (src_sizes_csum <= self.num_tok).argmin()
            if n_samples == 0:
                n_samples = next_head - head
            stop_sample = head + n_samples

        elif self.mode == 'n2n_block':
            if self.num_sent < next_head - head:
                n_samples = self.num_sent
            else:
                n_samples = next_head - head
            stop_sample = head + n_samples

        else: # self.mode == 'complete'
            stop_sample = next_head
            n_samples = next_head - head
        
        if stop_sample >= next_head: # doc fits perfectly or is shorter then max block size
            self.doc_indices.append([head, next_head])
            from_split = False      
        elif stop_sample < next_head: # doc will be split into subdocs
            self.doc_indices.append([head, stop_sample])
            from_split = True
        else:
            raise ValueError('stop_sample has a weird value')

        self.is_doc_split.append(from_split)
        self.sents_per_doc.append(n_samples)    
              
        return stop_sample, from_split

    def get_sizes(self):
        szs, tsz = [], []
        if self.mode != 'slide_n2one':
            for i, dii in enumerate(self.doc_indices):
                start, end = dii[0], dii[1]
                szs.append(sum(self.src_sizes[start:end]))
                tsz.append(sum(self.tgt_sizes[start:end]))
        else:
            tsz = self.tgt_sizes
            for i, dii in enumerate(self.doc_indices):
                start, end = dii[0], dii[1]
                szs.append(sum(self.src_sizes[start:end]))
    
        return szs, tsz

    def append_token_doc(self, doc, token, dic):
        assert isinstance(token, str)
        token_idx = dic.index(token)
        doc = torch.cat([doc, torch.LongTensor([token_idx])])
        return doc
    
    def prepend_token_doc(self, doc, size=None, token=None):
        if token is not None:
            assert isinstance(token, str)
            tok = self.src_dict.index(token) if self.src_dict else self.tgt_dict.index(token)
            doc = torch.cat([torch.LongTensor([tok]), doc])
            if size is not None:
                size += 1
        return doc, size

    # returns the actual num tokens of the concat sequence
    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.doc_src_sizes[index], self.doc_tgt_sizes[index])

    # maintained so we can filter sentences based on the size of the current sentence
    # def size(self, index):
    #     """Return an example's size as a float or tuple. This value is used when
    #     filtering a dataset with ``--max-positions``."""
    #     return (self.src_sizes[index],
    #             self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.doc_src_sizes[index], self.doc_tgt_sizes[index])

    def max_src_size(self):
        """Return the max src size in the dataset"""
        return max(self.doc_src_sizes)

    def max_tgt_size(self):
        """Return the max tgt size in the dataset"""
        return max(self.doc_tgt_sizes)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
         
        # sort by target length, then source length
        if self.doc_tgt_sizes is not None:
            indices = indices[np.argsort(self.doc_tgt_sizes[indices],
                                            kind='mergesort')]
        return indices[np.argsort(self.doc_src_sizes[indices],
                                    kind='mergesort')]
    
    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.
        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.doc_src_sizes, self.doc_tgt_sizes, indices, max_sizes,
        )
        
    # used as collate_fn by get_batch_iterator() in fairseq_task
    def collater(self, samples, pad_to_length=None):
        return collate(
            samples,
            num_sent=self.num_sent,
            pad_idx=self.src_dict.pad(),
            end_idx=self.tgt_dict.index('<END>'),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
            need_seg_label=self.need_seg_label,
        )