import os
import itertools
import logging
import math
import numpy as np

import torch

from fairseq import metrics, utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data.sent2doc_dataset import Sent2DocDataset
from fairseq.data import (ConcatDataset, data_utils, indexed_dataset)


logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

def load_langpair_dataset(
    data_path, split, src, src_dict, tgt, tgt_dict, mode, num_tok, num_sent,
    add_doc_markup, combine, dataset_impl, upsample_primary, left_pad_source,
    left_pad_target, pad_to_multiple, shuffle, current_tgt_only, input_feeding,
    need_seg_label,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(
            data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang)
        )
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    def load_doc_heads(doc_heads_path):
        heads = []
        with open(doc_heads_path) as infile:
            for line in infile:
                heads.append(int(line.strip()))
        if heads[0] == 0:
            return heads
        elif heads[0] == 1:
            return [h - 1 for h in heads]
        else:
            raise ValueError(
                'The first head should be the first line of a document, with id=0 or id=1'
            )

    # load indexed datasets and document heads depending on prefix
    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(
                data_path, '{}.{}-{}.'.format(split_k, src, tgt)
            )
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(
                data_path, '{}.{}-{}.'.format(split_k, tgt, src)
            )
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    'Dataset not found: {} ({})'.format(split, data_path)
                )

        # load src
        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        src_datasets.append(src_dataset)

        # load tgt
        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        # load doc heads and add fictional last doc head
        doc_heads = load_doc_heads(prefix + src + '.heads')
        doc_heads.append(len(src_datasets[-1]))

        logger.info(
            '{} {} {}-{} {} examples'.format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0
    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    return Sent2DocDataset(
        split,
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset.sizes,
        tgt_dict,
        doc_heads,
        mode=mode,
        num_tok=num_tok,
        num_sent=num_sent,
        add_doc_markup=add_doc_markup,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        pad_to_multiple=pad_to_multiple,
        shuffle=shuffle,
        current_tgt_only=current_tgt_only,
        input_feeding=input_feeding,
        need_seg_label=need_seg_label,
    )


@register_task('doc2doc_translation')
class Doc2DocTranslation(TranslationTask):
    """
    Translate documents from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    """
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.mode = args.mode
        self.num_tok = args.num_tok
        self.num_sent = args.num_sent
        self.context_discount = args.context_discount
        self.original_loss_for_stopping =  args.original_loss_for_stopping
        self.args.need_seg_label = utils.eval_bool(self.args.need_seg_label)
        logger.info('translating with {} mode'.format(self.mode))
        if self.args.need_seg_label:
            logger.info('adding segment labels to model input')
        if self.context_discount < 0:
            logger.info('Invalid context discount, setting its value back to 1, but still calculating current loss.')

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--mode',
            type=str,
            default=None,
            help=
            'one of: none, block, n2n_block, slide_n2one, slide_n2n, slide_block, complete'
        )
        parser.add_argument(
            '--num-tok',
            type=int,
            metavar='N',
            default=1000,
            help='max length of documents in number of tokens'
        )
        parser.add_argument(
            '--num-sent',
            default=None,
            type=int,
            metavar='N',
            help='max length of documents in number of sentences'
        )
        parser.add_argument(
            '--add-doc-markup',
            action='store_true',
            help=
            'add markup to documents i.e. special symbols <BEG>, <BRK>, <CNT>, <END>'
        )
        parser.add_argument(
            '--remove-markup-from-output',
            action='store_true',
            help='remove special symbols from generator output'
        ) # to use only at generation if --add-doc-markup was used    
        parser.add_argument(
            '--need-seg-label',
            default='False', type=str, metavar='BOOL',
            help='Compute segment labels for current segments (label=1) and context (label=2,3,...).'
        )
        parser.add_argument(
            '--context-discount',
            default=1,
            type=float,
            metavar='D',
            help=
            'discount weight to apply to the loss generated by the context sentences, 1 means no discount'
        )
        parser.add_argument(
            '--original-loss-for-stopping',
            action='store_true',
            help='Use the non-discounted loss as a stopping criterion even if context loss is being discounted.'
        )
        parser.add_argument(
            '--need-encoder-self-attn',
            type=int,
            default=0,
            help='if >1, retrieve encoder attention weights to analyze them'
        )
        parser.add_argument(
            '--need-decoder-self-attn',
            type=int,
            default=0,
            help='if >1, retrieve decoder attention weights to analyze them'
        )
        parser.add_argument(
            '--need-cross-attn',
            type=int,
            default=0,
            help='if >1, retrieve decoder attention weights to analyze them'
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)
        
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                'Could not infer language pair, please provide it explicitly'
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        # add markup tokens to dictionaries
        src_dict.add_symbol('<END>')  # END of concatenated sequence
        tgt_dict.add_symbol('<END>')  # or document
        if args.add_doc_markup:
            src_dict.add_symbol('<BEG>')  # BEG of document
            tgt_dict.add_symbol('<BEG>')
            src_dict.add_symbol('<BRK>')  # End of the concatenated sequence,
            tgt_dict.add_symbol('<BRK>')  # but the document is not ended
            src_dict.add_symbol('<CNT>')  # Continuation of a document that did
            tgt_dict.add_symbol('<CNT>')  # not end in the previous sequence

        logger.info(
            '[{}] dictionary: {} types'.format(args.source_lang, len(src_dict))
        )
        logger.info(
            '[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict))
        )

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        # check for correct combinations of loading parameters
        if (self.mode == 'none' or self.mode == 'block' or self.mode == 'slide_block') \
            and self.num_tok is None:
            raise ValueError(
                'Please specify \'num_tok\' for modes none, block and slide_block'
            )

        if (self.mode == 'n2n_block' or self.mode == 'slide_n2one' or self.mode == 'slide_n2n') \
            and self.num_sent is None:
            raise ValueError(
                'Please specify \'num_sent\' for modes n2n_block, slide_n2one, slide_n2n'
            )
        
        # determine wether the target example consists
        # in the current sentence only or its concatenation with context
        generating_n_sentences=(
            not getattr(self.args, "score_reference", False) and \
                (
                    split=='test' or \
                        self.args.gen_subset == 'valid' or \
                            (split=='valid' and self.args.eval_bleu)
                )
        )
        current_tgt_only=self.mode == 'slide_n2one' or \
            (self.mode == 'slide_n2n' and generating_n_sentences)

        self.datasets[split] = load_langpair_dataset(
            data_path, split,
            src, self.src_dict,
            tgt, self.tgt_dict,
            mode=self.mode,
            num_tok=self.num_tok,
            num_sent=self.num_sent,
            add_doc_markup=self.args.add_doc_markup,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            pad_to_multiple=self.args.required_seq_len_multiple,
            shuffle=(split!='test' and self.args.gen_subset != "valid"),
            current_tgt_only=current_tgt_only,
            input_feeding=(not generating_n_sentences),
            need_seg_label=self.args.need_seg_label,
        )
        logger.info(
            '{} {}-{} {} documents made from loaded examples'.format(
                split, src, tgt, len(self.datasets[split])
            )
        )
        logger.info(
            'max source size for {} split: {}'.format(
                split, self.datasets[split].max_src_size()
            )
        )
        logger.info(
            'max target size for {} split: {}'.format(
                split, self.datasets[split].max_tgt_size()
            )
        )

    def build_loss_weights(self, target, po_segment_labels, cd=None):
        """
        Build a tensor of discount weights to be applied to the loss generated
        by tokens belonging to context sentences.
        """
        weights = (po_segment_labels==1).long()
        weights[:,0] = 0
        first_current_target = weights.argmax(1) - 1
        row_idx = torch.arange(len(weights))
        weights[row_idx,first_current_target] = 1
        if self.context_discount < 1:
            # generate discount weights for the loss generated by context tokens
            weights = weights*(1-self.context_discount) + self.context_discount
        elif self.context_discount > 1:
            # generate multipliers for the loss generated by current tokens
            weights = weights*(self.context_discount-1) + 1

        return weights

    def train_step(
        self,
        sample,
        model,
        criterion,
        optimizer,
        update_num,
        ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            if self.context_discount == 1:
                # no context discounting
                loss, sample_size, logging_output = criterion(model, sample)
            else:
                _loss, sample_size, logging_output = criterion(
                    model, sample, reduce=False
                )
                # generate weights to discount the loss generated by context
                weights = self.build_loss_weights(sample['target'], sample['net_input']['po_segment_labels'])
                logging_output['full_loss']=logging_output['loss']
                # calculate weighted loss
                if self.context_discount < 0:
                    # cd<0 is useful for analysing the current vs context loss when there is no discount
                    loss = torch.matmul(torch.ones_like(weights).view(-1, 1).T, _loss).squeeze(-1).squeeze(-1)
                else:
                    loss = torch.matmul(weights.view(-1, 1).T, _loss).squeeze(-1).squeeze(-1)
                
                logging_output['current_loss'] = torch.matmul((weights==1).view(-1, 1).T.float(), _loss).squeeze(-1).squeeze(-1).detach().data
                if self.original_loss_for_stopping:
                    # in the logging, we call "loss" the undiscounted loss
                    # to be consistent with valid_step()
                    logging_output['weighted_loss'] = loss.data
                    logging_output['loss'] = logging_output['full_loss']
                else:
                    logging_output['loss']=loss.data
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        if not self.args.eval_bleu:
            with torch.no_grad():
                if self.context_discount == 1:
                    # no context discounting
                    loss, sample_size, logging_output = criterion(model, sample)
                else:
                    _loss, sample_size, logging_output = criterion(
                        model, sample, reduce=False
                    )
                    # generate weights to discount the loss generated by context
                    weights = self.build_loss_weights(sample['target'], sample['net_input']['po_segment_labels'])
                    logging_output['full_loss']=logging_output['loss']
                    # calculate weighted loss
                    if self.context_discount < 0:
                        loss = torch.matmul(torch.ones_like(weights).view(-1, 1).T, _loss).squeeze(-1).squeeze(-1)
                    else:
                        loss = torch.matmul(weights.view(-1, 1).T, _loss).squeeze(-1).squeeze(-1)
                    # calculate current loss
                    logging_output['current_loss'] = torch.matmul((weights==1).view(-1, 1).T.float(), _loss).squeeze(-1).squeeze(-1).detach().data
                    if self.original_loss_for_stopping:
                        # in the logging, we call "loss" the undiscounted loss
                        # so that this loss will be used as a stopping criterion
                        # instead of the actual (discounted) loss
                        logging_output['weighted_loss'] = loss.data
                        logging_output['loss'] = logging_output['full_loss']
                    else:
                        logging_output['loss']=loss.data
        else:
            # don't calculate loss when evaluating with BLEU
            loss=None
            sample_size=None
            logging_output={}
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs={}
    ):
        # at generation time, the eos token is used to decide if a hypothesis is finished or not
        # we pass our own eos to the generator (<END> token also used for doc markup)
        # so it does not stop the generation of a hypothesis after one sentence
        extra_gen_cls_kwargs['eos'] = self.tgt_dict.index('<END>')
        # to_strip = set()
        to_strip = {self.tgt_dict.index('<END>')}
        if self.args.remove_markup_from_output and self.args.add_doc_markup:
            to_strip.union(
                {
                    self.tgt_dict.index('<BEG>'),
                    self.tgt_dict.index('<BRK>'),
                    self.tgt_dict.index('<CNT>')
                }
            )
        extra_gen_cls_kwargs['symbols_to_strip_from_output'] = to_strip

        generator = super().build_generator(
            models, args, seq_gen_cls, extra_gen_cls_kwargs
        )
        return generator

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            hypos = generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints
            )

            if self.mode == 'slide_n2n' and not getattr(self.args, "score_reference", False):
                # select only current sentence tokens from best hypothesis
                for i in range(len(hypos)):
                    top_hypo_tokens = hypos[i][0]['tokens']
                    # find the eos tokens in the hypo
                    matches = (top_hypo_tokens == self.tgt_dict.eos()).nonzero()
                    if len(matches) != 0:
                        # keep the last sentence only
                        start_idx = matches[-1]
                        hypos[i][0]['tokens'] = top_hypo_tokens[start_idx + 1:]
            return hypos

    def match_pretrained_to_arch(self, state, model):
        """ Operations to match a pretrained state dict with the current model

            Args:
                state (dict): the pretrained state (e.g from load_checkpoint_to_cpu())
                model (torch.nn.Module): the current model to match
            Returns:
                dict: the modified pretrained state dict
         """
        state_dict = model.state_dict()
        pretr_state_dict = state['model']
        for key in state_dict.keys():
            if key not in pretr_state_dict:
                pretr_state_dict[key] = state_dict[key]
                logger.info('added {} to pre-trained architecture'.format(key))
            elif state_dict[key].size() > pretr_state_dict[key].size():
                pretr_size = pretr_state_dict[key].size()
                curr_size = state_dict[key].size()
                tensor = torch.empty(
                    curr_size, dtype=pretr_state_dict[key].dtype
                )
                torch.nn.init.normal_(tensor, mean=0, std=curr_size[-1]**-0.5)
                tensor[:-(curr_size[0] - pretr_size[0]
                         ), :pretr_size[1]] = pretr_state_dict[key]
                pretr_state_dict[key] = tensor
                logger.info(
                    'matched pretrained {} from ({}, {}) to ({}, {})'.format(
                        key, pretr_size[0], pretr_size[1], curr_size[0],
                        curr_size[1]
                    )
                )
                # # Test
                # pretr_state_dict[key][-1] = pretr_state_dict[key][self.src_dict.eos()]
                # logger.info('initialized new <EOS> parameters with </s> (eos) parameters')

        state['model'] = pretr_state_dict

        return state

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if logging_outputs[0].get('weighted_loss', None) is not None:
            sample_size = sum(log.get('sample_size', 1e-9) for log in logging_outputs)
            # log full loss
            weighted_loss = sum(log.get('weighted_loss', 1e-9) for log in logging_outputs)
            metrics.log_scalar('weighted_loss', weighted_loss / sample_size / math.log(2), sample_size, round=3)
            # log current loss
            current_loss = sum(log.get('current_loss', 1e-9) for log in logging_outputs)
            metrics.log_scalar('current_loss', current_loss / sample_size / math.log(2), sample_size, round=3)

        elif logging_outputs[0].get('full_loss', None) is not None:
            sample_size = sum(log.get('sample_size', 1e-9) for log in logging_outputs)
            # log full loss
            full_loss = sum(log.get('full_loss', 1e-9) for log in logging_outputs)
            metrics.log_scalar('full_loss', full_loss / sample_size / math.log(2), sample_size, round=3)
            # log current loss
            current_loss = sum(log.get('current_loss', 1e-9) for log in logging_outputs)
            metrics.log_scalar('current_loss', current_loss / sample_size / math.log(2), sample_size, round=3)
