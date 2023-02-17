# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import random
import torch

from torch import nn
import torch.nn.functional as F
from fairseq import utils

from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def compute_nce_loss(sample, imgs_embed, temperature=0.1):
    '''
    imgs_embed: list of tensors. contains 3 imgs features
              orignal, cropped ones, colored
              `(batch,encoder_dim)`
    margin: hyper-parameter
    '''
    bsz = imgs_embed[0].size(0)
    def compute_nce (a,b):
        norm_a = torch.nn.functional.normalize(a,p=2,dim=1) #normalize the tensors
        norm_b = torch.nn.functional.normalize(b,p=2,dim=1) #normalize the tensors

        cat_res = torch.cat((norm_a,norm_b),0)
        all_sim = torch.matmul(cat_res,cat_res.t()) #[2B, 2B]
        mask = torch.eye(all_sim.size(0),all_sim.size(1)).bool().cuda()
        all_sim.masked_fill_(mask,0.0)
        all_sim = torch.exp(all_sim/temperature)
        pos_sim = torch.exp(F.cosine_similarity(norm_a,norm_b)/temperature)
        batch_sim = torch.log(pos_sim.repeat(2)/all_sim.sum(1))
        loss = - batch_sim.sum()/(bsz*2)
        return loss

    loss = compute_nce(imgs_embed[1],imgs_embed[2])

    return loss

@register_criterion('nce_label_smoothed_cross_entropy')
class NCELabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.temperature = args.temperature
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--temperature', default=0.1, type=float, metavar='D',
                            help='temperature for NCE loss')
        # fmt: on

    def forward(self, model, sample, reduce=True, awl=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        imgs_embed = []
        for img in sample['image']:
            if img == 'orig':
               imgs_embed.append(model.img_encoder(sample['image'][img], flag='orig'))
            imgs_embed.append(model.img_encoder(sample['image'][img], flag='global'))
        #imgs_embed contain 4 features (9,300) and 3 (1,300)
        #order: orig local, orig global, crop global, color global

        net_output = model(**sample['net_input'], img_features=imgs_embed[0]) #use the orig features for nll loss
        loss, nll_loss, constrast_loss = self.compute_loss(model, net_output, sample,
                                         imgs_embed[1:], temperature=self.temperature, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        triplet_loss = constrast_loss*sample_size
        if awl is not None:
            loss = awl(loss, triplet_loss)
        else:
            loss = loss + triplet_loss

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'triplet_loss': utils.item(triplet_loss.data) if reduce else triplet_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, imgs_embed, reduce=True, temperature=1.0):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        constrast_loss = compute_nce_loss(sample, imgs_embed, temperature)

        return loss, nll_loss, constrast_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'triplet_loss': sum(log.get('triplet_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
