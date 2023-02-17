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

def cos_sim_loss(sample, imgs_embed, temperature=0.1):
    '''
    imgs_embed: list of tensors. contains 3 imgs features 
              cropped ones, colored
              (batch,encoder_dim) 
    Triplet margin loss: colored and cropped ones are positive samples of orignal,
    and two of the other samples in this batch are negative ones.
    '''
    #distance = nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    bsz = imgs_embed[0].size(0)
    tensor_imgs = torch.cat(tuple(imgs_embed),0) #(2*batch, dim)
    cos_all = tensor_imgs.new(bsz*2,bsz*2).zero_()

    for i in range(2*bsz):
      for j in range(2*bsz):
        cos_all[i][j] = torch.exp(cos(tensor_imgs[i],tensor_imgs[j])/temperature)

    def cos_loss(index1, index2):
        return -torch.log(cos_all[index1][index2]/ \
                           (cos_all[index1].sum(-1) - cos_all[index1][index1]))

    total_loss = torch.tensor(0.0).cuda()
    for bsz_id in range(bsz):
        cos_crop_color = cos_loss(bsz_id, bsz_id+bsz) 
        cos_color_crop = cos_loss(bsz_id+bsz, bsz_id)
        #print (cos_crop_color, cos_color_crop)
        total_loss += cos_crop_color + cos_color_crop
 
    loss = total_loss / (bsz*2) 
    #print ("loss", total_loss, loss)
    #print ("in margin 68 color and crop", colored_loss, crop_loss)

    return loss


@register_criterion('sim_label_smoothed_cross_entropy')
class SimLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.temperature = args.temperature
        self.alpha = args.loss_weight
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--temperature', default=0.1, type=float, metavar='D',
                            help='margin for triplet margin loss')
        parser.add_argument('--loss-weight', default=1.0, type=float, metavar='D',
                            help='weight for triplet margin loss')
        # fmt: on

    def forward(self, model, sample, awl=None, epoch=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        imgs_embed = []
        for img in sample['image']:
        #img: orig, crop_image, color_image
            if img == 'orig' :
                imgs_embed.append(model.img_encoder(sample['image'][img], flag='orig'))
                continue
            imgs_embed.append(model.img_encoder(sample['image'][img], flag='global'))
        #imgs_embed contain 3 features (9,300) and 2 (1,300)
        #order: orig local, crop global, color global

        net_output = model(**sample['net_input'], img_features=imgs_embed[0]) #use the orig features for nll loss
        loss, nll_loss, constrast_loss = self.compute_loss(model, net_output, sample,
                                         imgs_embed[1:], temperature=self.temperature, reduce=reduce, epoch=epoch)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        triplet_loss = constrast_loss*sample_size
        #print ("loss: {}  nll_loss: {} triplet: {}".format(loss, nll_loss, triplet_loss))
        if awl is not None and epoch>0:
            tloss = awl(loss, triplet_loss)
        else: 
            lambda1 = 1
            lambda2 = 1
            #if epoch <= 50:
            #    lambda1 = 0
            tloss = lambda1*loss + lambda2*triplet_loss

        logging_output = {
            'loss': utils.item(tloss.data) if reduce else tloss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'triplet_loss': utils.item(triplet_loss.data) if reduce else triplet_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return tloss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, imgs_embed, temperature, reduce=True, epoch=0):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        constrast_loss = cos_sim_loss(sample, imgs_embed, temperature)

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
