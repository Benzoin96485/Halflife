#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Customized loss functions.
'''

# Imports
# std libs
# ...

# 3rd party libs
from torch.nn.modules.loss import _Loss
from torch.nn import Softplus, NLLLoss
from torch.nn.functional import softmax, softplus
import torch
from torch import Tensor, log, lgamma, split, erfc, diff, ones_like, zeros_like, flatten
from numpy import pi, sqrt
from numpy.random import randn, rand

# my modules
# ...

class NIGLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', lamda=0) -> None:
        super(NIGLoss, self).__init__(size_average, reduce, reduction)
        self.lamda = lamda
        self.soft_plus = Softplus()
        self._name = "evidential"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        g, v_, a_, b_ = split(input, 1, dim=1) # v, a-1, b must be positive
        v = self.soft_plus(v_)
        a = self.soft_plus(a_) + 1
        b = self.soft_plus(b_)
        O = 2 * b * (1 + v)
        L_nll = -log(v / pi) / 2 - a * log(O) + (a + 0.5) * log((target - g) ** 2 * v + O) + lgamma(a) - lgamma(a + 0.5)
        L_R = torch.abs(target - g) * (2 * v + a)
        return L_nll + self.lamda * L_R


def cg_prob(input: Tensor, thresholds: Tensor, n_tasks: int=1):
    input_ = input.reshape(-1, n_tasks, 2)
    mu = input_[:, :, 0:1]
    sigma = softplus(input_[:, :, 1:2])
    normalized = (thresholds - mu) / sigma
    percentile = 1 - erfc(normalized / (2 ** 0.5)) / 2
    prob = diff(percentile, prepend=zeros_like(percentile[:, :, 0:1]), append=ones_like(percentile[:, :, 0:1]))
    return prob.view(-1, prob.shape[-1])


class CGCELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', thresholds = None, n_tasks = 2):
        super(CGCELoss, self).__init__(size_average, reduce, reduction)
        self.NLLLoss = NLLLoss(size_average=size_average, reduce=reduce, reduction=reduction)
        self.thresholds = thresholds
        self.n_tasks = n_tasks
        self._name = "cgce"

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        thresholds = torch.tensor(self.thresholds, device = input.device)
        return self.NLLLoss.forward(cg_prob(input, thresholds, self.n_tasks), target)


def ood_reweight_loss(glb_repr_hat, glb_wt_hat, Q=5):
    dim = glb_repr_hat.shape[1]
    device = glb_repr_hat.device
    f_omega = torch.randn((1, dim, Q), device=device)
    g_omega = torch.randn((1, dim, Q), device=device)
    f_phi = torch.rand((1, dim, Q), device=device) * 2 * pi
    g_phi = torch.rand((1, dim, Q), device=device) * 2 * pi
    fi = glb_repr_hat[:, :, None] * f_omega + f_phi
    gj = glb_repr_hat[:, :, None] * g_omega + g_phi
    wfi = fi * glb_wt_hat.reshape(-1, 1, 1)
    wgj = gj * glb_wt_hat.reshape(-1, 1, 1)
    wfi = wfi - torch.mean(wfi, dim=0)
    wgj = torch.transpose(wgj - torch.mean(wgj, dim=0), 1, 2)
    Cij = torch.sum(torch.matmul(wfi, wgj), dim=0) / (torch.sum(glb_wt_hat) - 1)
    diag_zero = torch.ones_like(Cij) - torch.diag_embed(torch.diag(Cij))
    return torch.sum(Cij * diag_zero) / 2

