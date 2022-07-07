#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
GNN train and evaluate.
'''

# Imports
# std libs

# 3rd party libs
from dgllife.utils import Meter
import torch
import numpy as np

# my modules


def train_epoch(device, epoch, model, num_epochs, data_loader,
                loss_criterion, optimizer,
                metric, logger, task_weight=None):
    model.train()
    train_meter = Meter()
    if task_weight:
        task_weight = task_weight.float().to(device)

    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, mask, extra = batch_data
        mask = mask.float().to(device)
        labels.float().to(device)
        bg = bg.to(device)
        if extra != None:
            extra = extra.to(device)
        atom_feats = bg.ndata.pop("atom").to(device)
        bond_feats = bg.edata.pop("bond").to(device)
        logits = model(bg, atom_feats, bond_feats, extra_embedding=extra, norm=None)[0]
        labels = labels.type_as(logits).reshape(logits.shape)
        mask = mask.reshape(logits.shape)
        if not task_weight:
            loss = (loss_criterion(logits, labels) * (mask != 0).float()).mean()
        else:
            loss = (torch.mean(loss_criterion(logits, labels) * (mask != 0).float(), dim=0) * task_weight).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #  ('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
        #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(logits, labels, mask)
        del bg, mask, labels, atom_feats, bond_feats, logits
        torch.cuda.empty_cache()

    logger.cur_dict()["loss"].append(loss.detach().cpu().numpy())
    try:
        train_score = np.mean(train_meter.compute_metric(metric))
    except:
        exit()
    logger.cur_dict()["train_score"].append(train_score)
    logger.report_train_score(epoch)

def eval_epoch(device, model, data_loader, metric):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_data in data_loader:
            smiles, bg, labels, mask, extra = batch_data
            labels = labels.float().to(device)
            mask = mask.float().to(device)
            bg = bg.to(device)
            if extra != None:
                extra = extra.to(device)
            atom_feats = bg.ndata.pop("atom").to(device)
            bond_feats = bg.edata.pop("bond").to(device)
            logits = model(bg, atom_feats, bond_feats, extra_embedding=extra, norm=None)[0]
            labels = labels.type_as(logits).reshape(logits.shape)
            mask = mask.reshape(logits.shape)
            eval_meter.update(logits, labels, mask)
            del smiles, bg, mask, labels, atom_feats, bond_feats, logits
            torch.cuda.empty_cache()

        return eval_meter.compute_metric(metric)