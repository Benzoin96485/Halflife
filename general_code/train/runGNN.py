#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
GNN train and evaluate.
'''

# Imports
# std libs

# 3rd party libs
import torch
from torch import split
import numpy as np
from torch.nn.functional import softplus
from torch.utils.data import DataLoader
from torch.optim import Adam

# my modules
from general_code.model.loss import ood_reweight_loss
from general_code.train.runNN import extract_output, Meter


def embedding_fusion(*embeddings):
    embedding_zip = list(map(tuple, zip(*embeddings)))
    embedding_fused = list(map(lambda e: torch.cat(e, dim=1), embedding_zip))
    return embedding_fused


def embedding_batch(embedding_model, device, bg, labels, mask):
    bg = bg.to(device)
    atom_feats = bg.ndata.pop("atom").to(device)
    bond_feats = bg.edata.pop("bond").to(device)
    return embedding_model(bg, atom_feats, bond_feats, norm=None)["mol_embedding"]


def train_epoch3(device, epoch, models, data_loader, loss_criterion, optimizer, metric, logger, task_weight=None, n_tasks=1):
    for model in models.values():
        model.train()

    train_meter = Meter()
    if task_weight:
        task_weight = task_weight.float().to(device)

    for batch_id, batch_data in enumerate(data_loader):
        bg = batch_data["bg"]
        bg_scaffold = batch_data["bg_scaffold"]
        bg_sidechain = batch_data["bg_sidechain"]
        labels = batch_data["labels"]
        mask = batch_data["mask"]
        embeddings = embedding_batch(
            embedding_model=models["GNN"], device=device, bg=bg, labels=labels, mask=mask
        )
        embeddings_scaffold = embedding_batch(
            embedding_model=models["scaffold_GNN"], device=device, bg=bg_scaffold, labels=labels, mask=mask
        )
        embeddings_sidechain = embedding_batch(
            embedding_model=models["sidechain_GNN"], device=device, bg=bg_sidechain, labels=labels, mask=mask
        )
        embedding_fused = embedding_fusion(embeddings, embeddings_scaffold, embeddings_sidechain)
        output = models["FCN"](embedding_fused)
        pred = output[:, 0].reshape(-1, n_tasks)
        labels = labels.type_as(pred).reshape(pred.shape)
        mask = mask.reshape(pred.shape)
        mask = mask.float().to(device)
        labels.float().to(device)
        if not task_weight:
            loss = (loss_criterion(output, labels) * (mask != 0).float()).mean()
        else:
            loss = (torch.mean(loss_criterion(output, labels) * (mask != 0).float(), dim=0) * task_weight).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(pred, labels, mask)
        del bg, bg_scaffold, bg_sidechain, labels, mask, output, pred
        torch.cuda.empty_cache()

    logger.cur_dict()["loss"].append(loss.detach().cpu().numpy())
    try:
        train_score = np.mean(train_meter.compute_metric(metric))
    except:
        exit()
    logger.cur_dict()["train_score"].append(train_score)
    logger.report_train_score(epoch)


def train_OOD_epoch(device, epoch, models: dict, data_loader: DataLoader, loss_criterion, optimizer, metric, logger, epochs_reweight=20, task_weight=None, n_tasks=1, gamma=0.9, **kwargs):
    for model in models.values():
        model.train()
    train_meter = Meter()
    if task_weight:
        task_weight = task_weight.float().to(device)
    
    gnn_out_nfeats = models["encoder"].gnn_out_nfeats
    batch_size = data_loader.batch_size
    total_length = len(data_loader.dataset)
    global_reprs = [torch.zeros((total_length, gnn_out_nfeats)).to(device) for i in range(n_tasks)]
    global_weights = [torch.ones(total_length).to(device) for i in range(n_tasks)]

    for batch_id, batch_data in enumerate(data_loader):
        batch_length = len(batch_data[0])
        smiles, bg, labels, mask, extra = batch_data
        embeddings = embedding_batch(
            embedding_model=models["encoder"], device=device, bg=bg, labels=labels, mask=mask
        )
        batch_reprs = [embedding.detach() for embedding in embeddings] 
        batch_weights = [
            torch.ones(batch_length).to(device).requires_grad_()
            for i in range(n_tasks)
        ]
        global_reprs_hat = [
            torch.cat((global_reprs[i], batch_reprs[i].detach())) 
            for i in range(n_tasks)
        ]
        global_weights_hat = [
            torch.cat((global_weights[i], batch_weights[i])) 
            for i in range(n_tasks)
        ]

        reweight_optimizer = Adam(batch_weights, lr=0.01)
        for sub_epoch in range(epochs_reweight):
            reweight_optimizer.zero_grad()
            global_weights_hat = [
                torch.cat((global_weights[i], batch_weights[i])) 
                for i in range(n_tasks)
            ]
            for i in range(n_tasks):
                sub_loss = ood_reweight_loss(global_reprs_hat[i], global_weights_hat[i])
                sub_loss.backward()
            reweight_optimizer.step()
        
        weights = torch.cat(batch_weights).detach().to(device)
        
        output = models["regressor"](embeddings)
        extract = extract_output(output, 
            criterion=loss_criterion._name if hasattr(loss_criterion, "_name") else None,
            metric=metric, 
            train=True,
            n_tasks=n_tasks,
            thresholds=loss_criterion.thresholds if hasattr(loss_criterion, "thresholds") else None
        )
        pred = extract["pred"]
        labels = labels.reshape(extract["shape"]).type_as(pred).to(device)
        mask = mask.reshape(extract["shape"]).float().to(device)
        if not task_weight:
            loss = (
                loss_criterion(output, labels) * (mask != 0).float() * weights
            ).mean()
        else:
            loss = (
                torch.mean(
                    loss_criterion(output, labels) * (mask != 0).float() * weights, dim=0
                ) * task_weight
            ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pointer = batch_id * batch_size
        for i in range(n_tasks):
            global_reprs[i][pointer: pointer + batch_length] += gamma * (batch_reprs[i] - global_reprs[i][pointer: pointer + batch_length])
            global_weights[i][pointer: pointer + batch_length] += gamma * (batch_weights[i].detach() - global_weights[i][pointer: pointer + batch_length])

        train_meter.update(pred, labels.reshape(pred.shape), mask.reshape(pred.shape))
        del bg, labels, mask, extra, output, pred, batch_weights, batch_reprs, weights
        torch.cuda.empty_cache()

    logger.cur_dict()["loss"].append(loss.detach().cpu().numpy())
    train_score = np.mean(train_meter.compute_metric(metric))
    logger.cur_dict()["train_score"].append(train_score)
    logger.report_train_score(epoch)


def train_epoch(device, epoch, model, data_loader,
                loss_criterion, optimizer,
                metric, logger, task_weight=None, n_tasks=1, **kwargs):
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
        output = model(bg, atom_feats, bond_feats, extra_embedding=extra, norm=None)[0]
        extract = extract_output(output, metric=metric, train=True,n_tasks=n_tasks)
        pred = extract["pred"]
        labels = labels.reshape(extract["shape"]).type_as(pred).to(device)
        mask = mask.reshape(extract["shape"]).float().to(device)
        if not task_weight:
            loss = (loss_criterion(output, labels) * (mask != 0).float()).mean()
        else:
            loss = (torch.mean(loss_criterion(output, labels) * (mask != 0).float(), dim=0) * task_weight).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #  ('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
        #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(pred, labels.reshape(pred.shape), mask.reshape(pred.shape))
        del bg, labels, mask, extra, atom_feats, bond_feats, output, pred
        torch.cuda.empty_cache()

    logger.cur_dict()["loss"].append(loss.detach().cpu().numpy())
    train_score = np.mean(train_meter.compute_metric(metric))
    logger.cur_dict()["train_score"].append(train_score)
    logger.report_train_score(epoch)


def eval_epoch3(device, models, data_loader, metric, extra_metrics=[], n_tasks=1):
    for model in models.values():
        model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_data in data_loader:
            bg = batch_data["bg"]
            bg_scaffold = batch_data["bg_scaffold"]
            bg_sidechain = batch_data["bg_sidechain"]
            labels = batch_data["labels"]
            mask = batch_data["mask"]
            labels = labels.float().to(device)
            mask = mask.float().to(device)
            embeddings = embedding_batch(
                embedding_model=models["GNN"], device=device, bg=bg, labels=labels, mask=mask
            )
            embeddings_scaffold = embedding_batch(
                embedding_model=models["scaffold_GNN"], device=device, bg=bg_scaffold, labels=labels, mask=mask
            )
            embeddings_sidechain = embedding_batch(
                embedding_model=models["sidechain_GNN"], device=device, bg=bg_sidechain, labels=labels, mask=mask
            )  
            embedding_fused = embedding_fusion(embeddings, embeddings_scaffold, embeddings_sidechain)
            output = models["FCN"](embedding_fused)
            pred = output[:, 0].reshape(-1, n_tasks)  
            labels = labels.type_as(pred).reshape(pred.shape)  
            mask = mask.reshape(pred.shape)
            eval_meter.update(pred, labels, mask)
            del bg, bg_scaffold, bg_sidechain, labels, mask, output, pred
            torch.cuda.empty_cache()
        return [eval_meter.compute_metric(metric)] + [eval_meter.compute_metric(extra_metric) for extra_metric in extra_metrics]


def eval_OOD_epoch(device, models, data_loader, metric, extra_metrics=[], n_tasks=1, loss_criterion=None):
    for model in models.values():
        model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_data in data_loader:
            smiles, bg, labels, mask, extra = batch_data
            labels = labels.float().to(device)
            mask = mask.float().to(device)
            embeddings = embedding_batch(
                embedding_model=models["encoder"], device=device, bg=bg, labels=labels, mask=mask
            )
            output = models["regressor"](embeddings)
            extract = extract_output(output, 
                criterion=loss_criterion._name if hasattr(loss_criterion, "_name") else None,
                metric=metric, 
                train=False,
                n_tasks=n_tasks,
                thresholds=loss_criterion.thresholds if hasattr(loss_criterion, "thresholds") else None
            )
            pred = extract["pred"]
            labels = labels.type_as(pred).reshape(pred.shape)  
            mask = mask.reshape(pred.shape)
            eval_meter.update(pred, labels, mask)
            del bg, labels, mask
            torch.cuda.empty_cache()
        return [eval_meter.compute_metric(metric)] + [eval_meter.compute_metric(extra_metric) for extra_metric in extra_metrics]

def eval_epoch(device, model, data_loader, metric, extra_metrics=[],
    give_result=False, evidential=False,
    result_dict={
        "smiles": [],
        "labels": [],
        "preds": [],
        "aleatoric": [],
        "epistemic": []
    }, n_tasks=1
):
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
            output = model(bg, atom_feats, bond_feats, extra_embedding=extra, norm=None)[0]
            extract = extract_output(
                output, 
                criterion = ("evidential" if evidential else None), metric=metric, 
                n_tasks=n_tasks
            )
            pred = extract["pred"]
            labels = labels.type_as(pred).reshape(pred.shape)
            mask = mask.reshape(pred.shape)
            eval_meter.update(pred, labels, mask)
            if give_result:
                result_dict["smiles"].extend(list(smiles))
                result_dict["labels"].extend(labels.reshape(-1).cpu().numpy())
                result_dict["preds"].extend(pred.numpy())
                if evidential:
                    result_dict["aleatoric"].extend(extract["aleatoric"].numpy())
                    result_dict["epistemic"].extend(extract["epistemic"].numpy())
            del bg, mask, extra, atom_feats, bond_feats
            torch.cuda.empty_cache()
    return [eval_meter.compute_metric(metric)] + [eval_meter.compute_metric(extra_metric) for extra_metric in extra_metrics]