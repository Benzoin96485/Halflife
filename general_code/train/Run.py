import torch
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score
import torch.nn.functional as F
import numpy as np
import time


class EarlyStopping(object):
    def __init__(self, model_path, task_name, pretrained_path, patience, earlystop_mode='higher', **kwargs):
        assert earlystop_mode in ['higher', 'lower']
        self.mode = earlystop_mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.model_path = model_path
        self.best_score = None
        self.early_stop = False
        self.pretrained_path = pretrained_path

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model):
        if not self.best_score:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when the metric on the validation set gets improved."""
        torch.save({'model_state_dict': model.state_dict()}, self.model_path)

    def load_checkpoint(self, model, device):
        """Load model saved with early stopping."""
        model.load_state_dict(torch.load(self.model_path, map_location=device)['model_state_dict'])

    def load_pretrained_model(self, model, device, param_filter_func=None):
        pretrained_model = torch.load(self.pretrained_path, map_location=device)
        model_dict = model.state_dict()
        if param_filter_func:
            pretrained_param = param_filter_func(pretrained_model['model_state_dict'].keys())
            pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_param}
        else:
            pretrained_dict = pretrained_model['model_state_dict']
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""

    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))
        return scores

    def return_pred_true(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        return y_pred, y_true

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_squared_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.
        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            AUC_PRC for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred)
            scores.append(auc(recall, precision))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'mae', 'roc_prc', 'r2', 'return_pred_true'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", "mae", "roc_prc", "r2", "return_pred_true", ' \
            'got {}'.format(metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'roc_prc':
            return self.roc_precision_recall_score()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'return_pred_true':
            return self.return_pred_true()


def train_epoch(device, epoch, model, num_epochs, data_loader,
                loss_r, optimizer,
                regression_metric_name='r2', task_weight=None, **kwargs):
    model.train()
    train_meter_r = Meter()
    if task_weight:
        task_weight = task_weight.float().to(device)

    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, mask = batch_data
        mask = mask.float().to(device)
        labels.float().to(device)
        bg = bg.to(device)
        atom_feats = bg.ndata.pop("atom").to(device)
        bond_feats = bg.edata.pop("bond").to(device)
        logits = model(bg, atom_feats, bond_feats, norm=None)[0]
        labels = labels.type_as(logits)
        if not task_weight:
            loss = (loss_r(logits, labels) * (mask != 0).float()).mean()
        else:
            loss = (torch.mean(loss_r(logits, labels) * (mask != 0).float(), dim=0) * task_weight).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
        #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter_r.update(logits, labels, mask)
        del bg, mask, labels, atom_feats, bond_feats, loss, logits
        torch.cuda.empty_cache()

    train_score = np.mean(train_meter_r.compute_metric(regression_metric_name))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, num_epochs, regression_metric_name, train_score))


def eval_epoch(device, model, data_loader, regression_metric_name, **kwargs):
    model.eval()
    eval_meter_r = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(device)
            mask = mask.float().to(device)
            bg = bg.to(device)
            atom_feats = bg.ndata.pop("atom").to(device)
            bond_feats = bg.edata.pop("bond").to(device)
            logits = model(bg, atom_feats, bond_feats, norm=None)[0]
            labels = labels.type_as(logits)
            eval_meter_r.update(logits, labels, mask)
            del smiles, bg, mask, labels, atom_feats, bond_feats, logits
            torch.cuda.empty_cache()

        return eval_meter_r.compute_metric(regression_metric_name)


def predict(device, model, dataset):
    pre_result = []
    model.eval()
    for data in dataset:
        smiles, bg, labels, mask = data
        bg = bg.to(device)
        atom_feats = bg.ndata.pop("atom").to(device)
        bond_feats = bg.edata.pop("bond").to(device)
        pre_result.append(model(bg, atom_feats, bond_feats, norm=None)[0].cpu().detach().numpy()[0][0])
    return pre_result

def print_dict(d_print, indent=""):
    print_str = ""
    for key, value in d_print.items():
        if type(value) == dict:
            print_str += indent + key + ": " + "\n" + print_dict(value, indent + "\t")
        else:
            print_str += indent + key + ": " + str(value) + "\n"
    return print_str


def print_log(settings, result):
    time_str = "_" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    settings["result_path"] = settings["result_folder"] + settings['task_name'] + time_str + '_result.csv'
    result.to_csv(settings["result_path"] , index=None)
    filename = settings["task_name"] + time_str + "_log.txt"
    mean_series = result.mean()
    std_series = result.std()
    n_tasks = settings["n_tasks"]
    metric_name = settings["regression_metric_name"]
    with open(settings["log_folder"] + filename, mode='w+') as f:
        f.write(print_dict(settings))
        f.write("\n")
        f.write(f"Train {metric_name}: \n")
        for i in range(n_tasks):
            f.write(
                f"{settings['label_name_list']}: mean:{mean_series[i]}, std:{std_series[i]}\n")
        f.write(f"Evaluate {metric_name}: \n")
        for i in range(n_tasks):
            f.write(
                f"{settings['label_name_list']}: mean:{mean_series[n_tasks + i]}, std:{std_series[n_tasks + i]}\n")
        f.write(f"Test {metric_name}: \n")
        for i in range(n_tasks):
            f.write(
                f"{settings['label_name_list']}: mean:{mean_series[2 * n_tasks + i]}, std:{std_series[2 * n_tasks + i]}\n")
    return mean_series[n_tasks: n_tasks * 2]
