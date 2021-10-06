import os
import shutil
import json
import matplotlib.pyplot as plt
import torch
import numpy as np

def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best: shutil.copyfile(filename, 'model_best.pth.tar')

def load_json(name):
    path = os.path.join("results", f"{name}.json")
    data_dict = {}
    if os.path.exists(path):
        with open(path, "r") as f: data_dict = json.load(f)
    return data_dict
    
def save_json(name, seed, res, pFlag=True):
    path = os.path.join("results", f"{name}.json")
    data_dict = {}
    if os.path.exists(path):
        with open(path, "r") as f: data_dict = json.load(f)
    data_dict[str(seed)] = res
    with open(path, "w") as f:
        json.dump(data_dict, f, indent=4)
    if pFlag:
        for k in res: print(k, res[k])

def save_config_file(model_checkpoints_folder, args):
    pass


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def tpr95(X1, Y1):
    #calculate the falsepositive error when tpr is 95%
    diff = [v + 1e-8 for v in X1] + [v + 1e-8 for v in Y1] + [1e-8, 1e8]
    diff = sorted(list(set(diff)))[::-1]
    total = 0.0
    fpr = 0.0
    for delta in diff:
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 <= delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fprBase = fpr/total

    return fprBase

def auroc(X1, Y1):
    diff = [v + 1e-8 for v in X1] + [v + 1e-8 for v in Y1] + [1e-8, 1e8]
    diff = sorted(list(set(diff)))[::-1]
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in diff:
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        aurocBase += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    aurocBase += fpr * tpr

    return aurocBase

def auprIn(X1, Y1):
    #calculate the AUPR
    diff = [v + 1e-8 for v in X1] + [v + 1e-8 for v in Y1] + [1e-8, 1e8]
    diff = sorted(list(set(diff)))[::-1]
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0
    for delta in diff:
        tp = np.sum(np.sum(X1 <= delta))
        fp = np.sum(np.sum(Y1 <= delta))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp / len(X1)
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase

def auprOut(X1, Y1):
    #calculate the AUPR
    diff = [v + 1e-8 for v in X1] + [v + 1e-8 for v in Y1] + [1e-8, 1e8]
    diff = sorted(list(set(diff)))[::-1]
    auprBase = 0.0
    recallTemp = 1.0
    for delta in diff[::-1]:
        fp = np.sum(np.sum(X1 > delta))
        tp = np.sum(np.sum(Y1 > delta))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp / len(Y1)
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision
        
    return auprBase


def detection(X1, Y1):
    #calculate the minimum detection error
    diff = [v + 1e-8 for v in X1] + [v + 1e-8 for v in Y1] + [1e-8, 1e8]
    diff = sorted(list(set(diff)))[::-1]
    errorBase = 1.0
    for delta in diff:
        tpr_ = np.sum(np.sum(X1 > delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr_+error2)/2.0)

    return errorBase

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res