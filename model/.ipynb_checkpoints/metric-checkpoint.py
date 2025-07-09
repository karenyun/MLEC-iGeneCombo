import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, mean_squared_error
from scipy.stats.stats import pearsonr
import numpy as np
import pickle


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def auc(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    return roc_auc_score(target, output)


def aupr(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    return average_precision_score(target, output)

def precision_at_5(output, target, top=0.05):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    sorted_index = np.argsort(-output)
    top_num = int(top * len(target))
    if top_num == 0:
        return -1
    sorted_targets = target[sorted_index[:top_num]]
    acc = float(sorted_targets.sum())/float(top_num)
    return acc

def precision_at_1(output, target, top=0.01):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    sorted_index = np.argsort(-output)
    top_num = int(top * len(target))
    if top_num == 0:
        return -1
    sorted_targets = target[sorted_index[:top_num]]
    acc = float(sorted_targets.sum())/float(top_num)
    return acc

def precision_at_10(output, target, top=0.10):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    sorted_index = np.argsort(-output)
    top_num = int(top * len(target))
    if top_num == 0:
        return -1
    sorted_targets = target[sorted_index[:top_num]]
    acc = float(sorted_targets.sum())/float(top_num)
    return acc

def corr(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
#     print("output is {}".format(output))
#     print("target is {}".format(target))
#     print("output is {}".format((output+1)*(3.932254644--6.027560721)/2+-6.027560721))
#     print("target is {}".format((target+1)*(3.932254644--6.027560721)/2+-6.027560721))
#     with open("./saved/output.pkl", "wb") as f:
#         pickle.dump(output, f)
#     with open("./saved/target.pkl", "wb") as f:
#         pickle.dump(target, f)
    return pearsonr(target, output)[0]

def mse(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    return mean_squared_error(target, output)