import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math
import numpy as np
import pandas as pd
from torch.optim import Optimizer
import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.multiprocessing as mp

from logger import *
from eval import *
from misc import *
import matplotlib.pyplot as plt

from normal_train import *
from final_util import *
from adam import Adam
from sgd import SGD

import torchvision.transforms as transforms
import torchvision.datasets as datasets

np.random.seed(2)
random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)
torch.cuda.manual_seed_all(2)

data_loc = "/home/cluster/data/cifar10/"
dataloc = "/home/cluster/data/cifar10/"

# load the train dataset

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2, 0.1994, 0.2010)),
    ]
)

data_train = datasets.CIFAR10(
    root=data_loc, train=True, download=False, transform=train_transform
)

data_test = datasets.CIFAR10(
    root=data_loc, train=False, download=False, transform=train_transform
)

X = []
Y = []
for i in range(len(data_train)):
    X.append(data_train[i][0].numpy())
    Y.append(data_train[i][1])

for i in range(len(data_test)):
    X.append(data_test[i][0].numpy())
    Y.append(data_test[i][1])

X = np.array(X)
Y = np.array(Y)

print("total data len: ", len(X))

all_indices = np.arange(len(X))
np.random.shuffle(all_indices)

X = X[all_indices]
Y = Y[all_indices]

nusers = 50
user_tr_len = 1000

total_tr_len = user_tr_len * nusers
val_len = 5000
te_len = 5000

print("total data len: ", len(X))

total_tr_data = X[:total_tr_len]
total_tr_label = Y[:total_tr_len]

val_data = X[total_tr_len : (total_tr_len + val_len)]
val_label = Y[total_tr_len : (total_tr_len + val_len)]

te_data = X[(total_tr_len + val_len) : (total_tr_len + val_len + te_len)]
te_label = Y[(total_tr_len + val_len) : (total_tr_len + val_len + te_len)]

val_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor)
val_label_tensor = torch.from_numpy(val_label).type(torch.LongTensor)

te_data_tensor = torch.from_numpy(te_data).type(torch.FloatTensor)
te_label_tensor = torch.from_numpy(te_label).type(torch.LongTensor)

user_tr_data_tensors = []
user_tr_label_tensors = []

for i in range(nusers):
    user_tr_data_tensor = torch.from_numpy(
        total_tr_data[user_tr_len * i : user_tr_len * (i + 1)]
    ).type(torch.FloatTensor)
    user_tr_label_tensor = torch.from_numpy(
        total_tr_label[user_tr_len * i : user_tr_len * (i + 1)]
    ).type(torch.LongTensor)

    user_tr_data_tensors.append(user_tr_data_tensor)
    user_tr_label_tensors.append(user_tr_label_tensor)


batch_size = 250
nepochs = 1200

nbatches = user_tr_len // batch_size
fed_lr = 0.5
criterion = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()

n_attacker = 10

arch = "alexnet"

epoch_num = 0
best_global_acc = 0
best_global_te_acc = 0

torch.cuda.empty_cache()
r = np.arange(user_tr_len)

fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

acc_point, los_point = [], []
while epoch_num <= nepochs:
    user_grads = []
    if not epoch_num and epoch_num % nbatches == 0:
        np.random.shuffle(r)
        for i in range(nusers):
            user_tr_data_tensors[i] = user_tr_data_tensors[i][r]
            user_tr_label_tensors[i] = user_tr_label_tensors[i][r]

    for i in range(n_attacker, nusers):
        inputs = user_tr_data_tensors[i][
            (epoch_num % nbatches)
            * batch_size : ((epoch_num % nbatches) + 1)
            * batch_size
        ]
        targets = user_tr_label_tensors[i][
            (epoch_num % nbatches)
            * batch_size : ((epoch_num % nbatches) + 1)
            * batch_size
        ]

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(
            targets
        )

        outputs = fed_model(inputs)
        loss = criterion(outputs, targets)
        fed_model.zero_grad()
        loss.backward(retain_graph=True)

        param_grad = []
        for param in fed_model.parameters():
            param_grad = (
                param.grad.data.view(-1)
                if not len(param_grad)
                else torch.cat((param_grad, param.grad.view(-1)))
            )

        user_grads = (
            param_grad[None, :]
            if len(user_grads) == 0
            else torch.cat((user_grads, param_grad[None, :]), 0)
        )

    malicious_grads = user_grads
    agg_grads = torch.mean(malicious_grads, dim=0)

    del user_grads

    start_idx = 0
    optimizer_fed.zero_grad()
    model_grads = []
    for i, param in enumerate(fed_model.parameters()):
        param_ = agg_grads[start_idx : start_idx + len(param.data.view(-1))].reshape(
            param.data.shape
        )
        start_idx = start_idx + len(param.data.view(-1))
        param_ = param_.cuda()
        model_grads.append(param_)
    optimizer_fed.step(model_grads)

    val_loss, val_acc = test(
        val_data_tensor, val_label_tensor, fed_model, criterion, use_cuda
    )
    te_loss, te_acc = test(
        te_data_tensor, te_label_tensor, fed_model, criterion, use_cuda
    )

    is_best = best_global_acc < val_acc

    best_global_acc = max(best_global_acc, val_acc)

    if is_best:
        best_global_te_acc = te_acc

    if epoch_num % 10 == 0 or epoch_num == nepochs - 1:
        print(
            "n_at %d n_mal_sel %d val loss %.4f val acc %.4f best val_acc %f te_acc %f"
            % (
                n_attacker,
                epoch_num,
                val_loss,
                val_acc,
                best_global_acc,
                best_global_te_acc,
            )
        )
        acc_point.append((epoch_num, te_acc))
        los_point.append((epoch_num, te_loss))
        plt.subplot(2, 1, 1)
        plt.plot([p[0] for p in acc_point], [p[1] for p in acc_point], color="red")
        plt.scatter([p[0] for p in acc_point], [p[1] for p in acc_point], color="red")
        plt.title("acc")
        plt.subplot(2, 1, 2)
        plt.plot([p[0] for p in los_point], [p[1] for p in los_point], color="green")
        plt.scatter([p[0] for p in los_point], [p[1] for p in los_point], color="green")
        plt.title("los")
        plt.show()

    epoch_num += 1
