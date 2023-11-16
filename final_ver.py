from __future__ import print_function
import torchvision.models as tvmodels
import argparse, os, sys, time, random, pickle, math
import numpy as np
from torch.optim import Optimizer
import torch.nn.functional as F
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import copy
import json

from adam import Adam
from sgd import SGD

sys.path.insert(0, "../utils/")
from logger import *
from eval import *
from misc import *
from normal_train import *
from final_util import *
from at_agr import *

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, to_tree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=30)
    parser.add_argument("--num_attackers", type=list, default=[10])
    parser.add_argument("--num_clusters", type=int, default=2)
    parser.add_argument("--gap", type=float, default=0.01)

    parser.add_argument(
        "--linkage_method",
        type=str,
        choices=["average", "single", "complete"],
        default="single",
    )
    parser.add_argument(
        "--abcd_method", type=str, choices=["diff", "min_los"], default="min_los"
    )

    parser.add_argument("--is_server", type=bool, default=False)

    parser.add_argument(
        "--agr",
        type=str,
        choices=[
            "average",
            "krum",
            "mkrum",
            "trmean",
            "fltrust",
            "foolsgold",
            "cluster",
        ],
        default="average",
    )
    parser.add_argument(
        "--at_agr", type=str, choices=["krum", "mkrum", "trmean"], default="krum"
    )
    parser.add_argument(
        "--at_type",
        type=str,
        choices=["none", "rev", "fang", "adaptive", "min_sum", "min_max"],
        default="none",
    )

    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--save", type=int, default=0)

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "mnist", "famnist"],
        default="famnist",
    )
    parser.add_argument(
        "--dev_type", type=str, choices=["sign", "std", "unit_vec"], default="std"
    )
    parser.add_argument("--dataloc", type=str, default="/home/cluster_defence/")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batchsize", type=int)
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--arch", type=int)
    parser.add_argument("--PATH", type=str)
    parser.add_argument("--PATH_txt", type=str)
    parser.add_argument("--schedule", type=list)
    args = parser.parse_args()
    return args


def init_args():
    args = parse_args()
    args.num_classes = {"cifar10": 10, "mnist": 10, "famnist": 10}[args.dataset]
    args.lr = {"cifar10": 0.01, "mnist": 0.001, "famnist": 0.001}[args.dataset]
    args.optimizer = {"cifar10": "SGD", "mnist": "Adam", "famnist": "Adam"}[
        args.dataset
    ]
    args.epoch = {"cifar10": 1200, "mnist": 500, "famnist": 500}[args.dataset]
    args.batchsize = {"cifar10": 64, "mnist": 1000, "famnist": 1000}[args.dataset]
    args.arch = {"cifar10": "resnet18", "mnist": "MLP", "famnist": "CNN"}[args.dataset]
    args.schedule = {"cifar10": [], "mnist": [], "famnist": []}[args.dataset]
    args.dev_type = {"cifar10": "std", "mnist": "std", "famnist": "std"}[args.dataset]
    args.dataloc = args.dataloc + args.dataset + "/"
    if args.agr == "fltrust":
        args.is_server = True
    if args.num_clients == 50:
        args.num_attackers = [15]
    PATH = args.dataset + "_" + args.at_type + "_"
    if args.at_type in ["adaptive", "min_max", "min_sum"]:
        PATH += args.dev_type + "_"
    PATH += args.at_agr + "_"
    PATH += args.agr + "_"
    if args.agr == "cluster":
        PATH += str(args.num_clusters) + "_"
    PATH += "n_clients" + str(args.num_clients) + "_" + "alpha" + str(args.alpha)

    args.PATH = "./checkpoint/" + PATH
    args.PATH_txt = "./result/" + PATH
    if args.agr == "cluster":
        args.PATH_txt = args.PATH_txt + "_" + str(args.gap)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return args


class cifar10_label_dis_skew_data:
    va_data_tensor = 0
    va_label_tensor = 0
    te_data_tensor = 0
    te_label_tensor = 0
    user_tr_data_tensors = 0
    user_tr_label_tensors = 0

    def split_noniid(self, train_idcs, train_labels, alpha, n_clients):
        n_classes = train_labels.max() + 1  # train_labels内的元素为[0,61],类数为62
        label_distribution = np.random.dirichlet(
            [alpha] * n_clients, n_classes
        )  # (类数K=62,客户端数N=10)，行向量是类K在每个客户端上的概率分布向量
        class_idcs = [
            np.argwhere(train_labels[train_idcs] == y).flatten()
            for y in range(n_classes)
        ]  # 把10000个数据中的62个类分离出来,得到62个idx组成的ndarray
        client_idcs = [[] for _ in range(n_clients)]  # （10，62）
        for c, fracs in zip(
            class_idcs, label_distribution
        ):  # c是label为k的样本的idx,fracs是个类k的概率分布向量
            for i, idcs in enumerate(
                np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
            ):  # cumsum是元素累加(类k在第1个客户端的概率，前2个的概率...),cumsum*len(c)是类k前n个客户端的数量, idcs是第k个类在第i个客户端李
                client_idcs[i] += [idcs]  # 把类k的数据的idx(max=10000)分到n个客户端中
        client_idcs = [
            train_idcs[np.concatenate(idcs)] for idcs in client_idcs
        ]  # concatenate:对数列合并，把62个类的训练样本idx(max=10000)合起来,换成所有样本下的idx(max=60000)
        return client_idcs  # (10),内容是客户端的样本在所有样本中的idx

    def split_noniid2(self, n_dis, train_labels, alpha, n_clients):
        n_classes = train_labels.max() + 1  # train_labels内的元素为[0,9],类数为10
        label_distribution = np.random.dirichlet(
            [alpha] * n_classes, n_dis
        )  # 从n_classes维对称狄利克雷分布中采n_clients个样本
        class_idx_list = [
            np.argwhere(train_labels == y).flatten().tolist() for y in range(n_classes)
        ]  # 把10个类分离出来,得到10个idx组成的ndarray

        client_data_num = len(train_labels) // n_clients  # 每个客户端有多少数据
        dis_each_class_num = client_data_num * label_distribution  # 每个客户端在每个类中有多少个数据
        dis_each_class_num = np.array(dis_each_class_num + 1).astype(
            int
        )  # 确保每个类都有一个样本，auc计算不报错
        client_idx_list = [[] for _ in range(n_clients)]
        for i in range(n_clients):
            for j in range(n_classes):
                # 客户端i的第j类数据的idx
                idx = random.sample(
                    class_idx_list[j],
                    dis_each_class_num[(i // (n_clients // n_dis)) % n_dis][j],
                )  # 从第j类样本的idx中取num个
                client_idx_list[i] += [idx]
        client_idx_list = [
            np.concatenate(idx) for idx in client_idx_list
        ]  # concatenate:对数列合并，把62个类的训练样本idx(max=10000)合起来,换成所有样本下的idx(max=60000)
        # return client_idx_list
        return sum(
            [
                [client_idx_list[i] for i in range(j, n_clients, n_clients // n_dis)]
                for j in range(n_clients // n_dis)
            ],
            [],
        )

    def __init__(self, dataloc, alpha, n_client):
        DIRICHLET_ALPHA = alpha
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        cifar10_train = datasets.CIFAR10(
            root=dataloc + "data", train=True, download=False, transform=train_transform
        )
        X, Y = [], []
        for i in range(len(cifar10_train)):  # 把训练集的50000个样本提取
            X.append(cifar10_train[i][0].numpy())
            Y.append(cifar10_train[i][1])
        X, Y = np.array(X), np.array(Y)
        train_labels = Y

        # n_dis = 5
        # client_idcs = self.split_noniid2(n_dis, train_labels, DIRICHLET_ALPHA, n_client)

        train_idx = np.arange(len(train_labels))
        client_idcs = self.split_noniid(
            train_idx, train_labels, alpha=DIRICHLET_ALPHA, n_clients=n_client
        )  # 根据对称狄利克雷参数划分noniid数据

        net_cls_counts = {}
        for net_i, dataidx in enumerate(client_idcs):
            unq, unq_cnt = np.unique(
                Y[dataidx], return_counts=True
            )  # 去除重复元素，得到类名和该类元素数量
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # 改为字典形式
            net_cls_counts[net_i] = tmp
        print("Data statistics Train:\n \t %s" % str(net_cls_counts))

        # 数据分布可视化
        # plt.figure(figsize=(20, 3))
        # plt.hist([train_labels[idc] for idc in client_idcs], stacked=True, bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1), label=["Client {}".format(i) for i in range(n_client)])
        # mapp = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # plt.xticks(np.arange(10), mapp)
        # plt.legend()
        # plt.show()

        user_tr_data_tensors = []
        user_tr_label_tensors = []
        for client_data_idx in client_idcs:
            user_tr_data_tensors.append(
                torch.from_numpy(X[client_data_idx]).type(torch.FloatTensor)
            )
            user_tr_label_tensors.append(
                torch.from_numpy(Y[client_data_idx]).type(torch.LongTensor)
            )

        cifar10_test = datasets.CIFAR10(
            root=dataloc + "data",
            train=False,
            download=False,
            transform=train_transform,
        )
        X = []
        Y = []
        for i in range(len(cifar10_test)):  # 把测试集的10000个样本提取
            X.append(cifar10_test[i][0].numpy())
            Y.append(cifar10_test[i][1])
        X = np.array(X)
        Y = np.array(Y)

        if not os.path.isfile("./cifar10_shuffle.pkl"):
            shuffle_idx = np.arange(len(X))
            np.random.shuffle(shuffle_idx)
            pickle.dump(shuffle_idx, open("./cifar10_shuffle.pkl", "wb"))
        else:
            shuffle_idx = pickle.load(open("./cifar10_shuffle.pkl", "rb"))
        X = X[shuffle_idx]
        Y = Y[shuffle_idx]

        test_labels = Y[:5000]
        test_data = X[:5000]
        te_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
        te_label_tensor = torch.from_numpy(test_labels).type(torch.LongTensor)

        unq, unq_cnt = np.unique(Y[:5000], return_counts=True)  # 去除重复元素，得到类名和该类元素数量
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # 改为字典形式
        print("Data statistics Test:\n \t %s" % str(tmp))

        val_labels = Y[5000:]
        val_data = X[5000:]
        va_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor)
        va_label_tensor = torch.from_numpy(val_labels).type(torch.LongTensor)

        self.va_data_tensor = va_data_tensor
        self.va_label_tensor = va_label_tensor
        self.te_data_tensor = te_data_tensor
        self.te_label_tensor = te_label_tensor
        self.user_tr_data_tensors = user_tr_data_tensors
        self.user_tr_label_tensors = user_tr_label_tensors


class mnist_label_dis_skew_data:
    va_data_tensor = 0
    va_label_tensor = 0
    te_data_tensor = 0
    te_label_tensor = 0
    user_tr_data_tensors = 0
    user_tr_label_tensors = 0

    def split_noniid(self, train_idcs, train_labels, alpha, n_clients):
        n_classes = train_labels.max() + 1  # train_labels内的元素为[0,61],类数为62
        label_distribution = np.random.dirichlet(
            [alpha] * n_clients, n_classes
        )  # (类数K=62,客户端数N=10)，行向量是类K在每个客户端上的概率分布向量
        class_idcs = [
            np.argwhere(train_labels[train_idcs] == y).flatten()
            for y in range(n_classes)
        ]  # 把10000个数据中的62个类分离出来,得到62个idx组成的ndarray
        client_idcs = [[] for _ in range(n_clients)]  # （10，62）
        for c, fracs in zip(
            class_idcs, label_distribution
        ):  # c是label为k的样本的idx,fracs是个类k的概率分布向量
            for i, idcs in enumerate(
                np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
            ):  # cumsum是元素累加(类k在第1个客户端的概率，前2个的概率...),cumsum*len(c)是类k前n个客户端的数量, idcs是第k个类在第i个客户端李
                client_idcs[i] += [idcs]  # 把类k的数据的idx(max=10000)分到n个客户端中
        client_idcs = [
            train_idcs[np.concatenate(idcs)] for idcs in client_idcs
        ]  # concatenate:对数列合并，把62个类的训练样本idx(max=10000)合起来,换成所有样本下的idx(max=60000)
        return client_idcs  # (10),内容是客户端的样本在所有样本中的idx

    def __init__(self, dataloc, alpha, n_client):
        DIRICHLET_ALPHA = alpha
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        mnist_train = datasets.MNIST(
            root=dataloc + "data", train=True, download=False, transform=train_transform
        )
        X, Y = [], []
        for i in range(len(mnist_train)):  # 把训练集的50000个样本提取
            X.append(mnist_train.data[i].numpy())
            Y.append(mnist_train.targets[i])
        X, Y = np.array(X), np.array(Y)
        train_labels = Y
        train_idx = np.arange(len(train_labels))
        client_idcs = self.split_noniid(
            train_idx, train_labels, alpha=DIRICHLET_ALPHA, n_clients=n_client
        )  # 根据对称狄利克雷参数划分noniid数据

        net_cls_counts = {}
        for net_i, dataidx in enumerate(client_idcs):
            unq, unq_cnt = np.unique(
                Y[dataidx], return_counts=True
            )  # 去除重复元素，得到类名和该类元素数量
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # 改为字典形式
            net_cls_counts[net_i] = tmp
        print("Data statistics Train:\n \t %s" % str(net_cls_counts))

        # 数据分布可视化
        # plt.figure(figsize=(20, 3))
        # plt.hist([train_labels[idc] for idc in client_idcs], stacked=True, bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1), label=["Client {}".format(i) for i in range(n_client)])
        # mapp = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # plt.xticks(np.arange(10), mapp)
        # plt.legend()
        # plt.show()

        user_tr_data_tensors = []
        user_tr_label_tensors = []
        for client_data_idx in client_idcs:
            user_tr_data_tensors.append(
                torch.from_numpy(X[client_data_idx]).type(torch.FloatTensor)
            )
            user_tr_label_tensors.append(
                torch.from_numpy(Y[client_data_idx]).type(torch.LongTensor)
            )

        mnist_test = datasets.MNIST(
            root=dataloc + "data",
            train=False,
            download=False,
            transform=train_transform,
        )
        X = []
        Y = []
        for i in range(len(mnist_test)):  # 把测试集的10000个样本提取
            X.append(mnist_test.data[i].numpy())
            Y.append(mnist_test.targets[i])
        X = np.array(X)
        Y = np.array(Y)

        if not os.path.isfile("./mnist_shuffle.pkl"):
            shuffle_idx = np.arange(len(X))
            np.random.shuffle(shuffle_idx)
            pickle.dump(shuffle_idx, open("./mnist_shuffle.pkl", "wb"))
        else:
            shuffle_idx = pickle.load(open("./mnist_shuffle.pkl", "rb"))
        X = X[shuffle_idx]
        Y = Y[shuffle_idx]

        test_labels = Y[:5000]
        test_data = X[:5000]
        te_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
        te_label_tensor = torch.from_numpy(test_labels).type(torch.LongTensor)

        unq, unq_cnt = np.unique(Y[:5000], return_counts=True)  # 去除重复元素，得到类名和该类元素数量
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # 改为字典形式
        print("Data statistics Test:\n \t %s" % str(tmp))

        val_labels = Y[5000:]
        val_data = X[5000:]
        va_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor)
        va_label_tensor = torch.from_numpy(val_labels).type(torch.LongTensor)

        self.va_data_tensor = va_data_tensor
        self.va_label_tensor = va_label_tensor
        self.te_data_tensor = te_data_tensor
        self.te_label_tensor = te_label_tensor
        self.user_tr_data_tensors = user_tr_data_tensors
        self.user_tr_label_tensors = user_tr_label_tensors


class famnist_label_dis_skew_data:
    va_data_tensor = 0
    va_label_tensor = 0
    te_data_tensor = 0
    te_label_tensor = 0
    user_tr_data_tensors = 0
    user_tr_label_tensors = 0

    def split_noniid(self, train_idcs, train_labels, alpha, n_clients):
        n_classes = train_labels.max() + 1  # train_labels内的元素为[0,61],类数为62
        label_distribution = np.random.dirichlet(
            [alpha] * n_clients, n_classes
        )  # (类数K=62,客户端数N=10)，行向量是类K在每个客户端上的概率分布向量
        class_idcs = [
            np.argwhere(train_labels[train_idcs] == y).flatten()
            for y in range(n_classes)
        ]  # 把10000个数据中的62个类分离出来,得到62个idx组成的ndarray
        client_idcs = [[] for _ in range(n_clients)]  # （10，62）
        for c, fracs in zip(
            class_idcs, label_distribution
        ):  # c是label为k的样本的idx,fracs是个类k的概率分布向量
            for i, idcs in enumerate(
                np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))
            ):  # cumsum是元素累加(类k在第1个客户端的概率，前2个的概率...),cumsum*len(c)是类k前n个客户端的数量, idcs是第k个类在第i个客户端李
                client_idcs[i] += [idcs]  # 把类k的数据的idx(max=10000)分到n个客户端中
        client_idcs = [
            train_idcs[np.concatenate(idcs)] for idcs in client_idcs
        ]  # concatenate:对数列合并，把62个类的训练样本idx(max=10000)合起来,换成所有样本下的idx(max=60000)
        return client_idcs  # (10),内容是客户端的样本在所有样本中的idx

    def __init__(self, dataloc, alpha, n_client):
        DIRICHLET_ALPHA = alpha
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        famnist_train = datasets.FashionMNIST(
            root=dataloc + "data", train=True, download=True, transform=train_transform
        )
        X, Y = [], []
        for i in range(len(famnist_train)):  # 把训练集的50000个样本提取
            X.append(famnist_train.data[i].numpy())
            Y.append(famnist_train.targets[i])
        X, Y = np.array(X), np.array(Y)
        train_labels = Y
        train_idx = np.arange(len(train_labels))
        client_idcs = self.split_noniid(
            train_idx, train_labels, alpha=DIRICHLET_ALPHA, n_clients=n_client
        )  # 根据对称狄利克雷参数划分noniid数据

        net_cls_counts = {}
        for net_i, dataidx in enumerate(client_idcs):
            unq, unq_cnt = np.unique(
                Y[dataidx], return_counts=True
            )  # 去除重复元素，得到类名和该类元素数量
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # 改为字典形式
            net_cls_counts[net_i] = tmp
        print("Data statistics Train:\n \t %s" % str(net_cls_counts))

        # 数据分布可视化
        # plt.figure(figsize=(20, 3))
        # plt.hist([train_labels[idc] for idc in client_idcs], stacked=True, bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1), label=["Client {}".format(i) for i in range(n_client)])
        # mapp = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # plt.xticks(np.arange(10), mapp)
        # plt.legend()
        # plt.show()

        user_tr_data_tensors = []
        user_tr_label_tensors = []
        for client_data_idx in client_idcs:
            user_tr_data_tensors.append(
                torch.from_numpy(X[client_data_idx]).type(torch.FloatTensor)
            )
            user_tr_label_tensors.append(
                torch.from_numpy(Y[client_data_idx]).type(torch.LongTensor)
            )

        famnist_test = datasets.FashionMNIST(
            root=dataloc + "data",
            train=False,
            download=True,
            transform=train_transform,
        )
        X = []
        Y = []
        for i in range(len(famnist_test)):  # 把测试集的10000个样本提取
            X.append(famnist_test.data[i].numpy())
            Y.append(famnist_test.targets[i])
        X = np.array(X)
        Y = np.array(Y)

        if not os.path.isfile("./famnist_shuffle.pkl"):
            shuffle_idx = np.arange(len(X))
            np.random.shuffle(shuffle_idx)
            pickle.dump(shuffle_idx, open("./famnist_shuffle.pkl", "wb"))
        else:
            shuffle_idx = pickle.load(open("./famnist_shuffle.pkl", "rb"))
        X = X[shuffle_idx]
        Y = Y[shuffle_idx]

        test_labels = Y[:5000]
        test_data = X[:5000]
        te_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
        te_label_tensor = torch.from_numpy(test_labels).type(torch.LongTensor)

        unq, unq_cnt = np.unique(Y[:5000], return_counts=True)  # 去除重复元素，得到类名和该类元素数量
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # 改为字典形式
        print("Data statistics Test:\n \t %s" % str(tmp))

        val_labels = Y[5000:]
        val_data = X[5000:]
        va_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor)
        va_label_tensor = torch.from_numpy(val_labels).type(torch.LongTensor)

        self.va_data_tensor = va_data_tensor
        self.va_label_tensor = va_label_tensor
        self.te_data_tensor = te_data_tensor
        self.te_label_tensor = te_label_tensor
        self.user_tr_data_tensors = user_tr_data_tensors
        self.user_tr_label_tensors = user_tr_label_tensors


class cifar10_iid:
    va_data_tensor = 0
    va_label_tensor = 0
    te_data_tensor = 0
    te_label_tensor = 0
    user_tr_data_tensors = 0
    user_tr_label_tensors = 0

    def __init__(self, dataloc, alpha, n_client):
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        cifar10_train = datasets.CIFAR10(
            root=dataloc, train=True, download=False, transform=train_transform
        )

        X = []
        Y = []
        for i in range(len(cifar10_train)):
            X.append(cifar10_train[i][0].numpy())
            Y.append(cifar10_train[i][1])
        X = np.array(X)
        Y = np.array(Y)

        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)

        X = X[all_indices]
        Y = Y[all_indices]

        user_tr_len = len(X) / n_client

        user_tr_data_tensors = []
        user_tr_label_tensors = []

        for i in range(n_client):
            user_tr_data_tensor = torch.from_numpy(
                X[user_tr_len * i : user_tr_len * (i + 1)]
            ).type(torch.FloatTensor)
            user_tr_label_tensor = torch.from_numpy(
                Y[user_tr_len * i : user_tr_len * (i + 1)]
            ).type(torch.LongTensor)

            user_tr_data_tensors.append(user_tr_data_tensor)
            user_tr_label_tensors.append(user_tr_label_tensor)
            print("user %d tr len %d" % (i, len(user_tr_data_tensor)))

        cifar10_test = datasets.CIFAR10(
            root=dataloc + "data",
            train=False,
            download=False,
            transform=train_transform,
        )
        X = []
        Y = []
        for i in range(len(cifar10_test)):  # 把测试集的10000个样本提取
            X.append(cifar10_test[i][0].numpy())
            Y.append(cifar10_test[i][1])
        X = np.array(X)
        Y = np.array(Y)

        if not os.path.isfile("./cifar10_shuffle.pkl"):
            shuffle_idx = np.arange(len(X))
            np.random.shuffle(shuffle_idx)
            pickle.dump(shuffle_idx, open("./cifar10_shuffle.pkl", "wb"))
        else:
            shuffle_idx = pickle.load(open("./cifar10_shuffle.pkl", "rb"))
        X = X[shuffle_idx]
        Y = Y[shuffle_idx]

        test_labels = Y[:5000]
        test_data = X[:5000]
        te_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
        te_label_tensor = torch.from_numpy(test_labels).type(torch.LongTensor)

        unq, unq_cnt = np.unique(Y[:5000], return_counts=True)  # 去除重复元素，得到类名和该类元素数量
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # 改为字典形式
        print("Data statistics Test:\n \t %s" % str(tmp))

        val_labels = Y[5000:]
        val_data = X[5000:]
        va_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor)
        va_label_tensor = torch.from_numpy(val_labels).type(torch.LongTensor)

        self.va_data_tensor = va_data_tensor
        self.va_label_tensor = va_label_tensor
        self.te_data_tensor = te_data_tensor
        self.te_label_tensor = te_label_tensor
        self.user_tr_data_tensors = user_tr_data_tensors
        self.user_tr_label_tensors = user_tr_label_tensors


def cluster_test(updates, train_tools):
    (
        te_data_tensor,
        te_label_tensor,
        fed_model,
        optimizer_fed,
        lr,
        optimizer_type,
    ) = train_tools
    criterion = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()
    true_los_list = []
    true_acc_list = []
    for _ in range(len(updates)):  # 对每一个cluster
        clients_model = copy.deepcopy(fed_model)
        if optimizer_type == "SGD":
            client_optimizer = SGD(
                clients_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
            )
        elif optimizer_type == "Adam":
            client_optimizer = Adam(clients_model.parameters(), lr=lr)
        client_optimizer.load_state_dict(optimizer_fed.state_dict())
        client_optimizer.zero_grad()
        model_grads = []
        start_idx = 0
        for i, param in enumerate(clients_model.parameters()):
            param_ = updates[_][
                start_idx : start_idx + len(param.data.view(-1))
            ].reshape(param.data.shape)
            start_idx = start_idx + len(param.data.view(-1))
            param_ = param_.cuda()
            model_grads.append(param_)
        client_optimizer.step(model_grads)  # 用包含恶意梯度的梯度来更新

        te_los, te_acc = test(
            te_data_tensor, te_label_tensor, clients_model, criterion, use_cuda
        )
        true_los_list.append(te_los.cpu())
        true_acc_list.append(te_acc.cpu())
    return [true_los_list, true_acc_list]


def abcd_leaf(mal_updates, train_tools, node, last_los_diff, gap):
    cluster_leaf = []
    if node.left is None:
        return set(node.pre_order())
    else:
        left_node = node.get_left()
        right_node = node.get_right()
        cluster_grads = [
            torch.mean(mal_updates[left_node.pre_order()], dim=0),
            torch.mean(mal_updates[right_node.pre_order()], dim=0),
        ]
        true_los_list, true_acc_list = cluster_test(cluster_grads, train_tools)
        new_los_diff = np.abs(true_los_list[0] - true_los_list[1])
        # 分不开就返回全部
        # if new_los_diff < last_los_diff and new_los_diff < gap:
        if new_los_diff < last_los_diff and new_los_diff < gap:
            return set(node.pre_order())
        # 分的开就继续分
        else:
            left_idx = abcd_leaf(mal_updates, train_tools, left_node, 10, gap)
            if isinstance(left_idx, list):  # 如果是列表，说明有子类，要extend
                cluster_leaf.extend(left_idx)
            else:  # 如果是集合，说明只有一个类，要加入列表
                cluster_leaf.extend([left_idx])
            right_idx = abcd_leaf(mal_updates, train_tools, right_node, 10, gap)
            if isinstance(right_idx, list):  # 如果是列表，说明有子类，要extend
                cluster_leaf.extend(right_idx)
            else:  # 如果是集合，说明只有一个类，要加入列表
                cluster_leaf.extend([right_idx])
            return cluster_leaf


def abcd_agg(mal_updates, train_tools, gap, linkage_method):
    num_clusters = 2
    x = mal_updates.cpu().numpy()
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters, metric="cosine", linkage=linkage_method
    ).fit(x)
    cluster_label = clustering.labels_
    label_m = cluster_label[0]
    index_m = np.where(cluster_label == label_m)
    index_zero = np.where(cluster_label == 0)
    cluster_label[index_zero] = label_m
    cluster_label[index_m] = 0
    grads_list = []
    for i in range(num_clusters):
        cluster_i_point_indices = np.where(cluster_label == i)[0]
        other_indices = list(set(np.arange(x.shape[0])) - set(cluster_i_point_indices))
        grads_list.append(torch.mean(mal_updates[cluster_i_point_indices], dim=0))  # 选择
        grads_list.append(torch.mean(mal_updates[other_indices], dim=0))  # 去除
    true_los_list, _ = cluster_test(grads_list, train_tools)
    true_los_list = np.array(
        [
            true_los_list[i] - true_los_list[i + 1]
            for i in range(0, len(true_los_list), 2)
        ]
    )

    bad_idx = np.argsort(np.array(true_los_list))[-1]
    good_idx = np.setdiff1d(np.arange(num_clusters), bad_idx)
    good_idx = np.concatenate(
        [np.where(cluster_label == good_idx[i])[0] for i in range(len(good_idx))]
    )
    good_agr_grads = mal_updates[good_idx]
    return torch.mean(good_agr_grads, dim=0)


def abcd_diff(mal_updates, train_tools, gap, linkage_method):
    x = mal_updates.cpu().numpy()
    Z = linkage(x, method=linkage_method, metric="cosine")
    root_node = to_tree(Z)
    if gap == 0:
        leaf_list = [
            set(root_node.get_left().pre_order()),
            set(root_node.get_right().pre_order()),
        ]
    else:
        leaf_list = abcd_leaf(mal_updates, train_tools, root_node, 0, gap)
    cluster_label = np.arange(len(mal_updates))
    grads_list = []
    for i in leaf_list:
        other_indices = list(set(cluster_label) - i)
        grads_list.append(torch.mean(mal_updates[list(i)], dim=0))  # 选择
        grads_list.append(torch.mean(mal_updates[other_indices], dim=0))
    true_los_list, _ = cluster_test(grads_list, train_tools)
    true_los_list = np.array(
        [
            true_los_list[i] - true_los_list[i + 1]
            for i in range(0, len(true_los_list), 2)
        ]
    )

    bad_idx = np.argsort(np.array(true_los_list))[-1]
    print(leaf_list)
    print(bad_idx)
    print(true_los_list)
    good_idx = sum(
        [
            list(i)
            for i in np.array(leaf_list)[
                np.setdiff1d(np.arange(len(leaf_list)), bad_idx)
            ]
        ],
        [],
    )
    good_agr_grads = mal_updates[good_idx]
    return torch.mean(good_agr_grads, dim=0)


def abcd_min_los(mal_updates, train_tools, gap, linkage_method):
    # def abcd(mal_updates, train_tools, gap, linkage_method):
    x = mal_updates.cpu().numpy()
    Z = linkage(x, method=linkage_method, metric="cosine")
    root_node = to_tree(Z)
    if gap == 0:
        leaf_list = [
            set(root_node.get_left().pre_order()),
            set(root_node.get_right().pre_order()),
        ]
    else:
        leaf_list = abcd_leaf(mal_updates, train_tools, root_node, 0, gap)
    grads_list = []

    # 去除1个后选最好的
    cluster_label = np.arange(len(mal_updates))
    for i in leaf_list:
        other_indices = list(set(cluster_label) - i)
        grads_list.append(torch.mean(mal_updates[other_indices], dim=0))
    true_los_list, _ = cluster_test(grads_list, train_tools)
    good_cluster = np.argsort(np.array(true_los_list))[0]
    # if len(mal_updates) == 30:
    #     n_attacker = 10
    # elif len(mal_updates) == 50:
    #     n_attacker = 15
    # with open(args.PATH_txt + f'nat{n_attacker}' + '.txt', 'a') as txt:
    #     print(leaf_list, file=txt)
    #     print(good_cluster, file=txt)
    #     print(true_los_list, file=txt)
    print(leaf_list)
    print(good_cluster)
    print(true_los_list)
    return grads_list[good_cluster]

    # 选择1个后去掉最坏的
    # for i in leaf_list:
    #     grads_list.append(torch.mean(mal_updates[list(i)], dim=0))  # 选择
    # true_los_list, _ = cluster_test(grads_list, train_tools)
    # bad_cluster = np.argsort(np.array(true_los_list))[-1]
    # good_grad = mal_updates[sum([list(i) for i in np.array(leaf_list)[np.setdiff1d(np.arange(len(leaf_list)), bad_cluster)]], [])]
    # return torch.mean(good_grad, dim=0)

    # true_los_list = np.array([true_los_list[i] - true_los_list[i + 1] for i in range(0, len(true_los_list), 2)])
    # bad_idx = np.argsort(np.array(true_los_list))[-1]
    #
    # good_idx = sum([list(i) for i in np.array(leaf_list)[np.setdiff1d(np.arange(len(leaf_list)), bad_idx)]], [])
    # good_agr_grads = mal_updates[good_idx]
    #
    # # good_agr_grads = mal_updates[n_attacker:]
    # if len(mal_updates) == 30:
    #     n_attacker = 10
    # elif len(mal_updates) == 50:
    #     n_attacker = 15
    # i = set(np.arange(n_attacker))
    # right_list = []
    # other_indices = list(set(cluster_label) - i)
    # right_list.append(torch.mean(mal_updates[list(i)], dim=0))  # 选择
    # right_list.append(torch.mean(mal_updates[other_indices], dim=0))
    # right_list.append(torch.mean(good_agr_grads, dim=0))
    # right_los_list, _ = cluster_test(right_list, train_tools)
    # right_los = right_los_list[1]
    # cluster_los = right_los_list[2]
    # right_los_list = right_los_list[:2]
    # right_los_list = np.array([right_los_list[i] - right_los_list[i + 1] for i in range(0, len(right_los_list), 2)])
    #
    # with open(args.PATH_txt + f'nat{n_attacker}' + '.txt', 'a') as txt:
    #     print(f'cluster loss dif {true_los_list[bad_idx]:.5f} right loss dif {right_los_list[0]:.5f} cluster loss {cluster_los:.5f} right loss {right_los:.5f}', file=txt)
    # print(f'cluster loss dif {true_los_list[bad_idx]:.5f} right loss dif {right_los_list[0]:.5f} cluster loss {cluster_los:.5f} right loss {right_los:.5f}')
    #
    # return torch.mean(good_agr_grads, dim=0)


# 攻击模板
def full_knowledge_attack(args, datatensors):
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    val_data_tensor = datatensors.va_data_tensor
    val_label_tensor = datatensors.va_label_tensor
    test_data_tensor = datatensors.te_data_tensor
    test_label_tensor = datatensors.te_label_tensor
    tr_data_tensors = datatensors.user_tr_data_tensors
    tr_label_tensors = datatensors.user_tr_label_tensors
    criterion = nn.CrossEntropyLoss().to(device)
    fed_lr = args.lr
    resume = args.resume
    save = args.save
    for n_attacker in args.num_attackers:
        if save:
            with open(args.PATH_txt + f"nat{n_attacker}" + ".txt", "a") as txt:
                if args.at_agr == "cluster":
                    print(f"gap {args.gap}", file=txt)
                print(
                    f"alpha {args.alpha} n_clients {args.num_clients} nat {n_attacker} at {args.at_type} at_agr {args.at_agr} agr {args.agr}",
                    file=txt,
                )
                print(f"resume {args.resume} save {args.save}", file=txt)
        if args.at_agr == "cluster":
            print(f"gap {args.gap}")
        print(
            f"dataset {args.dataset} alpha {args.alpha} n_clients {args.num_clients} nat {n_attacker} at {args.at_type} at_agr {args.at_agr} agr {args.agr}"
        )
        print(f"resume {args.resume} save {args.save}")
        epoch_num = 0
        best_global_acc = 0
        torch.cuda.empty_cache()

        if args.arch == "resnet18":
            # fed_model = ResNet18()
            fed_model = DenseNet()
        # fed_model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # fed_model.fc = nn.Linear(fed_model.fc.in_features, args.num_classes)
        elif args.arch == "CNN":
            fed_model = mnist_conv()
            fed_model.apply(weights_init)
        elif args.arch == "MLP":
            fed_model = MnistNet()
        fed_model = fed_model.to(device)
        if args.optimizer == "SGD":
            optimizer_fed = SGD(
                fed_model.parameters(), lr=fed_lr, momentum=0.9, weight_decay=5e-4
            )
        elif args.optimizer == "Adam":
            optimizer_fed = Adam(fed_model.parameters(), lr=fed_lr)
        client_shuffle_idcs = [[] for _ in range(args.num_clients)]

        if resume:  # 是否保存checkpoint，默认不保存
            epoch_num = resume
            PATH_tar = (
                args.PATH
                + "n_at"
                + str(n_attacker)
                + "epoch"
                + str(epoch_num)
                + ".pth.tar"
            )
            assert os.path.isfile(PATH_tar), (
                "Error: no user checkpoint at %s" % PATH_tar
            )
            if save:
                with open(args.PATH_txt + f"nat{n_attacker}" + ".txt", "a") as txt:
                    print(f"resumed from {PATH_tar}.pth.tar", file=txt)
            print(f"resumed from {PATH_tar}.pth.tar")
            checkpoint = torch.load(
                PATH_tar, map_location="cuda:%d" % torch.cuda.current_device()
            )
            epoch_num = checkpoint["epoch"] + 1
            fed_model.load_state_dict(checkpoint["state_dict"])
            optimizer_fed.load_state_dict(checkpoint["optimizer"])
            resume = 0
            client_shuffle_idcs = checkpoint["client_shuffle_idcs"]
            for i in range(len(client_shuffle_idcs)):
                tr_data_tensors[i] = tr_data_tensors[i][client_shuffle_idcs[i]]
                tr_label_tensors[i] = tr_label_tensors[i][client_shuffle_idcs[i]]
            best_global_acc = checkpoint["best_acc"]
            random_state = checkpoint["random_state"]
            np_random_state = checkpoint["np_random_state"]
            torch_rng_state = checkpoint["torch_rng_state"]
            random.setstate(random_state)
            np.random.set_state(np_random_state)
            torch.set_rng_state(torch_rng_state.cpu())

        acc_point, los_point = [], []
        while epoch_num <= args.epoch:  # 1200个epoch
            user_grads = []  # full_knowledge良性梯度
            # 良性客户端训练一次 & full knowledge保存梯度
            for i in range(n_attacker, args.num_clients):  # 和partial不同
                nbatches = (
                    len(tr_data_tensors[i]) // args.batchsize
                )  # 每个客户端一轮要训练几次，向下取整
                if epoch_num % nbatches == 0:  # 如果该客户端所有数据都训练完了就打乱再训练
                    r = np.arange(len(tr_data_tensors[i]))
                    np.random.shuffle(r)
                    client_shuffle_idcs[i] = (
                        client_shuffle_idcs[i][r] if len(client_shuffle_idcs[i]) else r
                    )
                    tr_data_tensors[i] = tr_data_tensors[i][r]
                    tr_label_tensors[i] = tr_label_tensors[i][r]
                inputs = tr_data_tensors[i][
                    (epoch_num % nbatches)
                    * args.batchsize : ((epoch_num % nbatches) + 1)
                    * args.batchsize
                ]
                targets = tr_label_tensors[i][
                    (epoch_num % nbatches)
                    * args.batchsize : ((epoch_num % nbatches) + 1)
                    * args.batchsize
                ]
                inputs, targets = inputs.to(device), targets.to(device)
                fed_model = fed_model.to(device)
                inputs, targets = torch.autograd.Variable(
                    inputs
                ), torch.autograd.Variable(
                    targets
                )  # 将tensor包装起来成为计算图中的节点(可以装载梯度信息)
                outputs = fed_model(inputs)  # 输入样本，输出预测值
                loss = criterion(outputs, targets)  # 计算预测值与标签的交叉熵损失
                fed_model.zero_grad()  # 清空模型的梯度
                loss.backward(retain_graph=True)  # 反向传播，计算当前梯度保留梯度，且梯度不free
                param_grad = []  # 把参数的梯度拼接起来
                for param in fed_model.parameters():  # 对于模型中的每个参数
                    param_grad = (
                        param.grad.data.view(-1)
                        if not len(param_grad)
                        else torch.cat((param_grad, param.grad.view(-1)))
                    )  # 将梯度拼接
                user_grads = (
                    param_grad[None, :]
                    if len(user_grads) == 0
                    else torch.cat((user_grads, param_grad[None, :]), 0)
                )  # 保存该轮良性客户端的梯度
            del inputs
            del targets
            benign_grads = user_grads  # 和partial不同

            if args.is_server:
                torch.cuda.empty_cache()
                server_inputs = val_data_tensor[:1000]
                server_targets = val_label_tensor[:1000]
                server_inputs, server_targets = server_inputs.to(
                    device
                ), server_targets.to(device)
                server_inputs, server_targets = torch.autograd.Variable(
                    server_inputs
                ), torch.autograd.Variable(server_targets)
                server_outputs = fed_model(server_inputs)  # 输入样本，输出预测值
                server_loss = criterion(
                    server_outputs, server_targets
                )  # 计算预测值与标签的交叉熵损失
                fed_model.zero_grad()  # 清空模型的梯度
                server_loss.backward(retain_graph=True)  # 反向传播，计算当前梯度保留梯度，且梯度不free
                param_grad = []  # 把参数的梯度拼接起来
                for param in fed_model.parameters():  # 对于模型中的每个参数
                    param_grad = (
                        param.grad.data.view(-1)
                        if not len(param_grad)
                        else torch.cat((param_grad, param.grad.view(-1)))
                    )  # 将梯度拼接
                server_grad = param_grad
            if epoch_num in args.schedule:
                for param_group in optimizer_fed.param_groups:
                    param_group["lr"] *= 0.5
                fed_lr *= 0.5
                if save:
                    with open(args.PATH_txt + f"nat{n_attacker}" + ".txt", "a") as txt:
                        print(
                            "New learnin rate ",
                            param_group["lr"],
                            "epoch",
                            epoch_num,
                            file=txt,
                        )
                print("New learnin rate ", param_group["lr"], "epoch", epoch_num)

            agg_grads = torch.mean(benign_grads, 0)
            if args.at_type == "none":
                malicious_grads = user_grads
            else:
                if args.at_type == "rev":
                    malicious_grads = reverse_attack(
                        benign_grads, agg_grads, n_attacker
                    )
                elif args.at_type == "fang":
                    if args.at_agr == "mkrum" or args.at_agr == "krum":
                        multi_k = True if args.agr == "mkrum" else False
                        malicious_grads = get_malicious_updates_fang(
                            benign_grads, agg_grads, n_attacker, multi_k
                        )
                    elif args.at_agr == "trmean":
                        malicious_grads = get_malicious_updates_fang_trmean(
                            benign_grads, agg_grads, n_attacker
                        )
                elif args.at_type == "adaptive":
                    if args.at_agr == "mkrum" or args.at_agr == "krum":
                        multi_k = True if args.agr == "mkrum" else False
                        malicious_grads = adaptive_attack_mkrum(
                            benign_grads,
                            agg_grads,
                            n_attacker,
                            multi_k,
                            dev_type=args.dev_type,
                        )
                    elif args.at_agr == "trmean":
                        malicious_grads = adaptive_attack_trmean(
                            benign_grads,
                            agg_grads,
                            n_attacker,
                            args.num_clients,
                            dev_type=args.dev_type,
                        )
                elif args.at_type == "min_max":
                    malicious_grads = adaptive_min_max(
                        benign_grads, agg_grads, n_attacker, args.dev_type
                    )
                elif args.at_type == "min_sum":
                    malicious_grads = adaptive_min_sum(
                        benign_grads, agg_grads, n_attacker, args.dev_type
                    )

                for _ in range(
                    1, n_attacker - 1
                ):  # 自己加的，不知道如何在一定欧式距离内随机采样，于是计算了高斯噪声后加上
                    distance = 1
                    noise_prop = 0.0001
                    while distance > 0.01:
                        noise = (
                            noise_prop * torch.randn(malicious_grads[0].size()).cuda()
                        )
                        distance = torch.norm(noise, p=2)
                        noise_prop *= 0.5
                    malicious_grads[_] += noise

            #  聚合
            if args.agr == "cluster":
                train_tool = [
                    val_data_tensor[:1000],
                    val_label_tensor[:1000],
                    fed_model,
                    optimizer_fed,
                    fed_lr,
                    args.optimizer,
                ]
                if args.abcd_method == "diff":
                    agg_grads = abcd_diff(
                        malicious_grads, train_tool, args.gap, args.linkage_method
                    )
                elif args.abcd_method == "min_los":
                    agg_grads = abcd_min_los(
                        malicious_grads, train_tool, args.gap, args.linkage_method
                    )
            elif args.agr == "fltrust":
                agg_grads = FLtrust(server_grad, malicious_grads)
            elif args.agr == "average":
                agg_grads = torch.mean(malicious_grads, dim=0)
            elif args.agr == "krum" or args.agr == "mkrum":
                multi_k = True if args.agr == "mkrum" else False
                agg_grads, krum_candidate = multi_krum(
                    malicious_grads, n_attacker, multi_k=multi_k
                )
            elif args.agr == "trmean":
                agg_grads = tr_mean(malicious_grads, n_attacker / args.num_clients)
            elif args.agr == "foolsgold":
                agg_grads = foolsgold(malicious_grads)

            del user_grads

            start_idx = 0
            optimizer_fed.zero_grad()  # 清空
            model_grads = []
            for i, param in enumerate(fed_model.parameters()):
                param_ = agg_grads[
                    start_idx : start_idx + len(param.data.view(-1))
                ].reshape(
                    param.data.shape
                )  # 从聚合的梯度(2000000,1)中取出每个节点的梯度
                start_idx = start_idx + len(param.data.view(-1))
                param_ = param_.cuda()
                model_grads.append(param_)
            optimizer_fed.step(model_grads)  # 更新网络参数
            if args.dataset == "mnist":
                save_epoch_num = 25
            elif args.dataset == "cifar10":
                save_epoch_num = 10
            elif args.dataset == "famnist":
                save_epoch_num = 10

            ####################################
            # test_loss, test_acc = test(test_data_tensor, test_label_tensor, fed_model, criterion, use_cuda)  # 用更新的模型测试
            # test_loss = test_loss.cpu()
            # test_acc = test_acc.cpu()
            # if test
            ####################################
            # for param_group in optimizer_fed.param_groups:
            #     param_group['lr'] *= 0.1
            # fed_lr *= 0.1
            # if save:
            #     with open(args.PATH_txt + f'nat{n_attacker}' + '.txt', 'a') as txt:
            #         print('New learnin rate ', param_group['lr'], 'epoch', epoch_num, file=txt)
            # print('New learnin rate ', param_group['lr'], 'epoch', epoch_num)
            if epoch_num % save_epoch_num == 0 or epoch_num == args.epoch - 1:
                test_loss, test_acc = test(
                    test_data_tensor, test_label_tensor, fed_model, criterion, use_cuda
                )  # 用更新的模型测试
                if math.isnan(loss):
                    if save:
                        with open(
                            args.PATH_txt + f"nat{n_attacker}" + ".txt", "a"
                        ) as txt:
                            print("loss nan", file=txt)
                    print("loss nan")
                    exit()
                test_loss = test_loss.cpu()
                test_acc = test_acc.cpu()
                best_global_acc = max(best_global_acc, test_acc.cpu())
                if save:
                    with open(args.PATH_txt + f"nat{n_attacker}" + ".txt", "a") as txt:
                        print(
                            "%s-%s: %s n_at %d epoch %d test loss %.4f test acc %.4f"
                            % (
                                args.at_type,
                                args.at_agr,
                                args.agr,
                                n_attacker,
                                epoch_num,
                                test_loss,
                                test_acc,
                            ),
                            file=txt,
                        )
                print(
                    "%s-%s: %s n_at %d epoch %d test loss %.4f test acc %.4f"
                    % (
                        args.at_type,
                        args.at_agr,
                        args.agr,
                        n_attacker,
                        epoch_num,
                        test_loss,
                        test_acc,
                    )
                )
                acc_point.append((epoch_num, test_acc))
                los_point.append((epoch_num, test_loss))

                # plt.subplot(2, 1, 1)
                # plt.plot([p[0] for p in acc_point], [p[1] for p in acc_point], color='red')
                # plt.scatter([p[0] for p in acc_point], [p[1] for p in acc_point], color='red')
                # plt.title('acc')
                # plt.subplot(2, 1, 2)
                # plt.plot([p[0] for p in los_point], [p[1] for p in los_point], color='green')
                # plt.scatter([p[0] for p in los_point], [p[1] for p in los_point], color='green')
                # plt.title('los')
                # plt.show()

                # if save:
                #     PATH_tar = args.PATH + 'n_at' + str(n_attacker) + 'epoch' + str(epoch_num) + '.pth.tar'
                #     torch.save({
                #         'epoch': epoch_num,
                #         'state_dict': fed_model.state_dict(),
                #         'optimizer': optimizer_fed.state_dict(),
                #         'acc': test_acc,
                #         'loss': test_loss,
                #         'client_shuffle_idcs': client_shuffle_idcs,
                #         'random_state': random.getstate(),
                #         'np_random_state': np.random.get_state(),
                #         'torch_rng_state': torch.get_rng_state(),
                #         'best_acc': best_global_acc
                #     }, PATH_tar)
            epoch_num += 1


if __name__ == "__main__":
    args = init_args()
    if args.dataset == "cifar10":
        # data_tensors_class = cifar10_label_dis_skew_data(args.dataloc, args.alpha, args.num_clients)
        if not os.path.isfile(
            f"./{args.dataset}_{args.num_clients}_{args.alpha}_dataset.pkl"
        ):
            data_tensors_class = cifar10_label_dis_skew_data(
                args.dataloc, args.alpha, args.num_clients
            )
            pickle.dump(
                data_tensors_class,
                open(
                    f"./{args.dataset}_{args.num_clients}_{args.alpha}_dataset.pkl",
                    "wb",
                ),
            )
        else:
            data_tensors_class = pickle.load(
                open(
                    f"./{args.dataset}_{args.num_clients}_{args.alpha}_dataset.pkl",
                    "rb",
                )
            )
    elif args.dataset == "mnist":
        if not os.path.isfile(
            f"./{args.dataset}_{args.num_clients}_{args.alpha}_dataset.pkl"
        ):
            data_tensors_class = mnist_label_dis_skew_data(
                args.dataloc, args.alpha, args.num_clients
            )
            pickle.dump(
                data_tensors_class,
                open(
                    f"./{args.dataset}_{args.num_clients}_{args.alpha}_dataset.pkl",
                    "wb",
                ),
            )
        else:
            data_tensors_class = pickle.load(
                open(
                    f"./{args.dataset}_{args.num_clients}_{args.alpha}_dataset.pkl",
                    "rb",
                )
            )
    elif args.dataset == "famnist":
        if not os.path.isfile(
            f"./{args.dataset}_{args.num_clients}_{args.alpha}_dataset.pkl"
        ):
            data_tensors_class = famnist_label_dis_skew_data(
                args.dataloc, args.alpha, args.num_clients
            )
            pickle.dump(
                data_tensors_class,
                open(
                    f"./{args.dataset}_{args.num_clients}_{args.alpha}_dataset.pkl",
                    "wb",
                ),
            )
        else:
            data_tensors_class = pickle.load(
                open(
                    f"./{args.dataset}_{args.num_clients}_{args.alpha}_dataset.pkl",
                    "rb",
                )
            )
    full_knowledge_attack(args, data_tensors_class)
