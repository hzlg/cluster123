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

# sys.path.insert(0, "../utils/")
# from logger import *
# from eval import *
# from misc import *
from normal_train import *
from final_util import *
from at_agr import *

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, to_tree


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据划分有关参数=============================================================================
    parser.add_argument("--train_alpha", type=int, default=1000)  # 控制狄利克雷分布
    parser.add_argument("--val_alpha", type=int, default=1000)  # 控制狄利克雷分布
    parser.add_argument("--num_dis", type=int, default=5)  # 分布数量
    parser.add_argument("--num_clients", type=int, default=50)  # 客户端数量
    parser.add_argument("--num_malicious", type=int, default=15)  # 攻击者数量
    parser.add_argument("--num_benign", type=int, default=35)  # 攻击者数量
    parser.add_argument("--num_test_data", type=int, default=5000)  # 测试集数量
    parser.add_argument("--num_val_data", type=int, default=5000)  # 验证集数量
    parser.add_argument("--dataloc", type=str, default="/home/cluster/")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "mnist", "famnist"],
        default="famnist",
    )
    parser.add_argument("--num_classes", type=int)
    # cluster有关参数=============================================================================
    parser.add_argument("--num_clusters", type=int, default=2)  # 聚类数量
    parser.add_argument("--gap", type=float, default=0.01)  # acc差距
    parser.add_argument(
        "--linkage_method",
        type=str,
        choices=["average", "single", "complete"],
        default="single",
    )
    parser.add_argument(
        "--abcd_method", type=str, choices=["diff", "min_los"], default="min_los"
    )
    # baseline有关参数=============================================================================
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
    # 训练参数=============================================================================
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batchsize", type=int)
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"])
    parser.add_argument("--lr", type=float)
    # 其他参数=============================================================================
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--PATH", type=str)
    parser.add_argument("--PATH_txt", type=str)
    args = parser.parse_args()
    return args


def init_args():
    args = parse_args()
    args.num_classes = {"cifar10": 10, "mnist": 10, "famnist": 10}[args.dataset]
    args.lr = {"cifar10": 0.01, "mnist": 0.001, "famnist": 0.001}[args.dataset]
    args.optimizer = {"cifar10": "SGD", "mnist": "Adam", "famnist": "Adam"}[
        args.dataset
    ]
    args.epoch = {"cifar10": 1200, "mnist": 350, "famnist": 500}[args.dataset]
    args.batchsize = {"cifar10": 250, "mnist": 512, "famnist": 512}[args.dataset]
    args.num_benign = args.num_clients - args.num_malicious
    args.dataloc = args.dataloc + "data/" + args.dataset + "/"

    if args.agr == "fltrust":
        args.is_server = True

    PATH = args.dataset + "_" + args.at_type + "_" + args.at_agr + "_" + args.agr + "_"
    if args.agr == "cluster":
        PATH += str(args.num_clusters) + "_"
    PATH += "n_clients" + str(args.num_clients) + "_" + "alpha" + str(args.train_alpha)
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


def init_data():
    # if not os.path.isfile(
    #     f"./{args.dataset}_{args.num_benign}_{args.train_alpha}_dataset.pkl"
    # ):
    #     data_tensors_class = LabelDisSkewData(args)
    #     pickle.dump(
    #         data_tensors_class,
    #         open(
    #             f"./{args.dataset}_{args.num_benign}_{args.train_alpha}_dataset.pkl",
    #             "wb",
    #         ),
    #     )
    # else:
    #     data_tensors_class = pickle.load(
    #         open(
    #             f"./{args.dataset}_{args.num_benign}_{args.train_alpha}_dataset.pkl",
    #             "rb",
    #         )
    #     )
    # return data_tensors_class
    return LabelDisSkewData(args)


class LabelDisSkewData:
    val_data_tensor = 0
    val_label_tensor = 0
    te_data_tensor = 0
    te_label_tensor = 0
    user_tr_data_tensors = 0
    user_tr_label_tensors = 0

    def split_noniid2(self, n_dis, train_labels, alpha, n_clients, client_data_num):
        n_classes = train_labels.max() + 1  # train_labels内的元素为[0,9],类数为10
        label_distribution = np.random.dirichlet(
            [alpha] * n_classes, n_dis
        )  # 从n_classes维对称狄利克雷分布中采n_clients个样本
        class_idx_list = [
            np.argwhere(train_labels == y).flatten().tolist() for y in range(n_classes)
        ]  # 把10个类分离出来,得到10个idx组成的ndarray (max=60000)
        dis_each_class_num = client_data_num * label_distribution  # 每个客户端在每个类中有多少个数据
        dis_each_class_num = np.array(dis_each_class_num).astype(int)
        for i in range(len(dis_each_class_num)):  # 微调一下样本个数
            diff = client_data_num - np.sum(dis_each_class_num[i])
            for _ in range(abs(diff)):
                if diff > 0:
                    dis_each_class_num[i][np.argmin(dis_each_class_num[i])] += 1
                else:
                    dis_each_class_num[i][np.argmax(dis_each_class_num[i])] -= 1
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
        ]  # concatenate:数列合并

        return client_idx_list
        # return sum(
        #     [
        #         [client_idx_list[i] for i in range(j, n_clients, n_clients // n_dis)]
        #         for j in range(n_clients // n_dis)
        #     ],
        #     [],
        # )  # 把客户端顺序打乱，eg:001122->012012

    def data_dis_plt_show(self, plot, num_classes):
        plt.figure(figsize=(20, 3))
        plt.hist(
            plot,
            stacked=True,
            bins=np.arange(-0.5, num_classes + 0.5, 1),
            label=["Client {}".format(i) for i in range(args.num_benign)],
        )
        mapp = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        plt.xticks(np.arange(10), mapp)
        plt.legend()
        plt.show()

    def __init__(self, args):
        # 加载数据集
        data_train = []
        data_test = []
        if args.dataset == "cifar10":
            train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            data_train = datasets.CIFAR10(
                root=args.dataloc,
                train=True,
                download=False,
                transform=train_transform,
            )
            data_test = datasets.CIFAR10(
                root=args.dataloc,
                train=False,
                download=False,
                transform=train_transform,
            )
        elif args.dataset == "mnist":
            train_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            data_train = datasets.MNIST(
                root=args.dataloc,
                train=True,
                download=False,
                transform=train_transform,
            )
            data_test = datasets.MNIST(
                root=args.dataloc,
                train=False,
                download=False,
                transform=train_transform,
            )
        elif args.dataset == "famnist":
            train_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            data_train = datasets.FashionMNIST(
                root=args.dataloc,
                train=True,
                download=False,
                transform=train_transform,
            )
            data_test = datasets.FashionMNIST(
                root=args.dataloc,
                train=False,
                download=False,
                transform=train_transform,
            )

        # 打乱数据集
        X, Y = [], []
        for i in range(len(data_train)):
            X.append(data_train[i][0].numpy())
            Y.append(data_train[i][1])
        for i in range(len(data_test)):
            X.append(data_test[i][0].numpy())
            Y.append(data_test[i][1])
        X, Y = np.array(X), np.array(Y)
        if not os.path.isfile(f"./{args.dataset}_all_shuffle.pkl"):
            all_indices = np.arange(len(X))
            np.random.shuffle(all_indices)
            pickle.dump(all_indices, open(f"./{args.dataset}_all_shuffle.pkl", "wb"))
        else:
            all_indices = pickle.load(open(f"./{args.dataset}_all_shuffle.pkl", "rb"))
        X, Y = X[all_indices], Y[all_indices]
        a = 1
        if a == 1:  # 新数据划分，dirichlet
            iid_data = X
            iid_label = Y
            te_idcs = self.split_noniid2(1, iid_label, 1000, 1, args.num_test_data)[
                0
            ]  # 随机生成1个分布，分配5000个样本
            te_data_tensor = torch.from_numpy(iid_data[te_idcs]).type(torch.FloatTensor)
            te_label_tensor = torch.from_numpy(iid_label[te_idcs]).type(
                torch.LongTensor
            )
            # # 统计每一类别的个数
            # unq, unq_cnt = np.unique(
            #     iid_label[te_idcs], return_counts=True
            # )  # 去除重复元素，得到类名和该类元素数量
            # tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # 改为字典形式
            # print("Data statistics Test:\n \t %s", str(tmp))

            X = np.delete(X, te_idcs, axis=0)
            Y = np.delete(Y, te_idcs, axis=0)

            val_idcs = self.split_noniid2(1, Y, args.val_alpha, 1, args.num_val_data)[
                0
            ]  # 随机生成1个分布，分配5000个样本
            val_data_tensor = torch.from_numpy(X[val_idcs]).type(torch.FloatTensor)
            val_label_tensor = torch.from_numpy(Y[val_idcs]).type(torch.LongTensor)
            # # 统计每一类别的个数
            # unq, unq_cnt = np.unique(Y[val_idcs], return_counts=True)  # 去除重复元素，得到类名和该类元素数量
            # tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # 改为字典形式
            # print("Data statistics Val:\n \t %s", str(tmp))

            X = np.delete(X, val_idcs, axis=0)
            Y = np.delete(Y, val_idcs, axis=0)

            client_idcs = self.split_noniid2(
                args.num_dis,
                Y,
                args.train_alpha,
                args.num_benign,
                len(Y) // args.num_benign,
            )
            # # 统计每一类别的个数
            # net_cls_counts = {}
            # for net_i, dataidx in enumerate(client_idcs):
            #     unq, unq_cnt = np.unique(
            #         Y[dataidx], return_counts=True
            #     )  # 去除重复元素，得到类名和该类元素数量
            #     tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}  # 改为字典形式
            #     net_cls_counts[net_i] = tmp
            # print("Data statistics Train:\n \t", end="")
            # for _ in range(len(net_cls_counts)):
            #     print(str(net_cls_counts[_]))
            # self.data_dis_plt_show(
            #     [Y[idc] for idc in client_idcs], args.num_classes
            # )

            user_tr_data_tensors = []
            user_tr_label_tensors = []
            for client_data_idx in client_idcs:
                user_tr_data_tensors.append(
                    torch.from_numpy(X[client_data_idx]).type(torch.FloatTensor)
                )
                user_tr_label_tensors.append(
                    torch.from_numpy(Y[client_data_idx]).type(torch.LongTensor)
                )
        else:  # 原本的数据划分shuffle
            nusers = 35
            user_tr_len = 50000 // 35
            total_tr_len = 50000
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

        self.val_data_tensor = val_data_tensor
        self.val_label_tensor = val_label_tensor
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
            client_optimizer = SGD(clients_model.parameters(), lr=lr)
        else:
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
        true_los_list.append(te_los)
        true_acc_list.append(te_acc)
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

    # print(leaf_list)
    # print(good_cluster)
    # print(true_los_list)
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
    # # good_agr_grads = mal_updates[args.num_malicious:]
    # if len(mal_updates) == 30:
    #     args.num_malicious = 10
    # elif len(mal_updates) == 50:
    #     args.num_malicious = 15
    # i = set(np.arange(args.num_malicious))
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
    # with open(args.PATH_txt + f'nat{args.num_malicious}' + '.txt', 'a') as txt:
    #     print(f'cluster loss dif {true_los_list[bad_idx]:.5f} right loss dif {right_los_list[0]:.5f} cluster loss {cluster_los:.5f} right loss {right_los:.5f}', file=txt)
    # print(f'cluster loss dif {true_los_list[bad_idx]:.5f} right loss dif {right_los_list[0]:.5f} cluster loss {cluster_los:.5f} right loss {right_los:.5f}')
    #
    # return torch.mean(good_agr_grads, dim=0)


# 攻击模板
def full_knowledge_attack(args, datatensors):
    val_data_tensor = datatensors.val_data_tensor
    val_label_tensor = datatensors.val_label_tensor
    test_data_tensor = datatensors.te_data_tensor
    test_label_tensor = datatensors.te_label_tensor
    tr_data_tensors = datatensors.user_tr_data_tensors
    tr_label_tensors = datatensors.user_tr_label_tensors

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    user_tr_len = len(tr_label_tensors[0])
    nbatches = user_tr_len // args.batchsize
    fed_lr = args.lr
    criterion = nn.CrossEntropyLoss().to(device)
    if args.save:
        with open(args.PATH_txt + f"nat{args.num_malicious}" + ".txt", "a") as txt:
            if args.at_agr == "cluster":
                print(f"gap {args.gap}", file=txt)
            print(
                f"alpha {args.train_alpha} n_clients {args.num_clients} nat {args.num_malicious} at {args.at_type} at_agr {args.at_agr} agr {args.agr}",
                file=txt,
            )
    if args.at_agr == "cluster":
        print(f"gap {args.gap}")
    print(
        f"dataset {args.dataset} alpha {args.train_alpha} n_clients {args.num_clients} nat {args.num_malicious} at {args.at_type} at_agr {args.at_agr} agr {args.agr}"
    )

    epoch_num = 0
    best_global_acc = 0
    torch.cuda.empty_cache()
    r = np.arange(user_tr_len)

    if args.dataset == "cifar10":
        fed_model = ResNet18()
        # fed_model = DenseNet()
        # fed_model, _ = return_model("alexnet", 0.1, 0.9, parallel=False)  # 不要再用alexnet了，会疯狂波动
        # fed_model.conv1 = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=False,
        # )
        # fed_model.fc = nn.Linear(fed_model.fc.in_features, args.num_classes)
    elif args.dataset == "famnist":
        fed_model = mnist_conv()
        fed_model.apply(weights_init)
    elif args.dataset == "mnist":
        fed_model = MnistNet()
    else:
        fed_model = MnistNet()
    fed_model = fed_model.to(device)

    if args.optimizer == "SGD":
        optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)
    elif args.optimizer == "Adam":
        optimizer_fed = Adam(fed_model.parameters(), lr=fed_lr)

    acc_point, los_point = [], []
    while epoch_num <= args.epoch:
        user_grads = []
        if not epoch_num and epoch_num % nbatches == 0:
            np.random.shuffle(r)
            for i in range(len(tr_data_tensors)):
                tr_data_tensors[i] = tr_data_tensors[i][r]
                tr_label_tensors[i] = tr_label_tensors[i][r]
        for i in range(args.num_benign):  # 和partial不同
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
            # fed_model = fed_model.to(device)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(
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
            # torch.cuda.empty_cache()
            server_inputs = val_data_tensor
            server_targets = val_label_tensor
            server_inputs, server_targets = server_inputs.to(device), server_targets.to(
                device
            )
            server_inputs, server_targets = torch.autograd.Variable(
                server_inputs
            ), torch.autograd.Variable(server_targets)
            server_outputs = fed_model(server_inputs)  # 输入样本，输出预测值
            server_loss = criterion(server_outputs, server_targets)  # 计算预测值与标签的交叉熵损失
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

        agg_grads = torch.mean(benign_grads, 0)
        if args.at_type == "none":
            malicious_grads = user_grads
        else:
            if args.at_type == "rev":
                malicious_grads = reverse_attack(
                    benign_grads, agg_grads, args.num_malicious
                )
            elif args.at_type == "fang":
                if args.at_agr == "mkrum" or args.at_agr == "krum":
                    multi_k = True if args.agr == "mkrum" else False
                    malicious_grads = get_malicious_updates_fang(
                        benign_grads, agg_grads, args.num_malicious, multi_k
                    )
                elif args.at_agr == "trmean":
                    malicious_grads = get_malicious_updates_fang_trmean(
                        benign_grads, agg_grads, args.num_malicious
                    )
            elif args.at_type == "adaptive":
                if args.at_agr == "mkrum" or args.at_agr == "krum":
                    multi_k = True if args.agr == "mkrum" else False
                    malicious_grads = adaptive_attack_mkrum(
                        benign_grads,
                        agg_grads,
                        args.num_malicious,
                        multi_k,
                        dev_type="std",
                    )
                elif args.at_agr == "trmean":
                    malicious_grads = adaptive_attack_trmean(
                        benign_grads,
                        agg_grads,
                        args.num_malicious,
                        dev_type="std",
                    )
            elif args.at_type == "min_max":
                malicious_grads = adaptive_min_max(
                    benign_grads, agg_grads, args.num_malicious
                )
            elif args.at_type == "min_sum":
                malicious_grads = adaptive_min_sum(
                    benign_grads, agg_grads, args.num_malicious
                )

            # for _ in range(
            #     1, args.num_malicious - 1
            # ):  # 自己加的，不知道如何在一定欧式距离内随机采样，于是计算了高斯噪声后加上
            #     distance = 1
            #     noise_prop = 0.0001
            #     while distance > 0.01:
            #         noise = (
            #             noise_prop * torch.randn(malicious_grads[0].size()).cuda()
            #         )
            #         distance = torch.norm(noise, p=2)
            #         noise_prop *= 0.5
            #     malicious_grads[_] += noise

        #  聚合
        if args.agr == "cluster":
            train_tool = [
                val_data_tensor,
                val_label_tensor,
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
                malicious_grads, args.num_malicious, multi_k=multi_k
            )
        elif args.agr == "trmean":
            agg_grads = tr_mean(malicious_grads, args.num_malicious)
        elif args.agr == "foolsgold":
            agg_grads = foolsgold(malicious_grads)
        elif args.agr == "median":
            agg_grads = torch.median(malicious_grads, dim=0)[0]
        elif args.agr == "bulyan":
            agg_grads, krum_candidate = bulyan(malicious_grads, args.num_malicious)

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

        save_epoch_num = 10

        test_loss, test_acc = test(
            test_data_tensor, test_label_tensor, fed_model, criterion, use_cuda
        )  # 用更新的模型测试
        if epoch_num % save_epoch_num == 0 or epoch_num == args.epoch - 1:
            if math.isnan(test_loss):  # mark loss 还是testloss
                if args.save:
                    with open(
                        args.PATH_txt + f"nat{args.num_malicious}" + ".txt", "a"
                    ) as txt:
                        print("loss nan", file=txt)
                print("loss nan")
                exit()
            best_global_acc = max(best_global_acc, test_acc)
            if args.save:
                with open(
                    args.PATH_txt + f"nat{args.num_malicious}" + ".txt", "a"
                ) as txt:
                    print(
                        "%s-%s: %s n_at %d epoch %d test loss %.4f test acc %.4f"
                        % (
                            args.at_type,
                            args.at_agr,
                            args.agr,
                            args.num_malicious,
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
                    args.num_malicious,
                    epoch_num,
                    test_loss,
                    test_acc,
                )
            )

            acc_point.append((epoch_num, test_acc))
            los_point.append((epoch_num, test_loss))

            plt.subplot(2, 1, 1)
            plt.plot([p[0] for p in acc_point], [p[1] for p in acc_point], color="red")
            plt.scatter(
                [p[0] for p in acc_point], [p[1] for p in acc_point], color="red"
            )
            plt.title("acc")
            plt.subplot(2, 1, 2)
            plt.plot(
                [p[0] for p in los_point], [p[1] for p in los_point], color="green"
            )
            plt.scatter(
                [p[0] for p in los_point], [p[1] for p in los_point], color="green"
            )
            plt.title("los")
            plt.show()

            # if args.save:
            #     PATH_tar = args.PATH + 'n_at' + str(args.num_malicious) + 'epoch' + str(epoch_num) + '.pth.tar'
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
        # if test_loss > 10:
        #     print("test loss %f too high" % test_loss)
        #     break
        epoch_num += 1


if __name__ == "__main__":
    args = init_args()
    full_knowledge_attack(args, LabelDisSkewData(args))
