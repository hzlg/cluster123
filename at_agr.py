import torch
import numpy as np
import sklearn.metrics.pairwise as smp
import torch.nn.functional as F
import scipy.stats

def reverse_attack(all_updates, model_re, n_attackers):
    deviation = model_re / torch.norm(model_re)  # 单位向量
    lamda = 3
    mal_update = (- lamda * deviation)  # 加入反向扰动
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)
    return mal_updates

def multi_krum(all_updates, n_attackers, multi_k=False):
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))
    if multi_k:
        while len(remaining_updates) > 2 * n_attackers + 2:
            torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:  # 在剩余的梯度中取梯度
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)  # 梯度i和其他梯度的欧式距离的平方
                distance = torch.Tensor(distance).float()  # 转换为tensor
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)  # 所有梯度之间的距离

            distances = torch.sort(distances, dim=1)[0]  # 对距离排序
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)  # 每个梯度与最接近的m-c-2个梯度的距离和
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]  # 对距离和排序，得到位置

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])  # 选中距离和最小的
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())  # 删除被选中的
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
    else:
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:  # 在剩余的梯度中取梯度
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)  # 梯度i和其他梯度的欧式距离的平方
            distance = torch.Tensor(distance).float()  # 转换为tensor
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]),
                                                                               0)  # 所有梯度之间的距离

        distances = torch.sort(distances, dim=1)[0]  # 对距离排序
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)  # 每个梯度与最接近的m-c-2个梯度的距离和
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]  # 对距离和排序，得到位置
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])  # 选中距离和最小的
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
    aggregate = torch.mean(candidates, dim=0)
    return aggregate, np.array(candidate_indices)
def compute_lambda_fang(all_updates, model_re, n_attackers):
    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1)
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, :n_benign - 2 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0])
    max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])
    return term_1 + max_wre_dist
def get_malicious_updates_fang(all_updates, model_re, n_attackers, multik):
    deviation = torch.sign(model_re)
    lamda = compute_lambda_fang(all_updates, model_re, n_attackers)  # 计算上界
    threshold = 1e-5
    mal_updates = []
    while lamda > threshold:
        mal_update = (- lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)
        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multik)
        if (multik and np.sum(krum_candidate < n_attackers) == n_attackers) or ((not multik) and krum_candidate < n_attackers):  # step4，Oracle=True，恶意梯度都被选中(idx小于10的有10个)
            return mal_updates
        lamda *= 0.5
    if not len(mal_updates):
        print(lamda, threshold)
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)
    return mal_updates
def adaptive_attack_mkrum(all_updates, model_re, n_attackers, multik, dev_type='unit_vec'):
    # 固定扰动类型
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # 单位向量
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)  # sign
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)  # 方差

    lamda = torch.Tensor([3.0]).cuda()

    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:  # step3
        mal_update = (model_re - lamda * deviation)  # 加入反向扰动
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)
        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=multik)  # agr,返回Krum聚合后的梯度，和被选中梯度的位置
        if(multik and np.sum(krum_candidate < n_attackers) == n_attackers) or ((not multik) and krum_candidate < n_attackers):  # step4，Oracle=True，恶意梯度都被选中(idx小于10的有10个)
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)
    # print(f"lamda_succ{lamda_succ}")
    return mal_updates


def tr_mean(all_updates, beta):
    return torch.tensor(scipy.stats.trim_mean(all_updates.cpu().numpy(), beta, axis=0)).cuda()
def get_malicious_updates_fang_trmean(all_updates, model_re, n_attackers):
    deviation = torch.sign(model_re)
    b = 2
    max_vector = torch.max(all_updates, 0)[0]  # wmaxj
    min_vector = torch.min(all_updates, 0)[0]  # wminj

    max_ = (max_vector > 0).type(torch.FloatTensor).cuda()  # 大于0的都置为1，其余为0
    min_ = (min_vector < 0).type(torch.FloatTensor).cuda()
    # 使得max_和min_里只有1/b和b
    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)  # (2472266,2) wmax和恶意梯度上界
    min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()  # 从(0,1)范围中随机采样 维度数*攻击者数 个随机数
    # [10个(攻击者数)wmax组成矩阵]+rand*[比原本多的部分] 最后结果落在[wmax,b*wmax]中
    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    # [b*wmin,wmin]中
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T
    # 恶意梯度 s=1就选小的,s=-1就选大的
    mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack([(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

    mal_updates = torch.cat((mal_vec, all_updates), 0)
    return mal_updates
def adaptive_attack_trmean(all_updates, model_re, n_attackers, n_clients, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).cuda()  # compute_lambda_adaptive(all_updates, model_re, n_attackers)

    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    iters = 0
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads = tr_mean(mal_updates, n_attackers/n_clients)

        loss = torch.norm(agg_grads - model_re)  # median聚合前后梯度的距离

        if prev_loss < loss:  # 距离增大了就看那能不能再大一点
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:  # 距离减小了就减小扰动(使恶意梯度不被修剪)
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss

    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)

    return mal_updates

def adaptive_min_max(all_updates, model_re, n_attackers, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).float().cuda()
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)  # 良性梯度之间的最大距离
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)

    return mal_updates
def adaptive_min_sum(all_updates, model_re, n_attackers, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).float().cuda()
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        score = torch.sum(distance)

        if score <= min_score:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    # print(lamda_succ)
    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)


    return mal_updates

def FLtrust(s_grad, all_grad):
    cos_sim = F.cosine_similarity(s_grad, all_grad)  # server梯度和所有梯度求余弦相似度
    relu_score = F.relu(cos_sim)  # relu
    norm_grad = [all_grad[i] * torch.norm(s_grad) * relu_score[i] / torch.norm(all_grad[i]) for i in range(all_grad.shape[0])]
    return torch.sum(torch.stack(norm_grad, dim=0), dim=0)/sum(relu_score)

def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads.cpu()) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 1e-8

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return torch.mm(torch.t(grads), torch.from_numpy(wv).type(torch.FloatTensor).unsqueeze(dim=1).cuda()).squeeze() / sum(wv)
