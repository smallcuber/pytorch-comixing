import torch
import torch.nn.functional as F
import numpy as np
from mixup import mixup_criterion


def loss_coteaching(criterion, y_1, y_2, t, label_a, label_b, forget_rate, ind, noise_or_not, lam):

    # loss_1 = F.cross_entropy(y_1, t, reduce=False)
    loss_1 = mixup_criterion(criterion, y_1, label_a, label_b, lam)
    loss_1 = loss_1.cpu()
    ind_1_sorted = np.argsort(loss_1.data)
    loss_1_sorted = loss_1[ind_1_sorted]

    # loss_2 = F.cross_entropy(y_2, t, reduce=False)
    loss_2 = mixup_criterion(criterion, y_2, label_a, label_b, lam)
    loss_2 = loss_2.cpu()
    ind_2_sorted = np.argsort(loss_2.data)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    if noise_or_not is None:
        pure_ratio_1, pure_ratio_2 = None, None
    else:
        pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(num_remember)
        pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(num_remember)

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    # loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    # loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    loss_1_update = mixup_criterion(criterion, y_1[ind_2_update], label_a[ind_2_update], label_b[ind_2_update], lam)
    loss_2_update = mixup_criterion(criterion, y_2[ind_1_update], label_a[ind_1_update], label_b[ind_1_update], lam)

    return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember, pure_ratio_1, pure_ratio_2
