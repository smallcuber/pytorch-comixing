import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from model import CNN
from loss import loss_coteaching

import random
import numpy as np
from PIL import Image
from mixup import mixup_data
import json
import os
from tqdm import tqdm
from loader_CIFAR import CifarDataloader
from loader_ANIMAL10N import Animal10N

# test the custom loaders for CIFAR
dataset = 'animal10n'  # either cifar10 or animal10n
data_path_animal = 'rawdata_ANIMAL10N'
data_path = 'rawdata_CIFAR10'  # path to the data file (don't forget to download the feature data and also put the noisy label file under this folder)

num_iter_per_epoch = 100
num_print_freq = 100
num_epoch = 200
num_batch_size = 16
num_gradual = 10
num_exponent = 1
num_forget_rate = 0.1
num_noise_rate = 0.2
num_workers = 8
num_classes = 10
num_learning_rate = 0.001
num_epoch_decay_start = 80
num_input_channel = 3

num_mixup_alpha = 0.1

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [num_learning_rate] * num_epoch
beta1_plan = [mom1] * num_epoch
for i in range(num_epoch_decay_start, num_epoch):
    alpha_plan[i] = float(num_epoch - i) / (num_epoch - num_epoch_decay_start) * num_learning_rate
    beta1_plan[i] = mom2

rate_schedule = np.ones(num_epoch) * num_forget_rate
rate_schedule[:num_gradual] = np.linspace(0, num_forget_rate ** num_exponent, num_gradual)

json_noise_file_names = {
    1: 'cifar10_noisy_labels_task1.json',
    2: 'cifar10_noisy_labels_task2.json',
    3: 'cifar10_noisy_labels_task3.json'
}
noise_file_name = json_noise_file_names[1]

if dataset == 'cifar10':
    loader = CifarDataloader(dataset, batch_size=128,
                             num_workers=10,
                             root_dir=data_path,
                             noise_file='%s/%s' % (data_path, noise_file_name))
    train_loader, noisy_labels, clean_labels = loader.run('train')
    noise_or_not = np.transpose(noisy_labels) == np.transpose(clean_labels)
    test_loader = loader.run('test')
else:
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset_train = Animal10N(split='train', transform=transform_train)
    dataset_test = Animal10N(split='test', transform=transform_test)

    train_loader = DataLoader(dataset_train, batch_size=num_batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset_test, batch_size=num_batch_size * 4, shuffle=False, num_workers=num_workers)
    noise_or_not = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_full_name = f'comixing_{dataset}_{noise_file_name}'


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # Only change beta1


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
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


def train(loader_train, epoch, model1, optimizer1, model2, optimizer2, criterion):
    print("Training... %s" % model_full_name)
    pure_ratio_list = []
    pure_ratio_1_list = []
    pure_ratio_2_list = []
    count_total_train1 = 0
    count_total_correct1 = 0
    count_total_train2 = 0
    count_total_correct2 = 0
    num_correct_1 = 0
    num_correct_2 = 0
    num_total = 0

    for i, (images, labels, indexes) in enumerate(loader_train):
        ind = indexes.cpu().numpy().transpose()
        if i > num_iter_per_epoch:
            break

        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        num_total = labels.size(0)

        images, label_a, label_b, lam = mixup_data(images, labels, num_mixup_alpha, True)

        # Forward Backward Optimize
        output1 = model1(images)
        _, predicted1 = torch.max(output1.data, 1)
        prec1, _ = accuracy(output1, labels, topk=(1, 5))
        count_total_train1 += 1
        count_total_correct1 += prec1

        output2 = model2(images)
        _, predicted2 = torch.max(output2.data, 1)
        prec2, _ = accuracy(output2, labels, topk=(1, 5))
        count_total_train2 += 1
        count_total_correct2 += prec2

        num_correct_1 += (lam * predicted1.eq(label_a.data).cpu().sum().float()
                          + (1 - lam) * predicted1.eq(label_b.data).cpu().sum().float())
        num_correct_2 += (lam * predicted2.eq(label_a.data).cpu().sum().float()
                          + (1 - lam) * predicted2.eq(label_b.data).cpu().sum().float())
        num_acc_1 = num_correct_1 / num_total
        num_acc_2 = num_correct_2 / num_total

        loss1, loss2, pure_ratio_1, pure_ratio_2 = loss_coteaching(criterion, output1, output2, labels, label_a,
                                                                   label_b, rate_schedule[epoch], ind, noise_or_not,
                                                                   lam)

        if pure_ratio_1 and pure_ratio_2 is not None:
            pure_ratio_1_list.append(100 * pure_ratio_1)
            pure_ratio_2_list.append(100 * pure_ratio_2)

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        if (i + 1) % num_print_freq == 0:
            if pure_ratio_1 and pure_ratio_2 is not None:
                str_calc_pure_ratio = 'Pure Ratio1: %.4f, Pure Ratio2 %.4f' % (
                np.sum(pure_ratio_1_list) / len(pure_ratio_1_list),
                np.sum(pure_ratio_2_list) / len(pure_ratio_2_list))
            else:
                str_calc_pure_ratio = 'Animal10N dataset without pure ratio'

            print(
                'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, %s'
                % (epoch + 1, num_epoch, i + 1, len(train_loader) // num_batch_size, num_acc_1, num_acc_2, loss1.item(),
                   loss2.item(), str_calc_pure_ratio))

    train_acc1 = float(count_total_correct1) / float(count_total_train1)
    train_acc2 = float(count_total_correct2) / float(count_total_train2)
    return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list


def evaluate(test_loader, model1, model2):
    print('Evaluating %s...' % model_full_name)
    # model1 = model1.to(device)  # Change model to 'eval' mode.
    # model2 = model2.to(device)
    correct1 = 0
    total1 = 0
    print("Start evaluating model 1")
    for images, labels in tqdm(test_loader):
        images = Variable(images).to(device)
        labels = labels.to(device)
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1 == labels).sum()

    print("Start evaluating model 2")
    model2.eval()  # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    for images, labels in tqdm(test_loader):
        images = Variable(images).to(device)
        labels = labels.to(device)

        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2 == labels).sum()

    acc1 = 100 * float(correct1) / float(total1)
    acc2 = 100 * float(correct2) / float(total2)
    return acc1, acc2


# Train the model

# Data Loader (Input Pipeline)
print('loading dataset...')

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=num_batch_size,
#                                            num_workers=num_workers,
#                                            drop_last=True,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=num_batch_size,
#                                           num_workers=num_workers,
#                                           drop_last=True,
#                                           shuffle=False)


# Define models
print('building model...')
cnn1 = CNN(input_channel=num_input_channel, n_outputs=num_classes)
cnn1.to(device)
# print(cnn1.parameters)
optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=num_learning_rate)

cnn2 = CNN(input_channel=num_input_channel, n_outputs=num_classes)
cnn2.to(device)
# print(cnn2.parameters)
optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=num_learning_rate)

criterion = torch.nn.CrossEntropyLoss(reduce=False)
mean_pure_ratio1 = 0
mean_pure_ratio2 = 0

# with open(txtfile, "a") as myfile:
#     myfile.write('epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n')

epoch = 0
train_acc1 = 0
train_acc2 = 0
# evaluate models with random weights
# test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
# print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, num_epoch, len(test_loader.dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
# save results
# with open(txtfile, "a") as myfile:
#     myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + "\n")


# training
for epoch in range(1, num_epoch):
    # train models
    cnn1.train()
    adjust_learning_rate(optimizer1, epoch)
    cnn2.train()
    adjust_learning_rate(optimizer2, epoch)
    train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = train(train_loader, epoch, cnn1, optimizer1, cnn2,
                                                                         optimizer2, criterion)
    # evaluate models
    test_acc1, test_acc2 = evaluate(test_loader, cnn1, cnn2)
    # save results
    if dataset == 'cifar10n':
        mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)
        str_ratio = 'Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f' % (mean_pure_ratio1, mean_pure_ratio2)
    else:
        str_ratio = 'animal10n without pure ratio.'
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, %s' % (
    epoch + 1, num_epoch, len(test_loader.dataset), test_acc1, test_acc2, str_ratio))
    # with open(txtfile, "a") as myfile:
    #     myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")
