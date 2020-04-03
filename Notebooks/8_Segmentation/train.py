import datetime
import os
import random

import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import VOCSegmentation
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from misc import evaluate, AverageMeter, colorize_mask, MaskToTensor, \
                 DeNormalize
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np


cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(train_args, model):
    print(train_args)

    net = model.to(device)

    train_args['best_record'] = {'epoch': 0,
                                 'val_loss': 1e10,
                                 'acc': 0, 'acc_cls': 0,
                                 'mean_iu': 0,
                                 'fwavacc': 0}

    net.train()

    mean_std = ([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])

    input_transform = transforms.Compose([
        transforms.Pad(200),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])

    train_transform = transforms.Compose([
        transforms.Pad(200),
        transforms.CenterCrop(320),
        MaskToTensor()])

    restore_transform = transforms.Compose([
        DeNormalize(*mean_std),
        transforms.ToPILImage(),
    ])

    visualize = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(400),
        transforms.ToTensor()
    ])

    train_set = VOCSegmentation(root='./',
                                image_set='train',
                                transform=input_transform,
                                target_transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=1,
                              num_workers=4,
                              shuffle=True)
    val_set = VOCSegmentation(root='./',
                              image_set='val',
                              transform=input_transform,
                              target_transform=train_transform)
    val_loader = DataLoader(val_set,
                            batch_size=1,
                            num_workers=4,
                            shuffle=False)

    criterion = CrossEntropyLoss(ignore_index=255, reduction='mean').to(device)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], momentum=train_args['momentum'])

    for epoch in range(1, train_args['epoch_num'] + 1):
        train(train_loader,
              net,
              criterion,
              optimizer,
              epoch,
              train_args)
        val_loss, imges = validate(val_loader,
                                   net,
                                   criterion,
                                   optimizer,
                                   epoch,
                                   train_args,
                                   restore_transform,
                                   visualize)
    return imges


def train(train_loader, net, criterion, optimizer, epoch, train_args):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:], ("inputs are {}"
                                                        "output is {}".format(
                                                           inputs.size()[2:],
                                                           labels.size()[1:]
                                                        ))
        N = inputs.size(0)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == 21

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data, N)

        curr_iter += 1

        if (i + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg
            ))


def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore,
             visualize):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []
    for data in val_loader:
        inputs, gts = data
        N = inputs.size(0)
        inputs = inputs.to(device)
        gts = gts.to(device)

        with torch.no_grad():
            outputs = net(inputs)

        predictions = outputs.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        val_loss.update(criterion(outputs, gts).data / N, N)

        if random.random() > train_args['val_img_sample_rate']:
            inputs_all.append(None)
        else:
            inputs_all.append(inputs.squeeze_(0).cpu())
        gts_all.append(gts.squeeze_(0).cpu().numpy())
        predictions_all.append(predictions)

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, 21)

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc

    val_visual = []
    for data in zip(inputs_all, gts_all, predictions_all):
        if data[0] is None:
            continue
        input_pil = restore(data[0])
        gt_pil = colorize_mask(data[1])
        predictions_pil = colorize_mask(data[2])
        val_visual.extend([visualize(input_pil.convert('RGB')),
                           visualize(gt_pil.convert('RGB')),
                           visualize(predictions_pil.convert('RGB'))])
    val_visual = torch.stack(val_visual, 0)
    val_visual = make_grid(val_visual, nrow=3, padding=5)

    print('--------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))

    print('--------------------------------------------------------------------')

    net.train()
    return val_loss.avg, val_visual
