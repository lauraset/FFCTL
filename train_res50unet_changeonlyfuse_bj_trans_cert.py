'''
2022.01.19
@Yinxia Cao
@function: only focus on changed samples
20220404: apply to beijing
consider fuse model
20220407: use trans1
20220421: use trans1 from the whole images, and use certainty label
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from torch.utils import data
from tensorboardX import SummaryWriter #change tensorboardX
from ZY3LC_dataset import dataloader_t1t2
from ZY3LC_loader import myImageFloder_8bit_t1t2
from metrics import SegmentationMetric, AverageMeter
import segmentation_models_pytorch_new as smp
import shutil
from myloss import BCE_DICE
import argparse
from collections import OrderedDict


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 1:
        lr = 0.001
    elif epoch <= 5:
        lr = 0.0001
    else:
        lr = 0.00001
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr #added


def main():

    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # Setup device
    device = 'cuda'

    # Setup Dataloader
    fcode = 'bj'
    datapath = os.path.join('changedata', fcode)
    train_img, train_lab, val_img, val_lab, _, _ = \
        dataloader_t1t2(datapath, split=(0.9, 0.1, 0), fcode=fcode) # 90% for training
    # use cert
    train_lab = [i.replace('lab', 'cert') for i in train_lab]
    val_lab = [i.replace('lab', 'cert') for i in val_lab]
    # test
    datapath = os.path.join(datapath, 'testdata')
    test_img, test_lab, _,_,_,_ = \
        dataloader_t1t2(datapath, split=(1.0,0,0),fcode=fcode) # 90% for training

    # batch_size = 16
    epochs = 10
    iroot = r'runs_change\res50cdo_fuse'
    logdir = os.path.join(iroot, fcode+'_1ta_cert') # change only
    writer = SummaryWriter(log_dir=logdir)
    # NUM_WORKERS = 4
    classes = 2 # 0,1
    nchannels = 4
    global best_acc
    best_acc = 0
    batch_size = 16
    numworkers = 4
    idir1 = 'img1ta'
    # train on the randomly cropped images
    traindataloader = torch.utils.data.DataLoader(
        myImageFloder_8bit_t1t2(train_img, train_lab, aug=True,
                                dir1=idir1, dir2='img2'),
        batch_size=batch_size, shuffle=True, num_workers=numworkers, pin_memory=True)

    # test on the whole images
    valdataloader = torch.utils.data.DataLoader(
        myImageFloder_8bit_t1t2(val_img, val_lab, aug=False,
                                dir1=idir1, dir2='img2'),
        batch_size=batch_size, shuffle=False, num_workers=numworkers, pin_memory=True)

    # test
    testdataloader = torch.utils.data.DataLoader(
        myImageFloder_8bit_t1t2(test_img, test_lab, aug=False,
                                dir1=idir1, dir2='img2'),
        batch_size=batch_size, shuffle=False, num_workers=numworkers, pin_memory=True)

    model = smp.UnetCDdiffuse(encoder_name="resnet50", encoder_weights="imagenet",
                             in_channels=nchannels, classes=1).to(device)
    # load pretrained models
    iroot2 = 'runs'
    # old version
    pretrainp = os.path.join(iroot2, 'res50build_update', 'scratch', 'model_best.tar') #
    pretrain = torch.load(pretrainp)
    print('loading pretrained epoch: %d'%pretrain['epoch'])
    pretrain = pretrain['state_dict']
    state_dict = model.state_dict()
    newstate = OrderedDict()
    for k, v in pretrain.items(): # loop over pretrained parameters
        if k in state_dict.keys():
            if v.shape==state_dict[k].shape: # bug, should add shape
                #print(k)
                newstate[k] = v
    state_dict.update(newstate)
    model.load_state_dict(state_dict)

    # print the model
    start_epoch = 0
    resume = os.path.join(logdir, 'checkpoint.tar')
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['test_iou']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = BCE_DICE()

    for epoch in range(epochs-start_epoch):
        epoch = start_epoch + epoch + 1 # current epochs
        adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        print('epoch %d, lr: %.6f'%(epoch, lr))
        train_loss, train_f1, train_iou = train_epoch(model, criterion,
                                                        traindataloader,
                                                          optimizer, device, epoch, classes)
        val_loss, val_f1, val_iou  = vtest_epoch(model, criterion, valdataloader, device, epoch, classes)
        test_loss, test_f1, test_iou  = vtest_epoch2(model, criterion, testdataloader, device, epoch, classes)
        # save every epoch
        savefilename = os.path.join(logdir, 'checkpoint.tar')
        torch.save({
            'epoch': epoch,
            'state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),  # multiple GPUs
            'val_f1': val_f1,
            'test_f1': test_f1,
            'test_iou': test_iou,
        }, savefilename)
        # save best one
        if test_iou>best_acc:
            best_acc = test_iou
            shutil.copy(savefilename, os.path.join(logdir, 'model_best.tar'))
        # write
        writer.add_scalar('lr', lr, epoch)
        writer.add_scalar('train/1.loss', train_loss,epoch)
        writer.add_scalar('train/2.f1', train_f1, epoch)
        writer.add_scalar('train/3.iou',train_iou, epoch)
        writer.add_scalar('val/1.loss', val_loss, epoch)
        writer.add_scalar('val/2.f1',val_f1, epoch)
        writer.add_scalar('val/3.iou', val_iou, epoch)
        writer.add_scalar('test/1.loss', test_loss, epoch)
        writer.add_scalar('test/2.f1',test_f1, epoch)
        writer.add_scalar('test/3.iou', test_iou, epoch)
    writer.close()


# train
def train_epoch(model, criterion, dataloader, optimizer, device, epoch, classes):
    model.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        output, unchange = model(images)
        output = torch.sigmoid(output)
        unchange = torch.sigmoid(unchange)

        # mask uncertain regions
        output = output.flatten()
        labels = labels.flatten()
        unchange = unchange.flatten()
        mask = (labels<3) # 0 (unchange), 1 (unchange), 2 (change), 3(uncertain)
        output = output[mask]
        labels = labels[mask]
        unchange = unchange[mask]

        loss = criterion(output, (labels==2).float()) + \
               criterion(unchange, (labels==1).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = (output > 0.5) # N C H W
        acc_total.addBatch(output, (labels==2)) # change
        losses.update(loss.item(), images.size(0))

        f1 = acc_total.F1score()[1]
        iou = acc_total.IntersectionOverUnion()[1]
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. F1 {f1:.3f}, IOU: {iou:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, f1=f1, iou=iou))
        pbar.update()
    pbar.close()
    f1 = acc_total.F1score()[1]
    iou = acc_total.IntersectionOverUnion()[1]
    print('epoch %d, train f1 %.3f, iou: %.3f' % (epoch, f1, iou))
    return losses.avg, f1, iou


# test
def vtest_epoch(model, criterion, dataloader, device, epoch, classes):
    model.eval()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            output, unchange = model(images)
            output = torch.sigmoid(output)
            unchange = torch.sigmoid(unchange)

            # mask uncertain regions
            output = output.flatten()
            labels = labels.flatten()
            unchange = unchange.flatten()
            mask = (labels < 3)  # 0 (unchange), 1 (unchange), 2 (change), 3(uncertain)
            output = output[mask]
            labels = labels[mask]
            unchange = unchange[mask]

            loss = criterion(output, (labels == 2).float()) + \
                   criterion(unchange, (labels == 1).float())

            output = (output > 0.5)  # N C H W
            acc_total.addBatch(output, (labels==2))
            losses.update(loss.item(), images.size(0))

            f1 = acc_total.F1score()[1]
            iou = acc_total.IntersectionOverUnion()[1]
            pbar.set_description(
                'Val Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. F1 {f1:.3f}, IOU: {iou:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, f1=f1, iou=iou))
            pbar.update()
        pbar.close()

    f1 = acc_total.F1score()[1]
    iou = acc_total.IntersectionOverUnion()[1]
    # print('epoch %d, train f1 %.3f, iou: %.3f' % (epoch, f1, iou))
    return losses.avg, f1, iou

# test
def vtest_epoch2(model, criterion, dataloader, device, epoch, classes):
    model.eval()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            output, _ = model(images)
            output = torch.sigmoid(output)
            # unchange = torch.sigmoid(unchange)

            loss = criterion(output, labels.float()) # + criterion(unchange, (labels == 1).float())

            output = (output > 0.5)  # N C H W
            acc_total.addBatch(output, labels)
            losses.update(loss.item(), images.size(0))

            f1 = acc_total.F1score()[1]
            iou = acc_total.IntersectionOverUnion()[1]
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. F1 {f1:.3f}, IOU: {iou:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, f1=f1, iou=iou))
            pbar.update()
        pbar.close()

    f1 = acc_total.F1score()[1]
    iou = acc_total.IntersectionOverUnion()[1]
    # print('epoch %d, train f1 %.3f, iou: %.3f' % (epoch, f1, iou))
    return losses.avg, f1, iou

if __name__=="__main__":
    main()