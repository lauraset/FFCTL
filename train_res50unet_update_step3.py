'''
2022.01.19
@Yinxia Cao
@function: used for training ZY3LC on beijing, 8 bit images
training with corrected labels from scratch
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
from ZY3LC_dataset import dataloader
from ZY3LC_loader import  myImageFloder_8bit_binary, myImageFloder_8bit_binary_update_scratch
from metrics import SegmentationMetric, AverageMeter
import segmentation_models_pytorch as smp
import shutil
from myloss import BCE_DICE
import argparse
import cv2


def get_arguments():
    parser = argparse.ArgumentParser(description="Test for binary class")
    parser.add_argument("--classname", type=str, default='build',
                        help="oisa|grass|tree|soil|build|water|road")
    args = parser.parse_args()
    return args

classdict = {'oisa': 1, 'grass': 2, 'tree': 3, 'soil': 4, 'build': 5, 'water': 6, 'road': 7}


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 20:
        lr = 0.001
    elif epoch <= 40:
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

    # Get args
    args = get_arguments()

    # Setup device
    device = 'cuda'

    # Setup Dataloader
    filepath = 'data' # data path
    train_img, train_lab, val_img, val_lab,_,_ = dataloader(filepath, split=(0.9, 0.1, 0)) # 90% for training

    epochs_scratch = 50
    iroot = 'runs'
    logdir = os.path.join(iroot, 'res50' + args.classname + '_update', 'scratch')
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)

    # storing updated labels
    updatepath = os.path.join(iroot, 'res50' + args.classname + '_update', 'update', 'pred')

    # NUM_WORKERS = 4
    classes = 2 # 0, 1, 2, 3, 4, 5, 6
    nchannels = 4
    imgsize = 256
    global best_acc
    best_acc = 0
    positive = 255 # values for buildings

    # train with updated labels
    traindataloader_scratch = torch.utils.data.DataLoader(
        myImageFloder_8bit_binary_update_scratch(train_img, updatepath, aug=True, imgsize=imgsize,
                                         channels=nchannels, positive=positive),
        batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    # test on the whole images
    valdataloader = torch.utils.data.DataLoader(
        myImageFloder_8bit_binary(val_img, val_lab, aug=False, imgsize=imgsize, channels=nchannels, positive=positive),
        batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
                             in_channels=nchannels, classes=1).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

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
        best_acc = checkpoint['best_acc']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")

    # get all parameters (model parameters + task dependent log variances)
    # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    #weights = torch.FloatTensor([1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0]).to(device) # defined
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = BCE_DICE()

    # train from scratch
    for epoch in range(epochs_scratch-start_epoch):
        epoch = start_epoch + epoch + 1 # current epochs
        # adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        print('epoch %d, lr: %.6f'%(epoch, lr))

        # train
        train_loss, train_f1, train_iou = \
            train_epoch(model, criterion, traindataloader_scratch,
                               optimizer, device, epoch, classes)

        # validate
        val_loss, val_f1, val_iou  = vtest_epoch(model, criterion, valdataloader, device, epoch, classes)
        # save every epoch
        savefilename = os.path.join(logdir, 'checkpoint.tar')
        is_best = val_f1 > best_acc
        best_acc = max(val_f1, best_acc)  # update
        torch.save({
            'epoch': epoch,
            'state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),  # multiple GPUs
            'val_f1': val_f1,
            'best_acc': best_acc,
        }, savefilename)
        # save every 10 epochs separately
        if epoch%10 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                'val_f1': val_f1,
                'best_acc': best_acc,
            }, os.path.join(logdir, 'checkpoint_'+str(epoch)+'.tar'))

        if is_best:
            shutil.copy(savefilename, os.path.join(logdir, 'model_best.tar'))
        # write
        writer.add_scalar('lr', lr, epoch)
        writer.add_scalar('train/1.loss', train_loss,epoch)
        writer.add_scalar('train/2.f1', train_f1, epoch)
        writer.add_scalar('train/3.iou',train_iou, epoch)
        writer.add_scalar('val/1.loss', val_loss, epoch)
        writer.add_scalar('val/2.f1',val_f1, epoch)
        writer.add_scalar('val/3.iou', val_iou, epoch)
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

        output = model(images)
        output = torch.sigmoid(output)

        loss = criterion(output, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = (output > 0.5) # N C H W
        acc_total.addBatch(output, labels)
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
        for idx, (x, y_true) in enumerate(dataloader):
            x = x.to(device, non_blocking =True)
            y_true = y_true.to(device, non_blocking =True).unsqueeze(1)
            ypred = model.forward(x)
            ypred = torch.sigmoid(ypred)

            loss = criterion(ypred, y_true.float())

            # ypred = ypred.argmax(axis=1)
            ypred = (ypred>0.5)
            acc_total.addBatch(ypred, y_true)

            losses.update(loss.item(), x.size(0))
            f1 = acc_total.F1score()[1]
            iou = acc_total.IntersectionOverUnion()[1]
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. F1 {f1:.3f}, IOU: {iou:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, f1=f1, iou=iou))
            pbar.update()
        pbar.close()

    return losses.avg, f1, iou


if __name__=="__main__":
    main()