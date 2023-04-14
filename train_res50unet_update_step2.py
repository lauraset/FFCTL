'''
2022.01.19
@Yinxia Cao
@function: used for training ZY3LC on beijing, 8 bit images
apply label correction
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
from ZY3LC_loader import myImageFloder_8bit_binarypath, myImageFloder_8bit_binary, myImageFloder_8bit_binary_update
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

    epochs_update = 10
    iroot= 'runs'
    logdirwarm = os.path.join(iroot, 'res50' + args.classname + '_warm')

    logdir = os.path.join(iroot, 'res50' + args.classname + '_update', 'update')
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)

    iepoch = '20'
    updatepath = os.path.join(logdir, 'pred') # for storing updated labels
    os.makedirs(updatepath, exist_ok=True)

    # NUM_WORKERS = 4
    classes = 2 # 0, 1, 2, 3, 4, 5, 6
    nchannels = 4
    imgsize = 256
    global best_acc
    best_acc = 0
    positive = 255 # values of buildings
    bs = 32
    nw = 8
    # train on the randomly cropped images
    traindataloader_path = torch.utils.data.DataLoader(
        myImageFloder_8bit_binarypath(train_img, train_lab, aug=False, imgsize=imgsize, # no augmentation
                                      channels=nchannels, positive=positive),
        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # train with updated labels
    traindataloader_update = torch.utils.data.DataLoader(
        myImageFloder_8bit_binary_update(train_img, train_lab, updatepath, aug=True, imgsize=imgsize,
                                         channels=nchannels, positive=positive),
        batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)

    # test on the whole images
    valdataloader = torch.utils.data.DataLoader(
        myImageFloder_8bit_binary(val_img, val_lab, aug=False, imgsize=imgsize, channels=nchannels, positive=positive),
        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
                             in_channels=nchannels, classes=1).to(device)
    # restore from warmup
    weight = torch.load(os.path.join(logdirwarm, 'checkpoint'+iepoch+'.tar'))
    print('loading the warmup params from epoch: %d'%weight['epoch'])
    model.load_state_dict(weight['state_dict'])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # print the model
    start_epoch = 0
    resume = os.path.join(logdir, 'model_best.tar')
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

    # test update_label function
    # update_label(model, traindataloader_path, device, updatepath)

    # correction
    for epoch in range(epochs_update-start_epoch):
        epoch = start_epoch + epoch + 1 # current epochs
        # adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        print('epoch %d, lr: %.6f'%(epoch, lr))
        # update
        print('update labels==>')
        update_label(model, traindataloader_path, device, updatepath)

        # train
        train_loss, train_f1, train_iou = \
            train_epoch_update(model, criterion, traindataloader_update,
                               optimizer, device, epoch, classes, alpha=0.2)

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



def update_label(net, dataloader, device, updatepath):
    net.eval()
    with torch.no_grad():
        for images, mask, imgpath in tqdm(dataloader):
            images = images.to(device, non_blocking=True) # N C H W
            output = net(images)
            output_pred = (torch.sigmoid(output)>0.5).long() # Prediction
            # select loss as uncertainty, mean values used as threshold
            # loss = criterion(output, mask.float()) # N C H W
            # imeans = torch.mean(loss, dim=[1,2,3], keepdim=True) # thresh N 1 1 1
            # imax = torch.max(loss, dim=3, keepdim=True)[0]
            # imax = K*torch.max(imax, dim=2, keepdim=True)[0] # N 1 1 1
            # ithresh = torch.max(imeans, imax)
            # mask_true = (loss<ithresh) # pred correct
            # mask_update = torch.where(mask_true, output_pred, mask)
            # save
            for idx, imgp in enumerate(imgpath):
                ibase = os.path.basename(imgp)[:-4]
                resname = os.path.join(updatepath, ibase)
                tmp = output_pred[idx].squeeze().cpu().numpy().astype('uint8') # H W, [0,1]
                diff = tmp.astype('float')- mask[idx].numpy().astype('float') # -1,0,1
                diff[diff==1] = 255 # pos
                diff[diff==-1] = 128 # neg
                cv2.imwrite(resname+'_up.png', tmp)
                cv2.imwrite(resname+'_upc.png', tmp*255)
                cv2.imwrite(resname + '_diff.png', diff)


def train_epoch_update(net, criterion, dataloader, optimizer, device, epoch, classes, alpha=0.5):
    net.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (images, mask, update) in enumerate(dataloader):
        images = images.to(device, non_blocking=True) # N C H W
        mask = mask.to(device, non_blocking=True).unsqueeze(1) # N 1 H W
        update = update.to(device, non_blocking=True).unsqueeze(1)

        output = net(images)
        output = torch.sigmoid(output)

        loss = alpha*criterion(output, mask.float()) + criterion(output, update.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = (output>0.5) # N C H W
        acc_total.addBatch(output, mask)
        losses.update(loss.item(), images.size(0))

        f1 = acc_total.F1score()[1]
        iou = acc_total.IntersectionOverUnion()[1]
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. F1 {f1:.3f}, IOU: {iou:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, f1=f1, iou=iou))
        pbar.update()
    pbar.close()

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