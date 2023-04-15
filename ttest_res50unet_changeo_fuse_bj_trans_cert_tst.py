'''
2022.01.19
@Yinxia Cao
@function: used for training ZY3LC on all cities, 8 bit images
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from torch.utils import data
from ZY3LC_dataset import dataloader_t1t2
from ZY3LC_loader import myImageFloder_8bit_t1t2
from metrics import SegmentationMetric, accprint, acc2file
import segmentation_models_pytorch_new as smp


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 10:
        lr = 0.001
    elif epoch <= 20:
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
    datapath = os.path.join('changedata',fcode, 'testdata')
    test_img, test_lab, _,_,_,_ = \
        dataloader_t1t2(datapath, split=(1.0,0,0),fcode=fcode) # 90% for training

    logdir = os.path.join(r'runs_change\res50cdo_fuse',
                          fcode+'_1ta_cert')
    classes = 2 # 0,1,2
    nchannels = 4

    # test on the whole images
    testdataloader = torch.utils.data.DataLoader(
        myImageFloder_8bit_t1t2(test_img, test_lab, aug=False, channels=nchannels,
                                dir1='img1ta', dir2='img2'),
        batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # model = get_efficientb0().to(device)
    model = smp.UnetCDdiffuse(encoder_name="resnet50", encoder_weights="imagenet",
                             in_channels=nchannels, classes=1).to(device)

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
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")

    id = str(start_epoch)
    txtpath = os.path.join(logdir, 'acc' + id + '_tst10.txt')  # save acc
    vtest_epoch(model, testdataloader, device, classes, start_epoch, txtpath, issave=False)


def vtest_epoch(model, dataloader, device, classes, epoch, txtpath, issave=False):
    model.eval()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloader):
            x = x.to(device, non_blocking =True)
            y_true = y_true.to(device, non_blocking =True).unsqueeze(1) # n c h w
            ypred, _ = model.forward(x)

            ypred = torch.sigmoid(ypred) # 0, 1 (change)

            ypred = (ypred > 0.5) # 0, 1 (change)
            acc_total.addBatch(ypred, y_true)

            f1 = acc_total.F1score()[1]
            iou = acc_total.IntersectionOverUnion()[1]
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}.  F1 {f1:.3f}, IOU: {iou:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, f1=f1, iou=iou))
            pbar.update()
        pbar.close()

    accprint(acc_total)
    if issave:
        acc2file(acc_total, txtpath)


if __name__=="__main__":
    main()