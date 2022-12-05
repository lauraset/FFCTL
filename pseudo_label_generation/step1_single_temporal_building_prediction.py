#!/usr/bin/env python
# coding: utf-8
# #### Test on a whole image on 2022.2.22

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import tifffile as tif
from os.path import join
import math
import time
import segmentation_models_pytorch as smp
import rasterio as rio


def predict_whole_image_over(model, image, r, c, num_class=1, grid=512, stride=256, device='device'):
    '''
    image: n,r,c,b  where n = 1
    model: FCN
    overlay prediction
    change pad to the out space
    r,c: original shape
    rows, cols: changed shape
    '''
    _, _, rows, cols = image.shape
    #     n,b,r,c = image.shape
    #     rows= math.ceil((r-grid)/(stride))*stride+grid
    #     cols= math.ceil((c-grid)/(stride))*stride+grid
    #     rows=math.ceil(rows)
    #     cols=math.ceil(cols)
    print('rows is {}, cols is {}'.format(rows, cols))
    # image_= np.pad(image,((0,0),(0,0),(0,rows-r), (0,cols-c), ),'symmetric')
    weight = np.zeros((rows, cols))
    res = np.zeros((num_class, rows, cols), dtype=np.float32)
    num_patch = len(range(0, rows, stride)) * len(range(0, cols, stride))
    print('num of patch is', num_patch)
    k = 0
    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            start = time.time()
            patch = image[0:, 0:, i:i + grid, j:j + grid]
            patch = torch.from_numpy(patch).float()
            with torch.no_grad():
                pred = model(patch.to(device))
            pred = pred.cpu().numpy()
            res[:, i:i + grid, j:j + grid] += np.squeeze(pred)  # H W
            weight[i:i + grid, j:j + grid] += 1
            end = time.time()
            k = k + 1
            if k % 500 == 0:
                print('patch [%d/%d] time elapse:%.3f' % (k, num_patch, (end - start)))
                # tif.imsave(os.path.join(ipath,'height{}_{}.tif'.format(i,j)),pred,dtype=np.float32)
    res = res / weight
    # res=np.argmax(res, axis=0)
    res = res[:, 0:r, 0:c].astype(np.float32)
    res = np.squeeze(res)
    return res


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_imgfile(ipath):
    filelist = []
    dirlist = os.listdir(ipath)
    for idir in dirlist:
        tmp = join(ipath, idir)
        p1 = join(ipath, idir, 'img18.tif')
        if os.path.isfile(p1):
            filelist.append(p1)

        p1 = join(ipath, idir, 'img28.tif')
        if os.path.isfile(p1):
            filelist.append(p1)
    return filelist


def main(filelist):
    device = 'cuda'
    nchannels = 4
    classes = 1
    net = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet",
                                 in_channels=nchannels, classes=classes).to(device)
    logdir = r'.\runs\res50build_update\scratch'
     # print the model
    resume = os.path.join(logdir, 'model_best.tar')
    try:
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> success '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
    except:
        print("resume fails")
    net.eval()

    # ### batch processing
    postfix = '_update_scratch'

    for file in filelist:
        iname = os.path.basename(file)[:-4]
        iroot = os.path.dirname(file)
        idir = os.path.join(iroot, "pred")
        os.makedirs(idir, exist_ok=True)
        respath = os.path.join(idir, iname + postfix + '.tif')

        if os.path.isfile(respath):
            print('exist then skip: ' + idir)
            continue

        # begin
        print('process:' + idir)
        # 1. load image
        mux = tif.imread(file)

        # 2. normalize to [0,1]
        mux = mux / 255.0

        # reshape to 1 C H W
        if mux.shape[2] > 4:
            mux = np.expand_dims(mux, axis=0)  # C H W
        else:
            mux = np.expand_dims(np.transpose(mux, (2, 0, 1)), axis=0)  # H W C

        # 3. predict
        grid = 1024
        stride = grid - 64
        # pad img
        n, b, r, c = mux.shape
        rows = math.ceil((r - grid) / (stride)) * stride + grid
        cols = math.ceil((c - grid) / (stride)) * stride + grid
        print('rows is {}, cols is {}'.format(rows, cols))
        mux = np.pad(mux, ((0, 0), (0, 0), (0, rows - r), (0, cols - c),), 'symmetric')

        res = predict_whole_image_over(net, mux, r=r, c=c, num_class=1, grid=grid, stride=stride, device=device)

        # 4. convert to [0,1]
        res = sigmoid(res)

        # 5. save
        rastermeta = rio.open(file).profile
        rastermeta.update(dtype='uint8', count=1, compress='lzw')
        iname = os.path.basename(file)[:-4]
        idir = os.path.join(os.path.dirname(file), "pred")
        os.makedirs(idir, exist_ok=True)

        # seg
        respath = os.path.join(idir, iname + postfix + '_seg.tif')
        res_seg = (res > 0.5).astype('uint8') * 255
        with rio.open(respath, mode="w", **rastermeta) as dst:
            dst.write(res_seg, 1)

        # prob: scale from [0,1] to [0,255]
        res = (res * 65535).astype('uint16')
        respath = os.path.join(idir, iname + postfix + '.tif')
        with rio.open(respath, mode="w", **rastermeta) as dst:
            dst.write(res, 1)

        # clear variable
        mux = 0
        res = 0
        res_seg = 0


if __name__=="__main__":
    # add file path
    filelist = ['Z:\\yinxcao\\change\\beijing\\img18.tif', 'Z:\\yinxcao\\change\\beijing\\img28.tif',
                'Z:\\yinxcao\\change\\shanghai\\img18.tif', 'Z:\\yinxcao\\change\\shanghai\\img28.tif'
                ]
    main(filelist)



