'''
load ZY3LC dataset filename
'''
from glob import glob
import numpy as np
import os
from os.path import join
import pandas as pd
# load classification
def dataloader_tif(filepath, split=0.95):
    # INPUT:
    # filepath: files for train
    # split: train/val dataset
    # RETUREN:
    # left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
    left = glob(join(filepath,'images', 'image*.tif'))
    cls = glob(join(filepath, 'labels', 'label*.tif'))

    assert len(left)==len(cls)
    left.sort()
    cls.sort()

    left = np.array(left)
    cls = np.array(cls)

    num = len(left)
    num_train = int(split * num)

    # File list
    seqpath = join(filepath, 'zy3lcseq.txt')
    if not os.path.exists(seqpath): # not exist
        # Shuffle the file list
        indices = np.arange(num)
        np.random.seed(0) # fixed
        np.random.shuffle(indices)
        np.savetxt(seqpath, indices, fmt='%d', delimiter=',')
    else:
        indices = np.loadtxt(seqpath, delimiter=',')

    indices= indices.astype(np.int32)
    train = indices[:num_train] # indices for training
    val = indices[num_train:] # indices for validation

    left_train = left[train]
    cls_train = cls[train]

    left_val = left[val]
    cls_val = cls[val]

    return left_train, cls_train, left_val, cls_val


# load classification in png form
def dataloader(filepath, split=(0.7,0.1,0.2), issave=False):
    # INPUT:
    # filepath: files for train
    # split: train/val dataset
    # RETUREN:
    # left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
    left = glob(join(filepath,'img', 'img*.tif'))
    cls = glob(join(filepath, 'lab', 'lab*.png'))

    assert len(left)==len(cls)
    left.sort()
    cls.sort()

    left = np.array(left)
    cls = np.array(cls)

    num = len(left)
    num_train = int(split[0] * num)
    num_val = int(split[1] * num)
    # File list
    seqpath = join(filepath, 'seq.txt')
    if not os.path.exists(seqpath): # not exist
        # Shuffle the file list
        indices = np.arange(num)
        np.random.seed(0) # fixed
        np.random.shuffle(indices)
        np.savetxt(seqpath, indices, fmt='%d', delimiter=',')
    else:
        indices = np.loadtxt(seqpath, delimiter=',')

    indices= indices.astype(np.int32)
    train = indices[:num_train] # indices for training
    val = indices[num_train:(num_train+num_val)] # indices for validation
    test = indices[(num_train+num_val):]

    left_train = left[train]
    cls_train = cls[train]

    left_val = left[val]
    cls_val = cls[val]

    left_test = left[test]
    cls_test = cls[test]

    if issave==True:
        ftrain = pd.DataFrame({'0':left_train, '1':cls_train})
        ftrain.to_csv(join(filepath,'trainlist.txt'), header=False, index=False)
        fval= pd.DataFrame({'0':left_val, '1':cls_val})
        fval.to_csv(join(filepath,'vallist.txt'), header=False, index=False)
        ftest = pd.DataFrame({'0':left_test, '1':cls_test})
        ftest.to_csv(join(filepath,'testlist.txt'), header=False, index=False)

    return left_train, cls_train, left_val, cls_val, left_test, cls_test


# load classification in png form
def dataloader_t1t2(filepath, split=(0.7,0.1,0.2), issave=False, fcode=''):
    # INPUT:
    # filepath: files for train
    # split: train/val dataset
    # RETUREN:
    # left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
    left = glob(join(filepath,'img1', 'img_'+ fcode+ '*.tif'))
    cls = glob(join(filepath, 'lab', 'lab_'+fcode+ '*.png'))

    assert len(left)==len(cls)
    left.sort()
    cls.sort()

    left = np.array(left)
    cls = np.array(cls)

    num = len(left)
    num_train = int(split[0] * num)
    num_val = int(split[1] * num)
    # File list
    seqpath = join(filepath, 'seq'+fcode+'.txt')
    if not os.path.exists(seqpath): # not exist
        # Shuffle the file list
        indices = np.arange(num)
        np.random.seed(0) # fixed
        np.random.shuffle(indices)
        np.savetxt(seqpath, indices, fmt='%d', delimiter=',')
    else:
        indices = np.loadtxt(seqpath, delimiter=',')

    indices= indices.astype(np.int32)
    train = indices[:num_train] # indices for training
    val = indices[num_train:(num_train+num_val)] # indices for validation
    test = indices[(num_train+num_val):]

    left_train = left[train]
    cls_train = cls[train]

    left_val = left[val]
    cls_val = cls[val]

    left_test = left[test]
    cls_test = cls[test]

    if issave==True:
        ftrain = pd.DataFrame({'0':left_train, '1':cls_train})
        ftrain.to_csv(join(filepath,'trainlist.txt'), header=False, index=False)
        fval= pd.DataFrame({'0':left_val, '1':cls_val})
        fval.to_csv(join(filepath,'vallist.txt'), header=False, index=False)
        ftest = pd.DataFrame({'0':left_test, '1':cls_test})
        ftest.to_csv(join(filepath,'testlist.txt'), header=False, index=False)

    return left_train, cls_train, left_val, cls_val, left_test, cls_test


def dataloader_split(leftp, clsp, seqpath, split=0.9):
    # for read imglist
    left = pd.read_csv(leftp, header=None)
    left = left[0].values.tolist()
    # for read lablist
    cls = pd.read_csv(clsp, header=None)
    cls = cls[0].values.tolist()

    left = np.array(left)
    cls = np.array(cls)

    num = len(left)
    num_train = int(split* num)

    # File list
    if not os.path.exists(seqpath): # not exist
        # Shuffle the file list
        indices = np.arange(num)
        np.random.seed(0) # fixed
        np.random.shuffle(indices)
        np.savetxt(seqpath, indices, fmt='%d', delimiter=',')
    else:
        indices = np.loadtxt(seqpath, delimiter=',')

    indices= indices.astype(np.int32)
    train = indices[:num_train] # indices for training
    val = indices[num_train:] # indices for validation

    left_train = left[train]
    cls_train = cls[train]

    left_val = left[val]
    cls_val = cls[val]

    return left_train, cls_train, left_val, cls_val


# choose specific city to train
def dataloader_city(filepath, cityname, split=(0.7,0.1,0.2), issave=False):
    # INPUT:
    # filepath: files for train
    # split: train/val dataset
    # RETUREN:
    # left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
    left = glob(join(filepath,'img', 'img_'+cityname+'_*.tif'))
    cls = glob(join(filepath, 'lab', 'lab_'+cityname+'_*.png'))

    assert len(left)==len(cls)
    left.sort()
    cls.sort()

    left = np.array(left)
    cls = np.array(cls)

    num = len(left)
    num_train = int(split[0] * num)
    num_val = int(split[1] * num)
    # File list
    seqpath = join(filepath, 'seq_'+cityname+'.txt')
    if not os.path.exists(seqpath): # not exist
        # Shuffle the file list
        indices = np.arange(num)
        np.random.seed(0) # fixed
        np.random.shuffle(indices)
        np.savetxt(seqpath, indices, fmt='%d', delimiter=',')
    else:
        indices = np.loadtxt(seqpath, delimiter=',')

    indices= indices.astype(np.int32)
    train = indices[:num_train] # indices for training
    val = indices[num_train:(num_train+num_val)] # indices for validation
    test = indices[(num_train+num_val):]

    left_train = left[train]
    cls_train = cls[train]

    left_val = left[val]
    cls_val = cls[val]

    left_test = left[test]
    cls_test = cls[test]

    if issave==True:
        ftrain = pd.DataFrame({'0':left_train, '1':cls_train})
        ftrain.to_csv(join(filepath,'trainlist_'+cityname+'.txt'), header=False, index=False)
        fval= pd.DataFrame({'0':left_val, '1':cls_val})
        fval.to_csv(join(filepath,'vallist_'+cityname+'.txt'), header=False, index=False)
        ftest = pd.DataFrame({'0':left_test, '1':cls_test})
        ftest.to_csv(join(filepath,'testlist_'+cityname+'.txt'), header=False, index=False)

    return left_train, cls_train, left_val, cls_val, left_test, cls_test


if __name__ == "__main__":

    ## 20220302 generate valid building patch
    # filepath = r'E:\yinxcao\ZY3LC\datanew8bit'
    # leftp = join(filepath, 'imglistvalid_train30_imgpath.csv')
    # clsp = join(filepath, 'imglistvalid_train30_labpath.csv')
    # seqpath = join(filepath, 'seqvalid_train30.txt')
    # left_train, cls_train, left_val, cls_val = dataloader_split(leftp, clsp, seqpath, split=0.9)
    # print(left_train[:10])
    # print(cls_train[:10])

    ## 20220402 generate test sample patches
    filepath = r'E:\yinxcao\ZY3LC\changedata\testdata'
    train_img, train_lab, _,_,_,_ = dataloader_t1t2(filepath, split=(1.0, 0, 0), issave=False, fcode='sh')
    print(train_img[:5])





