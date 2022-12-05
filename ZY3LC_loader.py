'''
2021.4.14 load ZY3LC
'''
import torch.utils.data as data
from PIL import Image, ImageOps
import tifffile as tif
import albumentations as A
import torch
import numpy as np
import random
from copy import deepcopy # add
import cv2
import os
from dataaug.aug import ToGrayMulti, ToColorJitter

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# for classification
image_transformed_cls = A.Compose([
    A.Flip(p=0.5),
    A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(p=0.5),
    # A.RandomBrightnessContrast(p=0.5), #wierd, for augcolor, have tested on 2022.2.15,should delete
]
)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


# img_mean and std
# VNIR
# for xian
IMG_MEAN_XIAN = [290.2186,  276.1452, 284.8843 , 372.6959]
IMG_STD_XIAN = [65.9909  , 74.0876 , 102.7238 , 102.0488]

# 2021.6.3
# for all cities: image has been normalized to SR
# IMG_MEAN_ALL = [1136.382247, 1179.705369, 1340.988426, 2290.007931]
# IMG_STD_ALL = [322.735247, 400.55218, 542.5424405, 941.419801]

# 2021.7.1: update mux images and add tlc
IMG_MEAN_ALL = [1142.05719085069, 1183.92746808790, 1324.37698042479, 2360.08189621090]
IMG_STD_ALL = [352.892230743533, 402.069966221899, 554.259982955950, 1096.14879868840]
TLC_MEAN_ALL = [440.312064755891, 387.339043098102, 444.801891941169]
TLC_STD_ALL = [270.286351619591, 202.061888100090, 216.688621196791]
# RGB, NIR
class myImageFloder_XIAN(data.Dataset):
    def __init__(self, left, cls, imgsize = 512, channels=3, aug=False):
        self.left = left
        self.cls = cls
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels]
        cls_img = tif.imread(cls)

        # Crop images
        w, h, _ = left_img.shape
        tw, th = self.imgsize, self.imgsize
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        left_img = left_img[x1:x1+tw, y1:y1+th]
        cls_img = cls_img[x1:x1+tw, y1:y1+th]

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = (left_img - IMG_MEAN_XIAN[:self.channels]) / IMG_STD_XIAN[:self.channels]
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float() # H W C ==> C H W

        # classification
        lab = cls_img-1 # [0, C-1]
        lab = torch.from_numpy(lab).long()

        return left_img, lab

    def __len__(self):
        return len(self.left)


# For all cities used for training
class myImageFloder(data.Dataset):
    def __init__(self, left, cls, imgsize = 512, channels=4, aug=False):
        self.left = left
        self.cls = cls
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = (left_img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float() # H W C ==> C H W

        # classification
        lab = cls_img-1 # [0, C-1]
        lab = torch.from_numpy(lab).long()

        return left_img, lab

    def __len__(self):
        return len(self.left)


# For all cities used for training
class myImageFloder_binary(data.Dataset):
    def __init__(self, left, cls, imgsize = 256, channels=4, aug=False, positive=0):
        self.left = left
        self.cls = cls
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.positive = positive

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = (left_img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float() # H W C ==> C H W

        # classification
        lab = np.zeros_like(cls_img, dtype=cls_img.dtype)
        lab[cls_img==self.positive] = 1

        return left_img, lab

    def __len__(self):
        return len(self.left)

# For all cities used for training
class myImageFloder_binarypath(data.Dataset):
    def __init__(self, left, cls, imgsize = 256, channels=4, aug=False, positive=0, num_sample=0):
        self.left = left
        self.cls = cls
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.positive = positive
        if num_sample > 0:
            self.left = left[:num_sample]
            self.cls = cls[:num_sample]

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = (left_img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float() # H W C ==> C H W

        # classification
        lab = np.zeros_like(cls_img, dtype=cls_img.dtype)
        lab[cls_img==self.positive] = 1

        return left_img, lab, left

    def __len__(self):
        return len(self.left)

# 2022.01.19
# for 8 bit images
class myImageFloder_8bit(data.Dataset):
    def __init__(self, left, cls, imgsize = 256, channels=4, aug=False):
        self.left = left
        self.cls = cls
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0
        # classification
        cls_img = cls_img-1 # [0, C-1]
        cls_img = torch.from_numpy(cls_img).long()

        return left_img, cls_img

    def __len__(self):
        return len(self.left)

# for binary
class myImageFloder_8bit_binary(data.Dataset):
    def __init__(self, left, cls, imgsize = 256, channels=4, aug=False, positive=0,
                 iscrop=False, istlc=False, ismabi=False):
        self.left = left
        self.cls = cls
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.positive = positive
        self.iscrop = iscrop
        self.istlc = istlc
        self.ismabi = ismabi

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG

        if self.istlc:
            iname = os.path.basename(left)
            idir = os.path.dirname(os.path.dirname(left))
            tlc = tif.imread(os.path.join(idir, 'tlc', 'tlc' + iname[3:]))
            left_img = np.concatenate((left_img, tlc), axis=2) # H W C
        if self.ismabi:
            iname = os.path.basename(left)
            idir = os.path.dirname(os.path.dirname(left))
            mabi = cv2.imread(os.path.join(idir, 'mabi', 'mabi' + iname[3:-4]+'.png'),
                              cv2.IMREAD_UNCHANGED) # 0-255, H W
            mabi = np.expand_dims(mabi, axis=2) # H W to H W 1
            left_img = np.concatenate((left_img, mabi), axis=2) # H W C
        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        if self.iscrop:
            left_img = A.center_crop(left_img, self.imgsize, self.imgsize)
            cls_img = A.center_crop(cls_img, self.imgsize, self.imgsize)

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0
        # classification

        lab = np.zeros_like(cls_img, dtype=cls_img.dtype)
        lab[cls_img==self.positive] = 1

        lab = torch.from_numpy(lab).long()

        return left_img, lab

    def __len__(self):
        return len(self.left)


# for binary
class myImageFloder_8bit_binarypath(data.Dataset):
    def __init__(self, left, cls, imgsize = 256, channels=4, aug=False, positive=0,num_sample=0,
                 iscrop=False, istlc=False, ismabi=False):
        self.left = left
        self.cls = cls
        if num_sample > 0:
            self.left = left[:num_sample]
            self.cls = cls[:num_sample]
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.positive = positive
        self.iscrop = iscrop
        self.istlc = istlc
        self.ismabi = ismabi

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG

        if self.istlc:
            iname = os.path.basename(left)
            idir = os.path.dirname(os.path.dirname(left))
            tlc = tif.imread(os.path.join(idir, 'tlc', 'tlc' + iname[3:]))
            left_img = np.concatenate((left_img, tlc), axis=2) # H W C

        if self.ismabi:
            iname = os.path.basename(left)
            idir = os.path.dirname(os.path.dirname(left))
            mabi = cv2.imread(os.path.join(idir, 'mabi', 'mabi' + iname[3:-4]+'.png'),
                              cv2.IMREAD_UNCHANGED) # 0-255, H W
            mabi = np.expand_dims(mabi, axis=2) # H W to H W 1
            left_img = np.concatenate((left_img, mabi), axis=2) # H W C

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        if self.iscrop:
            left_img = A.center_crop(left_img, self.imgsize, self.imgsize)
            cls_img = A.center_crop(cls_img, self.imgsize, self.imgsize)

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0
        # classification

        lab = np.zeros_like(cls_img, dtype=cls_img.dtype)
        lab[cls_img==self.positive] = 1

        lab = torch.from_numpy(lab).long()

        return left_img, lab, left

    def __len__(self):
        return len(self.left)


# 2022.9.18: only return img
class myImageFloder_8bit_binarypath_img(data.Dataset):
    def __init__(self, left, channels=4):
        self.left = left
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0
        return left_img, left

    def __len__(self):
        return len(self.left)


# for binary, return img and path
class myImageFloder_8bit_binarypath_gid(data.Dataset):
    def __init__(self, left,  imgsize = 256, channels=4, aug=False, positive=0, num_sample=0):
        self.left = left
        if num_sample > 0:
            self.left = left[:num_sample]
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.positive = positive

    def __getitem__(self, index):
        left = self.left[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img)
            left_img = transformed["image"]

        # scale images
        left_img = left_img[:,:,::-1].copy() # nir-rgb to bgr-nir
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0

        return left_img, left

    def __len__(self):
        return len(self.left)


# for binary update
class myImageFloder_8bit_binary_update(data.Dataset):
    def __init__(self, left, cls, updatepath, imgsize = 256, channels=4, aug=False,
                 positive=0, returnpath=False, istlc = False, ismabi=False):
        self.left = left
        self.cls = cls
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.positive = positive
        self.updatepath = updatepath
        self.returnpath = returnpath # add
        self.istlc = istlc
        self.ismabi = ismabi

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # base name
        ibase = os.path.basename(left)[:-4]

        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG
        update = cv2.imread(os.path.join(self.updatepath, ibase + '_up.png'), cv2.IMREAD_UNCHANGED)

        # add
        if self.istlc:
            iname = os.path.basename(left)
            idir = os.path.dirname(os.path.dirname(left))
            tlc = tif.imread(os.path.join(idir, 'tlc', 'tlc' + iname[3:]))
            left_img = np.concatenate((left_img, tlc), axis=2) # H W C

        if self.ismabi:
            iname = os.path.basename(left)
            idir = os.path.dirname(os.path.dirname(left))
            mabi = cv2.imread(os.path.join(idir, 'mabi', 'mabi' + iname[3:-4]+'.png'),
                              cv2.IMREAD_UNCHANGED) # 0-255, H W
            mabi = np.expand_dims(mabi, axis=2) # H W to H W 1
            left_img = np.concatenate((left_img, mabi), axis=2) # H W C

        ref = np.stack((cls_img, update), axis=2) # H W C

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=ref)
            left_img = transformed["image"]
            ref = transformed["mask"]

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0

        # lab and update
        cls_img = ref[:, :, 0]
        lab = np.zeros_like(cls_img, dtype=cls_img.dtype)
        lab[cls_img==self.positive] = 1
        lab  = torch.from_numpy(lab).long()

        update = torch.from_numpy(ref[:, :, 1]).long()
        if self.returnpath:
            return left_img, lab, update, left
        else:
            return left_img, lab, update

    def __len__(self):
        return len(self.left)

'''
# # 2022.3.8 consider strong data augmentation
# class myImageFloder_8bit_binary_update_sda(data.Dataset):
#     def __init__(self, left, cls, updatepath, imgsize = 256, channels=4, positive=0, returnpath=False):
#         self.left = left
#         self.cls = cls
#         # self.aug = aug # augmentation for images
#         self.imgsize = imgsize # used for training and validation
#         self.channels = channels
#         self.positive = positive
#         self.updatepath = updatepath
#         self.returnpath = returnpath # add
#         self.transform_w = A.Compose([
#             A.RandomResizedCrop(height=self.imgsize,width=self.imgsize, p=1),
#             A.Flip(p=0.5),
#             A.Rotate(p=0.5),
#         ])
#         self.transform_s =  A.Compose([
#             ToColorJitter(0.5, 0.5, 0.5, 0.25, p=0.8),
#             ToGrayMulti(p=0.2),
#             A.GaussianBlur(p=0.5),
#         ])
#
#     def __getitem__(self, index):
#         left = self.left[index]
#         cls = self.cls[index]
#         # base name
#         ibase = os.path.basename(left)[:-4]
#
#         # Read images
#         left_img = tif.imread(left)[:, :, :self.channels] # TIF
#         cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG
#         update = cv2.imread(os.path.join(self.updatepath, ibase + '_up.png'), cv2.IMREAD_UNCHANGED)
#
#         ref = np.stack((cls_img, update), axis=2) # H W C
#
#         # Augmentation
#         # weak: spatial transformations, apply to image and mask
#         transformed = self.transform_w(image=left_img, mask=ref)  # basic augmentation
#         left_img_w = transformed["image"]
#         ref = transformed["mask"]
#         # strong: spectral transformations, only apply to image
#         transformed = self.transform_s(image=left_img_w)
#         left_img_s = transformed["image"]
#
#         # scale images
#         left_img_w = torch.from_numpy(left_img_w).permute(2, 0, 1).float()/255.0 # H W C ==> C H W
#         left_img_s = torch.from_numpy(left_img_s).permute(2, 0, 1).float()/255.0 # H W C ==> C H W
#         left_img = {'weak': left_img_w, 'strong': left_img_s}
#
#         # lab and update
#         cls_img = ref[:, :, 0]
#         lab = np.zeros_like(cls_img, dtype=cls_img.dtype)
#         lab[cls_img==self.positive] = 1
#         lab  = torch.from_numpy(lab).long()
#
#         update = torch.from_numpy(ref[:, :, 1]).long()
#         if self.returnpath:
#             return left_img, lab, update, left
#         else:
#             return left_img, lab, update
#
#     def __len__(self):
#         return len(self.left)
'''

# for binary update 2 nets

# 2022.3.8 consider strong data augmentation
class myImageFloder_8bit_binary_update_sda(data.Dataset):
    def __init__(self, left, cls, updatepath, imgsize = 256, channels=4, positive=0, returnpath=False):
        self.left = left
        self.cls = cls
        # self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.positive = positive
        self.updatepath = updatepath
        self.returnpath = returnpath # add
        self.transform = A.Compose([
            A.Flip(p=0.5),
            A.RandomGridShuffle(grid=(2, 2), p=0.5),
            A.Rotate(p=0.5),
            # ToColorJitter(0.5, 0.5, 0.5, 0.25, p=0.8),
            # ToGrayMulti(p=0.2),
            A.GaussianBlur(p=0.5),
        ])

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # base name
        ibase = os.path.basename(left)[:-4]

        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG
        update = cv2.imread(os.path.join(self.updatepath, ibase + '_up.png'), cv2.IMREAD_UNCHANGED)

        ref = np.stack((cls_img, update), axis=2) # H W C

        # Augmentation
        transformed = self.transform(image=left_img, mask=ref)  # basic augmentation
        left_img = transformed["image"]
        ref = transformed["mask"]

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float()/255.0 # H W C ==> C H W

        # lab and update
        cls_img = ref[:, :, 0]
        lab = np.zeros_like(cls_img, dtype=cls_img.dtype)
        lab[cls_img==self.positive] = 1
        lab  = torch.from_numpy(lab).long()

        update = torch.from_numpy(ref[:, :, 1]).long()
        if self.returnpath:
            return left_img, lab, update, left
        else:
            return left_img, lab, update

    def __len__(self):
        return len(self.left)


class myImageFloder_8bit_binary_update2net(data.Dataset):
    def __init__(self, left, cls, updatepath1, updatepath2, imgsize = 256, channels=4, aug=False, positive=0):
        self.left = left
        self.cls = cls
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.positive = positive
        self.updatepath1 = updatepath1
        self.updatepath2 = updatepath2

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # base name
        ibase = os.path.basename(left)[:-4]

        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG
        update1 = cv2.imread(os.path.join(self.updatepath1, ibase + '_up.png'), cv2.IMREAD_UNCHANGED)
        update2 = cv2.imread(os.path.join(self.updatepath2, ibase + '_up.png'), cv2.IMREAD_UNCHANGED)

        ref = np.stack((cls_img, update1, update2), axis=2) # H W C

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=ref)
            left_img = transformed["image"]
            ref = transformed["mask"]

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0

        # lab and update
        cls_img = ref[:, :, 0]
        lab = np.zeros_like(cls_img, dtype=cls_img.dtype)
        lab[cls_img==self.positive] = 1
        lab  = torch.from_numpy(lab).long()

        update1 = torch.from_numpy(ref[:, :, 1]).long()
        update2 = torch.from_numpy(ref[:, :, 2]).long()

        return left_img, lab, update1, update2

    def __len__(self):
        return len(self.left)


# for binary update then scratch
class myImageFloder_8bit_binary_update_scratch(data.Dataset):
    def __init__(self, left, updatepath, imgsize = 256, channels=4,
                 aug=False, positive=0, ismabi=False):
        self.left = left
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.positive = positive
        self.updatepath = updatepath
        self.ismabi = ismabi

    def __getitem__(self, index):
        left = self.left[index]
        # base name
        ibase = os.path.basename(left)[:-4]

        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        update = cv2.imread(os.path.join(self.updatepath, ibase + '_up.png'), cv2.IMREAD_UNCHANGED)

        if self.ismabi:
            iname = os.path.basename(left)
            idir = os.path.dirname(os.path.dirname(left))
            mabi = cv2.imread(os.path.join(idir, 'mabi', 'mabi' + iname[3:-4]+'.png'),
                              cv2.IMREAD_UNCHANGED) # 0-255, H W
            mabi = np.expand_dims(mabi, axis=2) # H W to H W 1
            left_img = np.concatenate((left_img, mabi), axis=2) # H W C

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=update)
            left_img = transformed["image"]
            update = transformed["mask"]

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0

        # updated labels
        update = torch.from_numpy(update).long() # 0, 1

        return left_img, update

    def __len__(self):
        return len(self.left)

# for binary update then scratch
# 2022.3.10 add strong data augmentation
class myImageFloder_8bit_binary_update_scratch_sda(data.Dataset):
    def __init__(self, left, updatepath, imgsize = 256, channels=4, aug=False, positive=0):
        self.left = left
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.positive = positive
        self.updatepath = updatepath
        self.transform_s = A.Compose([
            A.Flip(p=0.5),
            A.RandomGridShuffle(grid=(2, 2), p=0.5),
            A.Rotate(p=0.5),
            ToColorJitter(0.5, 0.5, 0.5, 0.25, p=0.8),
            # ToGrayMulti(p=0.2),
            A.GaussianBlur(p=0.5),
        ])

    def __getitem__(self, index):
        left = self.left[index]
        # base name
        ibase = os.path.basename(left)[:-4]

        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        update = cv2.imread(os.path.join(self.updatepath, ibase + '_up.png'), cv2.IMREAD_UNCHANGED)

        # Augmentation
        if self.aug:
            transformed = self.transform_s(image=left_img, mask=update)
            left_img = transformed["image"]
            update = transformed["mask"]

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0

        # updated labels
        update = torch.from_numpy(update).long() # 0, 1

        return left_img, update

    def __len__(self):
        return len(self.left)


# for 8 bit images
class myImageFloder_8bitpath(data.Dataset):
    def __init__(self, left, cls, imgsize = 256, channels=4, aug=False, num_sample=0):
        self.left = left
        self.cls = cls
        if num_sample > 0:
            self.left = left[:num_sample]
            self.cls = cls[:num_sample]
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0
        # classification
        cls_img = cls_img-1 # [0, C-1]
        cls_img = torch.from_numpy(cls_img).long()

        return left_img, cls_img, left

    def __len__(self):
        return len(self.left)


# add boundary
class myImageFloder_bound(data.Dataset):
    def __init__(self, left, cls, bound, imgsize = 512, channels=4, aug=False):
        self.left = left
        self.cls = cls
        self.bound = bound
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        bound = self.bound[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG
        bound_img = cv2.imread(bound, cv2.IMREAD_UNCHANGED) # PNG
        cls = np.stack((cls_img, bound_img), axis=2) # w h 2
        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls)
            left_img = transformed["image"]
            cls = transformed["mask"]

        # scale images
        left_img = (left_img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float() # H W C ==> C H W

        # classification
        cls_img = cls[:, :, 0]
        lab = cls_img-1 # [0, C-1]
        lab = torch.from_numpy(lab).long()

        bound_img = cls[:, :, 1] # 0, 1
        bound_img = torch.from_numpy(bound_img).long()

        return left_img, lab, bound_img

    def __len__(self):
        return len(self.left)


# add coarse
class myImageFloder_coarse(data.Dataset):
    def __init__(self, left, cls, bound, imgsize = 512, channels=4, aug=False):
        self.left = left
        self.cls = cls
        self.bound = bound
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        bound = self.bound[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG
        bound_img = cv2.imread(bound, cv2.IMREAD_UNCHANGED) # PNG
        cls = np.stack((cls_img, bound_img), axis=2) # w h 2
        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls)
            left_img = transformed["image"]
            cls = transformed["mask"]

        # scale images
        left_img = (left_img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float() # H W C ==> C H W

        # classification
        lab = cls[:, :, 0]-1 # [0, C-1]
        lab = torch.from_numpy(lab).long()

        bound_img = cls[:, :, 1]-1 # 0, 1, 2, 3
        bound_img = torch.from_numpy(bound_img).long()

        return left_img, lab, bound_img

    def __len__(self):
        return len(self.left)


# add boundary, tlc
class myImageFloder_bound_tlc(data.Dataset):
    def __init__(self, left, tlc, cls, bound, imgsize = 512, channels=4, aug=False):
        self.left = left
        self.tlc = tlc
        self.cls = cls
        self.bound = bound
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        bound = self.bound[index]
        tlc = self.tlc[index]
        # 1. Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        tlc_img = tif.imread(tlc) # TIF
        img = np.concatenate((left_img, tlc_img), axis=2) # W H 7

        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG
        bound_img = cv2.imread(bound, cv2.IMREAD_UNCHANGED) # PNG
        cls = np.stack((cls_img, bound_img), axis=2) # W H 2

        # 2. Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=img, mask=cls)
            img = transformed["image"]
            cls = transformed["mask"]

        # mux + tlc
        img = img.astype(np.float32) # from uint16 to float32
        img[:, :, :self.channels] = (img[:, :, :self.channels] - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        img[:, :, self.channels:] = (img[:, :, self.channels:] - TLC_MEAN_ALL) / TLC_STD_ALL
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        # class
        cls_img = cls[:, :, 0]
        cls_img = cls_img-1 # [0, C-1]
        cls_img = torch.from_numpy(cls_img).long()
        # bound
        bound_img = cls[:, :, 1] # 0, 1
        bound_img = torch.from_numpy(bound_img).long()

        return img, cls_img, bound_img

    def __len__(self):
        return len(self.left)


# For all cities used for training
class myImageFloder_update(data.Dataset):
    def __init__(self, left, cls, cls_update, imgsize = 512, channels=4, aug=False):
        self.left = left
        self.cls = cls
        self.cls_update = cls_update
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        cls_update = self.cls_update[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img_old = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG
        cls_img_update = cv2.imread(cls_update, cv2.IMREAD_UNCHANGED) # PNG
        cls_img = np.stack((cls_img_old, cls_img_update), axis=2) # w h 2: noise, update
        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = (left_img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float() # H W C ==> C H W

        # classification
        cls_img = cls_img-1 # [0, C-1]
        cls_img = torch.from_numpy(cls_img).long()

        return left_img, cls_img[:,:,0], cls_img[:,:,1], os.path.basename(cls)

    def __len__(self):
        return len(self.left)


# For all cities used for eval train folder
# return img, lab, respath
class myImageFloder_evaltrain(data.Dataset):
    def __init__(self, left, cls, imgsize = 512, channels=4, aug=False):
        self.left = left
        self.cls = cls
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = (left_img - IMG_MEAN_ALL[:self.channels]) / IMG_STD_ALL[:self.channels]
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float() # H W C ==> C H W

        # classification
        lab = cls_img-1 # [0, C-1]
        lab = torch.from_numpy(lab).long()

        return left_img, lab, os.path.basename(cls)

    def __len__(self):
        return len(self.left)


# return building or non-building
class myImageFloder_build(data.Dataset):
    def __init__(self, left, cls, imgsize = 512, channels=3, aug=False):
        self.left = left
        self.cls = cls
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img = tif.imread(left)[:, :, :self.channels]
        cls_img = tif.imread(cls)

        # Crop images
        w, h, _ = left_img.shape
        tw, th = self.imgsize, self.imgsize
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        left_img = left_img[x1:x1+tw, y1:y1+th]
        cls_img = cls_img[x1:x1+tw, y1:y1+th]

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = (left_img - IMG_MEAN_XIAN[:self.channels]) / IMG_STD_XIAN[:self.channels]
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float() # H W C ==> C H W

        # classification
        lab = cls_img.copy()
        lab[lab!=1]=0 # 0: non-building; 1-building
        lab = torch.from_numpy(lab).long()

        return left_img, lab

    def __len__(self):
        return len(self.left)

# for change
class myImageFloder_8bit_t1t2(data.Dataset):
    def __init__(self, left, cls, imgsize = 256, channels=4, aug=False, returnpath=False,
                 numsample=0, positive=0, dir1='img1', dir2='img2', isprob=False):
        self.left = left
        self.cls = cls
        if numsample>0:
            self.left = left[:numsample]
            self.cls = cls[:numsample]
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.returnpath = returnpath
        self.positive = positive
        self.dir1 = dir1
        self.dir2 = dir2
        self.isprob = isprob

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        idir = os.path.dirname(os.path.dirname(left))
        iname = os.path.basename(left)
        left = os.path.join(idir, self.dir1, iname)
        left2 = os.path.join(idir, self.dir2, iname)
        # Read images
        left_img1 = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG
        left_img2 = tif.imread(left2)[:, :, :self.channels]  # TIF

        left_img = np.concatenate((left_img1, left_img2), axis=2) # H W C
        # add change probability
        if self.isprob:
            probp = os.path.join(idir, 'prob', 'prob'+iname[3:-4]+'.png')
            prob = cv2.imread(probp, cv2.IMREAD_UNCHANGED)
            prob = np.expand_dims(prob, axis=2) # H W 1
            left_img = np.concatenate((left_img, prob), axis=2)

        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0
        # classification
        if self.positive>0:
            cls_img = (cls_img==self.positive) # pos: 1, neg: 0
        cls_img = torch.from_numpy(cls_img).long()

        if not self.returnpath:
            return left_img, cls_img
        else:
            return left_img, cls_img, left

    def __len__(self):
        return len(self.left)

# for change
class myImageFloder_8bit_t1t1(data.Dataset):
    def __init__(self, left, cls, imgsize = 256, channels=4, aug=False, returnpath=False,
                 numsample=0):
        self.left = left
        self.cls = cls
        if numsample>0:
            self.left = left[:numsample]
            self.cls = cls[:numsample]
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.returnpath = returnpath

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        left_img1 = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG

        left_img = np.concatenate((left_img1, left_img1), axis=2) # H W C
        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0
        # classification
        cls_img = torch.from_numpy(cls_img).long()

        if not self.returnpath:
            return left_img, cls_img
        else:
            return left_img, cls_img, left

    def __len__(self):
        return len(self.left)

# for change
class myImageFloder_8bit_t2t2(data.Dataset):
    def __init__(self, left, cls, imgsize = 256, channels=4, aug=False, returnpath=False,
                 numsample=0):
        self.left = left
        self.cls = cls
        if numsample>0:
            self.left = left[:numsample]
            self.cls = cls[:numsample]
        self.aug = aug # augmentation for images
        self.imgsize = imgsize # used for training and validation
        self.channels = channels
        self.returnpath = returnpath

    def __getitem__(self, index):
        left = self.left[index]
        cls = self.cls[index]
        # Read images
        # left_img1 = tif.imread(left)[:, :, :self.channels] # TIF
        cls_img = cv2.imread(cls, cv2.IMREAD_UNCHANGED) # PNG
        left_img2 = tif.imread(left.replace('img1', 'img2'))[:, :, :self.channels]  # TIF

        left_img = np.concatenate((left_img2, left_img2), axis=2) # H W C
        # Augmentation
        if self.aug:
            transformed = image_transformed_cls(image=left_img, mask=cls_img)
            left_img = transformed["image"]
            cls_img = transformed["mask"]

        # scale images
        left_img = torch.from_numpy(left_img).permute(2, 0, 1) # H W C ==> C H W
        left_img = left_img.float()/255.0
        # classification
        cls_img = torch.from_numpy(cls_img).long()

        if not self.returnpath:
            return left_img, cls_img
        else:
            return left_img, cls_img, left

    def __len__(self):
        return len(self.left)

if __name__=="__main__":
    #test
    pass
    # data = myImageFloder()