'''
2021.4.14 load ZY3LC
'''
import torch.utils.data as data
import tifffile as tif
import albumentations as A
import torch
import numpy as np
import cv2
import os


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

# img_mean and std
# 2021.7.1: update mux images and add tlc
IMG_MEAN_ALL = [1142.05719085069, 1183.92746808790, 1324.37698042479, 2360.08189621090]
IMG_STD_ALL = [352.892230743533, 402.069966221899, 554.259982955950, 1096.14879868840]
TLC_MEAN_ALL = [440.312064755891, 387.339043098102, 444.801891941169]
TLC_STD_ALL = [270.286351619591, 202.061888100090, 216.688621196791]


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
