# A full-level fused cross-task transfer learning method for building change detection using noise-robust pretrained networks on crowdsourced labels
The paper is accepted by Remote Sensing of Environment recently, and the code will be available recently.   
Author: Yinxia Cao and Xin Huang
### Abstract
Accurate building change detection is crucial for understanding urban development. Although fully supervised deep learning-based methods for building change detection have made progress, they tend to fuse temporal information only at a single level (e.g., input, feature, or decision levels) to mitigate the data distribution differences between time-series images, which is highly prone to introduce a large number of pseudo changes. Moreover, these methods rely on a large number of high-quality pixel-level change labels with high acquisition costs. In contrast, available crowdsourced building data are abundant but are less considered for change detection. For example, OpenStreetMap (OSM), Google Map, and Gaode Map provide lots of available building labels, yet they usually contain noise such as false alarms, omissions, and mismatches, limiting their wide application. In addition, when the building extraction task is transferred to the building change detection task, the temporal and regional differences between different images may lead to undesired pseudo changes. Given these issues, we propose a full-level fused cross-task transfer learning method for building change detection using only crowdsourced building labels and high-resolution satellite imagery. The method consists of three steps: 1) noise-robust building extraction network pretraining; 2) uncertainty-aware pseudo label generation; and 3) full-level fused building change detection. We created building extraction and building change detection datasets. The former (building extraction dataset) contains 30 scenes of ZY-3 images covering 27 major cities in China and crowdsourced building labels from Gaode Map for training, while the latter (building change dataset) contains bi-temporal ZY-3 images in Shanghai and Beijing for testing. The results show that the proposed method can identify changed buildings more effectively and better balance false alarms and omissions, compared to the existing state-of-the-art methods. Further analysis indicates that the inclusion of samples from multiple cities with various spatial heterogeneities is helpful to improve the performance. The experiments show that it is promising to apply the proposed method to situations where true labels are completely lacking or limited, thus alleviating the issue of high label acquisition cost. The source code will be available at https://github.com/lauraset/FFCTL. 

[paper](https://www.sciencedirect.com/science/article/pii/S0034425722004771)
![image](https://user-images.githubusercontent.com/39206462/205588692-dc37ecbc-d11e-4c77-a4ea-6ee8e80b7858.png)


## Getting Started
### Prerequisites
```
python >=3.6
pytorch >= 1.7.0 (lower version may be also work)
GPU: one NVIDIA GTX 1080 Ti GPU (11G memory)
```

## Prepare datasets
1. Datasets: building detection and change detection datasets, which were organized as follows:    
```
building detection dataset:
    data/img # for storing images, the name of image: img_*.tif
    data/lab # for storing lab, the name of lab: lab_*.png
change detection dataset for training:
    changedata/bj/img1: original images at time t1 (called t1 images)
    changedata/bj/img2: original images at time t2 (called t2 images)
    changedata/bj/lab: pseudo labels from the objec-to-pixel analysis (see Step 2)
    changedata/bj/cert: pseudo labels from the objec-to-pixel comparison and uncertainty-aware analysis (see Step 2)
    changedata/bj/img1t: t1 images were transferred to t2 on each sample (with size of 256 x 256 pixels)
    changedata/bj/img1t_wallis: t1 images were transferred to t2 on each sample by the wallis method (see Section 5.3)
    changedata/bj/img1ta: t1 images were transferred to t2 on the whole images (i.e., the original large images)
    changedata/bj/img2ta: t2 images were transferred to t1 on the whole images (i.e., the original large images)
change detection dataset for testing:
    changedata/bj/testdata/img1: original images at time t1 (called t1 images)
    changedata/bj/testdata/img2: original images at time t2 (called t2 images)
    changedata/bj/testdata/lab: reference labels
    changedata/bj/testdata/img1ta: t1 images were transferred to t2 on the whole images (i.e., the original large images)
```
Note that `certc` or `labc` were only used for visualization and not training.   
In experiments, we found that `img1ta` performs slightly better than `img1t`, and therefore, we used the `img1ta`.   

2. Large area test data: see the path  `zhengzhou`
Due to the data privacy of shanghai and beijing, I presented a public region in zhengzheng of China.
The original data and preprocessed data has been provided, see the following baidu cloud pan (link: https://pan.baidu.com/s/1CMkKZHv__mARhHJ-JHrkvw , code:04qp)

## One-by-one step
### Step 1: noise-robust building extraction network pretraining   
The method includes three steps: 1) network initialization; 2) noisy label correction; and 3) network retraining.   
```
python train_res50unet_warm.py
python train_res50unet_update_step2.py
python train_res50unet_update_step3.py
```
The data and weight path for the above three files is the same:
```python
# Setup Dataloader
filepath = 'data' # the default data path, which depends on personal preferences
train_img, train_lab, val_img, val_lab,_,_ = dataloader(filepath, split=(0.9, 0.1,0), issave=True) # 90% for training
iroot = 'runs' # the default weight path, which depends on personal preferences
```
Note that the training epochs and learning rates should be set according to your own datasets.

### Step 2: uncertainty-aware pseudo label generation
The algorithm includes three steps:    
1) single-temporal building prediction;    
2) object-to-pixel multi-temporal comparison;   
3) uncertainty-aware analysis for reliable pseudo label generation.

step 1:   
```
cd pseudo_label_generation
python step1_single_temporal_building_prediction.py
```
step 2-3:   
```
demo_step2_change_region.m
demo_step3_0_generate_changeprob.m
demo_step3_1_obtain_reliable_label.m
```

### Step 3: full-level fused building change detection   
The network consists of three parts: 1) color transfer at the input level;    
2) layer-wise temporal difference at the feature level;   
3) simultaneous extraction of changed and unchanged buildings at the decision level. 

step 1:     
```
cd color_transfer
see the function Color_Transfer_CCS_multi.m for color transfer
```
step 2-3:   
for each training file below, the key data path is set as follows:
```
# Setup Dataloader
fcode = 'bj' # city name, bj denotes beijing, sh denotes shanghai
datapath = os.path.join('changedata', fcode)
# use pseudo labels from the object-to-pixel comparison and uncertainty-aware analysis
train_lab = [i.replace('lab', 'cert') for i in train_lab] 
val_lab = [i.replace('lab', 'cert') for i in val_lab]
# test data 
datapath = os.path.join(datapath, 'testdata')
# path for saving weights 
iroot = r'runs_change\res50cdo_fuse' # the default setting, depends on personal preferences
# the path of images with color transfer
idir1 = 'img1ta'
# load pretrained models
iroot2 = 'runs' # the path of storing building detection models, see Step 1.
```
After setting the correct path, run the model training:   
```
python train_res50unet_changeonlyfuse_bj_trans_cert.py # for beijing
python train_res50unet_changeonlyfuse_sh_trans_cert.py # for shanghai
```
Note that the training epochs and learning rates for the two cities are different, and they should be set according to your own datasets.

## Trained weights
the well-trained weights can be obtained through baidu cloud pan (link: https://pan.baidu.com/s/1CMkKZHv__mARhHJ-JHrkvw , code:04qp)

## Accuracy assessment
the testdata is put in `data/bj/testdata`, and run the following code:
```
python ttest_res50unet_changeo_fuse_bj_trans_cert_tst.py # for beijing
# python ttest_res50unet_changeo_fuse_sh_trans_cert_tst.py # for shanghai, the code is similar to that in beijing
```

## Large area samples
the original data and preprocessed one have been put in the baidu cloud pan, and then run the following code to generate predictions.
```
cd pseudo_label_generation
python step1_single_temporal_building_prediction.py
```
![image](https://user-images.githubusercontent.com/39206462/232196436-4e82c405-345d-4c45-abf1-f9df0c0acbf1.png)

## Acknowledgement
We used the package "segmentation_models_pytorch" and "pytorch-grad-cam".   
Thanks for their contributions.  
```
@misc{Yakubovskiy:2019,
  Author = {Pavel Yakubovskiy},
  Title = {Segmentation Models Pytorch},
  Year = {2020},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
```
## Citation
If you use our codes in your research, please cite our RSE 2023 paper.  
```
@article{WANG2022113058,
    title = {A full-level fused cross-task transfer learning method for building change detection using noise-robust pretrained networks on crowdsourced labels},
    journal = {Remote Sensing of Environment},
    volume = {284},
    pages = {113371},
    year = {2023},
    issn = {0034-4257},
    doi = {https://doi.org/10.1016/j.rse.2022.113371},
    url = {https://www.sciencedirect.com/science/article/pii/S0034425722004771},
    author = {Yinxia Cao and Xin Huang},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
