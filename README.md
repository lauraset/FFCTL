# A full-level fused cross-task transfer learning method for building change detection using noise-robust pretrained networks on crowdsourced labels
The paper is accepted by Remote Sensing of Environment recently, and the code will be available recently.   
Author: Yinxia Cao and Xin Huang
### Abstract
Accurate building change detection is crucial for understanding urban development. Although fully supervised deep learning-based methods for building change detection have made progress, they tend to fuse temporal information only at a single level (e.g., input, feature, or decision levels) to mitigate the data distribution differences between time-series images, which is highly prone to introduce a large number of pseudo changes. Moreover, these methods rely on a large number of high-quality pixel-level change labels with high acquisition costs. In contrast, available crowdsourced building data are abundant but are less considered for change detection. For example, OpenStreetMap (OSM), Google Map, and Gaode Map provide lots of available building labels, yet they usually contain noise such as false alarms, omissions, and mismatches, limiting their wide application. In addition, when the building extraction task is transferred to the building change detection task, the temporal and regional differences between different images may lead to undesired pseudo changes. Given these issues, we propose a full-level fused cross-task transfer learning method for building change detection using only crowdsourced building labels and high-resolution satellite imagery. The method consists of three steps: 1) noise-robust building extraction network pretraining; 2) uncertainty-aware pseudo label generation; and 3) full-level fused building change detection. We created building extraction and building change detection datasets. The former (building extraction dataset) contains 30 scenes of ZY-3 images covering 27 major cities in China and crowdsourced building labels from Gaode Map for training, while the latter (building change dataset) contains bi-temporal ZY-3 images in Shanghai and Beijing for testing. The results show that the proposed method can identify changed buildings more effectively and better balance false alarms and omissions, compared to the existing state-of-the-art methods. Further analysis indicates that the inclusion of samples from multiple cities with various spatial heterogeneities is helpful to improve the performance. The experiments show that it is promising to apply the proposed method to situations where true labels are completely lacking or limited, thus alleviating the issue of high label acquisition cost. The source code will be available at https://github.com/lauraset/FFCTL. 

![image](https://user-images.githubusercontent.com/39206462/205587423-f5e4fa8c-ad22-4c2a-aca0-2f9e9f585a0a.png)

## Getting Started
### Prerequisites

```
python >=3.6
pytorch >= 1.7.0 (lower version may be also work)
GPU: NVIDIA GTX 1080 Ti GPU (11G memory)
```

## Prepare datasets

## One-by-one step
### Step 1: noise-robust building extraction network pretraining   
The method includes three steps: 1) network initialization; 2) noisy label correction; and 3) network retraining.   
```
python train_res50unet_warm.py
python train_res50unet_update_step2.py
python train_res50unet_update_step3.py
```
### Step 2: uncertainty-aware pseudo label generation
The algorithm includes three steps:    
1) single-temporal building prediction;    
2) object-to-pixel multi-temporal comparison;   
3) uncertainty-aware analysis for reliable pseudo label generation.
```
TO DO: some codes still needs reorganization, and will be available recently.
```


### Step 3: full-level fused building change detection   
The network consists of three parts: 1) color transfer at the input level;    
2) layer-wise temporal difference at the feature level;   
3) simultaneous extraction of changed and unchanged buildings at the decision level.   
```
python train_res50unet_changeonlyfuse_bj_trans_cert.py
python train_res50unet_changeonlyfuse_sh_trans_cert.py
```

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


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
