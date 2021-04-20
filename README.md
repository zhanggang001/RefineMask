## RefineMask: Towards High-Quality Instance Segmentation <br>with Fine-Grained Features (CVPR 2021)

This repo is the official implementation of [RefineMask: Towards High-Quality Instance Segmentation with Fine-Grained Features](https://arxiv.org/abs/2104.08569).

## Framework
![image](https://user-images.githubusercontent.com/79979076/112595320-394a7280-8e45-11eb-90b1-3164accd0518.png)

## Main Results

### Results on COCO
Method | Backbone | Schedule | AP | AP<sup>*</sup> | Checkpoint
------ | -------- | -------- | -- | -------------- | ----------
Mask R-CNN | R50-FPN | 1x | 34.7 | 36.8 |
RefineMask | R50-FPN | 1x | 37.3 | 40.6 | [download](https://drive.google.com/file/d/1ad7YewfVLJIZa_xErCW4qJBzwS4yKCnS/view?usp=sharing)
Mask R-CNN | R50-FPN | 2x | 35.4 | 37.7
RefineMask | R50-FPN | 2x | 37.8 | 41.2 | [download](https://drive.google.com/file/d/1-UuTjM9b3EfINqgGH0jyJ9uWJ3P2_wgy/view?usp=sharing)
Mask R-CNN | R101-FPN | 1x | 36.1 | 38.4 |
RefineMask | R101-FPN | 1x | 38.6 | 41.8 | [download](https://drive.google.com/file/d/1JcpfBzXhrSWa4MwH0LFQ575WIYxCdUyz/view?usp=sharing)
Mask R-CNN | R101-FPN | 2x | 36.6 | 39.3
RefineMask | R101-FPN | 2x | 39.0 | 42.4 | [download](https://drive.google.com/file/d/1W6jdqziYqAqiyYide9SxvHE2y79A0KJC/view?usp=sharing)

Note: No data augmentations except standard horizontal flipping were used.

### Results on LVIS
Method | Backbone | Schedule | AP | AP<sub>r</sub> | AP<sub>c</sub> | AP<sub>f</sub> | Checkpoint
------ | -------- | -------- | -- | -------------- | -------------- | -------------- | ----------
Mask R-CNN | R50-FPN | 1x | 22.1 | 10.1 | 21.7 | 30.0
RefineMask | R50-FPN | 1x | 25.7 | 13.8 | 24.9 | 31.8 | [download](https://drive.google.com/file/d/1t10bX0S6II-PNdOP1z_hmXHs7K3fhJdv/view?usp=sharing)
Mask R-CNN | R101-FPN | 1x | 23.7 | 12.3 | 23.2 | 29.1
RefineMask | R101-FPN | 1x | 27.1| 15.6 | 26.2 | 33.1 | [download](https://drive.google.com/file/d/13cLKIFwlMg_QSAuXHEwISehm4PlB95eC/view?usp=sharing)

### Results on Cityscapes
Method | Backbone | Schedule | AP | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | Checkpoint
------ | -------- | -------- | -- | -------------- | -------------- | -------------- | ----------
Mask R-CNN | R50-FPN | 1x | 33.8 | 12.0 | 31.5 | 51.8
RefineMask | R50-FPN | 1x | 37.6 | 14.0 | 35.4 | 57.9 | [download](https://drive.google.com/file/d/1hEXRl2zqC0rUyKixU1FLoFTlNLqWvUCE/view?usp=sharing)

### Efficiency of RefineMask
Method | AP | AP<sup>*</sup> | FPS
------ | -- | -------------- | ---
Mask R-CNN | 34.7 | 36.8 | 15.7
PointRend | 35.6 | 38.7 | 11.4
HTC | 37.4 | 40.7 | 4.4
RefineMask | 37.3 | 40.9 | 11.4


## Usage

### Requirements
* Python 3.6+
* Pytorch 1.5.0
* mmcv-full 1.0.5

### Datasets
    data
      ├── coco
      |   ├── annotations
      │   │   │   ├── instances_train2017.json
      │   │   │   ├── instances_val2017.json
      │   │   │   ├── lvis_v0.5_val_cocofied.json
      │   ├── train2017
      │   │   ├── 000000004134.png
      │   │   ├── 000000031817.png
      │   │   ├── ......
      │   ├── val2017
      │   ├── test2017
      ├── lvis
      |   ├── annotations
      │   │   │   ├── lvis_v1_train.json
      │   │   │   ├── lvis_v1_val.json
      │   ├── train2017
      │   │   ├── 000000004134.png
      │   │   ├── 000000031817.png
      │   │   ├── ......
      │   ├── val2017
      │   ├── test2017
      ├── cityscapes
      |   ├── annotations
      │   │   │   ├── instancesonly_filtered_gtFine_train.json
      │   │   │   ├── instancesonly_filtered_gtFine_val.json
      │   ├── leftImg8bit
      │   |   ├── train
      │   │   ├── val
      │   │   ├── test

Note: We used the lvis-v1.0 dataset which consists of 1203 categories.

### Training
```
./scripts/dist_train.sh ./configs/refinemask/coco/r50-refinemask-1x.py 8
```
Note: The codes only support batch size 1 per GPU, and we trained all models with a total batch size 16x1. If you train models with a total batch size 8x1, the performance may drop. We will support batch size 2 or more per GPU later. You can use ./scripts/slurm_train.sh for training with multi-nodes.

### Inference
```
./scripts/dist_test.sh ./configs/refinemask/coco/r50-refinemask-1x.py xxxx.pth 8
```

## Citation
```
@article{zhang2021refinemask,
  title={RefineMask: Towards High-Quality Instance Segmentation with Fine-Grained Features},
  author={Gang, Zhang and Xin, Lu and Jingru, Tan and Jianmin, Li and Zhaoxiang, Zhang and Quanquan, Li and Xiaolin, Hu},
  journal={arXiv preprint arXiv:2104.08569},
  year={2021}
}
```
