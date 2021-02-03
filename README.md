# Spatio-Temporal Inception Graph Convolutional Networks for Skeleton-Based Action Recognition
This code provides a PyTorch implementation and pretrained models for **STIGCN**
as described in the paper
[Spatio-Temporal Inception Graph Convolutional Networks for Skeleton-Based
Action Recognition](https://arxiv.org/abs/2011.13322).

Block
[![yK0610.png](https://s3.ax1x.com/2021/02/03/yK0610.png)](https://imgchr.com/i/yK0610)

Framework
[![yK0IhR.png](https://s3.ax1x.com/2021/02/03/yK0IhR.png)](https://imgchr.com/i/yK0IhR)

STIGCN is an efficient and simple method for skeleton-based action recognition.
It overcomes the limitations of previous methods in extracting and synthesizing
information of different scales and transformations from different paths at
different levels (simiar to GoogLeNet). Extensive experiments demonstrate that
our network outperforms state-of-the-art methods by a significant margin with
only 1/5 of the parameters and 1/10 of the FLOPs.


## Data Preparation
For data preparation, please refer to
[2s-AGCN](https://github.com/lshiwjx/2s-AGCN) for more details.

## Training & Testing

Change the config file depending on what you want.

    `python main.py --config ./config/ntu-xsub.yaml`

    `python main.py --config ./config/test.yaml`


## Citation
Please cite the following paper if you use this repository in your reseach.

    @inproceedings{huang2020spatio,
        title={Spatio-Temporal Inception Graph Convolutional Networks for Skeleton-Based Action Recognition},
        author={Huang, Zhen and Shen, Xu and Tian, Xinmei and Li, Houqiang and Huang, Jianqiang and Hua, Xian-Sheng},
        booktitle={ACM MM},
        pages={2122--2130},
        year={2020}
    }

