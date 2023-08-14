<div align="center">


#  <img src="./assets/mpot_logo.ico" style="vertical-align: sub;" width="40"> **Multiple Planar Object Tracking**


<i>Zhicheng Zhang, Shengzhe Liu, Jufeng Yang</i>

<a href=" "><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Conference](https://img.shields.io/badge/ICCV-2023-orange)](https://iccv2023.thecvf.com/)
[![License](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)
</div>



**Key motivation:** *Tracking both location and pose of multiple planar objects (MPOT) is of great significance to numerous real-world applications, including industrial, education, geometric, art, and our daily life.*


This repository contains the official implementation of our work in ICCV 2023. **MPOT-3K** dataset and the pytorch training/validation code for **tracking framework PRTrack** are released. More details can be viewed in our paper.
[<a href="../assets/2023_ICCV_MPOT.pdf" target="_blank">PDF</a>]
[<a href="https://zzcheng.top/MPOT#Video" target="_blank">Video</a>]
[<a href="https://zzcheng.top/MPOT" target="_blank">Project Page</a>]
[<a href="https://github.com/nku-zhichengzhang/MPOT" target="_blank">Github</a>]
[<a href="https://github.com/nku-zhichengzhang/MPOT#mpot-3k-dataset" target="_blank">MPOT-3K Dataset</a>]
[<a href="https://zzcheng.top/MPOT#Demo" target="_blank">Demo</a>]

## Publication

>**Multiple Planar Object Tracking**<br>
Zhicheng Zhang, Shengzhe Liu, Jufeng Yang<br>
<i>Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023</i>.
</br>


- [ **Multiple Planar Object Tracking**](#-multiple-planar-object-tracking)
  - [Publication](#publication)
- [Abstract](#abstract)
- [DEPENDENCY](#dependency)
    - [Recommended Environment](#recommended-environment)
- [MPOT-3K DATASET](#mpot-3k-dataset)
  - [Data Source](#data-source)
  - [Data Format](#data-format)
  - [Annotations](#annotations)
- [SCRIPTS of PRTrack](#scripts-of-prtrack)
  - [Running](#running)
  - [Evaluation](#evaluation)
  - [Demo](#demo)
- [REFERENCE](#reference)
- [CITATION](#citation)


# Abstract

<img src="./assets/fig1o11.png" width="50%" align="right">Tracking both location and pose of multiple planar objects (MPOT) is of great significance to numerous real-world applications. The greater degree-of-freedom of planar objects compared with common objects makes MPOT far more challenging than well-studied object tracking, especially when occlusion occurs. To address this challenging task, we are inspired by amodal perception that humans jointly track visible and invisible parts of the target, and propose a tracking framework that unifies appearance perception and occlusion reasoning. Specifically, we present a dual branch network to track the visible part of planar objects, including vertexes and mask. Then, we develop an occlusion area localization strategy to infer the invisible part, i.e., the occluded region, followed by a two-stream attention network finally refining the prediction. To alleviate the lack of data in this field, we build the first large-scale benchmark dataset, namely MPOT-3K. It consists of 3,717 planar objects from 356 videos, and contains 148,896 frames together with 687,417 annotations. The collected planar objects have 9 motion patterns and the videos are shot in 6 types of indoor and outdoor scenes. Extensive experiments demonstrate the superiority of our proposed method on the newly developed MPOT-3K as well as other two popular single planar object tracking datasets.


# DEPENDENCY
You can set up the environments by running the following lines:

    conda env create -f environment.yml
    pip install -r requirements.txt

### Recommended Environment
* CUDA 11.3
* Python 3.6.13
* Pytorch 1.8.1 
* apex:0.1 (only for training)


# MPOT-3K DATASET
If you need the MPOT-3K dataset for academic purposes, please download the [application form](./assets/MPOT-3K_Data_Access_Form.docx) and fill out the request information, then send it to ***gloryzzc6@sina.com***.
We will process your application as soon as possible.
Please make sure that the email used comes from your educational institution.


## Data Source
The collected scenes include lib, gallery, house, streetview, buildings, village. Besides, the videos are shoot under nine motion patterns that involves camera motion and target movement as follow:

|Id|Motion Pattern|
|  ----  | ----  |
1|Far-near Movement
2|In-plane Rotation
3|Out-plane Rotation
4|In-plane Movement
5|Motion Blur 
6|Camera Occlusion 
7|Unconstrained 
8|Moving Objects 
9|Moving Occlusions


## Data Format
```
MPOT-3k
├── list #splits
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
├── train 
│   ├── buildings1-1
│   ├── ...
│   └── village7-9
├── test
│   ├── buildings5-1
│   ├── ...
│   └── village2-9
└── val 
    ├── buildings4-1
    ├── ...
    └── gallery2-9
        ├── gobjs #planar objects
        ├── gt #annotations
        │   ├── gt_init.txt #instances (initial frame)
        │   ├── gt_obj_init.txt #objects (initial frame)
        │   ├── gt_obj.txt #ground truth
        │   └── objects.txt
        ├── seq1 #images
        └── seqinfo.ini #video information
```

## Annotations
In gt_obj.txt, the annotation in each line include:

&emsp;1: frame id  
&emsp;2: instance id 
&emsp;3-10: 4 points  
&emsp;11: class id
&emsp;12: object id

A simple example is listed below.

|Frame|instance|point1_x|point1_y|point2_x|point2_y|point3_x|point3_y|point4_x|point4_y|class|object|
|  ----  | ----  |  ----  | ----  |  ----  | ----  |  ----  | ----  |  ----  | ----  |  ----  | ----  |
1|1|90.0|197.3|196.0|191.3|217.3|426.0|112.0|441.3|6|1
2|1|87.6|197.4|193.6|191.4|214.9|426.1|109.6|441.4|6|1




# SCRIPTS of PRTrack
## Running
You can easily train and evaluate the model by running the script below.


You can include more details such as epoch, , etc. Please refer to `config.yaml`.

~~~~
python train.py --cfg ./configs/config_train.yaml
~~~~


For testing

~~~~
python test.py --cfg ./configs/config_test.yaml
~~~~
    
## Evaluation
You can evaluate the model by running the command below. The pre-trained model can be found in [pretrained model](https://drive.google.com/file/d/1IgtUszVJjoa-VJ_UZVryFFcMouVJ__7d/view?usp=sharing). More details can be viewed in [eval](./evaluation).

~~~~
cd evaluation/MPOT
python evalMPOT.py
~~~~



## Demo
We built online demo on gradio at [here]().


# REFERENCE
We referenced the repos below for the code.

* [Cycle-STM](https://github.com/lyxok1/STM-Training)
* [SAMTrack](https://github.com/z-x-yang/Segment-and-Track-Anything)
* [MOT evaluation](https://github.com/shenh10/mot_evaluation)
* [POT](https://liangp.github.io/data/pot280/)


# CITATION

If you find this repo useful in your project or research, please consider citing the relevant publication.

````
@inproceedings{zhang2023multiple,
  title={Multiple Planar Object Tracking},
  author={Zhang, Zhichang and Liu, Shengzhe and Yang, Jufeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
````
