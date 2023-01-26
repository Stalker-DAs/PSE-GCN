# Progressive Structure Enhancement Graph Convolutional Network for Face clustering
## Introduction

This model presents a novel GCN-based face clustering framework, called as PSE-GCN, that optimizes graph structure and clustering features jointly for fully exploiting their complementarity. This framework includes multiple dynamic graph construction (DGC) blocks, which progressively denoise a graph while providing highly discriminative features for various clustering tasks. The proposed method outperforms the state-of-the-art methods on the large-scale face dataset MS-Celeb-1M and achieves competitive results on the DeepFashion and MSMT17 datasets.

The main framework of PSE-GCN is shown in the following:

<img src=image/fig.png width=1000 height=245 />

## Main Results
<img src=image/results.png width=900 height=355 />

## Requirements
* Python=3.6.8
* Pytorch=1.7.1
* Cuda=11.0
* faiss=1.5.3

## Hardware
The hardware we used in this work is as follows:
* NVIDIA GeForce RTX 3090
* Intel Xeon Gold 6226R CPU@2.90GHz

## Datasets
Create a new folder for dataset:
```
mkdir data
```
After that, follow the link below to download the dataset and construct the data directory as follows:
```
|——data
   |——features
      |——part0_train.bin
      |——part1_test.bin
      |——...
      |——part9_test.bin
   |——labels
      |——part0_train.meta
      |——part1_test.meta
      |——...
      |——part9_test.meta
   |——knns
      |——part0_train/faiss_k_80.npz
      |——part1_test/faiss_k_80.npz
      |——...
      |——part9_test/faiss_k_80.npz
```
The MS1M and DeepFashion dataset at https://github.com/yl-1993/learn-to-cluster/blob/master/DATASET.md#supported-datasets.
The MSMT17 dataset at https://github.com/damo-cv/Ada-NETS.

## Training
### Calculate re-ranking knn
This step will calculate the re-ranking knn based on the graph constructed by the original knn. The calculated knn will be saved in `./data`
```
cd PSE-GCN
python re-ranking.py 80, part0_train
```
### Training model
You can use the following command to train the dataset directly. Alternatively, you can find the model configuration file in `./config/cfg_train.py`.
```
cd PSE-GCN
python main.py
```

## Testing
### Calculate re-ranking knn
```
cd PSE-GCN
python re-ranking.py 80, part1_test
```
### Testing model
First, you need to create a folder to save the calculated features.
```
cd PSE-GCN/data
mkdir Out_feature
```
If you want to test model, please set the config file to `cfg_test.py` in `main.py` and make sure the weight file is in the `./saves` folder. The pre-trained weight is in the path `./saves/checkpoint.pth`. After that, you can get the test results with the following command:
```
cd PSE-GCN
python main.py
```
### Clustering
```
cd PSE-GCN
python clustering.py part1_test, ./data/Out_feature/cluster_feature.npy
```




