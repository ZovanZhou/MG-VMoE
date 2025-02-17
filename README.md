# MG-VMoE

[Title] Multimodal Graph-Based Variational Mixture of Experts Network for Zero-Shot Multimodal Information Extraction

## Preparation

1. Clone the repo to your local.
2. Download Python version: 3.6.13
3. Download the dataset from this [link](https://pan.baidu.com/s/1g4qda-y7SsPHZbxYElhyjQ ) and the extraction code is **1234**. Unzip the downloaded files into the ''**dataset**'' folder.
4. Download the [BERT](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip) pretrained models, and unzip into the ''**pretrain**'' folder.
5. Open the shell or cmd in this repo folder. Run this command to install necessary packages.

```cmd
pip install -r requirements.txt
```

## Experiments

1. In each folder "MET" or "MRE", we have shell scripts to run the training and evaluation procedures for the Linux systems. You can run the following command:

```cmd
cd ./MRE
./run.sh 0

cd ./MET
./run.sh 0 0
```

2. You can also input the following command to train the model. There are different choices for some hyper-parameters shown in square barckets. The meaning of these parameters are shown in the following tables.

|  Parameters | Value | Description|
|  ----  | ----  | ---- |
|seed|int|Random seed to initialize weights|
|batch_size|int|The number of a batch for training|
|lr|float|Learning rate|
|epoch|int|Maximum training iteration times|
|patience|int|Tolerance for early stopping|
|n_expert|int|The number of experts in VMoE|
|beta|float|The hyper-parameter for balancing loss|

```cmd
CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --seed 0 \
    --batch_size 16 \
    --lr 1e-5 \
    --epoch 20 \
    --patience 10 \
    --n_expert 8 \
    --beta 1.0 \
```
