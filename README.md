# Solving the Catastrophic Forgetting Problem in Generalized Category Discovery
PyTorch implementation of “Solving the Catastrophic Forgetting Problem in Generalized Category Discovery”.

## 💡 Abstract
Generalized Category Discovery (GCD) aims to identify a mix of known and novel categories within unlabeled data sets, providing a more realistic setting for image recognition.
Essentially, GCD needs to **remember** existing patterns thoroughly to recognize novel categories.
Recent state-of-the-art method SimGCD transfers the knowledge from known-class data to the learning of novel classes through debiased learning. 
However, some patterns are catastrophically **forgot** during adaptation and thus lead to poor performance in novel categories classification.
To address this issue, we propose a novel learning approach, LegoGCD, which is seamlessly integrated into previous methods to enhance the discrimination of novel classes while maintaining performance on previously encountered known classes.
Specifically, we design two types of techniques termed as **L**ocal **E**ntropy Re**g**ularization (LER) and Dual-views Kullback–Leibler divergence c**o**nstraint (DKL).
The LER optimizes the distribution of potential known class samples in unlabeled data, thus ensuring the preservation of knowledge related to known categories while learning novel classes.
Meanwhile, DKL introduces Kullback–Leibler divergence to encourage the model to produce a similar prediction distribution of two view samples from the same image.
In this way, it successfully avoids mismatched prediction and generates more reliable potential known class samples simultaneously.
Extensive experiments validate that the proposed LegoGCD effectively addresses the known category forgetting issue across all datasets, \eg, delivering a $\textbf{7.74\%}$ and $\textbf{2.51\%}$ accuracy boost on known and novel classes in CUB, respectively. 


## Models
You can find the training logs and checkpoints in the directory: "dev_outputs/", with a structure example:
```
.
├── 1698108877.7779174
│   └── events.out.tfevents.1698108877.deeplearning-v191204-deeplearn.205227.1
├── checkpoints
│   └── model.pt
├── events.out.tfevents.1698108877.deeplearning-v191204-deeplearn.205227.0
├── log.txt
├── Train\ ACC\ Unlabelled_v2_All
│   └── events.out.tfevents.1698108959.deeplearning-v191204-deeplearn.205227.4
├── Train\ ACC\ Unlabelled_v2_New
│   └── events.out.tfevents.1698108959.deeplearning-v191204-deeplearn.205227.3
└── Train\ ACC\ Unlabelled_v2_Old
    └── events.out.tfevents.1698108959.deeplearning-v191204-deeplearn.205227.2
```

## Config
It is a config example. You can change data_path and output positions according to your directory.
```
# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = 'data/cifar10'
cifar_100_root = 'data/cifar100'
cub_root = 'data/cub'
aircraft_root = '/data/fgvc-aircraft-2013b'
car_root = '/data/stanford_cars'
herbarium_dataroot = '/data/herbarium_19'
imagenet_root = 'ImageNet/train/ILSVRC2012_img_train/data'
imagenet_1k = 'ImageNet/train/ILSVRC2012_img_train/data'

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# logs and checkpoints dir
exp_root = 'dev_outputs'
```

## Training
- To train the dataset xxx, you can run:
```
./scripts/run_xxx.sh
```

