# ImageNet training in PyTorch

This code is slightly modified from the official pytorch examples from imagenet for use in classification on the rove-tree-11 dataset (original code: https://github.com/pytorch/examples/tree/main/imagenet ) Much of this readme is directly copied from there for reference.



## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, move and extract the training and validation images to labeled subfolders, using [the following shell script](extract_ILSVRC.sh)

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python main.py -a resnet50 [imagenet-folder with train and val folders]
```


The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
python main.py -a alexnet --lr 0.01 [imagenet-folder with train and val folders]
```

## Training of Rove-Tree-11
The following was run to complete the training on Rove-Tree-11:


```bash

# Species-level classifications, resnet18
python main.py data -a resnet18 -b 32 -s 0
python main.py data -a resnet18 -b 32 -s 1
python main.py data -a resnet18 -b 32 -s 2
# Species-level classifications, resnet50
python main.py data -a resnet50 -b 32 -s 0
python main.py data -a resnet50 -b 32 -s 1
python main.py data -a resnet50 -b 32 -s 2
# Species-level classifications, efficientnet_b0
python main.py data -a efficientnet_b0 -b 32 -s 0
python main.py data -a efficientnet_b0 -b 32 -s 1
python main.py data -a efficientnet_b0 -b 32 -s 2


# Genus-level classifications, resnet18
#    for this, data_genus is a folder with each genus as a subfolder containing all images
#    from that genus
python main.py data_genus -a resnet18 -b 32 -s genus_0
python main.py data_genus -a resnet18 -b 32 -s genus_1
python main.py data_genus -a resnet18 -b 32 -s genus_2
# Genus-level classifications, resnet50
python main.py data_genus -a resnet50 -b 32 -s genus_0
python main.py data_genus -a resnet50 -b 32 -s genus_1
python main.py data_genus -a resnet50 -b 32 -s genus_2
# Genus-level classifications, efficientnet_b0
python main.py data_genus -a efficientnet_b0 -b 32 -s genus_0
python main.py data_genus -a efficientnet_b0 -b 32 -s genus_1
python main.py data_genus -a efficientnet_b0 -b 32 -s genus_2


# Species-level classifications, resnet18, with background
#    for this, data_background is a folder with each unsegmented images used instead of segmented
python main.py data_background -a resnet18 -b 32 -s background_0
python main.py data_background -a resnet18 -b 32 -s background_1
python main.py data_background -a resnet18 -b 32 -s background_2
```

## Multi-processing Distributed Data Parallel Training

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

### Single node, multiple GPUs:

```bash
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```

### Multiple nodes:

Node 0:
```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
```

Node 1:
```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
```

## Usage

```
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE]
               [--rank RANK] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
               [--multiprocessing-distributed]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
```
