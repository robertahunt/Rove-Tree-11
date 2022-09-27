# Rove-Tree-11
Repository for Code for the Rove-Tree-11 paper/dataset

## Purpose
The Rove-Tree-11 dataset and code is provided to further research into automatic generation of phylogenetic trees (trees that show how closely related species are) from images using deep learning. To do this, segmented images of specimens of 13,887 beetles from 215 different species of Rove Beetles (Family Staphylinidae) are provided taken from the Natural History Museum of Denmark's collections. A current gold standard phylogeny is provided with the dataset.

## How to use - Generating the latent features

### Installation requirements


### Running the main code
python main.py --dataset=rove --suffix=tripD0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split

## How to use - Generating the Align Scores


The align scores are generated in two steps:
1. Running the latent features through RevBayes, which is a program to use bayesian inference to generate phylogenetic trees from continuous features.
2. Checking the resulting tree against the ground truth phylogeny using the align score


### Installation requirements
This requires revbayes to be installed. I did this on linux using the similarity image provided by revbayes. This requires Singularity to be installed. This can be done on ubuntu with:
```
sudo apt-get install -y singularity-container
```

The revbayes singularity image can then be downloaded from: https://revbayes.github.io/download

### Running the code
The alignment score is calculated in the jupyter notebook '00. Run RevBayes for Single Result'. You will need to change the paths to the output folders of your training runs

## Citations
If you use this code/dataset, please cite:
TBD

## License
This code is provided with an MIT license (see license file). The original code in the forked submodule (Revisiting_Deep_Metric_Learning_PyTorch) has the same MIT license. 
