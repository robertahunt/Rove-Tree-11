# Rove-Tree-11
Repository for Code for the Rove-Tree-11 paper/dataset

## Purpose
The Rove-Tree-11 dataset and code is provided to further research into automatic generation of phylogenetic trees (trees that show how closely related species are) from images using deep learning. To do this, segmented images of specimens of 13,887 beetles from 215 different species of Rove Beetles (Family Staphylinidae) are provided taken from the Natural History Museum of Denmark's collections. A current gold standard phylogeny is provided with the dataset.

## How to use - Running the deep metric learning models
The latent features are all generated using the forked submodule. If you are installing locally, there is a requirements.txt file you can use with pip. I used a docker container inside vscode. The relevant docker files are provided in the submodule. 


### Running the main code
Once you have a system with the python requirements installed, you can run the main code via the command line, an example is given below:
```
python main.py --dataset=rove --suffix=tripD0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=112 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split
```

If your gpu cannot run such a large batch size (I had issues with this), you can use gradient accumulation, with the gradient_accumulation_steps argument (be mindful it will slightly alter the results even if you are using the same seed):
```
python main.py --dataset=rove --suffix=tripD0 --source_path=/home/ngw861/01_abbey_rove/data --seed=0 --bs=8 --gradient_accumulation_steps=14 --samples_per_class=2  --loss=triplet --batch_mining=distance --arch=resnet50_frozen_normalize --augmentation=rove --use_tv_split
```
The runs for the main results provided in the paper are listed in sbatch.sh.


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


![alt text](https://github.com/robertahunt/Rove-Tree-11/blob/main/images/chart.png?raw=true)
