import os
import re
import sys
import copy
import ete3
import json
import shutil
import munkres
import subprocess

import numpy as np
import pandas as pd
import scipy.stats as st

from tqdm import tqdm
from glob import glob
from copy import deepcopy
from munkres import Munkres
from ete3 import Tree, TreeNode
from scipy.cluster import hierarchy
from scipy.optimize import linear_sum_assignment
from subprocess import Popen, PIPE, CalledProcessError


# Code modified from: Mary K. Kuhner, Jon Yamato, Practical Performance of Tree Comparison Metrics, Systematic Biology, Volume 64, Issue 2, March 2015, Pages 205–214, https://doi.org/10.1093/sysbio/syu085 
# which was provided under public domain license
def ars(set1, set2):
# intersection over union
    numer = len(set1.intersection(set2))
    denom = len(set1.union(set2))
    assert denom > 0   # don't divide by zero!
    return float(numer)/float(denom)

def treealign(intree1, intree2):    
    tree1 = copy.deepcopy(intree1)
    tree2 = copy.deepcopy(intree2)
    
    i = 0
    for node1 in tree1.iter_descendants("postorder"):
        if not node1.is_leaf():
            node1.name = i
            i += 1
    
    j = 0
    for node2 in tree2.iter_descendants("postorder"):
        if not node2.is_leaf():
            node2.name = j
            j += 1
    
    n = max(i,j)
    vals = np.zeros((i,j))
    
    allnodes = set(tree1.get_leaf_names())
    for node1 in tree1.iter_descendants("postorder"):
        if node1.is_leaf():
            continue
        i0 = set(node1.get_leaf_names())
        i1 = allnodes.difference(i0)
        for node2 in tree2.iter_descendants("postorder"):
            if node2.is_leaf():
                continue
            j0 = set(node2.get_leaf_names())
            j1 = allnodes.difference(j0)
            a00 = ars(i0,j0)
            a11 = ars(i1,j1)
            a01 = ars(i0,j1)
            a10 = ars(i1,j0)
            s = max(min(a00,a11),min(a01,a10))
            vals[node1.name][node2.name] = 1.0 - s
        m = Munkres()

    total = 0
    for row,column in zip(*linear_sum_assignment(vals)):
        value = vals[row][column]
        total += value


    return total

# from https://github.com/scipy/scipy/issues/8274
def get_newick(node, newick, parentdist, leaf_names):
    """
    Converts scipy tree to newick format
    """
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = get_newick(node.get_left(), newick, node.dist, leaf_names)
        newick = get_newick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick


def generate_random_tree(node_names, SEED=42):
    """
    Given a list of node names, generate a random binary tree with those nodes
    Does this by first generating a matrix of random node similarities 
    Then using scipy's implementation of single linkage to generate a tree
    """
    df = pd.DataFrame()
    df['class'] = node_names
    
    # Generate random matrix of size length of the node names
    np.random.seed(SEED)
    z = np.random.random([len(node_names),2])
    z_columns= list(range(len(z[0])))
    df[z_columns] = z
    df = df.set_index('class')
    
    # Generate Tree
    X = df[z_columns]
    Z = hierarchy.linkage(X, 'single')
    tree = hierarchy.to_tree(Z, False)

    # Convert tree to newick format so ete3 can read it
    newick = get_newick(tree, "", tree.dist, node_names)
    tree_to_check = Tree(newick)
    return tree_to_check


def read_nexus_sum_into_ete3(fp):
    """
    Reads RevBayes Output Summary into ete3 tree format

    fp should point to a OUTSUMFILE produced by RevBayes
    """
    assert os.path.exists(fp), f"Could not find filepath: {fp}"
    
    with open(fp,'r') as f:
        file_str = ''.join(f.readlines())

    match  = re.search("Begin trees;.+End;", file_str, re.MULTILINE | re.DOTALL)
    if match:
        tree_block =  match.group()
        tree_block =  re.sub("\[&[^\]]+\]", "", tree_block)

        for match in re.finditer("\s*tree\s+([^=]+)=([^;]+;)", tree_block, re.MULTILINE | re.DOTALL):
            tree_name = match.groups()[0].strip()
            nw = match.groups()[1]
            t = Tree(nw)
        
    assert t is not None
    
    for l in t.get_leaves():
        l.name = l.name.replace('_',' ').lower()    
    return t


def make_tree_from_results_using_linkage(folder):
    """
    Use single linkage of cluster center distances to generate tree
    """
    z = np.load(os.path.join(folder, 'features_discriminative_Test.npy'))
    y = np.load(os.path.join(folder, 'image_paths_discriminative_Test.npy'))
    # get only the species name
    y = [a[0].split('/')[-2].replace('_',' ').lower() for a in y]
    
    df = pd.DataFrame()
    df['class'] = y
    z_columns= list(range(len(z[0])))
    df[z_columns] = z
    
    species_means = df.groupby('class')[z_columns].mean()

    #species_means.index = species_means.index + '-'  + species_means['count'].map(str)
    
    names = species_means.index
    X = species_means[z_columns]
    Z = hierarchy.linkage(X, 'single')
    
    tree = hierarchy.to_tree(Z, False)
    newick = get_newick(tree, "", tree.dist, names)
    tree_to_check = Tree(newick)
    return tree_to_check


# This is a template used to generate a RevBayes file programmatically
#   which can then be run by python
# Modified from: Caroline Parins-Fukuchi, Use of Continuous Traits Can Improve Morphological Phylogenetics, Systematic Biology, Volume 67, Issue 2, March 2018, Pages 328–339, https://doi.org/10.1093/sysbio/syx072
# which was provided under public domain license

# _dir is the directory the summary files should be saved in,
# _set is the set the traits are from (ie 'Test', or 'Val')
revbayes_template = """##This is intended to be run on RevBayes v1.0.0
#v1.0.1 has changed several function names. see RevBayes documentation for more details. 
#This procedure was developed with the gratuitous aid of RevBayes example documents
#authored by Nicolas Lartillot, Michael Landis, and April Wright.
 
fl <- "{_dir}/traits_discriminative_{_set}.nex"  #continuous character set used for analysis in nexus format
outtr <- "{_dir}/OUTTREEFILE.trees" #file to write sampled trees
outlf <- "{_dir}/OUTLOGFILE" #MCMC log
outsm <- "{_dir}/OUTSUMFILE" #MAP summary tree file
contData<- readContinuousCharacterData(fl)

numTips = contData.ntaxa()
numNodes = numTips * 2 - 1
names = contData.names()
diversification ~ dnLognormal(0,1)
turnover = 0 #we are going to parameterise the BD prior
speciation := diversification + turnover
extinction := turnover 
sampling_fraction <- 1

psi ~ dnBirthDeath(lambda=speciation, mu=extinction, rho=sampling_fraction, rootAge=1, taxa=names) #instantiate a BD tree with the parameters set above
mvi = 0 #we are going to set tree rearrangement and node height scaling moves

#create our node height and topological rearrangement MCMC moves
moves[++mvi] = mvSubtreeScale(psi, weight=5.0)
moves[++mvi] = mvNodeTimeSlideUniform(psi, weight=5.0) # changed from 10 to 5
moves[++mvi] = mvNNI(psi, weight=5.0)
moves[++mvi] = mvFNPR(psi, weight=5.0)

monitors[3] = mnFile(filename=outtr, printgen=100,separator = TAB, psi)

logSigma ~ dnNormal(0,1) #place a prior on BM sigma parameter.
sigma := 10^logSigma
moves[++mvi] = mvSlide(logSigma, delta=1.0, tune=true, weight=2.0)

#specify that we are going calculate BM likelihood using the REML PIC algorithm (see Felsenstein 1973) 
traits ~ dnPhyloBrownianREML(psi, branchRates=1.0, siteRates=sigma, nSites=contData.nchar())

traits.clamp(contData) #match traits to tips
monitors[1] = mnScreen(printgen=50000, sigma)
monitors[2] = mnFile(filename=outlf, printgen=400, separator = TAB,sigma)
bmv = model(sigma) #link sigma param w/ BM model

#set up MCMC
chain = mcmc(bmv, monitors, moves)
chain.burnin(generations=50000,tuningInterval=500)

chain.run(500000)
treetrace = readTreeTrace(file = outtr, "clock")
treefl <-outsm

#set MAP tree (we will also summarise trees as MCC representation outside revbayes
map = mapTree( file=treefl, treetrace )
q()

"""
