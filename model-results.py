import argparse
import torch
import os.path
from subprocess import call
import uproot
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
import utils

import importlib
pt=importlib.import_module("pytorch-tutuorial")


def load_options(file):
    with open(file,'r') as f:
        n_classes=f.readline().strip()
        n_feats=f.readline().strip()
        hidden_size=f.readline().strip()
        layers=f.readline().strip()
        activation=f.readline().strip()
        dropout=f.readline().strip()
        f.close()

    return int(n_classes),int(n_feats),int(hidden_size),int(layers),activation,float(dropout)

def main():

    #print(load_options("NNopts.txt"))
    n_classes,n_feats,hidden_size,layers,activation,dropout=load_options("NNopts.txt")

    #load model
    PATH="/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/modelNN"

    model = pt.FeedforwardNetwork(
        n_classes,
        n_feats,
        hidden_size,
        layers,
        activation,
        dropout
    )
    #model.load_state_dict(torch.load(PATH))
    #model.eval()

if __name__ == '__main__':
    main()