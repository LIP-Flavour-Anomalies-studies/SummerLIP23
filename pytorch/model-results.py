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
        batch_size=f.readline().strip()
        f.close()

    return int(n_classes),int(n_feats),int(hidden_size),int(layers),activation,float(dropout),int(batch_size)

def load_data(batch_size):
    dir="/lstore/cms/boletti/ntuples/"
    filename="2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
    filename_mc="MC_JPSI_2018_preBDT_Nov21.root"

    data=uproot.open(dir+filename)
    data_mc=uproot.open(dir+filename_mc)

    #data = uproot.open('tmva_class_example.root')

    x,y = utils.prepdata(data,data_mc) 
    dataset = utils.ClassificationDataset(x,y)

    train_set,test_set,val_set =torch.utils.data.random_split(dataset, [0.5,0.25,0.25])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    dev_X, dev_y = val_set[:][0], val_set[:][1]
    test_X, test_y = test_set[:][0], test_set[:][1]

    n_classes = torch.unique(dataset.y).shape[0]  
    n_feats = dataset.X.shape[1]


def main():

    #print(load_options("NNopts.txt"))
    n_classes,n_feats,hidden_size,layers,activation,dropout,batch_size=load_options("output.txt")

    #load model
    PATH="/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/pytorch/modelNN"

    model = pt.FeedforwardNetwork(
        n_classes,
        n_feats,
        hidden_size,
        layers,
        activation,
        dropout
    )
    model.load_state_dict(torch.load(PATH))
    model.eval()

    load_data(batch_size)

if __name__ == '__main__':
    main()