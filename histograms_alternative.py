##LOAD EVERYTHING FIRST AND THEN PLOT

import numpy as np
import uproot
import os.path
from subprocess import call
import matplotlib.pyplot as plt
import awkward.operations as ak
import sys
import os
import pandas as pd


##THIS IS THE UPDATED VERSION OF THIS CODE

##MAKE OPTION FOR NORMALIZED OR NOT
##MAKE OPTION FOR SIMPLE HISTOGRAM PLOTS
##OPTION FOR ROOT FILE WITH OTHER TREENAME OTHER THAN NTUPLE

normalized=False
if (len(sys.argv) !=1) and (sys.argv[1]=="normalized"):
    normalized=True
else:
    normalized=False


dir="/lstore/cms/boletti/ntuples/"
filename="2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
filename2="MC_JPSI_2018_preBDT_Nov21.root"

data = uproot.open(dir+filename)
data_mc=uproot.open(dir+filename2)

variables_path = 'variables.xlsx'
df = pd.read_excel(variables_path,header=0)
df.columns = df.columns.str.strip()

Tree=data["ntuple"]
Tree_mc=data_mc["ntuple"]


def get_normal(v):
    signal=Tree.arrays(v,library="np")
    background=Tree_mc.arrays(v,library="np")
    return signal,background

def get_composite(v):
    signal=Tree.arrays(v,aliases={v:v},library="np")
    background=Tree_mc.arrays(v,aliases={v:v},library="np")
    return signal,background


def hist(v,signal,background, minv,maxv,bins,logscale,legend):

    plt.figure()
    
    data_hist, data_bin_edges = np.histogram(signal[v], bins=bins, range=(minv,maxv))
    mc_hist, mc_bin_edges = np.histogram(background[v], bins=bins, range=(minv,maxv))

    data_bin_centers = (data_bin_edges[:-1] + data_bin_edges[1:]) / 2
    mc_bin_centers = (mc_bin_edges[:-1] + mc_bin_edges[1:]) / 2

    if normalized:
        #Normalize the histogram bin counts
        data_hist = data_hist / np.sum(data_hist)
        mc_hist = mc_hist / np.sum(mc_hist)

    plt.bar(data_bin_centers, data_hist, width=np.diff(data_bin_edges), color='red', alpha=0.5, label='Background') # Data (Bkg)
    plt.bar(mc_bin_centers, mc_hist, width=np.diff(mc_bin_edges), color='blue', alpha=0.5, label='Signal') # MC (Signal)

    plt.legend(loc='upper right')
    plt.xlabel(legend)



    if logscale==1:
        plt.yscale('log')

    try:
        plt.xlim(minv,maxv)
    except ValueError:
        pass

    while True:
        try:
            plt.savefig(v+'hist.png')
            break
        except FileNotFoundError:
            print("bad name")
            v=v.replace("/", "_div_")





def save_all(file,folder): ## saves all histograms in folder

    variables_path = file #'variables.xlsx'
    df = pd.read_csv(variables_path, header=0)
    df.columns = df.columns.str.strip()
    os.chdir("/user/u/u23madalenablanc/flavour-anomalies/"+folder)
    print(df["var_name"])
    df["var_name"]=df["var_name"].str.strip()
    for v in df["var_name"]:   #.str.replace(" ", ""):
        composite_value = df.loc[df["var_name"] == v, "composite"].iloc[0]
        minv= df.loc[df["var_name"] == v, "min"].iloc[0]
        maxv= df.loc[df["var_name"] == v, "max"].iloc[0]
        bins=df.loc[df["var_name"] == v, "bin"].iloc[0]
        logscale=df.loc[df["var_name"] == v, "log"].iloc[0]
        legend=df.loc[df["var_name"] == v, "legend"].iloc[0]

        print(f"{v}: composite = {composite_value}")
        if composite_value==0:
            s,b=get_normal(v)
        elif (composite_value)==1:
            s,b=get_composite(v)
        hist(v,s,b,minv,maxv,bins,logscale,legend)



save_all("vars.csv","plots2") 