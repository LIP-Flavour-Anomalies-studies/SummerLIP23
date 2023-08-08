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

def hist_normal(v,minv,maxv):
    signal=Tree.arrays(v)
    signal_mc=Tree_mc.arrays(v)

    plt.figure()
    plt.hist(signal[v],bins=50,alpha=0.5,density=normalized,label="data")
    plt.title(v)
    plt.hist(signal_mc[v],bins=50,alpha=0.5,density=normalized,label="MC")
    plt.legend(loc='upper right')
    try:
        plt.xlim(minv,maxv)
    except ValueError:
        pass
    plt.savefig(v+'hist.png')

def hist_composite(v,minv,maxv):
    #print("i am composite", v)
    newvar1=Tree.arrays(v,aliases={v:v},library="pd")
    newvar2=Tree_mc.arrays(v,aliases={v:v},library="pd")

    plt.figure()
    plt.hist(newvar1,bins=50,alpha=0.5,density=normalized,label="data")
    plt.title(v)
    plt.hist(newvar2,bins=50,alpha=0.5,density=normalized,label="MC")
    plt.legend(loc='upper right')
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
    df = pd.read_excel(variables_path,header=0)
    df.columns = df.columns.str.strip()
    os.chdir("/home/t3cms/u23madalenablanc/flavour-anomalies/"+folder)
    print(df["var_name"])
    df["var_name"]=df["var_name"].str.strip()
    for v in df["var_name"]:   #.str.replace(" ", ""):
        composite_value = df.loc[df["var_name"] == v, "composite"].iloc[0]
        minv= df.loc[df["var_name"] == v, "min"].iloc[0]
        maxv= df.loc[df["var_name"] == v, "max"].iloc[0]
        print(f"{v}: composite = {composite_value}")
        if composite_value==0:
            hist_normal(v,minv,maxv)
        elif (composite_value)==1:
            hist_composite(v,minv,maxv)

save_all("variables.xlsx","plots") 