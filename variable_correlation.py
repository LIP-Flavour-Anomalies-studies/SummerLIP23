import numpy as np
import uproot
import os.path
from subprocess import call
import matplotlib.pyplot as plt
import awkward.operations as ak
import sys
import os
import pandas as pd
import seaborn as sns


dir="/lstore/cms/boletti/ntuples/"
filename="2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
filename2="MC_JPSI_2018_preBDT_Nov21.root"

data = uproot.open(dir+filename)
data_mc=uproot.open(dir+filename2)

variables_path = 'variables.xlsx'
df = pd.read_excel(variables_path,header=0)
df.columns = df.columns.str.strip()

#Tree=data["ntuple"]
#Tree_mc=data_mc["ntuple"]

def plot_heatmap(data_type):

    if data_type=="signal":
        Tree=data["ntuple"]
    elif data_type=="background":
        Tree=data_mc["ntuple"]
    else:
        print("not valid value")
        
   #read all variables
    combined_signal_df = pd.DataFrame()
    df["var_name"]=df["var_name"].str.strip()
    for v in df["var_name"]:  

        composite_value = df.loc[df["var_name"] == v, "composite"].iloc[0]

        if composite_value==0:
            signal=Tree.arrays(v,library="pd")
        elif (composite_value)==1:
            signal=Tree.arrays(v,aliases={v:v},library="pd")
        combined_signal_df = pd.concat([combined_signal_df, signal], axis=1)
    
    correlation_matrix = combined_signal_df.corr()

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))  # Set the figure size (optional)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f",cmap='coolwarm', center=0)

    # Add labels and title to the heatmap
    plt.title('Correlation Heatmap: '+data_type)
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    

    #remove error handling if it is not used
    while True:
        try:
            plt.savefig(data_type+'_correlation.png')
            break
        except FileNotFoundError:
            print("bad name")
            v=v.replace("/", "_div_")

plot_heatmap("signal")
plot_heatmap("background")