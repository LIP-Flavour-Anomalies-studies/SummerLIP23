import numpy as np
import uproot
import os.path
from subprocess import call
import matplotlib.pyplot as plt
import awkward.operations as ak
import sys
import os
import pandas as pd

##calculate initial and final FOM and final FOM with the cuts applied to the variables

##plot histogram showing the cut 


normalized=True

dir="/lstore/cms/boletti/ntuples/"
filename="2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
filename2="MC_JPSI_2018_preBDT_Nov21.root"

data = uproot.open(dir+filename)
data_mc=uproot.open(dir+filename2)

Tree=data["ntuple"]
Tree_mc=data_mc["ntuple"]




folder="plot_fom"
variables_path = 'vars.csv'
df = pd.read_csv(variables_path, header=0)
df.columns = df.columns.str.strip()


#os.chdir("/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/"+folder)




def get_signal_normal(v,sel_s,sel_b):
    signal=Tree.arrays(v,cut=sel_s,library="np")
    background=Tree_mc.arrays(v,cut=sel_b,library="np")
    return signal,background


def get_signal_composite(v,sel_s,sel_b):
    signal=Tree_mc.arrays(v,aliases={v:v},cut=sel_s,library="np")
    background=Tree.arrays(v,aliases={v:v},cut=sel_b,library="np")
    return signal,background



def calc_fom(v,signal,background,minv,maxv,cut): #on a signle point
    s=0;b=0;f=0
    i=cut
    #print(i)

    #print(signal.size)
    s = np.sum(signal[v] > i)
    #s=s*fs
    b = np.sum(background[v] > i)
    #b=b*fb

    f=s/(s+b)**0.5
    return f

def hist(v,signal,background, minv,maxv,bins,logscale,legend,cut):

    folder="before_n_after"

    os.chdir("/user/u/u23madalenablanc/flavour-anomalies/"+ folder)

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

    plt.axvline(x=cut, color='black', linestyle='dashed', linewidth=2, label='Cut')

    if logscale==1:
        plt.yscale('log')

    try:
        plt.xlim(minv,maxv)
    except ValueError:
        pass

    while True:
        try:
            plt.savefig(v+'.png')
            break
        except FileNotFoundError:
            print("bad name")
            v=v.replace("/", "_div_")

    
#this part will be in a different scipt

file=open("/user/u/u23madalenablanc/SummerLIP23/Fit results/B0Fit_3.5sigma_results.txt","r")

str1="left sideband edge"
str2="right sideband edge"
str3="background scaling factor"
str4="signal scaling factor "

def get_value(line):
    value=""
    
    for c in line:
        if c.isdigit() or c==".":
            value += c
        else:
            break
    return value


for line in file:
    if str1 in line:
        left_edge=get_value(line)
    elif str2 in line:
        right_edge=get_value(line)
    elif str3 in line:
        fb=float(get_value(line))
    elif str4 in line:
        fs=float(get_value(line))

#this is for background
sel_tagged_mass_b="(tagged_mass<" + left_edge+ ") | (tagged_mass>" +right_edge + ")"
sel_tagged_mass_s="(tagged_mass>" + left_edge+ ") & (tagged_mass<" +right_edge + ")"

df["var_name"]=df["var_name"].str.strip()

#get cuts
cuts_dict={}
for v in df["var_name"]:  
    cut=df.loc[df["var_name"] == v, "best_cut"].iloc[0]
    cuts_dict[v]=cut

selection_s=""
selection_b=""

selection_s+=sel_tagged_mass_s
selection_b+=sel_tagged_mass_b
for v,cut in cuts_dict.items():
    
    c=" | (" + v + "<" +str(cut) + ")"
    selection_b+=c

    c=" & (" + v + ">" +str(cut) + ")"
    selection_s+=c

print(selection_s)


fom_dict={}
for v in df["var_name"]:  
    print(v)
    composite_value = df.loc[df["var_name"] == v, "composite"].iloc[0]
    minv= df.loc[df["var_name"] == v, "min"].iloc[0]
    maxv= df.loc[df["var_name"] == v, "max"].iloc[0]
    bins=df.loc[df["var_name"] == v, "bin"].iloc[0]
    logscale=df.loc[df["var_name"] == v, "log"].iloc[0]
    legend=df.loc[df["var_name"] == v, "legend"].iloc[0]
    cut=df.loc[df["var_name"] == v, "best_cut"].iloc[0]
    #print(minv,maxv)
    if composite_value==0:
        s,b=get_signal_normal(v,sel_tagged_mass_s,sel_tagged_mass_b)
    elif (composite_value)==1:
        s,b=get_signal_composite(v,sel_tagged_mass_s,sel_tagged_mass_b)
    fom_before=calc_fom(v,s,b,minv,maxv,minv)
    print(fom_before)

    if composite_value==0:
        new_s,new_b=get_signal_normal(v,selection_s,selection_b)
    elif (composite_value)==1:
        new_s,new_b=get_signal_composite(v,selection_s,selection_b)
    fom_after=calc_fom(v,new_s,new_b,minv,maxv,minv)
    print(fom_after)

    fom_dict[v]=[fom_before,fom_after]

    if composite_value==0:
        signal,background=get_signal_normal(v,None,None)
    elif (composite_value)==1:
        signal,background=get_signal_composite(v,None,None)
    

    hist(v,signal,background, minv,maxv,bins,logscale,legend,cut)

    fig, ax = plt.subplots()

    # Create side-by-side bar plot for the current variable
    bars_before = ax.bar(['Before', 'After'], [fom_before, fom_after])

    # Add labels, title, and legend
    ax.set_ylabel('FOM')
    ax.set_title(f'FOM Before and After Cuts for {v}')
    ax.legend()

    plt.tight_layout()    

    while True:
        try:
            plt.savefig(v+'_compar.png')
            break
        except FileNotFoundError:
            print("bad name")
            v=v.replace("/", "_div_")

#bar plot to compare results



