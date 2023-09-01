import numpy as np
import uproot
import os.path
from subprocess import call
import matplotlib.pyplot as plt
import awkward.operations as ak
import sys
import os
import pandas as pd
import utils_fom
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

    

left_edge,right_edge,fb,fs=utils_fom.get_factors("/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/Fit_Results/B0Fit_3.5sigma_results.txt")

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

def parse_all(df):
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

        cut_b=selection_b+" | (" + v + "<" +str(cut) + ")"
        cut_s=selection_s+" & (" + v + ">" +str(cut) + ")"
        #print(minv,maxv)
        if composite_value==0:
            s,b=get_signal_normal(v,sel_tagged_mass_s,cut_b)
        elif (composite_value)==1:
            s,b=get_signal_composite(v,sel_tagged_mass_s,sel_tagged_mass_b)
        fom_before=calc_fom(v,s,b,minv,maxv,cut)
        print(fom_before)

        if composite_value==0:
            new_s,new_b=get_signal_normal(v,sel_tagged_mass_s,sel_tagged_mass_b)
        elif (composite_value)==1:
            new_s,new_b=get_signal_composite(v,sel_tagged_mass_s,sel_tagged_mass_b)

        fom_after=calc_fom(v,new_s,new_b,minv,maxv,cut)
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
        ax.set_title('FOM Before and After Cuts for {v}')
        ax.legend()

        plt.tight_layout()    

        while True:
            try:
                plt.savefig(v+'_compar.png')
                break
            except FileNotFoundError:
                print("bad name")
                v=v.replace("/", "_div_")


    # Create a single grouped bar plot
    x = np.arange(len(df["var_name"]))  # x-axis positions for variables
    width = 0.35  # Width of each bar

    fig, ax = plt.subplots()


    for idx, v in enumerate(df["var_name"]):
        fom_values = fom_dict.get(v, [0, 0])  # Default to [0, 0] if v not found
        bars_before = ax.bar(x[idx], fom_values[0], width, label=f'Before {v}', color='blue')
        bars_after = ax.bar(x[idx] + width, fom_values[1], width, label=f'After {v}', color='orange')

    # Add labels, title, and remove legend
    ax.set_ylabel('FOM')
    ax.set_title('FOM Before and After Cuts by Variable')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(df["var_name"], rotation=45, ha="right")
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    ax.legend().set_visible(False)  # Remove the legend

    plt.tight_layout()

    plt.savefig("compar.png")


parse_all(df)

