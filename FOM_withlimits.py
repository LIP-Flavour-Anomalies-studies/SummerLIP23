import pandas as pd
import uproot
import matplotlib.pyplot as plt
import numpy as np
import os
#var="bVtxCL"
#var="kstTrkmPt"
#var="bLBS"
#var = "bCosAlphaBS"

##APLICAR SCALING FACTORS
##APLICAR TAGGED MASS CUTS

#columns=["tagged_mass",var]

sel="(tagged_mass<5.1) | (tagged_mass>5.44)"



dir="/lstore/cms/boletti/ntuples/"
filename="2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
filename2="MC_JPSI_2018_preBDT_Nov21.root"

data = uproot.open(dir+filename)
data_mc=uproot.open(dir+filename2)

Tree=data["ntuple"]
Tree_mc=data_mc["ntuple"]




folder="plot_fom"
variables_path = 'variables.xlsx'
df = pd.read_excel(variables_path,header=0)
df.columns = df.columns.str.strip()


os.chdir("/home/t3cms/u23madalenablanc/"+folder)






def get_signal_normal(v,sel):
    background=Tree.arrays(v,cut=sel,library="pd")
    signal=Tree_mc.arrays(v,cut=sel,library="pd")
    return signal,background

def get_signal_composite(v,sel):
    signal=Tree_mc.arrays(v,aliases={v:v},cut=sel,library="pd")
    background=Tree.arrays(v,aliases={v:v},cut=sel,library="pd")
    return signal,background



def calc_fom(v,signal,background,minv,maxv,fs,fb):

    num_points=20

    while True:
        try:
            step = (maxv - minv) / (num_points - 1)
            var_range = np.arange(minv, maxv , step)
            break
        except ValueError:
            minv=signal[v].min()
            maxv=signal[v].max()
    
    fom=[]

    for i in var_range:
        s=0;b=0;f=0
        #print(i)

        #print(signal.size)
        s = np.sum(signal[v] > i)
        s=s*fs
        b = np.sum(background[v] > i)
        b=b*fb

        f=s/(s+b)**0.5
        #print(str(i)+" , figure of merit:"+str(f))
        fom.append(f)

    plt.figure()
    plt.plot(var_range,fom)
    plt.title(f"FOM for: {v}")
    while True:
        try:
            plt.savefig(v+'hist.png')
            break
        except FileNotFoundError:
            print("bad name")
            v=v.replace("/", "_div_")



#scaling factor, signal
fs=0.67752
#scaling factor, background
fb=0.74094

df["var_name"]=df["var_name"].str.strip()
for v in df["var_name"]:  
    print(v)
    composite_value = df.loc[df["var_name"] == v, "composite"].iloc[0]
    minv= df.loc[df["var_name"] == v, "min"].iloc[0]
    maxv= df.loc[df["var_name"] == v, "max"].iloc[0]
    print(minv,maxv)
    if composite_value==0:
        signal,back=get_signal_normal(v,sel)
    elif (composite_value)==1:
        signal,back=get_signal_composite(v,sel)
    calc_fom(v,signal,back,minv,maxv,fs,fb)


