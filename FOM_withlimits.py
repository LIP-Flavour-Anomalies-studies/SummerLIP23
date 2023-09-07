import pandas as pd
import uproot
import matplotlib.pyplot as plt
import numpy as np
import os
import utils_fom
#var="bVtxCL"
#var="kstTrkmPt"
#var="bLBS"
#var = "bCosAlphaBS"

##APLICAR SCALING FACTORS
##APLICAR TAGGED MASS CUTS

#columns=["tagged_mass",var]

#sel="(tagged_mass<5.1) | (tagged_mass>5.44)"



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
    background=Tree.arrays(v,cut=sel_b,library="pd")
    signal=Tree_mc.arrays(v,cut=sel_s,library="pd")
    return signal,background

def get_signal_composite(v,sel_s,sel_b):
    signal=Tree_mc.arrays(v,aliases={v:v},cut=sel_s,library="pd")
    background=Tree.arrays(v,aliases={v:v},cut=sel_b,library="pd")
    return signal,background



def calc_fom(v,signal,background,minv,maxv,fs,fb,show_upper,legend):
    #os.chdir("/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/"+folder)

    num_points=100

    while True:
        try:
            step = (maxv - minv) / (num_points - 1)
            var_range = np.arange(minv, maxv , step)

            break
        except ValueError:
            minv=signal[v].min()
            maxv=signal[v].max()
    
    fom=[]
    fom_opposite=[]

    print (np.sum(signal[v]>minv))
    for i in var_range:
        s=0;b=0;f=0
        #print(i)

        #print(signal.size)
        s = np.sum(signal[v] > i)
        #ss=signal[v].shape(0)
        print("s",s)
        s=s*fs
        b = np.sum(background[v] > i)
        b=b*fb

        s_opposite=np.sum(signal[v]<i)
        s_opposite*=fs
        b_opposite = np.sum(background[v] < i)
        b_opposite*=fb


        f=s/(s+b)**0.5
        #print(str(i)+" , figure of merit:"+str(f))
        fom.append(f)

        if (s_opposite>0 or b_opposite>0):
            fom_opposite.append(s_opposite/(s_opposite+b_opposite)**0.5)
        else:
            fom_opposite.append(0)


    if show_upper==1:
        label="upper cut"
        fom_to_plot=fom_opposite
    else:
        label="lower cut"
        fom_to_plot=fom


    os.chdir("/user/u/u23madalenablanc/flavour-anomalies/plot_fom/")

    plt.figure()
    
    
    plt.plot(var_range,fom_to_plot,label=label)

    #plt.title(f"FOM for: {v}")
    plt.legend()
    plt.xlabel(legend)
    plt.ylabel("FOM")

    while True:
        try:
            plt.savefig(v+'_fom.png')
            break
        except FileNotFoundError:
            print("bad name")
            v=v.replace("/", "_div_")    


def read_factors():

    left_edge,right_edge,fb,fs=utils_fom.get_factors("/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/Fit_Results/B0Fit_3.5sigma_results.txt")

    sel_b="(tagged_mass<" + left_edge+ ") | (tagged_mass>" +right_edge + ")"
    sel_s="(tagged_mass>" + left_edge+ ") & (tagged_mass<" +right_edge + ")"

    return sel_b,sel_s,fb,fs


def main():
    sel_b,sel_s,fb,fs=read_factors()
    df["var_name"]=df["var_name"].str.strip()
    for v in df["var_name"]:  
        print(v)
        composite_value = df.loc[df["var_name"] == v, "composite"].iloc[0]
        minv= df.loc[df["var_name"] == v, "min"].iloc[0]
        maxv= df.loc[df["var_name"] == v, "max"].iloc[0]
        show_upper=df.loc[df["var_name"] == v, "show_upper"].iloc[0]
        legend=df.loc[df["var_name"] == v, "legend"].iloc[0]
        print(minv,maxv)
        if composite_value==0:
            signal,back=get_signal_normal(v,sel_s,sel_b)
        elif (composite_value)==1:
            signal,back=get_signal_composite(v,sel_s,sel_b)
        calc_fom(v,signal,back,minv,maxv,fs,fb,show_upper,legend)

if __name__ == '__main__':
    main()