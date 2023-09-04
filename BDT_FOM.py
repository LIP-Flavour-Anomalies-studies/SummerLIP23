import pandas as pd
import uproot
import matplotlib.pyplot as plt
import numpy as np
import utils_fom


dir="/user/j/joaobss/SummerLIP23/dataset_norm1C/results/rootfiles/"
root_file="TMVA_BDT.root"

file = uproot.open(dir+root_file)

#Access the TestTree
directory = file["dataset_norm1C"]
testTree = directory["TestTree"]

#TBranches
classID_branch = testTree["classID"].array(library="pd")
BDTscore_branch = testTree["BDTs"].array(library="pd")

#signal and background BDT scores
signal = BDTscore_branch[classID_branch==0]
background = BDTscore_branch[classID_branch==1]

#min/max values of the BDT scores
minv=BDTscore_branch.min()
maxv=BDTscore_branch.max()


def calc_fom(signal,background,minv,maxv,fs,fb,show_upper=0):

    num_points=100

    step = (maxv - minv) / (num_points - 1)
    var_range = np.arange(minv, maxv, step)
    
    fom=[]
    fom_opposite=[]

    for i in var_range:
        s=0;b=0;f=0

        s = np.sum(signal > i)
        s=s*fs
        b = np.sum(background > i)
        b=b*fb

        #Not used in this var: BDT score
        s_opposite=np.sum(signal<i)
        s_opposite*=fs
        b_opposite = np.sum(background < i)
        b_opposite*=fb


        f=s/(s+b)**0.5
        fom.append(f)


        #Not used in this var: BDT score
        if (s_opposite>0 or b_opposite>0):
            fom_opposite.append(s_opposite/(s_opposite+b_opposite)**0.5)
        else:
            fom_opposite.append(0)


    # Find the index of the maximum value in the FOM array
    max_FOM_index = np.argmax(fom)
    
    # Get the corresponding BDT score (x coordinate)
    max_x = var_range[max_FOM_index]
    
    # Get the maximum FOM value (y coordinate)
    max_y = fom[max_FOM_index]

    print(f"Best cut: BDT score={max_x}, FOM={max_y}")

    f = open("/user/j/joaobss/SummerLIP23/dataset_norm1C/plots/BDT_fom.txt", "w")
    f.write(f"Best cut: \nBDT score = {max_x}, \nFOM = {max_y}")
    f.close()


    if show_upper==1:               #Not used in this var: BDT score
        label="upper cut"           #Not used in this var: BDT score
        fom_to_plot=fom_opposite    #Not used in this var: BDT score
    else:
        label="lower cut"
        fom_to_plot=fom


    plt.figure()
    plt.plot(var_range,fom_to_plot,label=label)


    plt.legend()
    plt.xlabel("BDT score")
    plt.ylabel("FOM")

    
    plt.savefig("/user/j/joaobss/SummerLIP23/dataset_norm1C/plots/BDT_fom.png")


    


#get the scale factors
_, _ , fb, fs = utils_fom.get_factors("/user/j/joaobss/SummerLIP23/Fit_Results/B0Fit_JPsi.txt")


#calculate the BDT FOM
calc_fom(signal,background,minv,maxv,fs,fb)


file.close()
