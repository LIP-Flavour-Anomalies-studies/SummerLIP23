import torch
import numpy as np
import utils_fom
import pandas as pd

class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):

        """
        data: the dict returned by utils.load_classification_data
        """
        
        train_X = data
        train_y = labels
        
        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def get_variables(file):
    var_list=[]
    df = pd.read_csv(file, header=0)
    df.columns = df.columns.str.strip()
    df["var_name"]=df["var_name"].str.strip()
    for v in df["var_name"]: 
        var_list.append(v)
    return var_list

    
def prepdata(data,data_mc):

    columns=get_variables("vars.csv")

    left_edge,right_edge,fb,fs=utils_fom.get_factors("/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/Fit_Results/B0Fit_3.5sigma_results.txt")

    sel_b="(tagged_mass<" + left_edge+ ") | (tagged_mass>" +right_edge + ")"
    sel_s="(tagged_mass>" + left_edge+ ") & (tagged_mass<" +right_edge + ")"
    
    TreeS=data_mc["ntuple"]
    TreeB=data["ntuple"]
    signal=TreeS.arrays(columns,cut=sel_s)
    background=TreeB.arrays(columns,cut=sel_b)

    stages=columns
    #stages=["var1","var2","var3","var4"]
    nsignal=len(signal[stages[0]])
    #nsignal=len(signal["var1"])
    nback=len(background[stages[0]])
    #nback=len(background["var1"])
    nevents=nsignal+nback
    x=np.zeros([nevents,len(stages)])
    y=np.zeros(nevents)
    y[:nsignal]=1
    for i,j in enumerate(stages):
        x[:nsignal,i]=signal[j]
        x[nsignal:,i]=background[j]
    
    return x,y

def prepdata_test(data,data_mc):

    columns=get_variables("vars.csv")

    left_edge,right_edge,fb,fs=utils_fom.get_factors("/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/Fit_Results/B0Fit_3.5sigma_results.txt")

    sel_b="(tagged_mass<" + left_edge+ ") | (tagged_mass>" +right_edge + ")"
    sel_s="(tagged_mass>" + left_edge+ ") & (tagged_mass<" +right_edge + ")"    

    TreeS=data_mc["ntuple"]
    TreeB=data["ntuple"]
    signal=TreeS.arrays(columns,cut=sel_s)
    background=TreeB.arrays(columns,cut=sel_b)

    stages=columns
    #stages=["var1","var2","var3","var4"]
    nsignal=len(signal[stages[0]])
    #nsignal=len(signal["var1"])
    nback=len(background[stages[0]])
    #nback=len(background["var1"])
    nevents=nsignal+nback
    x=np.zeros([nevents,len(stages)])
    y=np.zeros(nevents)
    y[:nsignal]=1
    for i,j in enumerate(stages):
        x[:nsignal,i]=signal[j]
        x[nsignal:,i]=background[j]


    print("this is the test function")