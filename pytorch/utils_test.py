import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/FOM/')
import utils_fom
import awkward as ak
import uproot

class RegressionDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):

        """
        data: the dict returned by utils.prepdata
        """
        
        train_X = data
        train_y = labels
        
        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def get_variables(file):
    var_list = []
    df = pd.read_csv(file, header=0)
    df.columns = df.columns.str.strip()
    df["var_name"] = df["var_name"].str.strip()
    for v in df["var_name"]: 
        var_list.append(v)
    return var_list

    
def prepdata(data, data_mc):

    columns = get_variables("/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/vars.csv")

    left_edge, right_edge, fb, fs = utils_fom.get_factors("/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/Fit_Results/B0Fit_3.5sigma_results.txt")

    sel_b = "(tagged_mass<" + left_edge + ") | (tagged_mass>" + right_edge + ")"
    sel_s = "(tagged_mass>" + left_edge + ") & (tagged_mass<" + right_edge + ")"

    TreeS = data_mc["ntuple"]
    TreeB = data["ntuple"]
    signal = TreeS.arrays(columns, cut=sel_s, entry_start=0, entry_stop=10000)
    background = TreeB.arrays(columns, cut=sel_b, entry_start=0, entry_stop=10000)

        # Assuming TreeS and TreeB are your TTree objects
    signal = ak.from_awkward0(TreeS.arrays(columns, cut=sel_s))
    background = ak.from_awkward0(TreeB.arrays(columns, cut=sel_b))

    # Flatten the arrays if needed (depends on your data structure)
    signal_flat = ak.flatten(signal)
    background_flat = ak.flatten(background)

    # Check for NaN values
    has_nan_signal = ak.any(ak.isnan(signal_flat), axis=None)
    has_nan_background = ak.any(ak.isnan(background_flat), axis=None)

    # Check for infinite values
    has_inf_signal = ak.any(ak.isinf(signal_flat), axis=None)
    has_inf_background = ak.any(ak.isinf(background_flat), axis=None)

    print("Signal has NaN:", has_nan_signal)
    print("Background has NaN:", has_nan_background)

    print("Signal has infinite values:", has_inf_signal)
    print("Background has infinite values:", has_inf_background)


    stages = columns
    nsignal = len(signal[stages[0]])
    nback = len(background[stages[0]])
    nevents = nsignal + nback
    x = np.zeros([nevents, len(stages)])
    y = np.zeros((nevents, len(stages)))
    y[:nsignal] = 1
    for i, j in enumerate(stages):
        x[:nsignal, i] = signal[j]
        x[nsignal:, i] = background[j]
    
    return x, y, columns

dir = "/lstore/cms/boletti/ntuples/"
filename = "2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
filename_mc = "MC_JPSI_2018_preBDT_Nov21.root"

data = uproot.open(dir + filename)
data_mc = uproot.open(dir + filename_mc)

x, y ,variable_names = prepdata(data, data_mc)