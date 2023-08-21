import ROOT
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Input PATH
dir = "/user/j/joaobss/SummerLIP23/ROOT_files/"
data_file = "Data2018_CompositeVars.root"
mc_file = "MC_JPSI_2018_CompositeVars.root"

#Output PATH
dir_out = "/user/j/joaobss/SummerLIP23/Correlation_Plots/"

#Open the ROOT files and get the TTrees
file1 = ROOT.TFile.Open(dir+data_file, "READ")
file2 = ROOT.TFile.Open(dir+mc_file, "READ")
dataTree = file1.Get("ntuple")
mcTree = file2.Get("ntuple")


# Access .csv file
vars_path = "/user/j/joaobss/SummerLIP23/vars.csv" #'vars.csv'
df = pd.read_csv(vars_path)
df.columns = df.columns.str.strip() # take off leading and trailing spaces
df["var_name"]=df["var_name"].str.strip()


# Create a list of variables in .csv file
variables = []

for v in df["var_name"]:
    composite_value = df.loc[df["var_name"] == v, "composite"].iloc[0]
    # Careful with the composite vars
    if composite_value == 1:
        if v == "(1-bCosAlphaBS)/bCosAlphaBSE":
            v = "bCosRatio"
        elif v == "bLBS/bLBSE":
            v = "bLBSRatio"
    variables.append(v)




def sideband_edges():
    # Define the path to the text file
    file_path = "/user/j/joaobss/SummerLIP23/Fit_Results/B0Fit_3.5sigma_results.txt"

    # Define the legend labels
    label1 = "left sideband edge"
    label2 = "right sideband edge"

    # Read the contents of the file and find the values for the specified legend labels
    left_desired_value = None
    right_desired_value = None
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("#")
            if len(parts) == 2:
                label = parts[1].strip()
                value = float(parts[0].strip())
                if label == label1:
                    left_desired_value = value
                elif label == label2:
                    right_desired_value = value
            
            # Exit the loop once both values are found
            if left_desired_value is not None and right_desired_value is not None:
                break
    
    return left_desired_value, right_desired_value





def correlation_matrix(data_type):

    # Get the left and right sideband edges
    sleft, sright = sideband_edges()

    data = {var: [] for var in variables}

    if data_type=="Background":
        for event in dataTree:
            if event.tagged_mass < sleft or event.tagged_mass > sright:
                for var in variables:
                    data[var].append(getattr(event, var))

    elif data_type=="Signal":
        for event in mcTree:
            if event.tagged_mass > sleft and event.tagged_mass < sright:
                for var in variables:
                    data[var].append(getattr(event, var))
    else:
        print("not valid value")



    df = pd.DataFrame(data)

    #Calculate the correlation matrix
    correlation_matrix = df.corr()

    #Create a heatmap of the correlation matrix using Seaborn
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(data_type + " Correlation Matrix")
    plt.savefig(dir_out + data_type + "_CorrelationMatrix.png")
    plt.close()





correlation_matrix("Signal")
correlation_matrix("Background")