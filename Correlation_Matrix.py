import ROOT
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Input PATH
dir = "/home/t3cms/joaobss/SummerLIP23/ROOT_files/"
data_file = "Data2018_CompositeVars.root"
mc_file = "MC_JPSI_2018_CompositeVars.root"

#Open the ROOT files and get the TTrees
file1 = ROOT.TFile.Open(dir+data_file, "READ")
file2 = ROOT.TFile.Open(dir+mc_file, "READ")
dataTree = file1.Get("ntuple")
mcTree = file2.Get("ntuple")

#Get the TBranches (14 variables) 
variables = ["bLBS","bLBSE","bLBSRatio","bCosAlphaBS","bCosAlphaBSE","bCosRatio","bVtxCL","kstTrkpDCABS","kstTrkmDCABS","mu1Pt","mu2Pt","kstTrkpPt","kstTrkmPt","tagB0"]


#Extract the data from the TBranches and store it in a Pandas DataFrame (Background)
data1 = {var: [] for var in variables}
for event in dataTree:
    if event.tagged_mass < 5.14 or event.tagged_mass > 5.41:
        for var in variables:
            data1[var].append(getattr(event, var))

df1 = pd.DataFrame(data1)

#Calculate the correlation matrix
correlation_matrix1 = df1.corr()

#Create a heatmap of the correlation matrix using Seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix1, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Background Correlation Matrix")
plt.savefig("CorrelationM_Bkg.png")
plt.close()




#Extract the data from the TBranches and store it in a Pandas DataFrame (Signal)
data2 = {var: [] for var in variables}
for event in mcTree:
    if event.tagged_mass > 5.14 and event.tagged_mass < 5.41:
        for var in variables:
            data2[var].append(getattr(event, var))

df2 = pd.DataFrame(data2)

# Calculate the correlation matrix
correlation_matrix2 = df2.corr()

# Create a heatmap of the correlation matrix using Seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix2, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Signal Correlation Matrix")
plt.savefig("CorrelationM_Sgn.png")
plt.close()