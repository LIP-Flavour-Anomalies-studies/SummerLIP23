import ROOT
from ROOT import TFile
from ROOT import TH1F
import numpy as np
import os
import pandas as pd


# Input PATH
dir = "/user/j/joaobss/SummerLIP23/ROOT_files/"
data_file = "Data2018_CompositeVars.root"
mc_file = "MC_JPSI_2018_CompositeVars.root"

# Output PATH
dir_out = "/user/j/joaobss/SummerLIP23/Variables_Plots/"



# Open the ROOT files and access the TTree for data and MC
data = TFile.Open(dir + data_file)
mc = TFile.Open(dir + mc_file)
dataTree = data.Get("ntuple")
mcTree = mc.Get("ntuple")




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






def create_histogram(branch,bin,bmin,bmax,blegend,blog):

    # Set the types
    branch = str(branch)
    bin = int(bin)
    bmin = float(bmin)
    bmax = float(bmax)
    blegend = str(blegend)
    blog = int(blog)

    # Get the left and right sideband edges
    sleft, sright = sideband_edges()

    # Create a TCanvas
    c = ROOT.TCanvas("c", "Histogram Canvas", 800, 600)

    # Create TH1F histograms
    h1 = ROOT.TH1F("h1", "", bin, bmin, bmax) # Data
    h2 = ROOT.TH1F("h2", "", bin, bmin, bmax) # MC

    # Draw data and fill it in the histograms
    dataTree.Draw(branch + ">>h1", "tagged_mass < " + str(sleft) + " || tagged_mass > " + str(sright)) # Background
    mcTree.Draw(branch + ">>h2", "tagged_mass > " + str(sleft) + " && tagged_mass < " + str(sright))   # Signal

    # Normalize
    h1.Scale(1./h1.Integral())
    h2.Scale(1./h2.Integral())

    # Set Background to red
    h1.SetLineColor(ROOT.kRed)

    if branch == "tagB0":
        h1.SetLineStyle(2)  # For dashed style

    # Remove the statistics box from the canvas
    h1.SetStats(0)
    h2.SetStats(0)

    # Maximum value
    h1max = h1.GetMaximum()
    h2max = h2.GetMaximum()

    # Draw h1 and h2
    if h1max > h2max:
        h1.Draw("HIST")
        h1.SetMaximum(1.1 * h1max) # Set the y-axis range to include both histograms
        h2.Draw("HIST same") # Draw h2 on top of h1
        h1.SetXTitle(blegend)
    else:
        h2.Draw("HIST")
        h2.SetMaximum(1.1 * h2max)
        h1.Draw("HIST same")
        h2.SetXTitle(blegend)

    # Create a legend and add entries for histo1 and histo2
    legend = ROOT.TLegend(0.69, 0.88, 0.89, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
    legend.AddEntry(h1, "Background", "l")  # "B" label for hVtx1 with line (l)
    legend.AddEntry(h2, "Signal", "l")  # "S" label for hVtx2 with line (l)
    legend.SetBorderSize(1)  # Remove border around the legend

    # Center the information in the legend and reduce the space after the text
    legend.SetTextAlign(22)   # Center-aligned text
    legend.SetTextSize(0.04)  # Adjust text size

    legend.Draw()

    # Set logarithmic y-axis
    if blog == 1:
        c.SetLogy()
    elif blog == 0:
        c.SetLogy(0)

    c.SaveAs(dir_out + branch + "_histo.png")

    # Clear the canvas
    c.Clear()






#create_histogram("bVtxCL",100,0,1,"hey",0)

def save_all(file): ## saves all histograms

    vars_path = file #'vars.csv'
    df = pd.read_csv(vars_path)
    df.columns = df.columns.str.strip() # take off leading and trailing spaces
    df["var_name"]=df["var_name"].str.strip()
    #os.chdir("/user/j/joaobss/SummerLIP23/")
    
    for v in df["var_name"]:
        composite_value = df.loc[df["var_name"] == v, "composite"].iloc[0]
        binv= df.loc[df["var_name"] == v, "bin"].iloc[0]
        minv= df.loc[df["var_name"] == v, "min"].iloc[0]
        maxv= df.loc[df["var_name"] == v, "max"].iloc[0]
        legendv= df.loc[df["var_name"] == v, "legend"].iloc[0]
        logv= df.loc[df["var_name"] == v, "log"].iloc[0]
        
        # Careful with the composite vars
        if composite_value == 1:
            if v == "(1-bCosAlphaBS)/bCosAlphaBSE":
                v = "bCosRatio"
            elif v == "bLBS/bLBSE":
                v = "bLBSRatio"
        
        create_histogram(v,binv,minv,maxv,legendv,logv)






save_all("vars.csv")

# Clean up
data.Close()
mc.Close()