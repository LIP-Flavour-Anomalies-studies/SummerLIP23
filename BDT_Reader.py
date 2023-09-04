import ROOT
from ROOT import TMVA, TFile, TTree, TCut
import array
import numpy as np


#input PATH
dir = "/lstore/cms/boletti/ntuples/"
data_file = "2018Data_passPreselection_passSPlotCuts_mergeSweights.root"

# Create a TMVA.Reader object
reader = TMVA.Reader()

# Create variables used for training the BDT
bLBS = array.array('f',[0])
bCosAlphaBS = array.array('f',[0])
bVtxCL = array.array('f',[0])
kstTrkmDCABS = array.array('f',[0])
kstTrkpDCABS = array.array('f',[0])
mu1Pt = array.array('f',[0])
mu2Pt = array.array('f',[0])
kstTrkmPt = array.array('f',[0])
kstTrkpPt = array.array('f',[0])
tagB0 = array.array('f',[0])

# Declare variables to the reader
reader.AddVariable("bLBS", bLBS)
reader.AddVariable("bCosAlphaBS", bCosAlphaBS)
reader.AddVariable("bVtxCL", bVtxCL)
reader.AddVariable("kstTrkmDCABS", kstTrkmDCABS)
reader.AddVariable("kstTrkpDCABS", kstTrkpDCABS)
reader.AddVariable("mu1Pt", mu1Pt)
reader.AddVariable("mu2Pt", mu2Pt)
reader.AddVariable("kstTrkmPt", kstTrkmPt)
reader.AddVariable("kstTrkpPt", kstTrkpPt)
reader.AddVariable("tagB0", tagB0)


# Book the MVA methods
reader.BookMVA("BDT method", "/user/j/joaobss/SummerLIP23/dataset_norm1C/weights/TMVAClassification_BDTs.weights.xml")


#Opening data ROOT file
root_file = TFile.Open(dir+data_file)
tree = root_file.Get("ntuple")

# Create a new ROOT file to save the selected events
dir_out = "/user/j/joaobss/SummerLIP23/dataset_norm1C/results/rootfiles/"
output_file = TFile(dir_out+"selected_taggedMass.root", "RECREATE")
output_tree = tree.CloneTree(0)

# Declare the tagged_mass variable
tagged_mass = array.array('f',[0])

# Attach the branches
tree.SetBranchAddress("tagged_mass", tagged_mass)
tree.SetBranchAddress("bLBS", bLBS)
tree.SetBranchAddress("bCosAlphaBS", bCosAlphaBS)
tree.SetBranchAddress("bVtxCL", bVtxCL)
tree.SetBranchAddress("kstTrkmDCABS", kstTrkmDCABS)
tree.SetBranchAddress("kstTrkpDCABS", kstTrkpDCABS)
tree.SetBranchAddress("mu1Pt", mu1Pt)
tree.SetBranchAddress("mu2Pt", mu2Pt)
tree.SetBranchAddress("kstTrkmPt", kstTrkmPt)
tree.SetBranchAddress("kstTrkpPt", kstTrkpPt)
tree.SetBranchAddress("tagB0", tagB0)

# Loop over the entries in the TTree
for i in range(tree.GetEntries()):
    tree.GetEntry(i)

    # Check for NaN values in the input variables
    if not np.isnan(bLBS[0]) and not np.isnan(bCosAlphaBS[0]) and not np.isnan(bVtxCL[0]) and not np.isnan(kstTrkmDCABS[0]) and not np.isnan(kstTrkpDCABS[0]) and not np.isnan(mu1Pt[0]) and not np.isnan(mu2Pt[0]) and not np.isnan(kstTrkmPt[0]) and not np.isnan(kstTrkpPt[0]) and not np.isnan(tagB0[0]):
        # Evaluate the BDT score for the current event
        bdt_score = reader.EvaluateMVA("BDT method")

        if not np.isnan(bdt_score) and bdt_score < -0.296:
            # If the condition is met, add the event to the output tree
            output_tree.Fill()
    else:
        # Handle events with NaN values (e.g., skip or log them)
        print(f"Event {i} contains NaN values and is skipped.")

# Save the selected events to the output file
output_file.Write()
output_file.Close()

# Close the input ROOT file
root_file.Close()

