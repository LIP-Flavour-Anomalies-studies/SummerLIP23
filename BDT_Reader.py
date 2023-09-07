import ROOT
from ROOT import TMVA, TFile, TTree, TCut
import array
import numpy as np


#input PATH
dir = "/lstore/cms/boletti/ntuples/"
data_file = "2018Data_passPreselection_passSPlotCuts_mergeSweights.root"

# Create a TMVA.Reader object
reader = TMVA.Reader()

# Create variables used for training the BDT - Reader
bLBS_r = array.array('f',[0])
bCosAlphaBS_r = array.array('f',[0])
bVtxCL_r = array.array('f',[0])
kstTrkmDCABS_r = array.array('f',[0])
kstTrkpDCABS_r = array.array('f',[0])
mu1Pt_r = array.array('f',[0])
mu2Pt_r = array.array('f',[0])
kstTrkmPt_r = array.array('f',[0])
kstTrkpPt_r = array.array('f',[0])
tagB0_r = array.array('f',[0])

# TTree
bLBS = array.array('d',[0])
bCosAlphaBS = array.array('d',[0])
bVtxCL = array.array('d',[0])
kstTrkmDCABS = array.array('d',[0])
kstTrkpDCABS = array.array('d',[0])
mu1Pt = array.array('d',[0])
mu2Pt = array.array('d',[0])
kstTrkmPt = array.array('d',[0])
kstTrkpPt = array.array('d',[0])
tagB0 = array.array('d',[0])

# Declare variables to the reader
reader.AddVariable("bLBS", bLBS_r)
reader.AddVariable("bCosAlphaBS", bCosAlphaBS_r)
reader.AddVariable("bVtxCL", bVtxCL_r)
reader.AddVariable("kstTrkmDCABS", kstTrkmDCABS_r)
reader.AddVariable("kstTrkpDCABS", kstTrkpDCABS_r)
reader.AddVariable("mu1Pt", mu1Pt_r)
reader.AddVariable("mu2Pt", mu2Pt_r)
reader.AddVariable("kstTrkmPt", kstTrkmPt_r)
reader.AddVariable("kstTrkpPt", kstTrkpPt_r)
reader.AddVariable("tagB0", tagB0_r)


# Book the MVA methods
reader.BookMVA("BDT method", "/user/j/joaobss/SummerLIP23/dataset_norm1C/weights/TMVAClassification_BDTs.weights.xml")


#Opening data ROOT file
root_file = TFile.Open(dir+data_file)
tree = root_file.Get("ntuple")

# Create a new ROOT file to save the selected events
dir_out = "/user/j/joaobss/SummerLIP23/dataset_norm1C/results/rootfiles/"
output_file = TFile(dir_out+"selected_taggedMass.root", "RECREATE")
output_tree = TTree("output_tree", "outputTree")

#print(output_tree.GetEntries())

# Declare the vars
tagged_mass = array.array('d',[0])
bdt_score = array.array('d',[0])

output_tree.Branch('tagged_mass', tagged_mass, 'tagged_mass/D')
output_tree.Branch('bdt_score', bdt_score, 'bdt_score/D')


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

    
    bLBS_r[0] = bLBS[0]
    bCosAlphaBS_r[0] = bCosAlphaBS[0]
    bVtxCL_r[0] = bVtxCL[0]
    kstTrkmDCABS_r[0] = kstTrkmDCABS[0]
    kstTrkpDCABS_r[0] = kstTrkpDCABS[0]
    mu1Pt_r[0] = mu1Pt[0]
    mu2Pt_r[0] = mu2Pt[0]
    kstTrkmPt_r[0] = kstTrkmPt[0]
    kstTrkpPt_r[0] = kstTrkpPt[0]
    tagB0_r[0] = tagB0[0]


    # Check for NaN values in the input variables
    if not np.isnan(bLBS[0]) and not np.isnan(bCosAlphaBS[0]) and not np.isnan(bVtxCL[0]) and not np.isnan(kstTrkmDCABS[0]) and not np.isnan(kstTrkpDCABS[0]) and not np.isnan(mu1Pt[0]) and not np.isnan(mu2Pt[0]) and not np.isnan(kstTrkmPt[0]) and not np.isnan(kstTrkpPt[0]) and not np.isnan(tagB0[0]):
        # Evaluate the BDT score for the current event
        bdt_score[0] = reader.EvaluateMVA("BDT method")

        # If the condition is met, add the event to the output tree
        output_tree.Fill()

    else:
        # Handle events with NaN values (e.g., skip or log them)
        print(f"Event {i} contains NaN values and is skipped.")



# Save the selected events to the output file
output_file.cd()
output_tree.Write("ntuple")
output_file.Close()

# Close the input ROOT file
root_file.Close()

