import ROOT
from ROOT import TMVA, TFile, TTree, TCut
import os.path
import sys


##run as 'python3 filename.py command_input'


#Get the input argument from the command line
if len(sys.argv) != 2:
    print("Insert a number after the python script")
    sys.exit(1)

input = sys.argv[1]


TMVA.Tools.Instance()               #need to run this two to load up TMVA
TMVA.PyMethodBase.PyInitialize()    #in PyROOT

#input PATH
dir = "/lstore/cms/boletti/ntuples/"
data_file = "2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
mc_file = "MC_JPSI_2018_preBDT_Nov21.root"
 
# Open the ROOT files and access the TTree for data and MC
data = TFile.Open(dir + data_file)
mc = TFile.Open(dir + mc_file)
background = data.Get("ntuple")
signal = mc.Get("ntuple")

# Create a ROOT output file where TMVA will store ntuples, histograms, correlationMatrix, etc
outfname='dataset_' + input + '/results/rootfiles/TMVA_BDT.root' 
output = TFile.Open(outfname, 'RECREATE')



def sideband_edges(fit_text_file):
    # Define the path to the text file
    file_path = "/user/j/joaobss/SummerLIP23/Fit_Results/" + fit_text_file

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


# Get the left and right sideband edges
sleft, sright = sideband_edges("B0Fit_JPsi.txt")

#Apply the necessary variable cuts
cuts="tagged_mass > %f && tagged_mass < %f" % (sleft, sright)   #MC
cutb="tagged_mass < %f || tagged_mass > %f" % (sleft, sright)   #data

#cuts=""
#cutb=""

mycutS=TCut(cuts)
mycutB=TCut(cutb)

#directories where the results will be stored
if not os.path.exists("dataset_" + input + "/results/rootfiles"):
    os.makedirs("dataset_" + input + "/results/rootfiles")
if not os.path.exists("dataset_" + input + "/weights"):
    os.makedirs("dataset_" + input + "/weights")
if not os.path.exists("dataset_" + input + "/plots"):
    os.makedirs("dataset_" + input + "/plots")



factory = TMVA.Factory('TMVAClassification', output, '!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification')

#loads the outputs to the dir 'dataset'
dataloader = TMVA.DataLoader('dataset_' + input)

# features to train the BDT
dataloader.AddVariable("bLBS")
dataloader.AddVariable("bCosAlphaBS")
dataloader.AddVariable("bVtxCL")
dataloader.AddVariable("kstTrkmDCABS")
dataloader.AddVariable("kstTrkpDCABS")
dataloader.AddVariable("mu1Pt")
dataloader.AddVariable("mu2Pt")
dataloader.AddVariable("kstTrkmPt")
dataloader.AddVariable("kstTrkpPt")
dataloader.AddVariable("tagB0")



# features not used to train the BDT
#dataloader.AddSpectator("bLBSE")
#dataloader.AddSpectator("bLBDRatio := bLBS / bLBSE")                        #Composite var
#dataloader.AddSpectator("bCosRatio := (1 - bCosAlphaBS) / bCosAlphaBSE")    #Composite var
#dataloader.AddSpectator("tagged_mass")

#signalWeight     = 0.6774
#backgroundWeight = 0.8138

signalWeight     = 1.0        #MC/signal
backgroundWeight = 1.0        #dataset sidebands/background

dataloader.AddSignalTree( signal, signalWeight )
dataloader.AddBackgroundTree( background, backgroundWeight )

sigCutEvents = signal.GetEntries(cuts)
bkgCutEvents = background.GetEntries(cutb)

#train 70% of the events, test 30%
sigTrain = int(sigCutEvents * 0.7)   
bkgTrain = int(bkgCutEvents * 0.7) 


dataloader.PrepareTrainingAndTestTree( mycutS, mycutB, "nTrain_Signal=%i:nTrain_Background=%i:SplitMode=Random:NormMode=NumEvents:!V" % (sigTrain, bkgTrain) )


# Book methods
factory.BookMethod(dataloader, TMVA.Types.kBDT, "BDTs",
                "!H:!V:NTrees=250:MinNodeSize=2.5%:MaxDepth=5:BoostType=AdaBoost:VarTransform=Decorrelate:SeparationType=GiniIndex:nCuts=30")


# Run training,BDT test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods() #add full dataset


# Plot ROC Curves AND OTHERS
roc = factory.GetROCCurve(dataloader)
roc.SaveAs('dataset_' + input + '/plots/ROC_ClassificationBDT.png')


#close the output file
output.Close()

#open the GUI interface
#TMVA.TMVAGui(outfname)
