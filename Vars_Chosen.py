import ROOT
from ROOT import TFile
from ROOT import TH1F
import numpy as np


#Run "python3 Vars_Chosen.py"

# Input PATH
dir = "/home/t3cms/joaobss/SummerLIP23/ROOT_files/"
data_file = "Data2018_CompositeVars.root"
mc_file = "MC_JPSI_2018_CompositeVars.root"

# Output PATH
dir_out = "/home/t3cms/joaobss/SummerLIP23/Variables_Plots/"


# Open the ROOT files and access the TTree for data and MC
data = TFile.Open(dir + data_file)
mc = TFile.Open(dir + mc_file)
dataTree = data.Get("ntuple")
mcTree = mc.Get("ntuple")



# Variable bVtxCL
# Create a TCanvas
c = ROOT.TCanvas("c", "Histogram Canvas", 800, 600)

# Create and plot the histogram for bVtxCL
hVtx1 = ROOT.TH1F("hVtx1", "", 100, 0, 1) # Data
hVtx2 = ROOT.TH1F("hVtx2", "", 100, 0, 1) # MC

# Draw the bVtxCL data and fill it in a histogram
dataTree.Draw("bVtxCL>>hVtx1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("bVtxCL>>hVtx2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

# Normalize
hVtx1.Scale(1./hVtx1.Integral())
hVtx2.Scale(1./hVtx2.Integral())

# Set Background to red
hVtx1.SetLineColor(ROOT.kRed)

# Remove the statistics box from the canvas
hVtx1.SetStats(0)
hVtx2.SetStats(0)

# Draw the histogram
hVtx1.Draw("HIST")

# Set the y-axis range to include both histograms
maximum = max(hVtx1.GetMaximum(), hVtx2.GetMaximum())
hVtx1.SetMaximum(1.1 * maximum)

# Draw hVtx2 on top of hVtx1
hVtx2.Draw("HIST same")

# Create a legend and add entries for histo1 and histo2
legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hVtx1, "B", "l")  # "B" label for hVtx1 with line (l)
legend.AddEntry(hVtx2, "S", "l")  # "S" label for hVtx2 with line (l)
legend.SetBorderSize(1)  # Remove border around the legend

# Center the information in the legend and reduce the space after the text
legend.SetTextAlign(22)   # Center-aligned text
legend.SetTextSize(0.04)  # Adjust text size

legend.Draw()

hVtx1.SetXTitle("Vertex CL")

c.SetLogy(0)

c.SaveAs(dir_out + "bVtxCL_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var bCosAlphaBS
hc1 = ROOT.TH1F("hc1", "", 100, 0.8, 1.) #Data
hc2 = ROOT.TH1F("hc2", "", 100, 0.8, 1.) #MC

dataTree.Draw("bCosAlphaBS>>hc1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("bCosAlphaBS>>hc2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hc1.Scale(1./hc1.Integral())
hc2.Scale(1./hc2.Integral())

hc1.SetLineColor(ROOT.kRed)

hc1.SetStats(0)
hc2.SetStats(0)

hc2.Draw("HIST")
maximum = max(hc1.GetMaximum(), hc2.GetMaximum())
hc2.SetMaximum(1.2 * maximum)
hc1.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hc1, "B", "l")  # "B" label 
legend.AddEntry(hc2, "S", "l")  # "S" label 
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hc2.SetXTitle("Cos(#alpha )")

c.SetLogy()
c.SaveAs(dir_out + "bCosAlphaBS_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var bLBS
hbLBS1 = ROOT.TH1F("hbLBS1", "", 100, 0, 1.2) #Data
hbLBS2 = ROOT.TH1F("hbLBS2", "", 100, 0, 1.2) #MC

dataTree.Draw("bLBS>>hbLBS1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("bLBS>>hbLBS2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hbLBS1.Scale(1./hbLBS1.Integral())
hbLBS2.Scale(1./hbLBS2.Integral())

hbLBS1.SetLineColor(ROOT.kRed)

hbLBS1.SetStats(0)
hbLBS2.SetStats(0)

hbLBS1.Draw("HIST")
maximum = max(hbLBS1.GetMaximum(), hbLBS2.GetMaximum())
hbLBS1.SetMaximum(1.1 * maximum)
hbLBS2.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hbLBS1, "B", "l")  # "B" label 
legend.AddEntry(hbLBS2, "S", "l")  # "S" label 
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hbLBS1.SetXTitle("Length from Beam Spot (cm)")

c.SetLogy(0)
c.SaveAs(dir_out + "bLBS_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var bLBSE
hbLBSE1 = ROOT.TH1F("hbLBSE1", "", 150, 0, 0.08) #Data
hbLBSE2 = ROOT.TH1F("hbLBSE2", "", 150, 0, 0.08) #MC

dataTree.Draw("bLBSE>>hbLBSE1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("bLBSE>>hbLBSE2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hbLBSE1.Scale(1./hbLBSE1.Integral())
hbLBSE2.Scale(1./hbLBSE2.Integral())

hbLBSE1.SetLineColor(ROOT.kRed)

hbLBSE1.SetStats(0)
hbLBSE2.SetStats(0)

hbLBSE1.Draw("HIST")
maximum = max(hbLBSE1.GetMaximum(), hbLBSE2.GetMaximum())
hbLBSE1.SetMaximum(1.3 * maximum)
hbLBSE2.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hbLBSE1, "B", "l")  # "B" label 
legend.AddEntry(hbLBSE2, "S", "l")  # "S" label 
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hbLBSE1.SetXTitle("Length from Beam Spot Error (cm)")

c.SetLogy()
c.SaveAs(dir_out + "bLBSE_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var kstTrkmDCABS
hkm1 = ROOT.TH1F("hkm1", "", 150, -1, 1) #Data
hkm2 = ROOT.TH1F("hkm2", "", 150, -1, 1) #MC

dataTree.Draw("kstTrkmDCABS>>hkm1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("kstTrkmDCABS>>hkm2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hkm1.Scale(1./hkm1.Integral())
hkm2.Scale(1./hkm2.Integral())

hkm1.SetLineColor(ROOT.kRed)

hkm1.SetStats(0)
hkm2.SetStats(0)

hkm1.Draw("HIST")
maximum = max(hkm1.GetMaximum(), hkm2.GetMaximum())
hkm1.SetMaximum(1.2 * maximum)
hkm2.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hkm1, "B", "l")  # "B" label
legend.AddEntry(hkm2, "S", "l")  # "S" label
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hkm1.SetXTitle("K* DCA from BS negative track (cm)")

c.SetLogy()
c.SaveAs(dir_out + "kstTrkmDCABS_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var kstTrkpDCABS
hkp1 = ROOT.TH1F("hkp1", "", 150, -1, 1) #Data
hkp2 = ROOT.TH1F("hkp2", "", 150, -1, 1) #MC

dataTree.Draw("kstTrkpDCABS>>hkp1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("kstTrkpDCABS>>hkp2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hkp1.Scale(1./hkp1.Integral())
hkp2.Scale(1./hkp2.Integral())

hkp1.SetLineColor(ROOT.kRed)

hkp1.SetStats(0)
hkp2.SetStats(0)

hkp1.Draw("HIST")
maximum = max(hkp1.GetMaximum(), hkp2.GetMaximum())
hkp1.SetMaximum(1.2 * maximum)
hkp2.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hkp1, "B", "l")  # "B" label
legend.AddEntry(hkp2, "S", "l")  # "S" label
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hkp1.SetXTitle("K* DCA from BS positive track (cm)")

c.SetLogy()
c.SaveAs(dir_out + "kstTrkpDCABS_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var mu1Pt 
hmu1_1 = ROOT.TH1F("hmu1_1", "", 150, 0, 50) #Data
hmu1_2 = ROOT.TH1F("hmu1_2", "", 150, 0, 50) #MC

dataTree.Draw("mu1Pt>>hmu1_1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("mu1Pt>>hmu1_2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hmu1_1.Scale(1./hmu1_1.Integral())
hmu1_2.Scale(1./hmu1_2.Integral())

hmu1_1.SetLineColor(ROOT.kRed) # Bkg

hmu1_1.SetStats(0)
hmu1_2.SetStats(0)

hmu1_2.Draw("HIST")
maximum = max(hmu1_1.GetMaximum(), hmu1_2.GetMaximum())
hmu1_2.SetMaximum(1.1 * maximum)
hmu1_1.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hmu1_1, "B", "l")  # "B" label
legend.AddEntry(hmu1_2, "S", "l")  # "S" label
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hmu1_2.SetXTitle("Muon1 P_{T} (GeV)")

c.SetLogy(0)
c.SaveAs(dir_out + "mu1Pt_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var mu2Pt 
hmu2_1 = ROOT.TH1F("hmu2_1", "", 150, 0, 22) #Data
hmu2_2 = ROOT.TH1F("hmu2_2", "", 150, 0, 22) #MC

dataTree.Draw("mu2Pt>>hmu2_1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("mu2Pt>>hmu2_2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hmu2_1.Scale(1./hmu2_1.Integral())
hmu2_2.Scale(1./hmu2_2.Integral())

hmu2_1.SetLineColor(ROOT.kRed) # Bkg

hmu2_1.SetStats(0)
hmu2_2.SetStats(0)

hmu2_2.Draw("HIST")
maximum = max(hmu2_1.GetMaximum(), hmu2_2.GetMaximum())
hmu2_2.SetMaximum(1.1 * maximum)
hmu2_1.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hmu2_1, "B", "l")  # "B" label
legend.AddEntry(hmu2_2, "S", "l")  # "S" label
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hmu2_2.SetXTitle("Muon2 P_{T} (GeV)")

c.SetLogy(0)
c.SaveAs(dir_out + "mu2Pt_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var kstTrkmPt 
hkmpt1 = ROOT.TH1F("hkmpt1", "", 50, 0, 20) #Data
hkmpt2 = ROOT.TH1F("hkmpt2", "", 50, 0, 20) #MC

dataTree.Draw("kstTrkmPt>>hkmpt1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("kstTrkmPt>>hkmpt2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hkmpt1.Scale(1./hkmpt1.Integral())
hkmpt2.Scale(1./hkmpt2.Integral())

hkmpt1.SetLineColor(ROOT.kRed)

hkmpt1.SetStats(0)
hkmpt2.SetStats(0)

hkmpt1.Draw("HIST")
maximum = max(hkmpt1.GetMaximum(), hkmpt2.GetMaximum())
hkmpt1.SetMaximum(1.1 * maximum)
hkmpt2.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hkmpt1, "B", "l")  # "B" label
legend.AddEntry(hkmpt2, "S", "l")  # "S" label
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hkmpt1.SetXTitle("K* P_{T} negative track (GeV)")

c.SetLogy(0)
c.SaveAs(dir_out + "kstTrkmPt_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var kstTrkpPt 
hkppt1 = ROOT.TH1F("hkppt1", "", 50, 0, 20) #Data
hkppt2 = ROOT.TH1F("hkppt2", "", 50, 0, 20) #MC

dataTree.Draw("kstTrkpPt>>hkppt1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("kstTrkpPt>>hkppt2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hkppt1.Scale(1./hkppt1.Integral())
hkppt2.Scale(1./hkppt2.Integral())

hkppt1.SetLineColor(ROOT.kRed)

hkppt1.SetStats(0)
hkppt2.SetStats(0)

hkppt1.Draw("HIST")
maximum = max(hkppt1.GetMaximum(), hkppt2.GetMaximum())
hkppt1.SetMaximum(1.1 * maximum)
hkppt2.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hkppt1, "B", "l")  # "B" label
legend.AddEntry(hkppt2, "S", "l")  # "S" label
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hkppt1.SetXTitle("K* P_{T} positive track (GeV)")

c.SetLogy(0)
c.SaveAs(dir_out + "kstTrkpPt_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var tagB0
htag1 = ROOT.TH1F("htag1", "", 10, -0.5, 1.5) #Data
htag2 = ROOT.TH1F("htag2", "", 10, -0.5, 1.5) #MC

dataTree.Draw("tagB0>>htag1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("tagB0>>htag2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

htag1.Scale(1./htag1.Integral())
htag2.Scale(1./htag2.Integral())

htag1.SetLineColor(ROOT.kRed)
htag1.SetLineStyle(2)  # Use integer value 2 for dashed style

htag1.SetStats(0)
htag2.SetStats(0)

htag1.Draw("HIST")
maximum = max(htag1.GetMaximum(), htag2.GetMaximum())
htag1.SetMaximum(1.1 * maximum)
htag2.Draw("HIST same")

legend = ROOT.TLegend(0.425, 0.88, 0.575, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(htag1, "B", "l")  # "B" label
legend.AddEntry(htag2, "S", "l")  # "S" label
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

htag1.SetXTitle("B-candidate tag")

c.SetLogy(0)
c.SaveAs(dir_out + "tagB0_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var tagged_mass
hmass1 = ROOT.TH1F("hmass1", "", 150, 5.0, 5.6) #Data
hmass2 = ROOT.TH1F("hmass2", "", 150, 5.0, 5.6) #MC

dataTree.Draw("tagged_mass>>hmass1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("tagged_mass>>hmass2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hmass1.Scale(1./hmass1.Integral())
hmass2.Scale(1./hmass2.Integral())

hmass1.SetLineColor(ROOT.kRed)

hmass1.SetStats(0)
hmass2.SetStats(0)

hmass1.Draw("HIST")
maximum = max(hmass1.GetMaximum(), hmass2.GetMaximum())
hmass1.SetMaximum(1.1 * maximum)
hmass2.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hmass1, "B", "l")  # "B" label
legend.AddEntry(hmass2, "S", "l")  # "S" label
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hmass1.SetXTitle("B-candidate mass (GeV)")

c.SetLogy(0)
c.SaveAs(dir_out + "tagged_mass_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Composite Vars
########################################################################################################

# Var bLBSRatio (Composite)
hRatio1 = ROOT.TH1F("hRatio1", "", 100, 0, 150) #Data
hRatio2 = ROOT.TH1F("hRatio2", "", 100, 0, 150) #MC

dataTree.Draw("bLBSRatio>>hRatio1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("bLBSRatio>>hRatio2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hRatio1.Scale(1./hRatio1.Integral())
hRatio2.Scale(1./hRatio2.Integral())

hRatio1.SetLineColor(ROOT.kRed)

hRatio1.SetStats(0)
hRatio2.SetStats(0)

hRatio1.Draw("HIST")
maximum = max(hRatio1.GetMaximum(), hRatio2.GetMaximum())
hRatio1.SetMaximum(1.1 * maximum)
hRatio2.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hRatio1, "B", "l")  # "B" label 
legend.AddEntry(hRatio2, "S", "l")  # "S" label 
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hRatio1.SetXTitle("bLBS Ratio")

c.SetLogy(0)
c.SaveAs(dir_out + "bLBSRatio_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Var bCosRatio (Composite)
hcRatio1 = ROOT.TH1F("hcRatio1", "", 100, 0, 0.05) #Data
hcRatio2 = ROOT.TH1F("hcRatio2", "", 100, 0, 0.05) #MC

dataTree.Draw("bCosRatio>>hcRatio1", "tagged_mass < 5.14 || tagged_mass > 5.41") # Background
mcTree.Draw("bCosRatio>>hcRatio2", "tagged_mass > 5.14 && tagged_mass < 5.41")   # Signal

hcRatio1.Scale(1./hcRatio1.Integral())
hcRatio2.Scale(1./hcRatio2.Integral())

hcRatio1.SetLineColor(ROOT.kRed)

hcRatio1.SetStats(0)
hcRatio2.SetStats(0)

hcRatio1.Draw("HIST")
maximum = max(hcRatio1.GetMaximum(), hcRatio2.GetMaximum())
hcRatio1.SetMaximum(1.2 * maximum)
hcRatio2.Draw("HIST same")

legend = ROOT.TLegend(0.73, 0.88, 0.88, 0.78)  # (x1, y1, x2, y2) coordinates for the legend box
legend.AddEntry(hcRatio1, "B", "l")  # "B" label 
legend.AddEntry(hcRatio2, "S", "l")  # "S" label 
legend.SetBorderSize(1)
legend.SetTextAlign(22)
legend.SetTextSize(0.04)
legend.Draw()

hcRatio1.SetXTitle("Cos(#alpha ) Ratio")

c.SetLogy()
c.SaveAs(dir_out + "bCosRatio_histo.png")

# Clear the canvas for the next histogram
c.Clear()





# Clean up
data.Close()
mc.Close()