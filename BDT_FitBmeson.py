import ROOT
from ROOT import TFile
from ROOT import TH1F
from ROOT import RooRealVar
from ROOT import RooDataHist
from ROOT import RooArgSet
from ROOT import RooDataSet
from ROOT import RooPlot
from ROOT import RooExponential
from ROOT import RooGaussian
from ROOT import RooCBShape
from ROOT import RooArgList
from ROOT import RooAddPdf
from ROOT import TLegend
from ROOT import RooFit
from ROOT import TLatex
from ROOT import RooChi2Var
from ROOT import RooFormulaVar
import numpy as np
import math


#PATH
dir = "/user/j/joaobss/SummerLIP23/dataset_norm1C/results/rootfiles/"
data_file = "selected_taggedMass.root"


#Setting the mass limits and binning
mmin = 5.0
mmax = 5.6
bins=120

#Opening data ROOT file
root_file = TFile.Open(dir+data_file)
tree = root_file.Get("ntuple")


#Create de B-candidate mass histogram
hmass = TH1F("hmass",";B-candidate mass (GeV)", bins, mmin, mmax)



def get_value(line):
    value=""
    
    for c in line:
        if c.isdigit() or c=="." or c=="-":
            value += c
        else:
            break
    return value


# Get the BDT score from the .txt file
file=open("/user/j/joaobss/SummerLIP23/dataset_norm1C/plots/BDT_fom.txt","r")
str1="BDT score"

# Get the BDT score from the .txt file
for line in file:
   if str1 in line:
      bdt_cut=get_value(line)


#print(bdt_cut)
   



#Fill it in
for index, event in enumerate(tree):
   if event.bdt_score > float(bdt_cut):
      hmass.Fill(event.tagged_mass)
      #if index > tree.GetEntries(): break


#list to store needed values
v=[]



def bmeson_fit():

   #Create a Mass variable that RooFit can use, and importing the relevant dataset
   mass = RooRealVar("mass", "B-candidate mass", mmin, mmax, "GeV")
   args = RooArgList(mass)
   dh = RooDataHist("dh", "dh", args, hmass)

   #Define a background model (exponential) and its parameters
   Lambda = RooRealVar("Lambda", "lambda", -0.3, -4.0, 0.0)
   background = RooExponential("background", "background", mass, Lambda)

   #Define a signal model (Gaussian + CB) and its parameters

   #Gaussian model
   mean = RooRealVar("mean", "mean", 0.5*(mmin+mmax), mmin, mmax)
   sigma1 = RooRealVar("sigma1", "sigma1", 0.1*(mmax-mmin),0.,0.5*(mmax-mmin))
   signal1 = RooGaussian("signal1", "signal1", mass, mean, sigma1)

   #Crystal Ball model
   sigma2 = RooRealVar("sigma2", "sigma2", 0.01 ,0., 0.5)
   alpha2 = RooRealVar("alpha2", "alpha2",  1.5, 0.3, 15.)
   n2 = RooRealVar("n2", "n2", 1.2, 0.3, 15.)
   signal2 = RooCBShape("signal2", "signal2", mass, mean, sigma2, alpha2, n2)


   #Define variables for the number of signal and background events
   n_signal_initial = 0.8*dh.sumEntries()
   n_back_initial = 0.2*dh.sumEntries()
   n_signal = RooRealVar("n_signal","n_signal",n_signal_initial,0.,dh.sumEntries())
   n_back = RooRealVar("n_back","n_back",n_back_initial,0.,dh.sumEntries())


   #Sum the signal components into a composite signal p.d.f. (gaussian + CB) with the form: c x gaussian + (1-c) x CB
   sig1_frac = RooRealVar("sig1_frac", "sig1_frac", 0.8, 0., 1.)
   sig2_frac = RooFormulaVar("sig2_frac", "1.0 - @0", RooArgList(sig1_frac))

   # Define the coefficient list containing both sig1_frac and sig2_frac
   coeff = RooArgList(sig1_frac, sig2_frac)

   # Sum the signal components into a composite signal p.d.f. (gaussian + CB)
   signal = RooAddPdf("signal", "signal", RooArgList(signal1,signal2), coeff)


   #Sum signal and background models
   model = RooAddPdf("model", "model", RooArgList(signal, background), RooArgList(n_signal, n_back))

   #Perform the fit
   model.fitTo(dh)



   #Plot the fit and display results
   
   frame = mass.frame()
   frame.SetTitle("")

   dh.plotOn(frame,RooFit.Name("dh"))
   model.plotOn(frame,RooFit.Name("modelSig"),RooFit.Components("signal"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(ROOT.kRed))
   model.plotOn(frame,RooFit.Name("modelBkg"),RooFit.Components("background"),RooFit.LineStyle(ROOT.kDashed),RooFit.LineColor(ROOT.kGreen))
   model.plotOn(frame,RooFit.Name("model"))

   c = ROOT.TCanvas("c","",800,600)
   frame.Draw()


   #Total average and std dev. for Gaussian + CB fit
   Mean = mean.getVal()
   Sigma = math.sqrt(sig1_frac.getVal() * sigma1.getVal()**2 + (1 - sig1_frac.getVal()) * sigma2.getVal()**2)

   #Sideband edges for 3.5sigma
   sideband_edge1 = Mean - 3.5 * Sigma
   sideband_edge2 = Mean + 3.5 * Sigma


   #Draw sideband boundaries
   binNumber1 = hmass.FindBin(sideband_edge1)

   l1 = ROOT.TLine(sideband_edge1, 0., sideband_edge1, hmass.GetBinContent(binNumber1)+300)
   l2 = ROOT.TLine(sideband_edge2, 0., sideband_edge2, hmass.GetBinContent(binNumber1)+300)

   #l1.DrawColorTable(15)
   l1.SetLineWidth(5)
   l1.SetLineStyle(2)
   #l2.DrawColorTable(15)
   l2.SetLineWidth(5)
   l2.SetLineStyle(2)

   l1.Draw()
   l2.Draw()


   #Draw a caption
   legend = TLegend(0.65,0.6,0.88,0.85)
   legend.SetBorderSize(0)
   legend.SetTextFont(40)
   legend.SetTextSize(0.03)
   legend.AddEntry(frame.findObject("dh"),"Data","1pe")
   legend.AddEntry(frame.findObject("modelBkg"),"Background fit","1pe")
   legend.AddEntry(frame.findObject("modelSig"),"Signal fit","1pe")
   legend.AddEntry(frame.findObject("model"),"Global fit","1pe")
   legend.Draw()


   #Display info and fit results
   L = TLatex()
   L.SetNDC()
   L.SetTextSize(0.03)
   L.DrawLatex(0.15,0.85,ROOT.Form("Y_{s}: %.0f #pm %.0f events" % (n_signal.getVal(),n_signal.getError())))
   L.DrawLatex(0.15,0.80,ROOT.Form("Y_{b}: %.0f #pm %.0f events" % (n_back.getVal(),n_signal.getError())))
   L.DrawLatex(0.15,0.75,ROOT.Form("#lambda: %.3f #pm %.3f GeV^{-1}" % (Lambda.getVal(),Lambda.getError())))
   L.DrawLatex(0.15,0.70,ROOT.Form("coeff1: %.3f #pm %.3f" % (sig1_frac.getVal(),sig1_frac.getError())))
   L.DrawLatex(0.15,0.65,ROOT.Form("mean: %.1f #pm %.1f MeV" % (mean.getVal()*1000,mean.getError()*1000)))
   L.DrawLatex(0.15,0.60,ROOT.Form("#sigma_{1}: %.1f #pm %.1f MeV" % (sigma1.getVal()*1000,sigma1.getError()*1000)))
   L.DrawLatex(0.15,0.55,ROOT.Form("#sigma_{2}: %.1f #pm %.1f MeV" % (sigma2.getVal()*1000,sigma2.getError()*1000)))
   L.DrawLatex(0.15,0.50,ROOT.Form("#alpha_{2}: %.3f #pm %.3f" % (alpha2.getVal(),alpha2.getError())))
   L.DrawLatex(0.15,0.45,ROOT.Form("n2: %.2f #pm %.2f" % (n2.getVal(),n2.getError())))
   chi = RooChi2Var("chi","chi^2",model,dh)
   variables = 8 # This is the number of free parameters in our model

   L.DrawLatex(0.15,0.40,ROOT.Form("#chi^{2}/ndf = %.2f" % frame.chiSquare(variables)))


   #Draw the last line of text and set its color to red
   L.SetTextColor(ROOT.kRed)
   L.DrawLatex(0.15, 0.35, "Gauss + CB")


   c.Draw()
   c.SaveAs("/user/j/joaobss/SummerLIP23/dataset_norm1C/plots/BDT_B0Fit_JPsi.png")


   v.append(sideband_edge1)
   v.append(sideband_edge2)
   v.append(n_back.getVal())
   v.append(n_signal.getVal())
 





def write_results():

   sideband_edge1, sideband_edge2, n_back_val, n_signal_val = v[0:4]

   #Write *.txt file
   f = open("/user/j/joaobss/SummerLIP23/dataset_norm1C/plots/BDT_B0Fit_JPsi.txt", "w")
   f.write(f"""{sideband_edge1}  #left sideband edge
{sideband_edge2}  #right sideband edge
{n_signal_val}  #signal yield
{n_back_val}  #background yield""")
   f.close()






bmeson_fit()
write_results()
