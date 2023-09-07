import pandas as pd
import uproot
import matplotlib.pyplot as plt
import numpy as np
import os
import utils_fom
import FOM_withlimits

dir="/lstore/cms/boletti/ntuples/"
filename="2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
filename2="MC_JPSI_2018_preBDT_Nov21.root"

data = uproot.open(dir+filename)
data_mc=uproot.open(dir+filename2)

Tree=data["ntuple"]
Tree_mc=data_mc["ntuple"]

v="bLBS"

left_edge,right_edge,fb,fs=utils_fom.get_factors("/user/u/u23madalenablanc/flavour-anomalies/SummerLIP23/Fit_Results/B0Fit_3.5sigma_results.txt")

sel_b="(tagged_mass<" + left_edge+ ") | (tagged_mass>" +right_edge + ")"
sel_s="(tagged_mass>" + left_edge+ ") & (tagged_mass<" +right_edge + ")"

background=Tree.arrays(v,cut=sel_b,library="pd")
signal=Tree_mc.arrays(v,cut=sel_s,library="pd")


minv=0
maxv=1.2

show_upper=0
legend="bLBS"

FOM_withlimits.calc_fom(v,signal,background,minv,maxv,fs,fb,show_upper,legend)