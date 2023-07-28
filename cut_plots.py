import numpy as np
import uproot
import os.path
import matplotlib.pyplot as plt
import sys

# run it as 'python3 filename.py var_name log'

# Open root file
dir = "/lstore/cms/boletti/ntuples/"
data_file = "2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
mc_file = "MC_JPSI_2018_preBDT_Nov21.root"

data = uproot.open(dir + data_file)
mc = uproot.open(dir + mc_file)

# Access TTree
dataTree = data["ntuple"]
mcTree = mc["ntuple"]

# Get the input argument from the command line
inputBranch = sys.argv[1]

# Access desired TBranch and turn into array
dataBranch = dataTree[inputBranch].array()
mcBranch = mcTree[inputBranch].array()

# Tagged B0 mass CUTS for pure Bkg (data) and pure signal (MC)
data_cut = dataBranch[(dataTree['tagged_mass'].array()>5.44) | (dataTree['tagged_mass'].array()<5.10)]
mc_cut = mcBranch[(mcTree['tagged_mass'].array()<=5.44) & (mcTree['tagged_mass'].array()>=5.10)]


#plt.hist(data_cut,bins=50)
#plt.hist(mc_cut,bins=50)

bin=150

# Compute the histograms
if (inputBranch == "tagged_mass"):
    data_hist, data_bin_edges = np.histogram(data_cut, bins=bin)
    mc_hist, mc_bin_edges = np.histogram(mc_cut, bins=bin)
else:
    data_hist, data_bin_edges = np.histogram(data_cut, bins=bin)
    mc_hist, mc_bin_edges = np.histogram(mc_cut, bins=data_bin_edges)

# Normalize the histogram bin counts
data_hist = data_hist / np.sum(data_hist)
mc_hist = mc_hist / np.sum(mc_hist)

# Compute the bin centers for bar plotting
data_bin_centers = (data_bin_edges[:-1] + data_bin_edges[1:]) / 2
mc_bin_centers = (mc_bin_edges[:-1] + mc_bin_edges[1:]) / 2

# Plot the histograms as filled histograms
plt.bar(data_bin_centers, data_hist, width=np.diff(data_bin_edges), color='red', alpha=0.5, label='B') # Data (Bkg)
plt.bar(mc_bin_centers, mc_hist, width=np.diff(mc_bin_edges), color='blue', alpha=0.5, label='S') # MC (Signal)


# Plot the histograms as step histograms
#plt.step(data_bin_edges[:-1], data_hist, where='post', color='red', label='Data (Bkg)')
#plt.step(mc_bin_edges[:-1], mc_hist, where='post', color='blue', label='MC (Signal)')


# Add labels and title
plt.xlabel(inputBranch)
plt.ylabel("Events (Norm)")
#plt.title("Histogram of " + inputBranch)
plt.legend()


# Set the y-axis to a logarithmic scale
if (sys.argv[2] == "log"):
    plt.yscale('log')


# Set the x-axis range
#plt.xlim(0, 3)



# Save the plot as a PNG file
plt.savefig(inputBranch + "_h1.png")

plt.show()