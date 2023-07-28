import numpy as np
import uproot
import os.path
import matplotlib.pyplot as plt
import sys

# File for tagged_mass cuts
# run it as 'python3 filename.py tagged_mass'

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

bin=150

# Compute the histograms
data_hist, data_bin_edges = np.histogram(dataBranch, bins=bin)
mc_hist, mc_bin_edges = np.histogram(mcBranch, bins=data_bin_edges)

# Normalize the histogram bin counts
data_hist_normalized = data_hist / np.sum(data_hist)
mc_hist_normalized = mc_hist / np.sum(mc_hist)

# Compute the bin centers for bar plotting
data_bin_centers = (data_bin_edges[:-1] + data_bin_edges[1:]) / 2
mc_bin_centers = (mc_bin_edges[:-1] + mc_bin_edges[1:]) / 2

# Plot the histograms as filled histograms
plt.bar(data_bin_centers, data_hist_normalized, width=np.diff(data_bin_edges), color='red', alpha=0.5, label='Data (Bkg+S)')
plt.bar(mc_bin_centers, mc_hist_normalized, width=np.diff(mc_bin_edges), color='blue', alpha=0.5, label='MC (Signal)')


# Add labels and title
plt.xlabel(inputBranch)
plt.ylabel("Events (Norm)")
#plt.title("Histogram of " + inputBranch)
plt.legend()

# Set the x-axis range
plt.xlim(4.98, 5.6)

# Add vertical lines at specific x-values
x_values = [5.10, 5.44]
for x in x_values:
    plt.axvline(x, color='green', linestyle='--', linewidth=1)


# Save the plot as a PNG file
plt.savefig(inputBranch + "_cut.png")

plt.show()


