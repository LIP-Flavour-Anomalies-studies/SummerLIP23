import numpy as np
import uproot
import os.path
import matplotlib.pyplot as plt


# File for tagged_mass cuts
# Run it as 'python3 filename.py'

# Open root file
dir = "/lstore/cms/boletti/ntuples/"
data_file = "2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
mc_file = "MC_JPSI_2018_preBDT_Nov21.root"

data = uproot.open(dir + data_file)
mc = uproot.open(dir + mc_file)

# Access TTree
dataTree = data["ntuple"]
mcTree = mc["ntuple"]

# Get the input argument
inputBranch = "tagged_mass"

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
plt.bar(data_bin_centers, data_hist_normalized, width=np.diff(data_bin_edges), color='red', alpha=0.5, label='Data (B+S)')
plt.bar(mc_bin_centers, mc_hist_normalized, width=np.diff(mc_bin_edges), color='blue', alpha=0.5, label='MC (S)')


# Add labels and title
plt.xlabel("B-candidate mass (GeV)")
plt.ylabel("Events (Norm)")
#plt.title("Histogram of " + inputBranch)
plt.legend()

# Set the x-axis range
plt.xlim(5.0, 5.6)

# Add vertical lines at specific x-values
x_values = [5.14, 5.41]
for x in x_values:
    plt.axvline(x, color='green', linestyle='--', linewidth=1)


# Save the plot as a PNG file
plt.savefig(inputBranch + "_cut.png")

plt.show()


