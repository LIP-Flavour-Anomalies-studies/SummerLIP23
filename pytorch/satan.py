import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utilsnew import prepdata, RegressionDataset
import numpy as np
import uproot

# Define the neural network
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return np.array(predictions), np.array(targets)


dir = "/lstore/cms/boletti/ntuples/"
filename = "2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
filename_mc = "MC_JPSI_2018_preBDT_Nov21.root"

data = uproot.open(dir + filename)
data_mc = uproot.open(dir + filename_mc)

# Load your data
data, labels, columns = prepdata(data, data_mc)

# Create a dataset and a DataLoader
dataset = RegressionDataset(data, labels)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Create the model
input_size = len(columns)
model = RegressionModel(input_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model (you can split your data into training and testing sets)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# predictions, targets = evaluate_model(model, test_loader)
