import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
from utilsnewer import prepdata, RegressionDataset
import numpy as np
import uproot

# Define the neural network for regression
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
    return torch.Tensor(predictions), torch.Tensor(targets)

def main():
    dir = "/lstore/cms/boletti/ntuples/"
    filename = "2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
    filename_mc = "MC_JPSI_2018_preBDT_Nov21.root"

    data = uproot.open(dir + filename)
    data_mc = uproot.open(dir + filename_mc)

    # Your data loading and preprocessing logic
    x, y, variable_names = prepdata(data, data_mc,"bLBS")

    # Create a dataset and DataLoader
    dataset = RegressionDataset(x, y)

    # Split the dataset into train, validation, and test sets
    train_set, test_set, val_set = random_split(dataset, [0.5, 0.25, 0.25])

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    dev_X, dev_y = val_set[:][0], val_set[:][1]
    test_X, test_y = test_set[:][0], test_set[:][1]

    # Initialize the model
    input_size = x.shape[1]
    model = RegressionModel(input_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_dataloader, criterion, optimizer, num_epochs=10)

    # Evaluate on the test set
    predictions, targets = evaluate_model(model, DataLoader(test_set, batch_size=64, shuffle=False))

    # Print predictions for each variable
    for variable_name, prediction in zip(variable_names, predictions.T):
        print(f"{variable_name}: {', '.join(map(str, prediction.numpy()))}\n")


if __name__ == "__main__":
    main()
