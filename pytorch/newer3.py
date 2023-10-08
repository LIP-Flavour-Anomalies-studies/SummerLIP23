import argparse
import torch
import os.path
from subprocess import call
import uproot
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, random_split
import utilsnew as utils  # Import your utility module for data preprocessing


class FeedforwardNetwork(nn.Module):
    def __init__(self, n_features, hidden_size, layers, dropout):
        super().__init__()

        self.first_layer = nn.Linear(n_features, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)  # Regression output for a single variable

        drop = nn.Dropout(p=dropout)

        self.order = nn.Sequential(self.first_layer, nn.ReLU(), drop)
        for _ in range(layers - 1):
            self.order.append(self.hidden_layers[_])
            self.order.append(nn.ReLU())
            self.order.append(drop)
        self.order.append(self.output_layer)

    def forward(self, x):
        x = self.order(x)
        return x


def train_batch(X, y, model, optimizer, criterion):
    optimizer.zero_grad()
    y_pred = model(X)

    # Ensure the target values have the same data type as the predictions
    y = y.float()

    # If the target tensor has only one dimension, expand it to match the shape of y_pred
    if y.dim() == 1:
        y = y.unsqueeze(1).expand_as(y_pred)

    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    return model(X)


def evaluate(model, X, y):
    model.eval()
    predictions = predict(model, X)
    mae = metrics.mean_absolute_error(y.numpy(), predictions.detach().numpy())
    rmse = metrics.mean_squared_error(y.numpy(), predictions.detach().numpy(), squared=False)
    model.train()
    return mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-hidden_size', type=int, default=64)
    parser.add_argument('-layers', type=int, default=3)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    opt = parser.parse_args()

    dir = "/lstore/cms/boletti/ntuples/"
    filename = "2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
    filename_mc = "MC_JPSI_2018_preBDT_Nov21.root"

    data = uproot.open(os.path.join(dir, filename))
    data_mc = uproot.open(os.path.join(dir, filename_mc))

    x, y, variable_names = utils.prepdata(data, data_mc)
    dataset = utils.RegressionDataset(x, y)

    print(variable_names)

    # Determine the number of variables dynamically
    n_vars = len(variable_names)

    # Split the dataset
    train_set, test_set, val_set = random_split(dataset, [0.5, 0.25, 0.25])
    train_dataloader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

    dev_X, dev_y = val_set[:][0], val_set[:][1]
    test_X, test_y = test_set[:][0], test_set[:][1]

    n_feats = dataset.X.shape[1]

    # Initialize a list to store models and their corresponding optimizers
    models = []
    optimizers = []

    for variable_index in range(n_vars):
        # Create a model for each variable
        model = FeedforwardNetwork(n_feats, opt.hidden_size, opt.layers, opt.dropout)
        models.append(model)

        # Create an optimizer for each model
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)
        optimizers.append(optimizer)

    # Get a loss criterion
    criterion = nn.MSELoss()

    # Training loop for each model
    for variable_index, (model, optimizer) in enumerate(zip(models, optimizers)):
        val_maes = []

        for epoch in range(1, opt.epochs + 1):
            print(f'Training epoch {epoch} for Variable {variable_index + 1}')
            for X_batch, y_batch in train_dataloader:
                # Use only the target variable corresponding to the model
                y_batch_variable = y_batch[:, variable_index].unsqueeze(1)

                loss = train_batch(X_batch, y_batch_variable, model, optimizer, criterion)
                # Assuming you have a function train_batch similar to the one you provided

            # Validation MAE
            model.eval()
            val_predictions = predict(model, dev_X)
            val_mae = mean_absolute_error(dev_y[:, variable_index].detach().numpy(),
                                        val_predictions.detach().numpy())
            val_maes.append(val_mae)
            model.train()

        # Store predictions for this variable after training
        val_predictions_final = predict(model, dev_X)
        mean_prediction = torch.mean(val_predictions_final)
        print(f'Variable {variable_index + 1} - Final Mean Prediction: {mean_prediction.item()}')

        # Print or save validation MAEs for each model
        print(f'Variable {variable_index + 1} - Validation MAEs: {val_maes}')

    # ... (rest of your code remains unchanged)



if __name__ == '__main__':
    main()
