import argparse
import torch
import os.path
from subprocess import call
import uproot
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, mean_absolute_error
import utilsnewer as utils

class FeedforwardNetwork(nn.Module):
    def __init__(self, n_features, hidden_size, layers, dropout, n_vars, **kwargs):
        super().__init__()

        self.first_layer = nn.Linear(n_features, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layers-1)])
        self.output_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_vars)])  # Separate output for each variable

        drop = nn.Dropout(p=dropout)

        self.order = nn.Sequential(self.first_layer, nn.ReLU(), drop)
        for _ in range(layers-1):
            self.order.append(self.hidden_layers[_])
            self.order.append(nn.ReLU())
            self.order.append(drop)

    def forward(self, x, **kwargs):
        x = self.order(x)
        outputs = [output_layer(x).squeeze() for output_layer in self.output_layers]
        return torch.stack(outputs, dim=1)  # Stack outputs along dimension 1

def train_batch(X, y, model, optimizer, criterion, **kwargs):
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

def plot(epochs, plottable, xlabel="Epoch", ylabel='', name=''):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')

def save_opts(fname, **kwargs):
    with open(fname, "w") as f:
        for key, value in kwargs.items():
            f.write(f"{value}\n")

def plot_roc_curve(fpr, tpr, auc_value):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-hidden_size', type=int, default=64)
    parser.add_argument('-layers', type=int, default=3)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-activation', choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='adam')
    opt = parser.parse_args()

    dir = "/lstore/cms/boletti/ntuples/"
    filename = "2018Data_passPreselection_passSPlotCuts_mergeSweights.root"
    filename_mc = "MC_JPSI_2018_preBDT_Nov21.root"

    data = uproot.open(dir + filename)
    data_mc = uproot.open(dir + filename_mc)

    x, y, variable_names = utils.prepdata(data, data_mc,"bLBS")
    dataset = utils.RegressionDataset(x, y)

    print(variable_names)

    # Determine the number of variables dynamically
    n_vars = len(variable_names)

    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [0.5, 0.25, 0.25])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

    dev_X, dev_y = val_set[:][0], val_set[:][1]
    test_X, test_y = test_set[:][0], test_set[:][1]

    n_feats = dataset.X.shape[1]
    n_variables = dataset.y.shape[1]

    # Initialize the model with the dynamically determined number of output variables
    model = FeedforwardNetwork(
        n_feats,
        opt.hidden_size,
        opt.layers,
        opt.dropout,
        n_variables
    )

    # Save options
    save_opts("output.txt", n_feats=n_feats, hidden_size=opt.hidden_size,
              layers=opt.layers, dropout=opt.dropout, batch_size=opt.batch_size)

    # Get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay
    )

    # Get a loss criterion
    criterion = nn.MSELoss()

    # Training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_losses = []
    val_maes = []  # Use a list to store validation MAEs
    for epoch in epochs:
        print(f'Training epoch {epoch}')
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print(f'Training loss: {mean_loss:.4f}')

        # Validation MAE
        model.eval()
        val_predictions = predict(model, dev_X)
        val_mae = mean_absolute_error(dev_y.detach().numpy(), val_predictions.detach().numpy())
        val_maes.append(val_mae)
        model.train()

    # Plot validation MAE
    plt.figure()
    plt.plot(epochs, val_maes, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE Over Epochs')
    plt.legend()
    plt.show()

    # Evaluate on the test set
    mae, rmse = evaluate(model, test_X, test_y)
    print(f'Test MAE: {mae:.4f}')
    print(f'Test RMSE: {rmse:.4f}')

    # Make predictions for the entire test set
    predictions = predict(model, test_X)

    # Print individual predictions for each variable
    for variable_name, variable_predictions in zip(variable_names, predictions.T):
        print(f"{variable_name} predictions: {variable_predictions}")

    # Print shapes
    print("Shapes:")
    print("predictions:", predictions.shape)

    # Calculate accuracy
    accuracy = accuracy_score(test_y.detach().numpy(), predictions.detach().round().numpy())
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
