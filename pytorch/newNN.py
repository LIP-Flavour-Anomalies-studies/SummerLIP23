import argparse
import torch
import os.path
from subprocess import call
import uproot
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
import utils

class FeedforwardNetwork(nn.Module):
    def __init__(self, n_features, hidden_size, layers, dropout, **kwargs):
        super().__init__()

        self.first_layer = nn.Linear(n_features, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layers-1)])
        self.output_layer = nn.Linear(hidden_size, 1)  # Regression output

        drop = nn.Dropout(p=dropout)

        self.order = nn.Sequential(self.first_layer, nn.ReLU(), drop)
        for _ in range(layers-1):
            self.order.append(self.hidden_layers[_])
            self.order.append(nn.ReLU())
            self.order.append(drop)
        self.order.append(self.output_layer)

    def forward(self, x, **kwargs):
        x = self.order(x)
        return x


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    optimizer.zero_grad()
    y_pred = model(X).squeeze()
    
    # Ensure the target values have the same data type as the predictions
    y = y.float()
    
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(model, X):
    return model(X).squeeze()

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

    x, y ,variable_names = utils.prepdata(data, data_mc)
    dataset = utils.ClassificationDataset(x, y)

    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [0.5, 0.25, 0.25])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

    dev_X, dev_y = val_set[:][0], val_set[:][1]
    test_X, test_y = test_set[:][0], test_set[:][1]

    n_feats = dataset.X.shape[1]

    # Initialize the model
    model = FeedforwardNetwork(
        n_feats,
        opt.hidden_size,
        opt.layers,
        opt.dropout
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
    for epoch in epochs:
        print(f'Training epoch {epoch}')
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print(f'Training loss: {mean_loss:.4f}')

    # Evaluate on the test set
    mae, rmse = evaluate(model, test_X, test_y)
    print(f'Test MAE: {mae:.4f}')
    print(f'Test RMSE: {rmse:.4f}')

    # Make predictions
    predictions = predict(model, test_X)

    # Create scatter plots for each variable
    n_variables = test_X.shape[1]
    for i in range(n_variables):
        plt.scatter(test_X[:, i].detach().numpy(), test_y.numpy(), label='Ground Truth', alpha=0.5)
        plt.scatter(test_X[:, i].detach().numpy(), predictions.detach().numpy(), label='Predictions', alpha=0.5)
        plt.xlabel(f'Variable {i + 1}')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f'Variable {i + 1} - Ground Truth vs Predictions')
        plt.show()
        plt.savefig("var"+str(i)+"preds.png")


    num_examples = len(test_X)
    print(f'Number of examples in the test set: {num_examples}')

    
     # Print some of the predictions with variable names
    print("Some of the predictions:")
    for i in range(min(10, len(predictions))):
        example_str = f"Example {i + 1}: "
        variable_prediction_str = " | ".join([f"{variable_names[j]}: {test_X[i, j].item()} - Prediction: {predictions[i].item()}" for j in range(len(variable_names))])
        print(example_str + variable_prediction_str + "\n")

    
if __name__ == '__main__':
    main()
