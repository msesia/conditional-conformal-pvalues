import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.tensor as tensor
from torch.utils.data import DataLoader, Dataset

from skgarden import RandomForestQuantileRegressor

from sklearn.model_selection import train_test_split

from tqdm.autonotebook import tqdm

import warnings

import pdb

class RegressionDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).float()

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

class NNet(nn.Module):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self, quantiles, num_features, num_hidden=64, dropout=0.2, no_crossing=False):
        """ Initialization
        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        num_features : integer, input signal dimension (p)
        num_hidden : integer, hidden layer dimension
        dropout : float, dropout rate
        no_crossing: boolean, whether to explicitly prevent quantile crossovers
        """
        super(NNet, self).__init__()

        self.no_crossing = no_crossing

        self.num_quantiles = len(quantiles)

        # Construct base network
        self.base_model = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, self.num_quantiles),
        )
        self.init_weights()

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        x = self.base_model(x)

        if self.no_crossing:
            def roll_1(x, n):
                return torch.cat((x[:, -n:], x[:, :-n]), dim=1)
            y = torch.zeros(x.shape)
            y[:,0] = x[:,0]
            y[:,1:self.num_quantiles] = nn.ReLU()(x[:,1:self.num_quantiles])
            y = torch.cumsum(y, dim=1)
            return y
        else:
            return x


class AllQuantileLoss(nn.Module):
    """ Pinball loss function
    """
    def __init__(self, quantiles):
        """ Initialize
        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ Compute the pinball loss
        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)
        Returns
        -------
        loss : cost function value
        """
        #assert not target.requires_grad
        #assert preds.size(0) == target.size(0)

        errors = target.unsqueeze(1)-preds
        Q = self.quantiles.unsqueeze(0)
        loss = torch.max((Q-1.0)*errors, Q*errors).mean()

        return loss


class QNet:
    """ Fit a neural network (conditional quantile) to training data
    """
    def __init__(self, quantiles, num_features, no_crossing=False, dropout=0.2, learning_rate=0.001,
                 num_epochs=100, batch_size=16, random_state=0, calibrate=False, verbose=False):
        """ Initialization
        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        num_features : integer, input signal dimension (p)
        learning_rate : learning rate
        random_state : integer, seed used in CV when splitting to train-test
        """

        # Detect whether CUDA is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Store input (sort the quantiles)
        quantiles = np.sort(quantiles)
        self.quantiles = torch.from_numpy(quantiles).float().to(self.device)
        self.num_features = num_features

        # Define NNet model
        self.model = NNet(self.quantiles, self.num_features, dropout=dropout, no_crossing=no_crossing)
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize loss function
        self.loss_func = AllQuantileLoss(self.quantiles)

        # Store variables
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.calibrate = calibrate

        # Initialize training logs
        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []

        # Validation
        self.val_period = 10

        self.verbose = verbose

    def fit(self, X, Y, return_loss=False):
        Y = Y.flatten().astype(np.float32)
        X = X.astype(np.float32)

        dataset = RegressionDataset(X, Y)
        num_epochs = self.num_epochs
        if self.calibrate:
            # Train with 80% of samples
            n_valid = int(np.round(0.2*X.shape[0]))
            X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=n_valid,
                                                                  random_state=self.random_state)
            train_dataset = RegressionDataset(X_train, Y_train)
            val_dataset = RegressionDataset(X_valid, Y_valid)
            loss_stats = self._fit(train_dataset, num_epochs, val_dataset=val_dataset)
            # Find optimal number of epochs
            num_epochs = self.val_period*(np.argmin(loss_stats['val'])+1)
            # Reset model
            self.model.init_weights()
            loss_stats_cal = loss_stats

        # Train with all samples
        loss_stats = self._fit(dataset, num_epochs)
        if self.calibrate:
            loss_stats = loss_stats_cal

        if return_loss:
          return loss_stats

    def _fit(self, train_dataset, num_epochs, val_dataset=None):
        batch_size = self.batch_size

        # Initialize data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        if val_dataset is not None:
            val_loader = DataLoader(dataset=val_dataset, batch_size=1)

        num_samples, num_features = train_dataset.X_data.shape
        print("Training with {} samples and {} features.". \
              format(num_samples, num_features))

        loss_stats = {'train': [], "val": []}

        X_train_batch = train_dataset.X_data.to(self.device)
        y_train_batch = train_dataset.y_data.to(self.device)

        for e in tqdm(range(1, num_epochs+1)):

            # TRAINING
            train_epoch_loss = 0
            self.model.train()

            if batch_size<500:

              for X_train_batch, y_train_batch in train_loader:
                  X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)
                  self.optimizer.zero_grad()

                  y_train_pred = self.model(X_train_batch).to(self.device)

                  train_loss = self.loss_func(y_train_pred, y_train_batch)

                  train_loss.backward()
                  self.optimizer.step()

                  train_epoch_loss += train_loss.item()

            else:
                self.optimizer.zero_grad()

                y_train_pred = self.model(X_train_batch).to(self.device)

                train_loss = self.loss_func(y_train_pred, y_train_batch)

                train_loss.backward()
                self.optimizer.step()

                train_epoch_loss += train_loss.item()

            # VALIDATION
            if val_dataset is not None:
                if e % self.val_period == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_epoch_loss = 0
                        for X_val_batch, y_val_batch in val_loader:
                            X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                            y_val_pred = self.model(X_val_batch).to(self.device)
                            val_loss = self.loss_func(y_val_pred, y_val_batch)
                            val_epoch_loss += val_loss.item()

                    loss_stats['val'].append(val_epoch_loss/len(val_loader))
                    self.model.train()

            else:
                loss_stats['val'].append(0)

            if e % self.val_period == 0:
                loss_stats['train'].append(train_epoch_loss/len(train_loader))

            if (e % 10 == 0) and (self.verbose):
                if val_dataset is not None:
                    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | ', end='')
                    print(f'Val Loss: {val_epoch_loss/len(val_loader):.5f} | ', flush=True)
                else:
                    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | ', flush=True)

        return loss_stats

    def predict(self, X):
        """ Estimate the label given the features
        Parameters
        ----------
        x : numpy array of training features (nXp)
        Returns
        -------
        ret_val : numpy array of predicted labels (n)
        """
        self.model.eval()
        ret_val = self.model(torch.from_numpy(X).to(self.device).float().requires_grad_(False))
        return ret_val.cpu().detach().numpy()

    def get_quantiles(self):
        return self.quantiles.cpu().numpy()

class QRF:
    """ Fit a random forest (conditional quantile) to training data
    """
    def __init__(self, quantiles, min_samples_leaf=5, n_estimators=100, n_jobs=1,
                 random_state=0, verbose=False):
        """ Initialization
        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        num_features : integer, input signal dimension (p)
        random_state : integer, seed used in quantile random forests
        """

        self.device = 'cpu'
        # Store input (sort the quantiles)
        self.quantiles = torch.from_numpy(np.sort(quantiles)).float().to(self.device)
        # Define RF model
        self.model = RandomForestQuantileRegressor(random_state=random_state,
                                                   min_samples_leaf=min_samples_leaf,
                                                   n_estimators=n_estimators,
                                                   n_jobs=n_jobs, verbose=verbose)

    def fit(self, X, Y, return_loss=None):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.model.fit(X, Y)
        warnings.filterwarnings("default", category=FutureWarning)
        return 0

    def predict(self, X):
        """ Estimate the label given the features
        Parameters
        ----------
        x : numpy array of training features (nXp)
        Returns
        -------
        ret_val : numpy array of predicted labels (n)
        """
        quantiles = self.quantiles.cpu()
        ret_val = np.zeros((X.shape[0], len(quantiles)))
        print("Predicting RF quantiles:")
        for i in tqdm(range(len(quantiles))):
            ret_val[:,i] = self.model.predict(X,quantile=100*quantiles[i])
        return ret_val

    def get_quantiles(self):
        return self.quantiles.cpu().numpy()
