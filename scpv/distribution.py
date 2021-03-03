import numpy as np
import sys
from scipy import interpolate
from statsmodels.distributions.empirical_distribution import ECDF
from scpv.utils_quantile import QNet, QRF
from scpv.utils_density import QDensity

import pdb

def eval_cdf(cdf_list, Y):
    return np.array([cdf(Y[ind]) for ind, cdf  in enumerate(cdf_list)]).squeeze()

def eval_inv_cdf(inv_cdf_list, T):
    return np.array([inv_cdf(T[ind]) for ind, inv_cdf in enumerate(inv_cdf_list)]).squeeze()

def compute_quantiles(inv_cdf_list, quantile_levels):
    output = np.array([eval_inv_cdf(inv_cdf_list,[q]*len(inv_cdf_list)) for q in quantile_levels]).T
    return output

def compute_ecdf(x, x_min=None, x_max=None, width=1e-4):
    x = np.sort(x)
    if x_min is None:
        x_min = np.min(x)-2*width
    if x_max is None:
        x_max = np.max(x)+2*width

    F = ECDF(x)
    x_grid = np.append([-100],np.unique(x))
    y_grid = F(x_grid)
    F_inv = interpolate.interp1d(y_grid, x_grid, kind="next", bounds_error=False, fill_value=(0,1), assume_sorted=True)
    return F, F_inv

def compute_sim_G_G_inv(scores, delta):
    """Simultaneous CDF calibration with DKW method"""
    G, inv_G = compute_ecdf(scores, x_min=0, x_max=1)
    epsilon = np.sqrt(np.log(2.0/delta)/(2.0*len(scores)))

    def sim_G(t):
        val = G(t)
        ret_val = 0.5
        if val<(0.5 - epsilon):
            ret_val = val + epsilon
        elif val>(0.5 + epsilon):
            ret_val = val - epsilon
        return ret_val

    def inv_sim_G(t):
        if t<0.5:
            ret_val = inv_G(np.maximum(t - epsilon,0))
        elif t>0.5:
            ret_val = inv_G(np.minimum(t + epsilon,1))
        else:
            ret_val = inv_G(0.5)
        return ret_val

    return sim_G, inv_sim_G

def calibrate_cdf(G, F_vec, inv_G, inv_F_vec):

    def compose_G_F(G, F_y_x, inv_G, inv_F_y_x):

        def cal_cdf(y):
            return G(F_y_x(y))

        def inv_cal_cdf(t):
            return inv_F_y_x(inv_G(t))

        return cal_cdf, inv_cal_cdf

    cal_cdf_list = []
    inv_cal_cdf_list = []
    for i in range(len(F_vec)):
        cal_cdf, inv_cal_cdf = compose_G_F(G, F_vec[i], inv_G, inv_F_vec[i])
        cal_cdf_list.append(cal_cdf)
        inv_cal_cdf_list.append(inv_cal_cdf)

    return cal_cdf_list, inv_cal_cdf_list


class ConditionalDistribution:
    """Conditional distribution estimator"""

    def __init__(self, grid_quantiles, y_min, y_max):
        # Define grid of conditional quantiles to be estimated
        self.grid_quantiles = np.arange(0.01,1,0.01)

        # Initialize conditional distribution function
        grid_density = np.linspace(y_min, y_max, 1000)
        self.qdens = QDensity(self.grid_quantiles, grid_density)

        # Initialize status indicators
        self.fitted = False
        self.calibrated = False

    def fit(self, X, Y, model):
        # Number of training samples
        n_train = X.shape[0]

        # Initialize black-box quantile regression model
        if model=="NNet":
            self.bbox = QNet(self.grid_quantiles, 1, no_crossing=True, batch_size=n_train, dropout=0.2,
                             num_epochs=3000, learning_rate=0.01, calibrate=False)
        elif model=="RF":
            self.bbox = QRF(self.grid_quantiles, n_estimators=10, min_samples_leaf=20, random_state=2020)
        else:
            err_msg = "Error: unknown black-box model: " + model
            sys.exit(err_mst)

        # Fit the black-box quantile regression model
        self.bbox.fit(X, Y)

        # Change status indicators
        self.fitted = True
        self.calibrated = False

    def predict_quantiles(self, X, quantiles):
        # Apply black-box quantile regression model
        Q_bbox = self.bbox.predict(X)

        # Compute conditional CDF and inverse conditional CDF
        F_hat, inv_F_hat = self.qdens.compute_cdf_and_inv_cdf(Q_bbox, smooth_tails=False, tau=0.1)

        # Compute desired quantiles
        Q = compute_quantiles(inv_F_hat, quantiles)

        return Q

    def calibrate_quantiles(self, X, Y, delta=0.05):
        # Check status indicators
        if not self.fitted:
            print("Error: you must first call the fit(...) method.")
            return None

        # Apply black-box quantile regression model on calibration data
        Q_bbox = self.bbox.predict(X)

        # Compute conditional CDF and inverse conditional CDF
        F_hat, inv_F_hat = self.qdens.compute_cdf_and_inv_cdf(Q_bbox, smooth_tails=True, tau=0.1)

        # Compute conformity scores
        scores_cal = eval_cdf(F_hat, Y)

        # Pointwise calibration
        self.G, self.inv_G = compute_ecdf(scores_cal, x_min=0, x_max=1)

        # Simultaneous calibration (DKW method)
        self.simG, self.inv_simG = compute_sim_G_G_inv(scores_cal, delta)

        # Change status indicators
        self.calibrated = True


    def predict_calibrated_quantiles(self, X, quantiles, simultaneous=False):
        # Check status indicators
        if not self.calibrated:
            print("Error: you must first call the calibrate_quantiles(...) method.")
            return None

        # Apply black-box quantile regression model
        Q_bbox = self.bbox.predict(X)

        # Compute conditional CDF and inverse conditional CDF
        F_hat, inv_F_hat = self.qdens.compute_cdf_and_inv_cdf(Q_bbox, smooth_tails=True, tau=0.1)

        # Apply calibration
        if simultaneous:
            calibrated_F, calibrated_inv_F = calibrate_cdf(self.simG, F_hat, self.inv_simG, inv_F_hat)
        else:
            calibrated_F, calibrated_inv_F = calibrate_cdf(self.G, F_hat, self.inv_G, inv_F_hat)

        # Compute desired quantiles
        Q = compute_quantiles(calibrated_inv_F, quantiles)

        return Q
