import numpy as np
import copy
from sklearn.model_selection import train_test_split
from .utils_calib import betainv_mc, compute_aseq, cdf_bound, betainv_simes, find_slope_EB, estimate_fs_correction, betainv_asymptotic

def calibrate_ccv(pval, n_cal, delta=0.01, method="MC", simes_kden=2, fs_correction=None, a=None, two_sided=False):

    if method=="Simes":
        k = int(n_cal/simes_kden)
        output = betainv_simes(pval, n_cal, k, delta)
        two_sided = False

    elif method=="DKWM":
        epsilon = np.sqrt(np.log(2.0/delta)/(2.0*n_cal))
        if two_sided==True:
            output = np.minimum(1.0, 2.0 * np.minimum(pval + epsilon, 1-pval + epsilon))
        else:
            output = np.minimum(1.0,pval + epsilon)

    elif method=="Linear":
        a = 10.0/n_cal #0.005
        b = find_slope_EB(n_cal, alpha=a, prob=1.0-delta)
        output_1 = np.minimum( (pval+a)/(1.0-b), (pval+a+b)/(1.0+b) )
        output_2 = np.maximum( (1-pval+a+b)/(1.0+b), (1-pval+a)/(1.0-b) )
        if two_sided == True:
            output = np.minimum(1.0, 2.0 * np.minimum(output_1, output_2))
        else:
            output = np.minimum(1.0, output_1)

    elif method=="MC":
        if fs_correction is None:
            fs_correction = estimate_fs_correction(delta,n_cal)
        output = betainv_mc(pval, n_cal, delta, fs_correction=fs_correction)
        two_sided = False

    elif method=="Asymptotic":
        k = int(n_cal/simes_kden)
        output = betainv_asymptotic(pval, n_cal, k, delta)
        two_sided = False

    else:
        raise ValueError('Invalid calibration method.')

    return output

class ConformalPvalues:
    def __init__(self, X, bbox, calib_size=0.5, delta=0.05, random_state=2020):
        self.bbox = copy.deepcopy(bbox)

        # Split data into training and calibration subsets
        X_train, X_calib = train_test_split(X, test_size=calib_size, random_state=random_state)

        # Fit the black-box one-class classification model
        self.bbox.fit(X_train)

        # Calibrate
        self.scores_cal = self.bbox.score_samples(X_calib)
        self.n_cal = len(self.scores_cal)
        self.delta = delta
        self.fs_correction = estimate_fs_correction(self.delta, self.n_cal)

    def predict(self, X_test, simes_kden=2):
        scores_test = self.bbox.score_samples(X_test)
        scores_mat = np.tile(self.scores_cal, (len(scores_test),1))
        tmp = np.sum(scores_mat <= scores_test.reshape(len(scores_test),1), 1)
        pvals = (1.0+tmp)/(1.0+self.n_cal)

        pvals_simes = calibrate_ccv(pvals, self.n_cal, delta=self.delta, method="Simes", simes_kden=simes_kden, two_sided=False)
        pvals_mc = calibrate_ccv(pvals, self.n_cal, delta=self.delta, method="MC", fs_correction=self.fs_correction, two_sided=False)
        pvals_as = calibrate_ccv(pvals, self.n_cal, delta=self.delta, method="Asymptotic", simes_kden=simes_kden)
        pvals_dkwm = calibrate_ccv(pvals, self.n_cal, delta=self.delta, method="DKWM")

        # Collect results
        output = {"Marginal" : pvals, "Monte Carlo":pvals_mc, "Simes" : pvals_simes, "Asymptotic" : pvals_as, "DKWM" : pvals_dkwm}

        return output


class CalibrationBound:
    def __init__(self, x, delta=0.1):
        self.n = len(x)
        self.delta = delta
        self.x_cal = np.sort(x)
        ## Sequence of a values for upper limit
        k = int(self.n/2)
        self.aseq_upper = compute_aseq(self.n, k, self.delta)

    def evaluate(self, x):
        x = np.sort(x)
        bound = cdf_bound(x, self.x_cal, self.aseq_upper)
        return bound, x
