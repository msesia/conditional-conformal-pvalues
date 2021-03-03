import numpy as np
import copy
from sklearn.model_selection import train_test_split
from .utils_calib import betainv_simes
from .utils_calib import compute_aseq, cdf_bound
from .utils_calib import find_slope_EB

def calibrate_simultaneous(pval, n_cal, delta=0.01, method="Simes", simes_kden=3, a=None, two_sided=False):

    if method=="Simes":
        k = int(n_cal/simes_kden)
        output = betainv_simes(pval, n_cal, k, delta)
        two_sided = False

    elif simultaneous=="DKW":
        epsilon = np.sqrt(np.log(2.0/delta)/(2.0*n_cal))
        if two_sided==True:
            output = np.minimum(1.0, 2.0 * np.minimum(pval + epsilon, 1-pval + epsilon))
        else:
            output = np.minimum(1.0,pval + epsilon)

    elif simultaneous=="Linear":
        a = 10.0/n_cal #0.005
        b = find_slope_EB(n_cal, alpha=a, prob=1.0-delta)
        output_1 = np.minimum( (pval+a)/(1.0-b), (pval+a+b)/(1.0+b) )
        output_2 = np.maximum( (1-pval+a+b)/(1.0+b), (1-pval+a)/(1.0-b) )
        if two_sided == True:
            output = np.minimum(1.0, 2.0 * np.minimum(output_1, output_2))
        else:
            output = np.minimum(1.0, output_1)

    else:
        raise ValueError('Invalid calibration method.')

    return output

class ConformalOutlierDetector:
    def __init__(self, X, bbox, calib_size=0.5, random_state=2020):
        self.bbox = copy.deepcopy(bbox)

        # Split data into training and calibration subsets
        X_train, X_calib = train_test_split(X, test_size=calib_size, random_state=random_state)

        # Fit the black-box one-class classification model
        self.bbox.fit(X_train)

        # Calibrate
        self.scores_cal = self.bbox.score_samples(X_calib)
        self.n_cal = len(self.scores_cal)

    def predict(self, X_test, delta=0.05, simes_kden=3):
        scores_test = self.bbox.score_samples(X_test)
        scores_mat = np.tile(self.scores_cal, (len(scores_test),1))
        tmp = np.sum(scores_mat <= scores_test.reshape(len(scores_test),1), 1)
        pvals = (1.0+tmp)/(1.0+self.n_cal)

        pvals_simes = calibrate_simultaneous(pvals, self.n_cal, delta=delta, method="Simes",
                                             simes_kden=simes_kden, two_sided=False)

        # Collect results
        output = {"Simes" : pvals_simes, "Pointwise" : pvals}

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
