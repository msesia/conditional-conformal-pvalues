import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from cccpv.methods import calibrate_ccv
from statsmodels.stats.multitest import multipletests
import copy

class ConformalOutlierDetector:
    def __init__(self, X, bbox, box_name, calib_size=0.5, random_state=2020):
        self.bbox = copy.deepcopy(bbox)
        self.box_name = box_name

        # Split data into training and calibration subsets
        X_train, X_calib = train_test_split(X, test_size=calib_size, random_state=random_state)

        # Fit the black-box model
        self.bbox.fit(X_train)

        self.n_cal = X.shape[0]
        self.oneclass = oneclass_novelty()
        tmp = self.oneclass.calibrate(X_calib, alpha=0.1, bbox=self.bbox, return_scores=True)

    def predict(self, X_test, delta=0.05, simes_kden=2):
        
        pvals = self.oneclass.get_pval(X_test)
        pvals_simes = calibrate_ccv(pvals, self.n_cal, delta=delta, method="Simes", simes_kden=simes_kden)
        pvals_mc = calibrate_ccv(pvals, self.n_cal, delta=delta, method="MC", fs_correction=fs_correction)
        pvals_as = calibrate_ccv(pvals, self.n_cal, delta=delta, method="Asymptotic")
        pvals_dkwm = calibrate_ccv(pvals, self.n_cal, delta=delta, method="DKWM", simes_kden=simes_kden)

        # Collect results
        output = pd.DataFrame({"Simes " + self.box_name : pvals_simes,
                               "Monte Carlo" + self.box_name : pvals_mc,
                               "Asymptotic" + self.box_name : pvals_as,
                               "DKWM" + self.box_name : pvals_dkwm,
                               "Pointwise " + self.box_name : pvals})

        return output


def evaluate_all_methods(pvals_one_class, is_nonnull, alpha=0.1, lambda_par=0.5):
    pi0_true = 1.0 - np.mean(is_nonnull)

    results_fdr = pd.DataFrame()

    for m in list(pvals_one_class):
        pval = pvals_one_class[m]

        for use_sbh in [True, False]:
            if use_sbh:
                pi = (1.0 + np.sum(pval>lambda_par)) / (len(pval)*(1.0 - lambda_par))
            else:
                pi = 1.0

            alpha_eff = alpha/pi
            reject, pvals_adj, _, _ = multipletests(pval, alpha=alpha_eff, method='fdr_bh')

            rejections = np.sum(reject)
            if rejections>0:
                fdp = 1-np.mean(is_nonnull[np.where(reject)[0]])
                power = np.sum(is_nonnull[np.where(reject)[0]]) / np.sum(is_nonnull)
            else:
                fdp = 0
                power = 0

            res_tmp = {'Method':m, 'Storey':use_sbh, 'Pi0-hat': pi, 'Pi0-true': pi0_true,
                       'Alpha':alpha, 'Rejections':rejections, 'FDR':fdp, 'Power':power}
            res_tmp = pd.DataFrame(res_tmp, index=[0])
            results_fdr = pd.concat([results_fdr, res_tmp])

    return results_fdr

def test_global(pvals_one_class, threshold=0.9):
    results_global = pd.DataFrame()

    for m in list(pvals_one_class):
        pval = np.array(pvals_one_class[m])
        results_global[m] = [combine_pvalues(pval)[1]] # Fisher's test of global null
        #results_global[m] = [combine_pvalues(pval, method="fisher")]

    return results_global
