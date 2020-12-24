import numpy as np
from scipy import interpolate
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pdb

def _estim_dist(quantiles, percentiles, y_min, y_max, smooth_tails, tau):

    noise = np.random.uniform(low=0.0, high=1e-8, size=((len(quantiles),)))
    noise_monotone = np.sort(noise)
    quantiles = quantiles + noise_monotone

    def interp1d(x, y, a, b):
        return interpolate.interp1d(x, y, bounds_error=False, fill_value=(a, b), assume_sorted=True)

    cdf = interp1d(quantiles, percentiles, 0.0, 1.0)
    inv_cdf = interp1d(percentiles, quantiles, y_min, y_max)

    tau_lo = tau
    tau_hi = tau

    if smooth_tails:
        mu = inv_cdf(0.75) - inv_cdf(0.25)
        q_lo = inv_cdf(tau_lo)
        q_hi = inv_cdf(1-tau_hi)
        def new_cdf(y):
            if y<q_lo:
                ret_val = tau_lo*np.exp( (y-q_lo)/(mu+1e-5) )
            elif y>=q_lo and y<=q_hi:
                ret_val = cdf(y)
            else:
                ret_val = 1.0 - tau_hi*np.exp( (q_hi - y)/(mu+1e-5) )
            return ret_val

        def new_inv_cdf(t):
            if t==1:
                return -(mu+1e-5) * np.log(1e-10/tau_hi) + q_hi
            if t==0:
                return (mu+1e-5) * np.log(1e-10/tau_lo) + q_lo
            if t<tau:
                ret_val = (mu+1e-5) * np.log(t/tau_lo) + q_lo
            elif t>=tau and t<=1.0-tau:
                ret_val = inv_cdf(t)
            else:
                ret_val = -(mu+1e-5) * np.log((1-t)/tau_hi) + q_hi
            return ret_val

        return new_cdf, new_inv_cdf

    return cdf, inv_cdf

class QDensity():
    def __init__(self, percentiles, breaks):
        self.percentiles = percentiles
        self.breaks = breaks

    def compute_density(self, quantiles, ymin, ymax):
        n = quantiles.shape[0]
        B = len(self.breaks)-1
        f_hat = np.zeros((n,B))
        #percentiles = self.percentiles
        percentiles = np.concatenate(([0],self.percentiles,[1]))
        quantiles = np.pad(quantiles, ((0,0),(1, 1)), 'constant', constant_values=(ymin,ymax))
        breaks = self.breaks

        def interp1d(x, y, a, b):
            return interpolate.interp1d(x, y, bounds_error=False, fill_value=(a, b), assume_sorted=True)

        for i in range(n):
            cdf, inv_cdf = _estim_dist(quantiles[i], percentiles, y_min=ymin, y_max=ymax, smooth_tails=True, tau=0.01)
            cdf_hat = cdf(breaks)
            f_hat[i] = np.diff(cdf_hat)
            f_hat[i] = (f_hat[i]+1e-6) / (np.sum(f_hat[i]+1e-6))

        return f_hat

    def compute_cdf(self, quantiles, ymin, ymax):
        f_hat = self.compute_density(quantiles, ymin, ymax)
        cdf_hat = np.cumsum(f_hat,axis=1)
        cdf_hat = np.clip(cdf_hat, 0, 1)
        return cdf_hat

    def compute_cdf2(self, quantiles):
        n = quantiles.shape[0]
        B = len(self.breaks)
        cdf_hat = np.zeros((n,B))
        percentiles = self.percentiles
        for i in range(n):
            cur_cdf = np.interp(self.breaks, quantiles[i], percentiles)
            cur_cdf[0] = 0
            cur_cdf[-1] = 1
            cdf_hat[i] = cur_cdf
        cdf_hat = np.clip(cdf_hat, 0, 1)
        return cdf_hat

    def compute_cdf_and_inv_cdf(self, quantiles, smooth_tails=True, tau=0.1):
        n = quantiles.shape[0]
        cdf_list = []
        inv_cdf_list = []
        for i in range(n):
            cdf, inv_cdf = _estim_dist(quantiles[i], self.percentiles, y_min=self.breaks[0], y_max=self.breaks[-1],
                                       smooth_tails=smooth_tails, tau=tau)

            cdf_list.append(cdf)
            inv_cdf_list.append(inv_cdf)

        return cdf_list, inv_cdf_list
