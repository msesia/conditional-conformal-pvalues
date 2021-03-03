import numpy as np
from scipy import interpolate
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pdb

def plot_func(x, y, quantiles=None, quantile_labels=None, max_show=5000,
              shade_color="", method_name="", title="", filename=None, save_figures=False):
    
    """ Scatter plot of (x,y) points along with the constructed prediction interval 
    
    Parameters
    ----------
    x : numpy array, corresponding to the feature of each of the n samples
    y : numpy array, target response variable (length n)
    quantiles : numpy array, the estimated prediction. It may be the conditional mean,
                or low and high conditional quantiles.
    shade_color : string, desired color of the prediciton interval
    method_name : string, name of the method
    title : string, the title of the figure
    filename : sting, name of the file to save the figure
    save_figures : boolean, save the figure (True) or not (False)
    
    """
    
    x_ = x[:max_show]
    y_ = y[:max_show]
    if quantiles is not None:
        quantiles = quantiles[:max_show]
    
    fig = plt.figure()
    inds = np.argsort(np.squeeze(x_))
    plt.plot(x_[inds], y_[inds], 'k.', alpha=.2, markersize=10, fillstyle='none')
    
    if quantiles is not None:
        num_quantiles = quantiles.shape[1]
    else:
        num_quantiles = 0  
    
    if quantile_labels is None:
        pred_labels = ["NA"] * num_quantiles
    for k in range(num_quantiles):
        label_txt = 'Quantile {q}'.format(q=quantile_labels[k])
        plt.plot(x_[inds], quantiles[inds,k], '-', lw=2, alpha=0.75, label=label_txt)
    
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    if quantile_labels is not None:
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.title(title)
    if save_figures and (filename is not None):
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    plt.show()


def plot_func_est_cal(x, y, quantiles_est=None,
                      quantiles_calibrated=None,
                      quantile_labels=None,
                      max_show=5000,
                      shade_color="",
                      method_name="",
                      title="",
                      filename=None,
                      save_figures=False):

    
    x_ = x[:max_show]
    y_ = y[:max_show]
    if quantiles_est is not None:
        quantiles_est = quantiles_est[:max_show]
        quantiles_calibrated = quantiles_calibrated[:max_show]
    
    fig = plt.figure()
    inds = np.argsort(np.squeeze(x_))
    plt.plot(x_[inds], y_[inds], 'k.', alpha=.2, markersize=10, fillstyle='none')
    
    if quantiles_est is not None:
        num_quantiles = quantiles_est.shape[1]
    else:
        num_quantiles = 0  
    
    if quantile_labels is None:
        pred_labels = ["NA"] * num_quantiles
        
    
    # Select the color map named rainbow
    cmap = cm.get_cmap(name='tab10')
    grid = inds[::100]

    for k in range(num_quantiles):
        
        label_txt = 'Est. quantile {q}'.format(q=quantile_labels[k])
        label_txt_cal = 'Cal. quantile {q}'.format(q=quantile_labels[k])

        plt.plot(x_[grid], quantiles_calibrated[grid,k], "-", lw=2, alpha=0.9, color=cmap(k), label=label_txt_cal)
        plt.plot(x_[grid], quantiles_est[grid,k], "v", markersize=5, color=cmap(k), fillstyle='none', alpha=0.9, label=label_txt)
        
    
#     plt.ylim([-2, 20])
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.title(title)
    if save_figures and (filename is not None):
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    plt.show()
