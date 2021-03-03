import numpy as np

def betainv_generic(pvals, aseq):
    n = len(aseq)
    idx = np.maximum(1, np.floor((n + 1) * (1 - pvals))).astype(int)
    out = 1 - aseq[idx-1]
    return out

def compute_aseq(n, k, delta):
    def movingaverage (values, window):
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'valid')
        return sma

    k = int(k)
    fac1 = np.log(delta) / k - np.mean(np.log(np.arange(n-k+1,n+1)))
    fac2 = movingaverage(np.log(np.arange(1,n+1)), k)
    aseq = np.concatenate([np.zeros((k-1,)), np.exp(fac2 + fac1)])
    return aseq

def betainv_simes(pvals, n, k, delta):
    aseq = compute_aseq(n, k, delta)
    out = betainv_generic(pvals, aseq)
    return out

def cdf_bound(x, x_cal, aseq):
    n_cal = len(x_cal)
    x = np.reshape(np.array(x), [len(x),1])
    jseq = np.maximum(0, np.sum(x > x_cal, 1)-1)
    g_hat = 1-aseq[n_cal-1-jseq]
    return g_hat

# Empirical bound
def find_slope_EB(n, alpha=None, prob=0.9, n_sim=5000):
    if alpha is None:
        alpha = 10.0/n
    U = np.random.uniform(size=(n_sim, n))
    U = np.sort(U,1)

    def get_a(n, beta, alpha=None):
        # function on bottom of page 13 of Shorack and Wellner
        # for symmetric, affine linear bands
        if alpha is None:
            alpha = 1.0/n
        a1 = (np.arange(1,n+1)/n - alpha) / (1.0+beta) #curve to the left of 1/2
        a2 = 1.0 + (np.arange(1,n+1)/n - 1.0 - alpha)/(1.0-beta)
        return np.maximum(a1, a2)

    def get_b(n, beta, alpha=None):
        # function on bottom of page 13 of Shorack and Wellner
        # for symmetric, affine linear bands
        if alpha is None:
            alpha = 1.0/n
        b1 = (np.arange(1,n+1)/n - 1.0/n + alpha) / (1.0-beta)
        b2 = 1.0 + (np.arange(1,n+1)/n - 1.0/n + alpha - 1.0)/(1.0+beta)
        return np.minimum(b1, b2)

    def compute_coverage(U, alpha, beta):
        n = U.shape[1]
        a = get_a(n, beta, alpha=alpha)
        b = get_b(n, beta, alpha=alpha)
        Uo = np.max(np.maximum(U>b,U<a),1)
        covg = 1.0 - np.mean(Uo)
        return covg

    # Search for beta parameter that gives the right coverage
    n2 = n*n
    beta_max = 1.0-1.0/n2
    beta_min = 1.0/n2
    while beta_max-beta_min > 1e-6:
        beta = 0.5*(beta_min+beta_max)
        covg = compute_coverage(U, alpha, beta)
        if covg>prob:
            beta_max = beta
        else:
            beta_min = beta

    return beta
