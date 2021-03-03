import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

class ToyModel:
    def __init__(self, p, amplitude, random_state=2020):
        np.random.seed(random_state)
        self.p = p
        self.a = amplitude
        self.Z = np.random.uniform(low=-3, high=3, size=(p,p))

    def _sample_clean(self, n):
        p = self.p
        X = np.random.randn(n, p)
        cluster_idx = np.random.choice(self.Z.shape[0], n, replace=True)
        X = X + self.Z[cluster_idx,]
        return X

    def _sample_outlier(self, n):
        p = self.p
        X = np.sqrt(self.a) * np.random.randn(n, p)
        cluster_idx = np.random.choice(self.Z.shape[0], n, replace=True)
        X = X + self.Z[cluster_idx,]
        return X

    def sample(self, n, purity=1, random_state=2020):
        p = self.p
        np.random.seed(random_state)
        purity = np.clip(purity, 0, 1)
        n_clean = np.round(n * purity).astype(int)
        n_outlier = n - n_clean
        X_clean = self._sample_clean(n_clean)
        is_outlier = np.zeros((n,))
        if n_outlier > 0:
            X_outlier = self._sample_outlier(n_outlier)
            idx_clean, idx_outlier = train_test_split(np.arange(n), test_size=n_outlier)
            X = np.zeros((n,p))
            X[idx_clean,:] = X_clean
            X[idx_outlier,:] = X_outlier
            is_outlier[idx_outlier] = 1
        else:
            X = X_clean
        return X, is_outlier.astype(int)
