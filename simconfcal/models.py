import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

class OutlierDataModel:
    def __init__(self, p, a=1, random_state=2020):
        np.random.seed(random_state)
        self.p = p
        self.a = a
        self.Z = np.random.uniform(low=-self.a, high=self.a, size=(p,p))
    
    def _sample_clean(self, n):
        p = self.p
        X = np.random.randn(n, p)
        cluster_idx = np.random.choice(p, n, replace=True)
        X = X + self.Z[cluster_idx,]
        return X
    
    def _sample_outlier(self, n):
        X = np.random.uniform(low=-self.a, high=self.a, size=(n, self.p))
        return X
    
    def sample(self, n, purity=1, random_state=2020):
        np.random.seed(random_state)
        purity = np.clip(purity, 0, 1)
        n_clean = np.round(n * purity).astype(int)
        n_outlier = n - n_clean
        X_clean = self._sample_clean(n_clean)
        X_outlier = self._sample_outlier(n_outlier)
        idx_clean, idx_outlier = train_test_split(np.arange(n), test_size=n_outlier)
        X = np.zeros((n,self.p))
        X[idx_clean,:] = X_clean
        X[idx_outlier,:] = X_outlier
        is_outlier = np.zeros((n,))
        is_outlier[idx_outlier] = 1
        return X, is_outlier.astype(int)
