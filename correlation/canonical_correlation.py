# 正準相関係数(canonical correlation)
def CCA(x1, x2, scale=True):
    """正準相関係数(canonical correlation)"""
    
    if scale == True:
        x1_mean, x2_mean = np.mean(x1, axis=0), np.mean(x2, axis=0)
        x1_std, x2_std = np.std(x1, ddof=1), np.std(x2, ddof=1)
        x1 = np.array((x1 - x1_mean) / x1_std)
        x2 = np.array((x2 - x2_mean) / x2_std)
    else:
        x1 = np.array(x1)
        x2 = np.array(x2)
    
    x1_cov, x2_cov = np.cov(x1, rowvar=False), np.cov(x2, rowvar=False)
    R = np.dot(x1.T, x2) / (x1.shape[0] - 1)
    A_l = ((np.linalg.inv(x1_cov) @ R) @ (np.linalg.inv(x2_cov))) @ R.T
    
    eign, _ = np.linalg.eig(A_l)
    
    return np.sqrt(eign[0])

  
# テスト
import numpy as np
from sklearn.datasets import load_iris
data = load_iris()
X = data.data[:, :2]
Y = data.data[:, 2:4]
print(f"正準相関係数：{CCA(X, Y, scale=True)}")
