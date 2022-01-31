# 統計クラス
class Statistics:
    def __init__(self, data):
        self.data = data
        self.length = len(data)
        self.mean = np.mean(data)
        self.var_p = np.var(data)
        self.var_s = np.var(data, ddof=1)
        self.std_p = np.std(data)
        self.std_s = np.std(data, ddof=1)
        self.std_e = self.std_p / self.mean
        self.min = np.amin(data)
        self.per1 = np.percentile(data, 25)
        self.median = np.median(data)
        self.per3 = np.percentile(data, 75)
        self.max = np.amax(data)
        self.quartile_range = self.per3 - self.per1
        
    def sort_asc(self):
        return np.sort(self.data)
    
    def sort_desc(self):
        return np.sort(self.data)[::-1]
        
    def cov_p(self, data):
        return np.cov(self.data, data.data, bias=True)[0][1]
        
    def cov_s(self, data):
        return np.cov(self.data, data.data)[0][1]


# テスト
import numpy as np
import pandas as pd
data = pd.DataFrame(np.random.randint(-100, 100, 100), columns=['data'])['data']
data2 = pd.DataFrame(np.random.randint(-100, 100, 100), columns=['data2'])['data2']
model = Statistics(data)
model2 = Statistics(data2)

print("data:", model.data)
print("sort_asc:", model.sort_asc())
print("sort_desc:", model.sort_desc())
print("length:", model.length)
print("mean:", model.mean)
print("var_p:", model.var_p)
print("var_s:", model.var_s)
print("std_p:", model.std_p)
print("std_s:", model.std_s)
print("std_e:", model.std_e)
print("min:", model.min)
print("per1:", model.per1)
print("median:", model.median)
print("per3:", model.per3)
print("max:", model.max)
print("quartile_range:", model.quartile_range)
print("cov_p:", model.cov_p(model2))
print("cov_s:", model.cov_s(model2))
