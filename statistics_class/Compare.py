# 指定値と比較する統計的検定クラスCompare
class Compare:
    def __init__(self, data):
        self.data = data
        self.length = len(self.data)
        self.mean = np.mean(self.data)
        self.var_p = np.var(self.data)
        self.var_s = np.var(self.data, ddof=1)
        self.std_p = np.std(self.data)
        self.std_s = np.std(self.data, ddof=1)
        
    def mean_ztest(self, com_mean):
        return (self.mean-com_mean) / (self.std_p/(self.length)**0.5)
    
    def mean_ttest(self, com_mean):
        return (self.mean-com_mean) / (self.std_s/(self.length)**0.5)
    
    def var_x2test(self, com_var):
        return ((self.length-1)*self.var_s) / com_var


# 使用例
data = pd.DataFrame(np.random.randint(-100, 101, 100), columns=['data'])
com_mean = 2
com_var = 3000
model = Compare(data)

# 比較値と平均値の差の検定
print(model.mean)
print(model.mean_ztest(com_mean))
print(model.mean_ttest(com_mean))
print()

# 比較値と分散の差の検定
print(model.var_s)
print(model.var_x2test(com_var))
