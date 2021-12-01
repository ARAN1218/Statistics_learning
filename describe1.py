# 一行データ記述統計
def describe1(data):
    # 必要なライブラリをインポート
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 元データ出力
    print(data)
    
    # サンプルサイズ
    def length(data):
        count = 0
        for d in data:
            count += 1
        return count
    
    # 合計値
    def sum_value(data):
        sum_value = 0
        for d in data:
            sum_value += d
        return sum_value
    
    # 平均値
    def mean(data):
        return sum_value(data) / length(data)
    
    # 母分散
    def var_p(data):
        var_p = 0
        m = mean(data)
        for d in data:
            var_p += (d - m)**2
        return var_p / length(data)
    
    # 不偏分散
    def var_s(data):
        var_s = 0
        m = mean(data)
        for d in data:
            var_s += (d - m)**2
        return var_s / (length(data)-1)
    
    # 母標準偏差
    def std_p(data):
        return var_p(data)**0.5
    
    # 不偏標準偏差..........................未完成？
    # 単に不偏分散の平方根を取ったものは、厳密には「母標準偏差の不偏推定量」ではないらしい
    def std_s(data):
        return var_s(data)**0.5
    
    # 標準誤差
    def std_e(data):
        return std_s(data) / length(data)**0.5
    
    # ソート（クイックソート）
    def quick_sort(data):
        n = length(data)
        pivot = data[n//2]
        left, middle, right = [], [], []
        for i in range(n):
            if data[i] < pivot:
                left = left + [data[i]]
            elif pivot < data[i]:
                right = right + [data[i]]
            else:
                middle = middle + [data[i]]
        if left:
            left = quick_sort(left)
        if right:
            right = quick_sort(right)
        return left + middle + right
    
    # 最小値
    def min_value(data):
        data_sorted = quick_sort(data)
        return data_sorted[0]
    
    # 最大値
    def max_value(data):
        data_sorted = quick_sort(data)
        return data_sorted[length(data)-1]
        
    # 第一四分位数
    def quartile1(data):
        data_sorted = quick_sort(data)
        q1 = length(data) * (1/4)
        return data_sorted[int(q1)]
        
    # 中央値
    def median(data):
        data_sorted = quick_sort(data)
        q2 = length(data) / 2
        if q2%1 != 0:
            return data_sorted[int(q2)]
        else:
            return (data_sorted[int(q2)] + data_sorted[int(q2)+1]) / 2
        
    # 第三四分位数
    def quartile3(data):
        data_sorted = quick_sort(data)
        q3 = length(data) * (3/4)
        return data_sorted[int(q3)]
    
    # 四分位範囲
    def quartile_range(data):
        return quartile3(data) - quartile1(data)
    
    
    # 結果出力
    display(pd.DataFrame({
        'count':length(data),
        'sum':sum_value(data),
        'mean':mean(data),
        'var.p':var_p(data),
        'var.s':var_s(data),
        'std.p':std_p(data),
        'std.s':std_s(data),
        'std_e':std_e(data),
        'min':min_value(data),
        '25%':quartile1(data),
        '50%':median(data),
        '75%':quartile3(data),
        'max':max_value(data),
        '25-75%':quartile_range(data)
    }, index=["descriptive statistics"]).T)
    
    # 種々のグラフをプロット
    # グラフ
    plt.plot(data)
    plt.title("Plot")
    plt.show()
    
    # 散布図
    plt.scatter([i for i in range(len(data))], data)
    plt.title("Scatter")
    plt.show()
    
    # ヒストグラム
    plt.hist(data, bins=(1+int(np.log2(length(data))))) #階級幅はスタージェスの公式で定める
    plt.title("Histogram")
    plt.show()
    
    # ヒストグラム(累積)
    plt.hist(data, bins=(1+int(np.log2(length(data)))), cumulative=True) #階級幅はスタージェスの公式で定める
    plt.title("Histogram(cumulative)")
    plt.show()
    
    # 円グラフ(データに負の値があると使えないらしいので一旦停止)
    #plt.pie(data, radius=3, center=(4, 4), wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)
    #plt.title("Piechart")
    #plt.show()
    
    # 箱ひげ図
    plt.boxplot(data)
    plt.title("Boxplot")
    plt.show()
    
    # バイオリンプロット
    plt.violinplot(data, widths=2, showmeans=True, showmedians=True, showextrema=True)
    plt.title("Violinplot")
    plt.show()
    
    # イベントプロット
    plt.eventplot(data, orientation="vertical")
    plt.title("Eventplot")
    plt.show()


# a以上b未満のランダム乱数size個...np.random.randint(a, b, size)
# 最小値low, 最大値highの一様乱数size個...np.random.uniform(low, high, size)
# 平均a, 標準偏差bの正規分布乱数size個...np.random.normal(a, b, size)
# ある期間に平均してscale回起こる現象が、次に起こるまでの期間Xが従う指数分布乱数size個...np.random.exponential(scale, size)
# 成功する確率がpのベルヌーイ試行をn回やる時の二項分布乱数size個...np.random.binomial(n, p, size)
# ベルヌーイ試行を何回か繰り返すときに、初めて成功するまでの回数を確率変数X、各試行の成功確率をpとしたときの幾何分布乱数size個...np.random.geometric(p, size)     
# 単位時間あたりにある事象が平均してlam回起こる場合に、その事象がx回起こる確率を示すポアソン分布size個...np.random.poisson(lam, size)
# 試行回数n回の結果がpvals通りある時の多項分布乱数size個...np.random.multinomial(n, pvals, size)...（当たり前だけど）リストで返ってくるので、今回の実験に向かない
# パラメータa,bのベータ分布size個...np.random.beta(a, b, size)
# 形状母数shape、尺度母数scaleのガンマ分布に従う乱数size個np.random.gamma(shape, scale, size)
# 自由度dfのカイ二乗分布に従う乱数size個...np.random.chisquare(df, size)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.DataFrame(np.random.normal(50, (250)**0.5, 1000), columns=["data"])["data"]
describe1(data)
