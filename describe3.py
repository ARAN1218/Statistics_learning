# 三行データ記述統計
def describe3(data1, data2, data3):
    # 必要なライブラリをインポート
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 元データ出力
    print(data1, data2, data3)

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
    
    # 母共分散
    def cov_p(data1, data2):
        m1 = mean(data1)
        m2 = mean(data2)
        cov_p = 0
        for d1, d2 in zip(data1, data2):
            cov_p += (d1-m1) * (d2-m2)
        return cov_p / length(data1)
    
    # 不偏共分散
    def cov_s(data1, data2):
        m1 = mean(data1)
        m2 = mean(data2)
        cov_s = 0
        for d1, d2 in zip(data1, data2):
            cov_s += (d1-m1) * (d2-m2)
        return cov_s / (length(data1)-1)
    
    # ピアソン相関係数
    def pearson_cor(data1, data2):
        return cov_s(data1, data2) / (std_s(data1)*std_s(data2))
    
    # ピアソン無相関検定(検定統計量)
    # 自由度n-2のt分布に従う
    # 帰無仮説：母集団のピアソン相関係数は0である(相関はない)
    # 対立仮説：母集団のピアソン相関係数は0以外である(相関がある)
    # 注：値がマイナスになるが、両側検定だから絶対値で考えてよい
    def pearson_cor_test(data1, data2):
        cor = pearson_cor(data1, data2)
        if cor > 0:
            return (cor*((length(data1)-2)**0.5)) / ((1-(cor**2))**0.5)
        else:
            return (-cor*(length(data1)-2)**0.5) / ((1-cor**2)**0.5)
        
    # リストのコピー生成
    def copy(data):
        copyed = []
        for d in data:
            copyed += [d]
        return copyed
    
    # ランク付け
    # 同順位は平均順位をつけ、同順位の数も返り値で返す
    def rank(data):
        ranking = 0
        rank_num_list = []
        l = length(data)
        reference, ranked = copy(data), copy(data)
        for d in data:
            count, rank_sum, rank_num = 0, 0, 0
            count_list = []
            max_val = max_value(reference)
            if max_val == -999999: break
            for d in data:
                if max_val == reference[count]:
                    reference[count] = -999999
                    count_list += [count]
                    rank_sum += l - ranking
                    rank_num += 1
                    ranking += 1
                count += 1
            for c in count_list:
                ranked[c] = rank_sum / rank_num
            if rank_num != 0:
                rank_num_list += [rank_num]
        return ranked, rank_num_list
    
    # スピアマン順位相関係数........................ちょっとだけズレがある
    def spearman_cor(data1, data2):
        n = length(data1)
        rank1, a = rank(data1)
        rank2, b = rank(data2)
        spearman = 0
        for d1, d2 in zip(rank1, rank2):
            spearman += (d1 - d2)**2
        return 1 - ((6*spearman) / (n*(n**2-1)))
    
    # スピアマン無相関検定(検定統計量)
    # 自由度n-2のt分布に従う
    # 帰無仮説：母集団のスピアマン順位相関係数は0である(相関はない)
    # 対立仮説：母集団のスピアマン順位相関係数は0以外である(相関がある)
    def spearman_cor_test(data1, data2):
        spearman =  spearman_cor(data1, data2)
        return spearman * ((length(data1)-1)/(1-spearman**2))**0.5
    
    # 一元配置分散分析(ANalysis Of VAriance)
    def anova(data1, data2, data3):
        l1, l2, l3 = length(data1), length(data2), length(data3)
        m1, m2, m3 = mean(data1), mean(data2), mean(data3)
        m = (m1+m2+m3) / 3
        
        sst = 0
        for data in [data1, data2, data3]:
            for d in data:
                sst += (d - m)**2
        ssw = ((m1-m)**2)*l1 + ((m2-m)**2)*l2 + ((m3-m)**2)*l3
        ssb = sst - ssw
        
        dft = (l1+l2+l3) - 1
        dfw = 3 - 1
        dfb = dft - dfw
        
        msw = ssw / dfw
        msb = ssb / dfb
        
        f = msw / msb
        
        return f, dfw, dfb
    
    # クラスカル=ウォリス検定-----------------未完成
    # 値がおかしい
    def kruskalwallis_test(data1, data2, data3):
        l1, l2, l3 = length(data1), length(data2), length(data3)
        ls = l1 + l2 + l3
        rank_sum, dup_sum = 0, 0
        
        for data, l in zip([data1, data2, data3], [l1, l2, l3]):
            ranked, dup = rank(data)
            print(sum_value(ranked)**2 / l)
            rank_sum += sum_value(ranked)**2 / l
        
            if dup != []:
                for d in dup:
                    dup_sum += d*d*d - d
                    
        print("dup_sum:", dup_sum)
        print("rank_sum", rank_sum)
        pre_h = (12 / (ls*(ls+1))) * rank_sum - (3*(ls+1))
        c = 1 - (dup_sum / (l*(l**2-1)))
        h = pre_h / c
        k = 3 - 1
        return h, k

    
    anova_f, anova_dfw, anova_dfb = anova(data1, data2, data3)
    kw_h, kw_k = kruskalwallis_test(data1, data2, data3)
    
    df1 = pd.DataFrame({
        'count':length(data1),
        'sum':sum_value(data1),
        'mean':mean(data1),
        'var.p':var_p(data1),
        'var.s':var_s(data1),
        'std.p':std_p(data1),
        'std.s':std_s(data1),
        'min':min_value(data1),
        '25%':quartile1(data1),
        '50%':median(data1),
        '75%':quartile3(data1),
        'max':max_value(data1),
        '25-75%':quartile_range(data1),
        'cov.p':cov_p(data2, data3),
        'cov.s':cov_s(data2, data3),
        'pearson_cor':pearson_cor(data2, data3),
        'pearson_cor_test':pearson_cor_test(data2, data3), #自由度n1-n2+2のt分布表を見ること
        'spearman_cor':spearman_cor(data2, data3),
        'spearman_cor_test':spearman_cor_test(data2, data3),
        'anova':anova_f,
        'kw_test':kw_h
    }, index=["data1"]).T
    
    df2 = pd.DataFrame({
        'count':length(data2),
        'sum':sum_value(data2),
        'mean':mean(data2),
        'var.p':var_p(data2),
        'var.s':var_s(data2),
        'std.p':std_p(data2),
        'std.s':std_s(data2),
        'min':min_value(data2),
        '25%':quartile1(data2),
        '50%':median(data2),
        '75%':quartile3(data2),
        'max':max_value(data2),
        '25-75%':quartile_range(data2),
        'cov.p':cov_p(data1, data3),
        'cov.s':cov_s(data1, data3),
        'pearson_cor':pearson_cor(data1, data3),
        'pearson_cor_test':pearson_cor_test(data1, data3), #自由度n1-n2+2のt分布表を見ること
        'spearman_cor':spearman_cor(data1, data3),
        'spearman_cor_test':spearman_cor_test(data1, data3),
        'anova':anova_dfw,
        'kw_test':kw_k
    }, index=["data2"]).T
    
    df3 = pd.DataFrame({
        'count':length(data3),
        'sum':sum_value(data3),
        'mean':mean(data3),
        'var.p':var_p(data3),
        'var.s':var_s(data3),
        'std.p':std_p(data3),
        'std.s':std_s(data3),
        'min':min_value(data3),
        '25%':quartile1(data3),
        '50%':median(data3),
        '75%':quartile3(data3),
        'max':max_value(data3),
        '25-75%':quartile_range(data3),
        'cov.p':cov_p(data1, data2),
        'cov.s':cov_s(data1, data2),
        'pearson_cor':pearson_cor(data1, data2),
        'pearson_cor_test':pearson_cor_test(data1, data2), #自由度n1-n2+2のt分布表を見ること
        'spearman_cor':spearman_cor(data1, data2),
        'spearman_cor_test':spearman_cor_test(data1, data2),
        'anova':anova_dfb
    }, index=["data3"]).T
    
    return display(pd.concat([df1, df2, df3], axis=1))

from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ランダム分布
data1 = pd.DataFrame([int(random() * 11) for i in range(100)], columns=["data1"])["data1"]
data2 = pd.DataFrame([int(random() * 11) for i in range(100)], columns=["data2"])["data2"]
data3 = pd.DataFrame([int(random() * 11) for i in range(100)], columns=["data3"])["data3"]
describe3(data1, data2, data3)

# 既存ライブラリで検証
from scipy import stats as st
print("PearsonrResult", st.pearsonr(data1, data2))
print(st.spearmanr(data1, data2))
print(st.f_oneway(data1, data2, data3))
print(st.kruskal(data1, data2, data3))
