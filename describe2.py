# 二行データ記述統計
def describe2(data1, data2):
    # 必要なライブラリをインポート
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 元データ出力
    print(data1, data2)
    
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
    
    # 不偏標準偏差
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
    
    # 等分散の検定(F比)
    # 帰無仮説：2群間の母分散に差がない(等分散である)
    # 対立仮説：2群間の母分散に差がある(等分散でない)
    def f_test(data1, data2):
        s1, s2 = var_s(data1), var_s(data2)
        return (s1 / s2) if (s1 < s2) else (s2 / s1), "({}, {})".format(length(data1)-1, length(data2)-1)
    
    # 対応なし2標本t検定
    # 自由度n1+n2-1のt分布に従う
    # 帰無仮説：2群間の母平均値に差がない(母平均値が等しい)
    # 対立仮説：2群間の母平均値に差がある(母平均値が異なる)
    def independent_ttest(data1, data2):
        n1, n2 = length(data1), length(data2)
        m1, m2 = mean(data1), mean(data2)
        s1, s2 = std_s(data1), std_s(data2)
        s = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)
        return (m1-m2) / (s * (1/n1 + 1/n2))**0.5
    
    # 対応あり2標本t検定
    # 自由度n1+n2-1のt分布に従う
    # 帰無仮説：2群間の母平均値に差がない(母平均値が等しい)
    # 対立仮説：2群間の母平均値に差がある(母平均値が異なる)
    def dependent_ttest(data1, data2):
        diff = data1 - data2
        n = length(diff)
        m = mean(diff)
        s = var_s(diff)
        return m / (s/n)**0.5
    
    # ウェルチのt検定
    # ウェルチ=サタスウェイト（Welch=Satterthwaite）の式により近似自由度のt分布に従う
    # 帰無仮説：2群間の母平均値に差がない(母平均値が等しい)
    # 対立仮説：2群間の母平均値に差がある(母平均値が異なる)
    def welch_ttest(data1, data2):
        n1, n2 = length(data1), length(data2)
        m1, m2 = mean(data1), mean(data2)
        s1, s2 = var_s(data1), var_s(data2)
        t = (m1-m2) / ((s1/n1) + (s2/n2))**0.5
        v = int(((s1/n1) + (s2/n2))**2 / (s1**2/(n1**2*(n1-1)) + s2**2/(n2**2*(n2-1))))
        return t, v
    
    # マン=ホイットニーのU検定-----------------未完成
    # 2群間の中央値に差があるかを検定する
    # サンプルサイズが十分に大きい時(n>20)、検定統計量zは標準正規分布に従うかどうかを考え、その有意性は正規分布表で確認できる
    # 帰無仮説：2群間の母集団は同じである
    # 対立仮説：2群間の母集団は異なる
    def mannwhitney_utest(data1, data2): # 標準正規分布に従うかどうかを考え、その有意性は正規分布表で確認できる
        n1, n2 = length(data1), length(data2)
        r1, r2 = sum_value(rank(data1)[0]), sum_value(rank(data2)[0])
        #u1 = n1*n2 + ((n1*(n1+1)) / 2) - r1
        #u2 = n1*n2 + ((n2*(n2+1)) / 2) - r2
        #u1 = r1 - (n1*(n1+1))/2
        #u2 = r2 - (n2*(n2+1))/2
        #u = u1 if (u1 < u2) else u2
        if (n1 <= 20 and n2 <= 20):
            num_bigger_value = []

            for i in data1:
                # i よりも大きい値がの個数
                bigger_temp = len(np.where(data2>i)[0])

                # i と同じ値の個数（平均をとるので２で割る）
                equal_temp = len(np.where(data2==i)[0]) / 2
                num_bigger_value.append(bigger_temp + equal_temp)

            U = np.array(num_bigger_value).sum()
            return U
            
        else:
            num_bigger_value = []

            for i in data1:
                bigger_temp = len(np.where(data2>i)[0])
                equal_temp = len(np.where(data2==i)[0]) / 2
                num_bigger_value.append(bigger_temp + equal_temp)

            U = np.array(num_bigger_value).sum()
            return U
        
    # ウィルコクソンの順位和検定-----------------未完成
    # 2群間の中央値に差があるかを検定する
    # ウィルコクソンの順位和検定の数表を参照して有意差があるかどうか判定する
    # 帰無仮説：2群間の母集団は同じである
    # 対立仮説：2群間の母集団は異なる
    def wilcoxon_rstest(data1, data2):
        n1, n2 = length(data1), length(data2)
        r1, r2 = sum_value(rank(data1)[0]), sum_value(rank(data2)[0])
        if r1 < r2:
            w = r1
            ew = (n1*(n1+n2+1)) / 2
        else:
            w = r2
            ew = (n2*(n1+n2+1)) / 2
        vw = (n1*n2*(n1+n2+1)) / 12
        return (ew-w) / (vw)**0.5
            
    
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
        'cov.p':cov_p(data1, data2),
        'cov.s':cov_s(data1, data2),
        'pearson_cor':pearson_cor(data1, data2),
        'pearson_cor_test':pearson_cor_test(data1, data2), #自由度n1-n2+2のt分布表を見ること
        'spearman_cor':spearman_cor(data1, data2),
        'spearman_cor_test':spearman_cor_test(data1, data2),
        'f_test':f_test(data1, data2)[0],
        'indep_ttest.t':independent_ttest(data1, data2),
        'dep_ttest.t':dependent_ttest(data1, data2),
        'welch.ttest':welch_ttest(data1, data2)[0], #自由度vのt分布表を見ること
        'mw_utest.u':mannwhitney_utest(data1, data2),
        'w_rstest.tw':wilcoxon_rstest(data1, data2)
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
        'f_test':f_test(data1, data2)[1],
        'welch.ttest':welch_ttest(data1, data2)[1]
    }, index=["data2"]).T
    
    return display(pd.concat([df1, df2], axis=1))

from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ランダム分布
data1 = pd.DataFrame([int(random() * 101) for i in range(100)], columns=["data1"])["data1"]
data2 = pd.DataFrame([int(random() * 101) for i in range(100)], columns=["data2"])["data2"]
describe2(data1, data2)

# 既存のライブラリで検証
from scipy import stats as st
print("PearsonrResult", st.pearsonr(data1, data2))
print(st.spearmanr(data1, data2))
print("indep_ttest", st.ttest_ind(data1, data2, equal_var=True))
print("dep_ttest", st.ttest_rel(data1, data2))
print("Welch", st.ttest_ind(data1, data2, equal_var=False))
print(st.mannwhitneyu(data1, data2, alternative='two-sided'))
print(st.wilcoxon(data1, data2))
