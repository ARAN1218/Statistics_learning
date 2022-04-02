# 三行データ記述統計
def describe3(data1, data2, data3):
    # 必要なライブラリをインポート
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats as st
    
    # 元データ出力
    print(data1, data2, data3, sep="\n")

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
    
    # 算術平均
    def mean(data):
        return sum_value(data) / length(data)
    
    # 幾何平均
    # 比率や割合等について適用する
    # 0以下の値が含まれていた場合、規格外としてErrorを出す
    def geometric_mean(data):
        try:
            ds = 1
            n = length(data)
            for d in data:
                ds *= d
            return (ds)**(1/n)
        except:
            return 'Error'
    
    # 調和平均
    # 時速の平均等に適用する
    # 0以下の値が含まれていた場合、規格外としてErrorを出す
    def harmonic_mean(data):
        try:
            ds = 0
            n = length(data)
            for d in data:
                ds += 1/d
            return 1 / ((1/n) * ds)
        except:
            return 'Error'
    
    # 平均偏差
    def meand(data):
        n = length(data)
        m = mean(data)
        s = 0
        for d in data:
            s += abs(d - m)
        return s / n
    
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
    
    # 母平均の95%信頼区間(母分散既知)
    def mean_95cl_known(data):
        n = length(data)
        m = mean(data)
        s = std_p(data)
        return "({:.2f}, {:.2f})".format(m-1.96*(s/n)**0.5, m+1.96*(s/n)**0.5)
    
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
    
    # 四分位偏差
    def quartile_range(data):
        return quartile3(data) - quartile1(data)
    
    # ミッドレンジ
    def mid_range(data):
        return (max_value(data) + min_value(data)) / 2
    
    # レンジ
    def all_range(data):
        return max_value(data) - min_value(data)
    
    # 最頻値
    def mode(data):
        # 個数を数える
        data_sorted = quick_sort(data)
        discoverd = {}
        for d in data_sorted:
            if d not in discoverd.keys():
                discoverd[d] = 1
            else:
                discoverd[d] += 1
        
        # 個数が最大のデータを検索する(複数個ある場合は全て出力する)
        discoverd_keys, discoverd_values = list(discoverd.keys()), list(discoverd.values())
        max_cnt = max_value(discoverd_values)
        mode = []
        for i, value in enumerate(discoverd_values):
            if max_cnt == value:
                mode += [discoverd_keys[i]]
        
        return str(mode)
    
    # 変動係数(coefficient of variance)
    # 単位が無く、直接の比較が困難な場合に平均を考慮した上での比率の比較ができる
    # 比率の比較なので、比例尺度のみ使用できる
    # 標準偏差は母分散を用いて計算する(統計学入門のp38式より)
    def cov(data):
        return std_p(data) / mean(data)
    
    # ジニ係数(gini coefficient)
    # 不平等度の指標として用いられる
    # 面積を表すので普通は負の値を取らないが、定義通りに計算して出力する。
    def gini(data):
        n = length(data)
        m = mean(data)
        s = 0
        for d1 in data:
            for d2 in data:
                s += abs(d1 - d2)
        return s / (2 * n**2 * m)
    
    # 歪度(skewness)
    # 母集団の分布の非対称性の程度・方向性を示す推定統計量
    # 値が正なら右の裾が長く、負なら左の裾が長い分布となっている
    def skewness(data):
        n = length(data)
        m = mean(data)
        s = std_s(data)
        
        three = 0
        for d in data:
            three += ((d-m) / s)**3
        
        return (n/((n-1)*(n-2))) * three
    
    # 尖度(kurtosis)
    # 母集団の分布の中心周囲部分の尖り具合を表す推定統計量
    # 値が正なら正規分布より尖っており、負なら正規分布より丸く鈍い形をしている
    def kurtosis(data):
        n = length(data)
        m = mean(data)
        s = std_s(data)
        
        four = 0
        for d in data:
            four += (d-m)**4 / s**4
        
        return ((n*(n+1))/((n-1)*(n-2)*(n-3))) * four - (3*(n-1)**2)/((n-2)*(n-3))
    
    # ジャック-ベラ検定(Jarque–Bera test)
    # 標本が正規分布に従っているかどうかを検定する
    # 帰無仮説：標本が正規分布に従う
    # 対立仮説：標本が正規分布に従わない
    # ソース：https://ja.wikipedia.org/wiki/ジャック–ベラ検定
    def jarque_bera(data):
        n = length(data)
        m = mean(data)
        
        s2, s3, s4 = 0, 0, 0
        for d in data:
            s2 += (d - m)**2
            s3 += (d - m)**3
            s4 += (d - m)**4
            
        # 標本歪度
        S = (s3/n) / (s2/n)**(3/2)
        # 標本尖度
        K = (s4/n) / (s2/n)**2
        
        JB = (n/6) * (S**2 + (1/4)*(K-3)**2)
        return JB
    
    # 母共分散
    # サンプルサイズが異なる場合、Errorを返す
    def cov_p(data1, data2):
        if length(data1) != length(data2): return 'Error'
        m1 = mean(data1)
        m2 = mean(data2)
        cov_p = 0
        for d1, d2 in zip(data1, data2):
            cov_p += (d1-m1) * (d2-m2)
        return cov_p / length(data1)
    
    # 不偏共分散
    # サンプルサイズが異なる場合、Errorを返す
    def cov_s(data1, data2):
        if length(data1) != length(data2): return 'Error'
        m1 = mean(data1)
        m2 = mean(data2)
        cov_s = 0
        for d1, d2 in zip(data1, data2):
            cov_s += (d1-m1) * (d2-m2)
        return cov_s / (length(data1)-1)
    
    # ピアソンの積率相関係数(Pearson product-moment correlation coefficient)
    # サンプルサイズが異なる場合、Errorを返す
    def pearson_cor(data1, data2):
        if length(data1) != length(data2): return 'Error'
        return cov_s(data1, data2) / (std_s(data1)*std_s(data2))
    
    # ピアソン無相関検定(検定統計量t)
    # 自由度n-2のt分布に従う
    # 帰無仮説：母集団のピアソン相関係数は0である(相関はない)
    # 対立仮説：母集団のピアソン相関係数は0以外である(相関がある)
    # 注：値がマイナスになるが、両側検定だから絶対値で考えてよい
    # サンプルサイズが異なる場合、Errorを返す
    def pearson_cor_test(data1, data2):
        cor = pearson_cor(data1, data2)
        if cor == 'Error':
            return 'Error'
        elif cor > 0:
            return (cor*((length(data1)-2)**0.5)) / ((1-(cor**2))**0.5)
        else:
            return (-cor*(length(data1)-2)**0.5) / ((1-cor**2)**0.5)
        
    # 偏相関係数(Partial correlation)
    # 第三引数のデータの影響を消した第一引数・第二引数のデータ間の相関関係を表す
    # 正規性を仮定しているので、スピアマンの順位相関係数では使えないはず
    # サンプルサイズが異なる場合、Errorを返す
    def partial_cor(data1, data2, data3):
        r12, r13, r23 = pearson_cor(data1, data2), pearson_cor(data1, data3), pearson_cor(data2, data3)
        if r12=='Error' or r13=='Error' or r23=='Error': return 'Error'
        return (r12 - r13*r23) / ((1-r13**2)**0.5 * (1-r23**2)**0.5)
    
    # 偏相関係数の無相関検定(検定統計量t)
    # 母偏相関係数が0であるかどうかを検定する
    # 帰無仮説：母偏相関係数が0である
    # 対立仮説：母偏相関係数が0ではない
    # サンプルサイズが異なる場合、Errorを返す
    def partial_cor_test(data1, data2, data3):
        p_cor = partial_cor(data1, data2, data3)
        if p_cor == 'Error': return 'Error'
        
        n = length(data1)
        return (abs(p_cor)*(n-3-2)**0.5) / (1-(p_cor)**2)**0.5
        
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
    
    # スピアマンの順位相関係数(Spearman's rank correlation coefficient)........................ちょっとだけズレがある
    # サンプルサイズが異なる場合、Errorを返す
    def spearman_cor(data1, data2):
        if length(data1) != length(data2): return 'Error'
        n = length(data1)
        rank1, a = rank(data1)
        rank2, b = rank(data2)
        spearman = 0
        for d1, d2 in zip(rank1, rank2):
            spearman += (d1 - d2)**2
        return 1 - ((6*spearman) / (n*(n**2-1)))
    
    # スピアマン無相関検定(検定統計量t)
    # 自由度n-2のt分布に従う
    # 帰無仮説：母集団のスピアマン順位相関係数は0である(相関はない)
    # 対立仮説：母集団のスピアマン順位相関係数は0以外である(相関がある)
    # サンプルサイズが異なる場合、Errorを返す
    def spearman_cor_test(data1, data2):
        spearman =  spearman_cor(data1, data2)
        if spearman == 'Error': return 'Error'
        return spearman * ((length(data1)-1)/(1-spearman**2))**0.5
    
    # ケンドールの順位相関係数(Kendall's rank correlation coefficient)
    # サンプルサイズが異なる場合、Errorを返す
    # ソース：https://manabitimes.jp/math/1286
    # ソース2：https://bellcurve.jp/statistics/glossary/1307.html
    def kendall_cor(data1, data2):
        n1, n2 = length(data1), length(data2)
        if n1 != n2: return 'Error'
        
        plus, minus, zero1, zero2 = 0, 0, 0, 0
        for i in range(n1-1):
            for j in range(i+1, n2):
                d = (data1[i]-data1[j])*(data2[i]-data2[j])
                if d == 0:
                    if data1[i] == data1[j]: zero1 += 1
                    else: zero2 += 1
                elif d > 0: plus += 1
                else: minus += 1
        
        N = (n1*(n1-1)) / 2
        T = (plus - minus) / ((N-zero1)**0.5 * (N-zero2)**0.5)
        return T
    
    # ケンドール無相関検定(検定統計量z)
    # 検定統計量zの値から標準正規分布表を読み、p値を読み取る
    # サンプルサイズが異なる場合、Errorを返す
    # ソース：https://oceanone.hatenablog.com/entry/2020/04/28/022222
    def kendall_cor_test(data1, data2):
        kendall = kendall_cor(data1, data2)
        if kendall == 'Error': return 'Error'
        n = length(data1)
        return kendall / ((2*(2*n+5)) / (9*n*(n-1)))**0.5
    
#     # コルモゴロフ-スミルノフ検定のために累積分布のリストを作成する
#     def cumulative_distribution(data, data_len):
#         data_cd = [0]
#         data_sorted = quick_sort(data)
#         #data_sorted[0], data_sorted[data_len-1] = 0, 1.0
        
#         for i in range(1, data_len-1):
#             if data_sorted[i]!=data_sorted[i-1]:
#                 data_cd.append(i / data_len)
#             else:
#                 data_cd.append(data_cd[i-1])
#         data_cd.append(1.0)
        
#         return data_cd
    
#     # コルモゴロフ-スミルノフ検定(Kolmogorov-Smirnov test)
#     # サンプルサイズが同じなら再現できたけど、それ以外だと難しい...
#     # 帰無仮説：2つの母集団の累積分布関数が等しい
#     # 対立仮説：2つの母集団の累積分布関数が異なる
#     def kolmogorov_smirnov(data1, data2):
#         n1, n2 = length(data1), length(data2)
#         data1_cd, data2_cd = cumulative_distribution(data1, n1), cumulative_distribution(data2, n2)
        
#         data_cd_sub = [abs(d1-d2) for d1,d2 in zip(data1_cd, data2_cd)]
#         D = max_value(data_cd_sub)
#         print(D)
#         K = D * ((n1*n2) / (n1+n2))**0.5
#         return "({}, {})".format(K, 1.35)
    
    # バートレット検定(Bartlett test)
    # 母集団に正規性がある3群標本間の等分散性を検定する
    # 自由度df=k-1のX2分布に従う
    # 帰無仮説：各群の分散は均一である
    # 対立仮説：各群の分散は均一でない
    # ソース：https://kusuri-jouhou.com/statistics/bartlett.html
    def bartlett(data1, data2, data3):
        n1, n2, n3 = length(data1), length(data2), length(data3)
        v1, v2, v3 = var_s(data1), var_s(data2), var_s(data3)
        N = n1 + n2 + n3
        k = 3
        
        se2 = ((n1-1)*v1 + (n2-1)*v2 + (n3-1)*v3) / ((n1+n2+n3) - k)
        M = (N-k)*np.log(se2) - ((n1-1)*np.log(v1) + (n2-1)*np.log(v2) + (n3-1)*np.log(v3))
        C = 1 + 1/(3*(k-1)) * (1/(n1-1) + 1/(n2-1) + 1/(n3-1) - 1/(N-k))
        X2 = M / C
        return X2, k-1
    
    # リストの要素を絶対値に直す
    # ルビーン検定のために実装
    def make_abs(data):
        abs_data = []
        for d in data:
            if d >= 0: abs_data += [d]
            else: abs_data += [-d]
        return abs_data
    
    # ルビーン検定(Levene's test)
    # 母集団に正規性がない3群標本間の等分散性を検定する
    # 自由度df=(k-1, N-k)のF分布に従う
    # 帰無仮説：各群の分散は均一である
    # 対立仮説：各群の分散は均一でない
    # ソース：https://ja.wikipedia.org/wiki/ルビーン検定
    # scipyの計算結果と合わないが、一応このサイトの例題と検定統計量の値が同じになった：https://istat.co.jp/sk_commentary/variance/Rubins-test
    def levene(data1, data2, data3):
        n1, n2, n3 = length(data1), length(data2), length(data3)
        m1, m2, m3 = mean(data1), mean(data2), mean(data3)
        N = n1 + n2 + n3
        k = 3
        z1, z2, z3 = make_abs(data1-m1), make_abs(data2-m2), make_abs(data3-m3)
        zm1, zm2, zm3 = mean(z1), mean(z2), mean(z3)
        zm = mean(list(z1) + list(z2) + list(z3))
        
        sz1, sz2, sz3 = 0, 0, 0
        for z in z1: sz1 += (z - zm1)**2
        for z in z2: sz2 += (z - zm2)**2
        for z in z3: sz3 += (z - zm3)**2
        
        W = ((N-k) / (k-1)) * ((n1*(zm1-zm)**2 + n2*(zm2-zm)**2 + n3*(zm3-zm)**2) / (sz1 + sz2 + sz3))
        return W, "({},{})".format(k-1, N-k)
    
    # ブラウン・フォーサイス検定(Brown-Forsythe test)
    # 母集団に正規性がない3群標本間の等分散性を検定する
    # ルビーン検定とは違いzの計算に中央値を用いるため、より非正規性に対してロバストである
    # 自由度df=(k-1, N-k)のF分布に従う
    # 帰無仮説：各群の分散は均一である
    # 対立仮説：各群の分散は均一でない
    # ソース：https://en.wikipedia.org/wiki/Brown–Forsythe_test
    def brown_forsythe(data1, data2, data3):
        n1, n2, n3 = length(data1), length(data2), length(data3)
        m1, m2, m3 = median(data1), median(data2), median(data3)
        N = n1 + n2 + n3
        k = 3
        z1, z2, z3 = make_abs(data1-m1), make_abs(data2-m2), make_abs(data3-m3)
        zm1, zm2, zm3 = mean(z1), mean(z2), mean(z3)
        zm = mean(list(z1) + list(z2) + list(z3))
        
        sz1, sz2, sz3 = 0, 0, 0
        for z in z1: sz1 += (z - zm1)**2
        for z in z2: sz2 += (z - zm2)**2
        for z in z3: sz3 += (z - zm3)**2
        
        W = ((N-k) / (k-1)) * ((n1*(zm1-zm)**2 + n2*(zm2-zm)**2 + n3*(zm3-zm)**2) / (sz1 + sz2 + sz3))
        return W, "({},{})".format(k-1, N-k)
    
    # 一元配置分散分析(One-way ANalysis Of VAriance)
    # 自由度(dfw, dfb)のF分布に従う
    def anova(data1, data2, data3):
        l1, l2, l3 = length(data1), length(data2), length(data3)
        m1, m2, m3 = mean(data1), mean(data2), mean(data3)
        m = mean(list(data1) + list(data2) + list(data3))
        
        sst = 0
        for data, mn in zip([data1, data2, data3], [m1, m2, m3]):
            for d in data:
                sst += (d - mn)**2
        ssw = ((m1-m)**2)*l1 + ((m2-m)**2)*l2 + ((m3-m)**2)*l3
        
        dft = (l1+l2+l3) - 1
        dfw = 3 - 1
        dfb = dft - dfw
        
        msw = ssw / dfw
        msb = sst / dfb
        
        f = msw / msb
        
        return f, dfw, dfb
    
    # 一元配置反復測定分散分析(One-way Repeated Measures ANOVA)
    # 自由度(df_model, df_error)のF分布に従う
    # ソース：https://www.spss-tutorials.com/repeated-measures-anova/
    # scipyライブラリに反復測定分散分析を計算するモジュールが無かったが、以下のURLのテストデータで答えが一致した
    # https://s-nako.work/ja/2020/01/paired-one-way-anova-and-multiple-comparisons-in-python/
    # サンプルサイズが異なる場合、Errorを返す
    def rm_anova(data1, data2, data3):
        n = length(data1)
        n2 = length(data2)
        n3 = length(data3)
        if n!=n2 or n!=n3 or n2!=n3: return 'Error', 'Error', 'Error'
        
        ss_within, count = 0, 0
        for d1, d2, d3 in zip(data1, data2, data3):
            m = mean([d1, d2, d3])
            ss_within += (d1 - m)**2
            ss_within += (d2 - m)**2
            ss_within += (d3 - m)**2
            
        m1, m2, m3 = mean(data1), mean(data2), mean(data3)
        m_grand = mean([m1, m2, m3])
        ss_model = 0
        for m in [m1, m2, m3]:
            ss_model += (m - m_grand)**2
        ss_model *= n
        
        ss_error = ss_within - ss_model
        
        k = 3
        df_model = k - 1
        df_error = (k-1)*(n-1)
        
        ms_model = ss_model / df_model
        ms_error = ss_error / df_error
        
        F = ms_model / ms_error
        return F, df_model, df_error
        
    
    # 全データ同一ランク付け
    # クラスカル=ウォリス検定の為に実装
    def rank_all(data1, data2, data3):
        data_linked = list(data1) + list(data2) + list(data3)
        data_linked, dup = rank(data_linked)
        data1_len, data2_len = length(data1), length(data2)
        data1_dec = data_linked[:data1_len]
        data2_dec = data_linked[data1_len:data1_len+data2_len]
        data3_dec = data_linked[data1_len+data2_len:]
        return data1_dec, data2_dec, data3_dec, dup
    
    # クラスカル=ウォリス検定(Kruskal-Wallis test)
    # 同順位がある時の補正がかかっているため、少しだけ高めに検定統計量が計算される
    # ソース：統計学図鑑
    def kruskal_wallis(data1, data2, data3):
        l1, l2, l3 = length(data1), length(data2), length(data3)
        ls = l1 + l2 + l3
        rank_sum, dup_sum = 0, 0
        data1_dec, data2_dec, data3_dec, dup = rank_all(data1, data2, data3)
        
        for data_ranked, l in zip([data1_dec, data2_dec, data3_dec], [l1, l2, l3]):
            rank_sum += sum_value(data_ranked)**2 / l
            
        if dup != []:
            for d in dup:
                dup_sum += d*d*d - d
        
        pre_h = (12 / (ls*(ls+1))) * rank_sum - (3*(ls+1))
        c = 1 - (dup_sum / (l*(l**2-1)))
        H = pre_h / c
        k = 3 - 1
        return H, k
    
    # インデックス毎のランク付け
    # フリードマン検定の為に実装
    def rank_index(data1, data2, data3):
        data1_ranked, data2_ranked, data3_ranked = [], [], []
        for d1, d2, d3 in zip(data1, data2, data3):
            data_ranked, dup = rank([d1, d2, d3])
            data1_ranked += [data_ranked[0]]
            data2_ranked += [data_ranked[1]]
            data3_ranked += [data_ranked[2]]
        return data1_ranked, data2_ranked, data3_ranked
    
    # フリードマン検定(Friedman test)
    # ソフトウェアの計算では、同順位があった際に検定統計量を少しだけ大きくする調整をしているらしい
    # ソース：https://sixsigmastudyguide.com/friedman-non-parametric-hypothesis-test/
    # サンプルサイズが異なる場合、Errorを返す
    def friedman_test(data1, data2, data3):
        n = length(data1)
        n2 = length(data2)
        n3 = length(data3)
        if n!=n2 or n!=n3 or n2!=n3: return 'Error', 'Error'
        k = 3
        data1_ranked, data2_ranked, data3_ranked = rank_index(data1, data2, data3)
        
        r2 = 0
        for data_ranked in [data1_ranked, data2_ranked, data3_ranked]:
            rank_sum = sum_value(data_ranked)**2
            r2 += rank_sum
            
        x20 = (12 / (n*k*(k+1))) * r2 - 3*n*(k+1)
        return x20, k-1
    
    # テューキー・クレーマー検定(Tukey-Kramer test).........................他の分析ライブラリ等で正確性が検査されていない
    # 群数が多い場合はボンフェローニより有意差が出やすい
    # 母集団の正規性と等分散性であることを要求される
    # 検定統計量はq分布に従い、読み取った値を√2で割った値と比較して高ければ帰無仮説を棄却する。
    # テューキーHSD法(Tukey’s honestly significant difference test)ではサンプルサイズが同数であることも要求される
    # ソース①：統計学図鑑
    def tukey_kramer(data1, data2, data3):
        l1, l2, l3 = length(data1), length(data2), length(data3)
        l = l1 + l2 + l3
        k = 3
        m1, m2, m3 = mean(data1), mean(data2), mean(data3)
        m = mean(list(data1) + list(data2) + list(data3))
        
        sst = 0
        for data, mn in zip([data1, data2, data3], [m1, m2, m3]):
            for d in data:
                sst += (d - mn)**2
        dfw = (l1+l2+l3) - 3
        MSe = sst / dfw #群内分散
        
        t1 = abs(m2-m3) / (MSe * ((1/l2) + (1/l3)))**0.5
        t2 = abs(m1-m3) / (MSe * ((1/l1) + (1/l3)))**0.5
        t3 = abs(m1-m2) / (MSe * ((1/l1) + (1/l2)))**0.5
        return "({:.3f}, ({}, {}))".format(t1, k, (l-k)), "({:.3f}, ({}, {}))".format(t2, k, (l-k)), "({:.3f}, ({}, {}))".format(t3, k, (l-k))
    
    # 全データ同一ランク付け
    # スティール・ドゥワス検定、スティール検定のために作成
    def rank_all_sd(data1, data2):
        data_linked = list(data1) + list(data2)
        data_linked, dup = rank(data_linked)
        data1_len = length(data1)
        data1_dec = data_linked[:data1_len]
        data2_dec = data_linked[data1_len:]
        return [i**2 for i in data1_dec], [i**2 for i in data2_dec], data1_dec
    
    # スティール・ドゥワス検定(Steel-Dwass test)
    # テューキー・クレーマー法のノンパラ版
    # ウィルコクソンの順位和検定が基礎となっている
    # (群数k, ∞)のt分布表の値を√2で割った値が棄却限界値(ソース参照)
    # ソース：https://imnstir.blogspot.com/2012/06/steel-dwassexcel.html
    def steel_dwass(data1, data2, data3):
        l1, l2, l3 = length(data1), length(data2), length(data3)
        n = l1 + l2 + l3
        k = 3
        E, V = [], []
        for first, second in [((data2,l2), (data3,l3)), ((data1,l1), (data3,l3)), ((data1,l1), (data2,l2))]:
            N = first[1] + second[1]
            E += [(first[1]*(N+1)) / 2]
            
            f, s, f2 = rank_all_sd(first[0], second[0])
            f, s, f2 = sum_value(f), sum_value(s), sum_value(f2)
            V += [(((first[1]*second[1])/(N*(N-1))) * ((f+s) - ((N*(N+1)**2)/4)), f2)]
            
        sd23 = "({:.3f}, {:.3f})".format(abs(V[0][1]-E[0])/(V[0][0])**0.5, 3.31/(2)**0.5)
        sd13 = "({:.3f}, {:.3f})".format(abs(V[1][1]-E[1])/(V[1][0])**0.5, 3.31/(2)**0.5)
        sd12 = "({:.3f}, {:.3f})".format(abs(V[2][1]-E[2])/(V[2][0])**0.5, 3.31/(2)**0.5)
        return sd23, sd13, sd12
    
    # ダネット検定(Dunnett's test)
    # 3群以上の比較にて対象群(1つ)と処理群(2つ以上)の組について検定したい時、通常の多重比較よりも棄却域の制限を緩めて検定ができる
    # 母集団の正規性・等分散性を仮定する
    # 帰無仮説：対象群と処理群で母平均値に差がない
    # 対立仮説：対象群と処理群で母平均値に差がある
    def dunnett(data1, data2, data3):
        # 統計検定量を求める
        N = 3
        n1, n2, n3 = length(data1), length(data2), length(data3)
        m1, m2, m3 = mean(data1), mean(data2), mean(data3)
        s2_1, s2_2, s2_3 = var_s(data1), var_s(data2), var_s(data3)
        
        s2 = ((n1-1)*s2_1 + (n2-1)*s2_2 + (n3-1)*s2_3) / ((n1+n2+n3)-N)
        t12 = abs(m1 - m2) / ((1/n1 + 1/n2)*s2)**0.5
        t13 = abs(m1 - m3) / ((1/n1 + 1/n3)*s2)**0.5
        
        return t12, t13
    
    # スティール検定(Steel test)
    # 3群以上の比較にて対象群(1つ)と処理群(2つ以上)の組について検定したい時、通常の多重比較よりも棄却域の制限を緩めて検定ができる
    # ダネット検定のノンパラ版
    # 母集団の正規性・等分散性を仮定しない
    # 帰無仮説：対象群と処理群で母平均値に差がない
    # 対立仮説：対象群と処理群で母平均値に差がある
    def steel(data1, data2, data3):
        n1, n2, n3 = length(data1), length(data2), length(data3)
        n = n1 + n2 + n3
        E, V = [], []
        
        for first, second in [((data1,n1), (data2,n2)), ((data1,n1), (data3,n3))]:
            N = first[1] + second[1]
            E += [(first[1]*(N+1)) / 2]
            
            f, s, f2 = rank_all_sd(first[0], second[0])
            f, s, f2 = sum_value(f), sum_value(s), sum_value(f2)
            V += [(((first[1]*second[1])/(N*(N-1))) * ((f+s) - ((N*(N+1)**2)/4)), f2)]
            
        t12 = abs(V[0][1]-E[0])/(V[0][0])**0.5
        t13 = abs(V[1][1]-E[0])/(V[1][0])**0.5
        
        return t12, t13

    
    bl_x2, bl_df = bartlett(data1, data2, data3)
    le_w, le_df = levene(data1, data2, data3)
    bf_w, bf_df = brown_forsythe(data1, data2, data3)
    anova_f, anova_dfw, anova_dfb = anova(data1, data2, data3)
    rm_anova_f, rm_anova_df_model, rm_anova_df_error = rm_anova(data1, data2, data3)
    kw_h, kw_k = kruskal_wallis(data1, data2, data3)
    fm_x20, fm_k = friedman_test(data1, data2, data3)
    tk1, tk2, tk3 = tukey_kramer(data1, data2, data3)
    sd23, sd13, sd12 = steel_dwass(data1, data2, data3)
    dun12, dun13 = dunnett(data1, data2, data3)
    st12, st13 = steel(data1, data2, data3)
    
    
    df1 = pd.DataFrame({
        'count':length(data1),
        'sum':sum_value(data1),
        'mean':mean(data1),
        'g_mean':geometric_mean(data1),
        'h_mean':harmonic_mean(data1),
        'meand':meand(data1),
        'var.p':var_p(data1),
        'var.s':var_s(data1),
        'std.p':std_p(data1),
        'std.s':std_s(data1),
        'std_e':std_e(data1),
        'mean_95cl_known':mean_95cl_known(data1),
        'min':min_value(data1),
        '25%':quartile1(data1),
        '50%':median(data1),
        '75%':quartile3(data1),
        'max':max_value(data1),
        '25-75%':quartile_range(data1),
        'mid-range':mid_range(data1),
        'range':all_range(data1),
        'mode':mode(data1),
        'cov':cov(data1),
        'gini':gini(data1),
        'skewness':skewness(data1),
        'kurtosis':kurtosis(data1),
        'cov.p':cov_p(data2, data3),
        'cov.s':cov_s(data2, data3),
        'Pearson_cor':pearson_cor(data2, data3),
        'Pearson_cor_test.t':pearson_cor_test(data2, data3),
        'partial_cor':partial_cor(data2, data3, data1),
        'partial_cor_test.t':partial_cor_test(data2, data3, data1),
        'Spearman_cor':spearman_cor(data2, data3),
        'Spearman_cor_test.t':spearman_cor_test(data2, data3),
        'Kendall_cor':kendall_cor(data2, data3),
        'Kendall_cor_test.z':kendall_cor_test(data2, data3),
        #'Kolmogorov-Smirnov':kolmogorov_smirnov(data2, data3),
        'Jarque-Bara_test.x2':jarque_bera(data1),
        "bartlett.x2":bl_x2,
        "levene.F":le_w,
        "Brown-Forsythe_test.F":bf_w,
        'ANOVA.F':anova_f,
        'RM_ANOVA.F':rm_anova_f,
        'Kruskal-Wallis_test.H':kw_h,
        "Friedman_test.Q":fm_x20,
        "Tukey-kramer_test.q":tk1,
        "Steel-dwass_test.t":sd23
    }, index=["data1"]).T
    
    df2 = pd.DataFrame({
        'count':length(data2),
        'sum':sum_value(data2),
        'mean':mean(data2),
        'g_mean':geometric_mean(data2),
        'h_mean':harmonic_mean(data2),
        'meand':meand(data2),
        'var.p':var_p(data2),
        'var.s':var_s(data2),
        'std.p':std_p(data2),
        'std.s':std_s(data2),
        'std_e':std_e(data2),
        'mean_95cl_known':mean_95cl_known(data2),
        'min':min_value(data2),
        '25%':quartile1(data2),
        '50%':median(data2),
        '75%':quartile3(data2),
        'max':max_value(data2),
        '25-75%':quartile_range(data2),
        'mid-range':mid_range(data2),
        'range':all_range(data2),
        'mode':mode(data2),
        'cov':cov(data2),
        'gini':gini(data2),
        'skewness':skewness(data2),
        'kurtosis':kurtosis(data2),
        'cov.p':cov_p(data1, data3),
        'cov.s':cov_s(data1, data3),
        'Pearson_cor':pearson_cor(data1, data3),
        'Pearson_cor_test.t':pearson_cor_test(data1, data3),
        'partial_cor':partial_cor(data1, data3, data2),
        'partial_cor_test.t':partial_cor_test(data1, data3, data2),
        'Spearman_cor':spearman_cor(data1, data3),
        'Spearman_cor_test.t':spearman_cor_test(data1, data3),
        'Kendall_cor':kendall_cor(data1, data3),
        'Kendall_cor_test.z':kendall_cor_test(data1, data3),
        #'Kolmogorov-Smirnov':kolmogorov_smirnov(data1, data3),
        'Jarque-Bara_test.x2':jarque_bera(data2),
        "bartlett.x2":bl_df,
        "levene.F":le_df,
        "Brown-Forsythe_test.F":bf_df,
        'ANOVA.F':anova_dfw,
        'RM_ANOVA.F':rm_anova_df_model,
        'Kruskal-Wallis_test.H':kw_k,
        "Friedman_test.Q":fm_k,
        "Tukey-kramer_test.q":tk2,
        "Steel-dwass_test.t":sd13,
        "Dunnett_test.t":dun12,
        "Steel_test.t":st12
    }, index=["data2"]).T
    
    df3 = pd.DataFrame({
        'count':length(data3),
        'sum':sum_value(data3),
        'mean':mean(data3),
        'g_mean':geometric_mean(data3),
        'h_mean':harmonic_mean(data3),
        'meand':meand(data3),
        'var.p':var_p(data3),
        'var.s':var_s(data3),
        'std.p':std_p(data3),
        'std.s':std_s(data3),
        'std_e':std_e(data3),
        'mean_95cl_known':mean_95cl_known(data3),
        'min':min_value(data3),
        '25%':quartile1(data3),
        '50%':median(data3),
        '75%':quartile3(data3),
        'max':max_value(data3),
        '25-75%':quartile_range(data3),
        'mid-range':mid_range(data3),
        'range':all_range(data3),
        'mode':mode(data3),
        'cov':cov(data3),
        'gini':gini(data3),
        'skewness':skewness(data3),
        'kurtosis':kurtosis(data3),
        'cov.p':cov_p(data1, data2),
        'cov.s':cov_s(data1, data2),
        'Pearson_cor':pearson_cor(data1, data2),
        'Pearson_cor_test.t':pearson_cor_test(data1, data2),
        'partial_cor':partial_cor(data1, data2, data3),
        'partial_cor_test.t':partial_cor_test(data1, data2, data3),
        'Spearman_cor':spearman_cor(data1, data2),
        'Spearman_cor_test.t':spearman_cor_test(data1, data2),
        'Kendall_cor':kendall_cor(data1, data2),
        'Kendall_cor_test.z':kendall_cor_test(data1, data2),
        #'Kolmogorov-Smirnov':kolmogorov_smirnov(data1, data2),
        'Jarque-Bara_test.x2':jarque_bera(data3),
        'ANOVA.F':anova_dfb,
        'RM_ANOVA.F':rm_anova_df_error,
        "Tukey-kramer_test.q":tk3,
        "Steel-dwass_test.t":sd12,
        "Dunnett_test.t":dun13,
        "Steel_test.t":st13
    }, index=["data3"]).T
    
    # 結果出力
    display(pd.concat([df1, df2, df3], axis=1))
    
    # 種々のグラフをプロット
    # ヒストグラム
    fig, axes= plt.subplots(1,3)
    axes[0].hist(data1, bins=(1+int(np.log2(length(data1)))))
    axes[1].hist(data2, bins=(1+int(np.log2(length(data2)))))
    axes[2].hist(data3, bins=(1+int(np.log2(length(data3)))))
    plt.show()
    
    # ヒストグラム(累積)
    fig, axes= plt.subplots(1,3)
    axes[0].hist(data1, bins=(1+int(np.log2(length(data1)))), cumulative=True)
    axes[1].hist(data2, bins=(1+int(np.log2(length(data2)))), cumulative=True)
    axes[2].hist(data3, bins=(1+int(np.log2(length(data3)))), cumulative=True)
    plt.show()
    
    # Q-Qプロット
    # 点が一直線に並んでいれば、そのデータの母集団の分布は正規分布に従う
    fig, axes= plt.subplots(1,3)
    st.probplot(data1, dist="norm", plot=axes[0])
    st.probplot(data2, dist="norm", plot=axes[1])
    st.probplot(data3, dist="norm", plot=axes[2])
    plt.show()
    
    # 箱ひげ図
    plt.boxplot([data1, data2, data3], positions=[1, 2, 3])
    plt.title("Boxplot")
    plt.show()
    
    # バイオリンプロット
    plt.violinplot([data1, data2, data3], positions=[1.2, 1.8, 2.4])
    plt.title("Violinplot")
    plt.show()
    
    # イベントプロット
    plt.eventplot([data1, data2, data3], orientation="vertical", lineoffsets=[2, 4, 6], linewidth=0.75)
    plt.title("Eventplot")
    plt.show()
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st

# ランダム分布
data1 = pd.DataFrame(np.random.randint(-100, 101, 100), columns=["data1"])["data1"]
data2 = pd.DataFrame(np.random.randint(-100, 101, 100), columns=["data2"])["data2"]
data3 = pd.DataFrame(np.random.randint(-100, 101, 100), columns=["data3"])["data3"]
describe3(data1, data2, data3)

# 既存ライブラリで検証
print("PearsonrResult" + str(st.pearsonr(data1, data2)))
print(st.spearmanr(data1, data2))
print(st.kendalltau(data1, data2))
print(st.kstest(data1, st.norm(loc=np.mean(data1), scale=np.std(data1)).cdf))
print(st.bartlett(data1, data2, data3))
print(st.levene(data1, data2, data3))
print(st.f_oneway(data1, data2, data3))
print(st.kruskal(data1, data2, data3))
print(st.friedmanchisquare(data1, data2, data3))
