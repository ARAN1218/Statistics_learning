# 二行データ記述統計
def describe2(data1, data2):
    # 必要なライブラリをインポート
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 元データ出力
    print(data1, data2, sep="\n")
    
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
    # 検定統計量は自由度n-2のt分布に従う
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
        
    # リストのコピー生成
    def copy(data):
        copyed = []
        for d in data:
            copyed += [d]
        return copyed
    
    # ランク付け
    # 同順位は平均順位をつけ、同順位の数も返り値で返す
    # 引数deliteはウィルコクソンの符号順位検定にて順位の整合性を合わせる為に指定した
    def rank(data, delite=0):
        ranking = 0
        rank_num_list = []
        l = length(data) - delite
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
        n = length(data1)
        if n != length(data2): return 'Error'
        rank1, a = rank(data1)
        rank2, b = rank(data2)
        spearman = 0
        for d1, d2 in zip(rank1, rank2):
            spearman += (d1 - d2)**2
        return 1 - ((6*spearman) / (n*(n**2-1)))
    
    # スピアマン無相関検定(検定統計量t)
    # 検定統計量は自由度n-2のt分布に従う
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
    # 検定統計量は標準正規分布に従う
    # サンプルサイズが異なる場合、Errorを返す
    # ソース：https://oceanone.hatenablog.com/entry/2020/04/28/022222
    def kendall_cor_test(data1, data2):
        kendall = kendall_cor(data1, data2)
        if kendall == 'Error': return 'Error'
        n = length(data1)
        return kendall / ((2*(2*n+5)) / (9*n*(n-1)))**0.5
    
    # 単回帰(simple regression)
    # data1を独立変数、data2を従属変数として、最小二乗法を用いて直線への当てはめを行う
    # y = bx + aとした時、bを回帰係数、aをy切片と呼び、これらを求める
    # また、独立変数が従属変数を決定する度合いとして決定係数も計算する(ピアソンの積率相関係数の二乗)
    # サンプルサイズが異なる場合、Errorを返す
    def simple_regression(data1, data2):
        nx, ny = length(data1), length(data2)
        if nx!=ny: return np.nan, np.nan, 'Error'
        mx, my = mean(data1), mean(data2)
        sx2 = sum_value(data1**2)
        sxy = 0
        for x, y in zip(data1, data2):
            sxy += x * y
            
        b = (sxy - nx*mx*my) / (sx2 - nx*mx**2)
        a = my - b*mx
        r2 = pearson_cor(data1, data2)**2
        return b, a, r2
    
    # 単回帰係数の無相関検定(検定統計量t)
    # 単回帰における回帰係数が0でない(=きちんと影響がある)事を検定する
    # 検定統計量は自由度n-2のt分布に従う
    # 帰無仮説：回帰係数が0である
    # 対立仮説：回帰係数が0ではない
    # サンプルサイズが異なる場合、Errorを返す
    def simple_regression_test(data1, data2):
        b, a, r2 = simple_regression(data1, data2)
        if r2 == 'Error': return np.nan, np.nan
        
        n, mx = length(data1), mean(data1)
        xx_, yy_ = 0, 0
        for d1, d2 in zip(data1, data2):
            xx_ += (d1 - mx)**2
            yy_ += (d2 - (b*d1 + a))**2
            
        return (b*(xx_)**0.5) / (yy_/(n-2))**0.5, n-2

    # 等分散の検定(F比)
    # 帰無仮説：2群間の母分散に差がない(等分散である)
    # 対立仮説：2群間の母分散に差がある(等分散でない)
    def f_test(data1, data2):
        s1, s2 = var_s(data1), var_s(data2)
        return (s1 / s2) if (s1 < s2) else (s2 / s1), "({}, {})".format(length(data1)-1, length(data2)-1)
    
    # 効果量1(Cohen's d)
    # 対応なし2標本t検定において、それがどの程度の効果を持つのかという指標
    # 目安...0.2:小、0.5:中、0.8:大
    # ソースでは母分散を使用していたが、母集団を推定するという役割を考え、自分の関数は普遍分散を使用している
    # ソース：https://bellcurve.jp/statistics/course/12765.html
    def ind_cohen_d(data1, data2):
        n1, n2 = length(data1), length(data2)
        bool_s = ((n1*var_s(data1) + n2*var_s(data2)) / (n1 + n2))**0.5
        return (mean(data1) - mean(data2)) / bool_s
    
    # 対応なし2標本t検定(Student t-test)
    # サンプルサイズの合計が100未満の時、検定統計量Tは自由度n1+n2-1のt分布に従う
    # サンプルサイズの合計が100以上の時、検定統計量zは標準正規分布表を参照して
    # 帰無仮説：2群間の母平均値に差がない(母平均値が等しい)
    # 対立仮説：2群間の母平均値に差がある(母平均値が異なる)
    def independent_ttest(data1, data2):
        n1, n2 = length(data1), length(data2)
        m1, m2 = mean(data1), mean(data2)
        s1, s2 = std_s(data1), std_s(data2)
        
        s = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)
        return (m1-m2) / (s * (1/n1 + 1/n2))**0.5, (n1+n2-1)
        
    # z検定(対応なし)
    # 2標本の母平均値に差があるかを検定する
    # 母分散が既知、またはサンプルサイズの合計が100以上の時、検定統計量zは標準正規分布表を参照して検定する
    # 帰無仮説：2群間の母平均値に差がない(母平均値が等しい)
    # 対立仮説：2群間の母平均値に差がある(母平均値が異なる)
    def independent_ztest(data1, data2):
        n1, n2 = length(data1), length(data2)
        m1, m2 = mean(data1), mean(data2)
        s1, s2 = std_p(data1), std_p(data2)
        
        z = (m1-m2) / ((s1/n1) + (s2/n2))**0.5
        return z
    
    # 効果量2(Cohen's d)
    # 対応あり2標本t検定において、それがどの程度の効果を持つのかという指標
    # 目安...0.2:小、0.5:中、0.8:大
    # サンプルサイズが異なる場合、Errorを返す
    # ソース：https://bookdown.org/sbtseiji/lswjamoviJ/ch-ttest.html#sec:cohensd
    def dep_cohen_d(data1, data2):
        if length(data1) != length(data2): return 'Error'
        diff = data1 - data2
        return mean(diff) / std_s(diff)
    
    # 対応あり2標本t検定(Paired Samples t-test)
    # 自由度n1-1のt分布に従う
    # 帰無仮説：2群間の母平均値に差がない(母平均値が等しい)
    # 対立仮説：2群間の母平均値に差がある(母平均値が異なる)
    # サンプルサイズが異なる場合、Errorを返す
    def dependent_ttest(data1, data2):
        if length(data1) != length(data2): return 'Error', 'Error'
        diff = data1 - data2
        n = length(diff)
        m = mean(diff)
        s = var_s(diff)
        return m / (s/n)**0.5, (n-1)
    
    # z検定(対応あり)
    # 2標本の母平均値に差があるかを検定する
    # 母分散が既知、またはサンプルサイズの合計が100以上の時、検定統計量zは標準正規分布表を参照して検定する
    # 帰無仮説：2群間の母平均値に差がない(母平均値が等しい)
    # 対立仮説：2群間の母平均値に差がある(母平均値が異なる)
    # サンプルサイズが異なる場合、Errorを返す
    def dependent_ztest(data1, data2):
        if length(data1) != length(data2): return 'Error'
        diff = data1 - data2
        n = length(diff)
        m = mean(diff)
        s = var_p(diff)
        
        d = sum_value(diff) / n
        return d / (s/n)**0.5
    
    # ウェルチのt検定(Welch t-test)
    # 対応なし2標本t検定であるが、こちらは等分散を仮定できない際に使用する
    # 一説によれば、標本が等分散かどうかによらず利用しても問題ないらしい
    # ウェルチ=サタスウェイト（Welch=Satterthwaite）の式により近似自由度vのt分布に従う
    # 帰無仮説：2群間の母平均値に差がない(母平均値が等しい)
    # 対立仮説：2群間の母平均値に差がある(母平均値が異なる)
    def welch_ttest(data1, data2):
        n1, n2 = length(data1), length(data2)
        m1, m2 = mean(data1), mean(data2)
        s1, s2 = var_s(data1), var_s(data2)
        t = (m1-m2) / ((s1/n1) + (s2/n2))**0.5
        v = int(((s1/n1) + (s2/n2))**2 / (s1**2/(n1**2*(n1-1)) + s2**2/(n2**2*(n2-1))))
        return t, v
    
    # 全データ同一ランク付け
    def rank_all(data1, data2):
        data_linked = list(data1) + list(data2)
        data_linked, dup = rank(data_linked)
        data1_len = length(data1)
        data1_dec = data_linked[:data1_len]
        data2_dec = data_linked[data1_len:]
        return data1_dec, data2_dec
    
    # ムード検定(Mood test)
    # 2標本の中央値を比較する
    # サンプルサイズが小さい時(20未満)はムード検定表を参照し、大きい時(20以上)は標準正規分布表を参照する。
    # 帰無仮説：2群間の母集団の中央値に差がない(母集団の中央値が等しい)
    # 対立仮説：2群間の母集団の中央値に差がある(母集団の中央値が異なる)
    def mood_test(data1, data2):
        m, n = length(data1), length(data2)
        N = m + n
        data_ranked = quick_sort(rank_all(data1, data2)[0])
        
        # サンプルサイズが小さい時の検定統計量(20未満)
        MD = 0
        for R in data_ranked:
            MD += (R - ((N+1)/2))**2
            
        # サンプルサイズが大きい時の検定統計量(20以上)
        E = (m*(N**2 - 1)) / 12
        V = (m*n*(N+1)*(N**2 - 4)) / 180
        
        return (MD - E) / V**0.5 if N>=20 else MD
    
    # マン=ホイットニーのU検定(Mann-Whitney u-test)
    # 対応のない2群間の中央値に差があるかを検定する
    # サンプルサイズが小さい時(n<=20)、検定統計量UはMann-Whitney検定表に基づいてp値を算出する
    # サンプルサイズが十分に大きい時(n>20)、検定統計量zは標準正規分布に従うかどうかを考え、正規分布表に基づいてp値を算出する
    # 注意1：通常とは違い、検定統計量 < Mann-Whitney検定表の棄却域の時に帰無仮説を棄却する
    # 注意2：2つの分布の分散が異なる場合などに頑健でないことが知られている
    # 両側検定なので、検定統計量が負の値でも絶対値を取って考えれば良い
    # 帰無仮説：2群間の母集団は同じである
    # 対立仮説：2群間の母集団は異なる
    def mannwhitney_utest(data1, data2):
        n1, n2 = length(data1), length(data2)
        r1, r2 = rank_all(data1, data2)
        r1, r2 = sum_value(r1), sum_value(r2)
        # u1, u2の求め方はどちらでも良い
        #u1 = n1*n2 + ((n1*(n1+1)) / 2) - r1
        #u2 = n1*n2 + ((n2*(n2+1)) / 2) - r2
        u1 = r1 - (n1*(n1+1))/2
        u2 = r2 - (n2*(n2+1))/2
        U = u1 if (u1 < u2) else u2
        if (n1 <= 20 and n2 <= 20):
            return U
        else:
            z = (U - ((n1*n2)/2)) / (((n1*n2*(n1+n2+1)) / 12)**0.5)
            return z
        
    # ウィルコクソンの順位和検定(Wilcoxon rank sum test)
    # 対応のない2群間の中央値に差があるかを検定する
    # ウィルコクソンの順位和検定の数表を参照して有意差があるかどうか判定する
    # 数表から読み取ったa/bという表示は、検定統計量がa以下またはb以上の時に帰無仮説を棄却できるという意味である
    # マン=ホイットニーのU検定と実質的に同じ計算をしているため、同じ結果が返ってくるはずである
    # 注意：2つの分布の分散が異なる場合などに頑健でないことが知られている
    # 帰無仮説：2群間の母集団は同じである
    # 対立仮説：2群間の母集団は異なる
    def wilcoxon_rstest(data1, data2):
        n1, n2 = length(data1), length(data2)
        r1, r2 = rank_all(data1, data2)
        r1, r2 = sum_value(r1), sum_value(r2)
        if r1 < r2:
            w = r1
            ew = (n1*(n1+n2+1)) / 2
        else:
            w = r2
            ew = (n2*(n1+n2+1)) / 2
        
        vw = (n1*n2*(n1+n2+1)) / 12
        return (ew-w) / (vw)**0.5
    
    # ブルンナームンツェル検定(Brunner-Munzel test)
    # Wilcoxon-Mann-Whitney検定とは違い、対象の等分散性が仮定できなくてもロバストな検定手法
    # 参考：https://oku.edu.mie-u.ac.jp/~okumura/stat/brunner-munzel.html
    def brunner_munzel_test(data1, data2):
        data1_ranked, data2_ranked = rank(data1, delite=0)[0], rank(data2, delite=0)[0]
        data1_ranked_all, data2_ranked_all = rank_all(data1, data2)
        n1, n2 = length(data1), length(data2)
        m1_all, m2_all = mean(data1_ranked_all), mean(data2_ranked_all)
        m1, m2 = (n1+1)/2, (n2+1)/2
        s21 = (1/(n1-1)) * sum_value((pd.Series(data1_ranked_all) - pd.Series(data1_ranked) - m1_all + m1)**2)
        s22 = (1/(n2-1)) * sum_value((pd.Series(data2_ranked_all) - pd.Series(data2_ranked) - m2_all + m2)**2)
        
        W = (n1*n2*(m2_all-m1_all)) / ((n1+n2)*(n1*s21 + n2*s22)**0.5)
        return W
    
    # 符号検定(Sign test)
    # 対応のある2群間の中央値に差があるかを検定する(質的データである名義・順位尺度にも使える)
    # サンプルサイズが25以下の場合、二項定理から直接確率(p値)を計算する
    # サンプルサイズが25より大きい場合、標準正規分布表からp値を読み取る
    # 帰無仮説：2組の標本の中央値に差はない
    # 対立仮説：2組の標本の中央値に差がある
    # ソース：https://kusuri-jouhou.com/statistics/fugou.html
    def sign_test(data1, data2):
        n1, n2 = length(data1), length(data2)
        if n1 != n2: return 'Error'
        
        sign_data = data1 - data2
        plus, minus = 0, 0
        for d in sign_data:
            if d == 0: continue
            elif d > 0: plus += 1
            else: minus += 1
        n = plus + minus
        sign = plus if plus < minus else minus
        c = 0.5 if sign < (n/2) else -0.5
        
        if n <= 25:
            p = 1
            for i in range(n): p *= 0.5
            C = p
            for r in range(1, sign+1):
                nn, over, under = n, 1, 1
                for j in range(1, r+1):
                    over *= nn
                    under *= j
                    nn -= 1
                C += (over / under) * p
            return "({:.3f}, {})".format(C, 'p')
        else:
            m = n / 2
            s = n**0.5 / 2
            z = ((sign+c) - m) / s
            return "({:.3f}, {})".format(z, 'z')

    # ウィルコクソンの符号順位検定(Wilcoxon signed rank test)
    # 対応のある2群間の中央値に差があるかを検定する(量的データにのみ使用可能)
    # 帰無仮説：2組の標本の中央値に差はない
    # 対立仮説：2組の標本の中央値に差がある
    # サンプルサイズが異なる場合、Errorを返す
    def wilcoxon_srtest(data1, data2):
        if length(data1) != length(data2): return 'Error', 'Error'
        data_diff = data1 - data2
        count_p, count_n, delite = [], [], 0
        for i in range(length(data1)):
            if data_diff[i] > 0:
                count_p += [i]
            elif data_diff[i] < 0:
                count_n += [i]
                data_diff[i] = -data_diff[i]
            else: #差が0の場合は取り除く
                data_diff[i] = -999999
                delite += 1
        data_diff_ranked, dup = rank(data_diff, delite=delite)
        
        p_num, n_num = length(count_p), length(count_n)
        r_sum = 0
        if p_num >= n_num:
            for n in count_n:
                r_sum += data_diff_ranked[n]
            return r_sum, (p_num+n_num) #順位が付けられたデータの組の数がNである
        else:
            for p in count_p:
                r_sum += data_diff_ranked[p]
            return r_sum, (p_num+n_num)


    sr_b, sr_a, sr_r2 = simple_regression(data1, data2)
    srt_t, srt_df = simple_regression_test(data1, data2)
    f_test_f, f_test_df = f_test(data1, data2)
    ind_ttest_t, ind_ttest_df = independent_ttest(data1, data2)
    dep_ttest_t, dep_ttest_df = dependent_ttest(data1, data2)
    welch_ttest_t, welch_ttest_v = welch_ttest(data1, data2)
    wilcoxon_srtest_Tw, wilcoxon_srtest_df = wilcoxon_srtest(data1, data2)
    
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
        'cov.p':cov_p(data1, data2),
        'cov.s':cov_s(data1, data2),
        'Pearson_cor.t':pearson_cor(data1, data2),
        'Spearman_cor.t':spearman_cor(data1, data2),
        'Kendall_cor.z':kendall_cor(data1, data2),
        'simple_regression':"({:.3f}, {:.3f})".format(sr_b, sr_a),
        'simple_regression_test.t':srt_t,
        'Jarque-Bara_test.x2':jarque_bera(data1),
        'F_test.F':f_test_f,
        'ind_ztest.z':independent_ztest(data1, data2),
        'ind_ttest.t':ind_ttest_t,
        'ind_cohen_d':ind_cohen_d(data1, data2),
        'dep_ztest.z':dependent_ztest(data1, data2),
        'dep_ttest.t':dep_ttest_t,
        'dep_cohen_d':dep_cohen_d(data1, data2),
        'Welch_ttest.t':welch_ttest_t,
        'Mood_test.z':mood_test(data1, data2),
        'Mann-Whitney_utest.U':mannwhitney_utest(data1, data2),
        'Wilcoxon_rstest.Tw':wilcoxon_rstest(data1, data2),
        'Brunner-Munzel.W':brunner_munzel_test(data1, data2),
        'sign_test.r':sign_test(data1, data2),
        'Wilcoxon_srtest.Tw':wilcoxon_srtest_Tw
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
        'Pearson_cor.t':pearson_cor_test(data1, data2),
        'Spearman_cor.t':spearman_cor_test(data1, data2),
        'Kendall_cor.z':kendall_cor_test(data1, data2),
        'simple_regression':sr_r2,
        'simple_regression_test.t':srt_df,
        'Jarque-Bara_test.x2':jarque_bera(data2),
        'F_test.F':f_test_df,
        'ind_ttest.t':ind_ttest_df,
        'dep_ttest.t':dep_ttest_df,
        'Welch_ttest.t':welch_ttest_v,
        'Wilcoxon_srtest.Tw':wilcoxon_srtest_df
    }, index=["data2"]).T
    
    # 結果出力
    display(pd.concat([df1, df2], axis=1))
    
    # 種々のグラフをプロット
    # ヒストグラム
    fig, axes= plt.subplots(1,2)
    axes[0].hist(data1, bins=(1+int(np.log2(length(data1)))))
    axes[1].hist(data2, bins=(1+int(np.log2(length(data2)))))
    plt.show()
    
    # ヒストグラム(累積)
    fig, axes= plt.subplots(1,2)
    axes[0].hist(data1, bins=(1+int(np.log2(length(data1)))), cumulative=True)
    axes[1].hist(data2, bins=(1+int(np.log2(length(data2)))), cumulative=True)
    plt.show()
    
    # Q-Qプロット
    # 点が一直線に並んでいれば、そのデータの母集団の分布は正規分布に従う
    fig, axes= plt.subplots(1,2)
    st.probplot(data1, dist="norm", plot=axes[0])
    st.probplot(data2, dist="norm", plot=axes[1])
    plt.show()

    # 箱ひげ図
    plt.boxplot([data1, data2], positions=[1, 2])
    plt.title("Boxplot")
    plt.show()
    
    # バイオリンプロット
    plt.violinplot([data1, data2], positions=[1.2, 1.8])
    plt.title("Violinplot")
    plt.show()
    
    # イベントプロット
    plt.eventplot([data1, data2], orientation="vertical", lineoffsets=[2, 4], linewidth=0.75)
    plt.title("Eventplot")
    plt.show()
    
    # サンプルサイズが同じ場合のみ出力
    try:
        # グラフ
        plt.plot(data1, data2)
        plt.title("plot")
        plt.xlabel("data1")
        plt.ylabel("data2")
        plt.show()

        # 散布図
        plt.scatter(data1, data2)
        plt.title("Scatter")
        plt.xlabel("data1")
        plt.ylabel("data2")
        plt.show()

        # ステムプロット
        plt.stem(data1, data2)
        plt.title("Stem")
        plt.xlabel("data1")
        plt.ylabel("data2")
        plt.show()

        # ステッププロット
        plt.step(data1, data2)
        plt.title("Step")
        plt.xlabel("data1")
        plt.ylabel("data2")
        plt.show()

        # hist2d
        plt.hist2d(data1, data2)
        plt.title("Hist2d")
        plt.xlabel("data1")
        plt.ylabel("data2")
        plt.show()

    except:
        pass
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st

# 確率分布からデータを生成
data1 = pd.DataFrame(np.random.randint(-100, 101, 100), columns=["data1"])["data1"]
data2 = pd.DataFrame(np.random.randint(-100, 101, 100), columns=["data2"])["data2"]
describe2(data1, data2)

# 既存のライブラリで検証
print("PearsonrResult", st.pearsonr(data1, data2))
print(st.spearmanr(data1, data2))
print("indep_ttest", st.ttest_ind(data1, data2, equal_var=True))
print("dep_ttest", st.ttest_rel(data1, data2))
print("Welch", st.ttest_ind(data1, data2, equal_var=False))
print("Mood", st.mood(data1, data2))
print(st.mannwhitneyu(data1, data2, alternative='two-sided'))
print(st.wilcoxon(data1, data2))
