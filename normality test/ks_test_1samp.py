def ks_test_1samp(data, cdf_dist='norm', args=(), is_lilliefors=False):
    """
    1標本コルモゴロフ＝スミルノフ検定（リリフォース補正対応版）。

    Args:
        data (pd.Series or np.ndarray): 検定対象のサンプルデータ。
        cdf_dist (str, optional): 検定したい理論分布の名前 ('norm', 'uniform' 등)。
                                 デフォルトは 'norm'。
        args (tuple, optional): 分布のパラメータ。指定しない場合、データから推定。
        is_lilliefors (bool, optional): Trueの場合、正規性検定に対してリリフォース補正p値を計算。
                                     この場合、cdf_distは'norm'である必要があります。
                                     デフォルトは False。

    Returns:
        tuple: (D統計量, p値)
    """
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    from statsmodels.stats.diagnostic import lilliefors

    # リリフォース補正は正規分布（パラメータ推定時）専用
    if is_lilliefors and cdf_dist != 'norm':
        raise ValueError("リリフォース補正は正規分布の検定でのみ利用可能です。")

    # --- Step 0: 各種パラメータ及びデータの準備 ---
    y = np.sort(np.asarray(data))
    n = len(y)
    
    # cdf_dist文字列から実際のCDF関数を取得
    try:
        cdf_function = getattr(st, cdf_dist).cdf
    except AttributeError:
        raise ValueError(f"'{cdf_dist}' はscipy.statsに存在する有効な分布名ではありません。")

    # argsが指定されていない場合、データからパラメータを推定
    if not args:
        if cdf_dist == 'norm':
            # 正規分布の場合、平均と標準偏差を推定
            args = (np.mean(y), np.std(y, ddof=1))

    # --- Step 1: ECDFと理論CDFの値を計算 ---
    ecdf_upper = np.arange(1, n + 1) / n
    ecdf_lower = np.arange(0, n) / n
    theoretical_cdf = cdf_function(y, *args)
    
    # --- Step 2: 検定統計量Dを計算 ---
    # ECDFと理論CDFの垂直距離の最大値
    d_plus = np.max(ecdf_upper - theoretical_cdf)
    d_minus = np.max(theoretical_cdf - ecdf_lower)
    D_statistic = np.max([d_plus, d_minus])

    # --- Step 3: p値を計算 ---
    if is_lilliefors:
        # リリフォース補正を適用する場合(=検定する理論的な分布のパラメータが不明な場合)
        _, p_value = lilliefors(y, dist='norm', pvalmethod='approx')
    else:
        # 通常のK-S検定の場合(=N(0,1)等、検定する理論的な分布のパラメータが明らかな場合)
        p_value = st.kstwobign.sf(D_statistic * np.sqrt(n))

    return D_statistic, p_value


# テスト
import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.diagnostic import lilliefors

np.random.seed(0)
normal_data = np.random.normal(loc=5, scale=2, size=100)

print("--- 正規分布データの正規性検定比較 ---")

# 1. 通常のK-S検定 (パラメータはデータから推定)
D1, p1 = ks_test_1samp(normal_data, cdf_dist='norm', is_lilliefors=False)
print(f"通常K-S検定:       D統計量 = {D1:.4f}, p値 = {p1:.4f}  <-- 過度に保守的なp値")

# 2. リリフォース補正を適用したK-S検定
D2, p2 = ks_test_1samp(normal_data, cdf_dist='norm', is_lilliefors=True)
print(f"リリフォース補正版: D統計量 = {D2:.4f}, p値 = {p2:.4f}  <-- より正確なp値")

# 3. 参考: scipy.stats.kstestの結果
mean_hat = np.mean(normal_data)
std_hat = np.std(normal_data, ddof=1)
res_scipy = st.kstest(normal_data, 'norm', args=(mean_hat, std_hat))
print(f"scipy.kstest:      D統計量 = {res_scipy.statistic:.4f}, p値 = {res_scipy.pvalue:.4f}")

if lilliefors:
    # 4. 参考: statsmodels.stats.diagnostic.lilliefors の結果
    res_sm = lilliefors(normal_data, dist='norm', pvalmethod='approx')
    print(f"statsmodels.lilliefors: D統計量 = {res_sm[0]:.4f}, p値 = {res_sm[1]:.4f}")
