def shapiro_wilk(data):
    """
    Roystonの近似アルゴリズムに基づき、シャピロ＝ウィルク検定を実装する。
    ※サンプルサイズ n が 3 <= n <= 5000 の場合に有効

    Args:
        data (pd.Series or np.ndarray): 検定対象のデータ

    Returns:
        tuple: (W統計量, p値)
    """
    import numpy as np
    import pandas as pd
    import scipy.stats as st

    # Step 0: 各種パラメータの計算
    y = np.sort(np.asarray(data))
    n = len(y)

    if n < 3:
        raise ValueError("サンプルサイズは3以上である必要があります。")
    if n > 5000:
        print("警告: サンプルサイズが5000を超えており、このアルゴリズムの精度が低下する可能性があります。")

    # --- Step 1: 係数 a_i の計算 ---
    m = st.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    u = 1.0 / np.sqrt(n)
    poly_1 = np.array([-2.706056e-5, 4.434685e-4, -0.207119e-2, -0.147981e-2, 0.221157e-1])
    poly_2 = np.array([-0.2547844e-4, 0.536277e-3, -0.310821e-2, -0.093762e-2, 0.042981e-1])
    
    a = np.zeros(n)
    if n > 5:
        a[-1] = m[-1] + np.polyval(poly_1, u)
        a[0]  = -a[-1]
        a[-2] = m[-2] + np.polyval(poly_2, u)
        a[1] = -a[-2]
        a[2:-2] = m[2:-2]
    else:
        a[:] = m[:]
        
    norm = np.sqrt(np.sum(a**2))
    a = a / norm

    # --- Step 2: W統計量の計算 ---
    y_mean = np.mean(y)
    b = np.sum(a * y)**2
    d = np.sum((y - y_mean)**2)
    
    if d < 1e-10:
        return 1.0, 1.0
        
    W = b / d
    W = min(W, 1.0)

    # --- Step 3: W統計量の正規化(Z値) (Royston's Algorithm AS R94) ---
    lw = np.log(1.0 - W)
    
    if n <= 11:
        mu = 0.0038915 * n**3 - 0.025054 * n**2 + 0.39978 * n - 0.5440
        sigma = np.exp(0.0020322 * n**3 - 0.062767 * n**2 + 0.77857 * n - 1.3822)
        Z = (lw - mu) / sigma
    else: # n >= 12
        ln_n = np.log(n)
        mu = 0.0038915 * ln_n**3 - 0.083751 * ln_n**2 - 0.31082 * ln_n - 1.5861
        sigma = np.exp(0.0030302 * ln_n**2 + 0.082676 * ln_n - 0.4803)
        Z = (lw - mu) / sigma

    # Step 4: p値の算出
    # Z値より小さい値が出る確率（分布の左側の面積）を求める必要がある。
    p_value = 1-st.norm.cdf(Z)

    return W, p_value


# テスト
import numpy as np
import pandas as pd
import scipy.stats as st

# サンプルデータを作成
# 1. 正規分布に従うデータ (n=50)
np.random.seed(1)
normal_data = pd.Series(np.random.normal(loc=10, scale=2, size=50))

# 2. 指数分布に従うデータ (n=200)
exp_data = pd.Series(np.random.exponential(scale=1.0, size=200))

# --- Royston実装版で検定 ---
print("--- Roystonの近似アルゴリズムによる実装 ---")
W_normal, p_normal = royston_shapiro_wilk(normal_data)
print(f"正規分布データ (n=50): W統計量 = {W_normal:.4f}, p値 = {p_normal:.4f}")

W_exp, p_exp = royston_shapiro_wilk(exp_data)
print(f"指数分布データ (n=200): W統計量 = {W_exp:.4f}, p値 = {p_exp:.4f}")
print("-" * 50)

# --- scipy版の結果と比較 ---
print("--- scipy.stats.shapiro の結果（比較用） ---")
res_normal = st.shapiro(normal_data)
print(f"正規分布データ (n=50): W統計量 = {res_normal.statistic:.4f}, p値 = {res_normal.pvalue:.4f}")

res_exp = st.shapiro(exp_data)
print(f"指数分布データ (n=200): W統計量 = {res_exp.statistic:.4f}, p値 = {res_exp.pvalue:.4f}")
