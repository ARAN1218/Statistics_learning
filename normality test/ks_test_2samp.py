def ks_test_2samp(data1, data2):
    """
    2標本K-S検定

    Args:
        data1 (list or tuple): 1つ目のサンプルデータ。
        data2 (list or tuple): 2つ目のサンプルデータ。

    Returns:
        tuple: (D統計量, p値)
    """
    import numpy as np
    import pandas as pd
    import scipy.stats as st

    # --- Step 1: 各データをソートし、サンプルサイズを取得 ---
    data1_sorted = sorted(list(data1))
    data2_sorted = sorted(list(data2))
    n1 = len(data1_sorted)
    n2 = len(data2_sorted)

    # --- Step 2: 全てのデータポイントを結合・ソートし、ユニークな値を取得 ---
    # 2つのリストを結合し、set()で重複を除去した後、再びソートして
    # ECDFを評価するための「チェックポイント」のリストを作成します。
    all_points = sorted(list(set(data1_sorted + data2_sorted)))

    # --- Step 3 & 4: ECDFの値を計算し、最大距離Dを求める ---
    # D統計量を記録する変数を0で初期化
    max_diff = 0.0

    # ループを使って、各チェックポイントでECDFの差を計算します
    count1 = 0
    count2 = 0
    for x in all_points:
        # data1の中で、現在のチェックポイントx以下のデータがいくつあるかを数える
        while count1 < n1 and data1_sorted[count1] <= x:
            count1 += 1
        
        # data2の中で、現在のチェックポイントx以下のデータがいくつあるかを数える
        while count2 < n2 and data2_sorted[count2] <= x:
            count2 += 1
        
        # 各データセットのECDFの値を計算（割合を算出）
        ecdf1 = count1 / n1
        ecdf2 = count2 / n2
        
        # 2つのECDFの値の差（垂直距離）を計算
        current_diff = abs(ecdf1 - ecdf2)
        
        # これまでの最大距離よりも大きければ、値を更新
        if current_diff > max_diff:
            max_diff = current_diff
            
    D_statistic = max_diff

    # --- Step 5: p値を計算 ---
    # この部分は統計的な専門知識が必要なため、scipyの力を借ります
    en = np.sqrt(n1 * n2 / (n1 + n2))
    p_value = st.kstwobign.sf(D_statistic * en)
    
    return D_statistic, p_value


# テスト
np.random.seed(0)

# --- 例1: 同じ分布から来た2つのサンプル ---
sample_a = np.random.normal(loc=5, scale=2, size=140)
sample_b = np.random.normal(loc=5, scale=2, size=150)

print("--- 例1: 同じ分布からの2サンプルの比較 ---")
D1, p1 = ks_test_2samp_manual(sample_a, sample_b)
print(f"手動実装版 (漸近近似): D統計量 = {D1:.4f}, p値 = {p1:.4f}")

# Scipyのデフォルト（高精度）
res1_auto = st.ks_2samp(sample_a, sample_b, method='auto')
print(f"scipy (method='auto'): D統計量 = {res1_auto.statistic:.4f}, p値 = {res1_auto.pvalue:.4f}")

# Scipyに手動実装と同じ「漸近近似」を使わせる
res1_asymp = st.ks_2samp(sample_a, sample_b, method='asymp')
print(f"scipy (method='asymp'): D統計量 = {res1_asymp.statistic:.4f}, p値 = {res1_asymp.pvalue:.4f} <-- p値が一致！")
print("-" * 50)

# --- 例2: 異なる分布から来た2つのサンプル ---
sample_c = np.random.normal(loc=0, scale=1, size=200)
sample_d = np.random.uniform(low=-3, high=3, size=220)

print("--- 例2: 異なる分布からの2サンプルの比較 ---")
D2, p2 = ks_test_2samp_manual(sample_c, sample_d)
print(f"手動実装版 (漸近近似): D統計量 = {D2:.4f}, p値 = {p2:.4f}")

res2_auto = st.ks_2samp(sample_c, sample_d, method='auto')
print(f"scipy (method='auto'): D統計量 = {res2_auto.statistic:.4f}, p値 = {res2_auto.pvalue:.4f}")

res2_asymp = st.ks_2samp(sample_c, sample_d, method='asymp')
print(f"scipy (method='asymp'): D統計量 = {res2_asymp.statistic:.4f}, p値 = {res2_asymp.pvalue:.4f} <-- p値が一致！")
