def Anscombe_Glynn_kurtosistest(df):
  """
  尖度による正規性の検定
  ※歪度によるダゴスティーノ検定をより正確にした統計的検定手法(1983)

  params:
    df: Series

  return:
    Z: float64
    p: float64
  """
  import numpy as np
  import pandas as pd
  import scipy.stats as st

  # Step 0: 各種パラメータを計算
  n = len(df)
  mean = np.mean(df)

  # Step 1: 標本尖度 (b2) を計算する
  # 2次の中心モーメント (m2) と4次の中心モーメント (m4) を計算
  m2 = (1/n) * np.sum([(x - mean)**2 for x in df])
  m4 = (1/n) * np.sum([(x - mean)**4 for x in df])
  b2 = m4 / (m2)**2 # 標本尖度

  # Step 2: 標本尖度の標準化 (x の計算)
  # 正規分布を仮定した場合のb2の期待値（平均）と分散を使って標準化する
  E_b2 = (3 * (n - 1)) / (n + 1)
  Var_b2 = (24*n * (n-2) * (n-3)) / ((n+1)**2 * (n+3) * (n+5))
  x = (b2 - E_b2) / np.sqrt(Var_b2)

  # Step 3: 標準化した尖度をさらに変換する（Zの計算)
  # 中間変数を計算する
  beta_1 = ((6 * (n**2 - 5*n + 2)) / ((n+7) * (n+9))) * np.sqrt((6 * (n+3) * (n+5)) / (n * (n-2) * (n-3)))
  A = 6 + (8/beta_1) * ((2/beta_1) + np.sqrt(1 + (4/beta_1**2)))
  # より標準正規分布に近い値Zに変換する(Wilson-Hilferty変換)
  Z = (1 - (2 / (9*A)) - ((1 - (2/A)) / (1 + x * np.sqrt(2 / (A-4))))**(1/3)) / np.sqrt(2 / (9*A))

  # Step 4: p値を計算する
  p_value = 2 * st.norm.sf(np.abs(Z))

  return Z, p_value


# テスト
# 有意にならないデータ (正規分布に近いデータ) - 尖度は0に近い
np.random.seed(123) # シード値を変更
data_not_significant_kurtosis = np.random.normal(loc=0, scale=1, size=100)

# 有意になるデータ (尖度のあるデータ) - 例えばラプラス分布など
data_significant_kurtosis = np.random.laplace(loc=0, scale=1, size=100)

# データをpandas DataFrameにまとめる
data_df_kurtosis = pd.DataFrame({
    '有意にならないデータ (尖度)': data_not_significant_kurtosis,
    '有意になるデータ (尖度)': data_significant_kurtosis
})

# 各データに対してダゴスティーノの尖度検定を実行
print("--- ダゴスティーノの尖度検定結果 ---")
for col in data_df_kurtosis.columns:
    data_series_kurtosis = data_df_kurtosis[col]
    statistic_kurtosis, p_value_kurtosis = Anscombe_Glynn_kurtosistest(data_series_kurtosis)
    print(f"{col} - 検定統計量: {statistic_kurtosis:.4f}, P値: {p_value_kurtosis:.4f}")

# データの可視化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data_df_kurtosis['有意にならないデータ (尖度)'], bins=20, density=True, alpha=0.6, color='g')
plt.title('有意にならないデータ (尖度)')
plt.xlabel('値')
plt.ylabel('頻度')

plt.subplot(1, 2, 2)
plt.hist(data_df_kurtosis['有意になるデータ (尖度)'], bins=20, density=True, alpha=0.6, color='r')
plt.title('有意になるデータ (尖度あり)')
plt.xlabel('値')
plt.ylabel('頻度')

plt.tight_layout()
plt.show()
