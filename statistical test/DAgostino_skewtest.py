def DAgostino_skewtest(df):
  """
  歪度による正規性の検定(歪度によるダゴスティーノ検定)

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

  # Step 1: 標本歪度 (g1) を計算する
  # 2次の中心モーメント (m2) と3次の中心モーメント (m3) を計算
  m2 = (1/n) * np.sum([(x - mean)**2 for x in df])
  m3 = (1/n) * np.sum([(x - mean)**3 for x in df])
  g1 = m3 / (m2)**(3/2) # 標本歪度
  y = g1 * np.sqrt(((n+1) * (n+3)) / (6 * (n-2))) # 標本歪度に特殊な変形を施した値

  # Step 2: 各種中間変数を計算
  Beta2_g1 = (3 * (n**2 + 27*n - 70) * (n+1) * (n+3)) / ((n-2) * (n+5) * (n+7) * (n+9))
  W2 = np.sqrt(2*(Beta2_g1 - 1)) - 1
  epsilon = 1 / np.sqrt(np.log(np.sqrt(W2)))
  alpha = np.sqrt(2 / (W2 - 1))

  # Step 3: 検定統計量 (Z) を計算する
  Z = epsilon * np.log((y/alpha) + np.sqrt((y/alpha)**2 + 1))

  # Step 4: p値を計算する
  p_value = 2 * st.norm.sf(np.abs(Z))

  return Z, p_value


# テスト
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# 有意にならないデータ (正規分布に近いデータ)
np.random.seed(42)
data_not_significant = np.random.normal(loc=0, scale=1, size=100)

# 有意になるデータ (歪度のあるデータ)
data_significant = np.random.exponential(scale=1, size=100)

# データをpandas DataFrameにまとめる
data_df = pd.DataFrame({
    '有意にならないデータ': data_not_significant,
    '有意になるデータ': data_significant
})

# 各データに対してダゴスティーノ検定を実行
print("--- ダゴスティーノ検定結果 ---")
for col in data_df.columns:
    data_series = data_df[col]
    statistic, p_value = DAgostino_skewtest(data_series)
    print(f"{col} - 検定統計量: {statistic:.4f}, P値: {p_value:.4f}")

# データの可視化 (オプション)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data_df['有意にならないデータ'], bins=20, density=True, alpha=0.6, color='g')
plt.title('有意にならないデータ (正規分布に近い)')
plt.xlabel('値')
plt.ylabel('頻度')

plt.subplot(1, 2, 2)
plt.hist(data_df['有意になるデータ'], bins=20, density=True, alpha=0.6, color='r')
plt.title('有意になるデータ (歪度あり)')
plt.xlabel('値')
plt.ylabel('頻度')

plt.tight_layout()
plt.show()
