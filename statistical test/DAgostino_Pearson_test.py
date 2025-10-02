def DAgostino_Pearson_test(df):
  """
  歪度と尖度による正規性の検定

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

  # Step 1: 歪度によるダゴスティーノ検定の検定統計量を計算する
  Z_w, _ = DAgostino_skewtest(df)
  
  # Step 2: Anscombe-Glynn検定の検定統計量を計算する
  Z_k, _ = Anscombe_Glynn_kurtosistest(df)

  # Step 3: オムニバス検定の検定統計量を計算する
  K = Z_w**2 + Z_k**2

  # Step 4: p値の算出と判定(カイ二乗分布)
  p_value = st.chi2.sf(K, df=1)

  return K, p_value


# テスト
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# 有意にならないデータ (正規分布に近いデータ)
np.random.seed(42)
data_not_significant_omnibus = np.random.normal(loc=0, scale=1, size=100)

# 有意になるデータ (正規分布から離れたデータ、例えばガンマ分布)
data_significant_omnibus = np.random.gamma(shape=2, scale=1, size=100)

# データをpandas DataFrameにまとめる
data_df_omnibus = pd.DataFrame({
    '有意にならないデータ (オムニバス)': data_not_significant_omnibus,
    '有意になるデータ (オムニバス)': data_significant_omnibus
})

# 各データに対してダゴスティーノ・ピアソン検定を実行
print("--- ダゴスティーノ・ピアソン検定結果 ---")
for col in data_df_omnibus.columns:
    data_series_omnibus = data_df_omnibus[col]
    statistic_omnibus, p_value_omnibus = DAgostino_Pearson_test(data_series_omnibus)
    print(f"{col} - 検定統計量: {statistic_omnibus:.4f}, P値: {p_value_omnibus:.4f}")

# データの可視化 (オプション)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data_df_omnibus['有意にならないデータ (オムニバス)'], bins=20, density=True, alpha=0.6, color='g')
plt.title('有意にならないデータ (オムニバス)')
plt.xlabel('値')
plt.ylabel('頻度')

plt.subplot(1, 2, 2)
plt.hist(data_df_omnibus['有意になるデータ (オムニバス)'], bins=20, density=True, alpha=0.6, color='r')
plt.title('有意になるデータ (オムニバス、非正規)')
plt.xlabel('値')
plt.ylabel('頻度')

plt.tight_layout()
plt.show()
