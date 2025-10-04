def mcnemar_test(data, x1, x2):
  """
  マクネマー検定
  対応のあるペアの2値データを用いて2つの処理あるいは
  2つの調査の結果に差があるかどうかを検定する。

  Args:
    data (pd.DataFrame): 
    x1 (string): 
    x2 (string): 

  Return: (tuble)
    Chi2 (float): 検定統計量
    p-value (float): p値
    Chi2_yates (float): 検定統計量（イェーツの補正有）
    p-value_yates (float): p値（イェーツの補正有）
    CL_lower (float): 信頼区間の下限
    CL_upper (float): 信頼区間の上限
  """

  import numpy as np
  import pandas as pd
  import scipy.stats as st

  # 各種パラメータを計算する
  n = len(data)
  df = 1

  # 交差表を作成する
  cross_table = pd.crosstab(data[x1], data[x2])
  b = cross_table.iloc[0,1]
  c = cross_table.iloc[1,0]
  display(cross_table)

  # 検定統計量とp値を求める
  Chi2 = (b - c)**2 / (b + c)
  p = st.chi2.sf(Chi2, df)
  Chi2_yates = (abs(b - c) - 1)**2 / (b + c)
  p_yates = st.chi2.sf(Chi2_yates, df)

  # 2標本における割合の差の信頼区間を求める
  D = (b - c) / n
  CL_lower = D - 1.96 * (1/n) * np.sqrt((b + c) - ((b - c)**2 / n))
  CL_upper = D + 1.96 * (1/n) * np.sqrt((b + c) - ((b - c)**2 / n))

  return Chi2, p, Chi2_yates, p_yates, CL_lower, CL_upper


# テスト
import pandas as pd

data = {'飲む前': ['○', '×', '×', '○', '×', '○', '×', '×', '×', '○', '×', '×', '×', '×', '×', '○', '×', '×', '×', '○'],
        '飲んだ後': ['○', '×', '○', '○', '×', '○', '○', '×', '○', '○', '○', '×', '○', '○', '○', '○', '○', '○', '○', '×']}
df = pd.DataFrame(data)

print("入力：")
display(df)

print("出力：")
mcnemar_test(df, '飲む前', '飲んだ後')
