def MANOVA(df, x, y_list):
  import numpy as np
  import pandas as pd
  import scipy.stats as st
  # ex.) x=教育法、y_list=["数学", "国語"]
  # Step 0: 水準数を確認しよう
  N = len(df) # サンプルサイズ
  p = len(y_list) # 従属変数の数
  level_list = df[x].unique() # 水準数
  k = len(level_list) # グループ数

  # Step 1: 平均ベクトルを計算しよう
  df_mean = df.copy()
  df_mean = df.groupby(x).mean()
  mean_all = np.array(df[y_list].mean())

  # Step 2: 「変動」を行列で捉えよう
  def calc_W(df, df_mean):
    """
    郡内変動行列Wを計算する。

    params:
    df: DataFrame
    df_mean: DataFrame

    return: np.array
    """

    # 結果を格納する変数を作る
    W = 0

    for level in level_list:
      level_df = df[df[x] == level].drop(x, axis=1) #  特定の水準に絞る
      level_df = level_df - df_mean.loc[level] # 各値から平均ベクトルを引く

      level_array = np.array(level_df) # 行列にする
      # 行列に対して、一行ずつ取り出し、変動行列を計算する。
      for i in range(level_array.shape[0]):
        W += level_array[i].reshape(1, len(level_array[i])).T * level_array[i]

    return W


  def calc_B(df, df_mean):
    """
    群間変動行列Bを計算する。

    params:
    df: DataFrame
    df_mean: DataFrame

    return: np.array
    """
    B = 0

    # 水準毎の平均偏差ベクトルを計算する
    for level in level_list:
      mean_deviation_vector = np.array(df_mean.loc[level]) - mean_all # 各水準の平均ベクトルから全体平均ベクトルを引く
      B += len(df[df[x] == level]) * mean_deviation_vector.reshape(1, len(mean_deviation_vector)).T * mean_deviation_vector # 群間変動を求める

    return B

  W = calc_W(df, df_mean)
  B = calc_B(df, df_mean)


  # Step 3: 固有値λを計算しよう
  eigenvalues = np.linalg.eigvals(np.linalg.inv(W) @ B)

  # Step 4: 各種検定統計量を計算しよう
  # ウィルクスのラムダ (Λ)
  Wilks_lambda = np.prod(1 / (1 + eigenvalues))

  # ピライのトレース (Pillai's Trace, V)
  pillais_trace_V = np.sum(eigenvalues / (1 + eigenvalues))

  # ホテリング・ローリーのトレース (Hotelling-Lawley Trace, T)
  hotelling_lawley_trace_T = np.sum(eigenvalues)

  # ロイの最大根 (Roy's Greatest Root, θ)
  roys_greatest_root_theta = np.max(eigenvalues)


  # Step 5: 有意性を確かめよう
  def Rao_F_approximation_formula():
    s = np.sqrt((p**2 * (k-1)**2 - 4)/(p**2 + (k-1)**2 - 5)) if (p**2 + (k-1)**2 - 5)!=0 else 1
    m = N - 1 - (p+k)/2
    df1_W = p * (k-1)
    df2_W = m * s - (p * (k-1))/2 + 1
    y = Wilks_lambda**(1/s)
    F_value_W = ((1-y)/y) * (df2_W/df1_W)
    return F_value_W, df1_W, df2_W

  def Pillai_F_approximation_formula():
    s = min(p, k-1)
    m = (abs(p - (k-1))-1) / 2
    n = (N - k - p - 1) / 2
    df1_P = s * (2*m + s + 1)
    df2_P = s * (2*n + s + 1)
    F_value_P = (pillais_trace_V / (s - pillais_trace_V)) * (df2_P/df1_P)
    return F_value_P, df1_P, df2_P

  def Hotelling_F_approximation_formula():
    s = min(p, k-1)
    m = (abs(p- (k-1))-1) / 2
    n = (N - k - p - 1) / 2
    df1_H = s * (2*m + s + 1)
    df2_H = 2 * (s*n + 1)
    F_value_H = (hotelling_lawley_trace_T / s) * (df2_H/df1_H)
    return F_value_H, df1_H, df2_H

  def Roys_F_approximation_formula():
    df1_R = max(p, k-1)
    df2_R = N - max(p, k-1) - 1
    F_value_R = roys_greatest_root_theta * (df2_R/df1_R)
    return F_value_R, df1_R, df2_R

  # 各検定統計量のF値を求める
  F_value_W, df1_W, df2_W = Rao_F_approximation_formula()
  F_value_P, df1_P, df2_P = Pillai_F_approximation_formula()
  F_value_H, df1_H, df2_H = Hotelling_F_approximation_formula()
  F_value_R, df1_R, df2_R = Roys_F_approximation_formula()

  # 求めたF値からp値を求める  
  p_value_W = 1 - st.f.cdf(F_value_W, dfn=df1_W, dfd=df2_W)
  p_value_P = 1 - st.f.cdf(F_value_P, dfn=df1_P, dfd=df2_P)
  p_value_H = 1 - st.f.cdf(F_value_H, dfn=df1_H, dfd=df2_H)
  p_value_R = 1 - st.f.cdf(F_value_R, dfn=df1_R, dfd=df2_R)


  # Step 5: 分散分析表を作ろう
  manova_results = {
      "Statistic": ["Wilks' Lambda", "Pillai's Trace", "Hotelling-Lawley Trace", "Roy's Greatest Root"],
      "Value": [Wilks_lambda, pillais_trace_V, hotelling_lawley_trace_T, roys_greatest_root_theta],
      "Num DF": [df1_W, df1_P, df1_H, df1_R], # 近似による自由度(分母)
      "Den DF": [df2_W, df2_P, df2_H, df2_R],  # 近似による自由度(分子)
      "F Value": [F_value_W, F_value_P, F_value_H, F_value_R],
      "p Value": [p_value_W, p_value_P, p_value_H, p_value_R],
  }
  manova_df = pd.DataFrame(manova_results)

  return manova_df


# テスト1
import numpy as np
import pandas as pd
import scipy.stats as st

df = pd.DataFrame([[0,8,9]
                     ,[0,9,8]
                     ,[0,7,6]
                     ,[1,5,5]
                     ,[1,6,5]
                     ,[1,4,3]]).set_axis(['教育法', '数学', '国語'], axis=1)
print("入力：")
display(df)

test = MANOVA(df, x="教育法", y_list=["数学", "国語"])
print("出力：")
display(test)


# テスト2
import numpy as np
import pandas as pd
import scipy.stats as st

df = pd.DataFrame({
    '教育法': [0, 0, 0, 1, 1, 1, 1, 2, 2],
    '数学': [8, 9, 10, 8, 9, 7, 4, 5, 6],
    '理科': [9, 8, 10, 5, 4, 6, 3, 5, 4]
})
print("入力：")
display(df)

test = MANOVA(df, x="教育法", y_list=["数学", "理科"])
print("出力：")
display(test)
