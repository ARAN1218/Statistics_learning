def MANCOVA(df, x, y_list, cov_list):
  import numpy as np
  import pandas as pd
  import scipy.stats as st
  from scipy.optimize import brentq
  """
  MANCOVAを実行する関数

  params:
    df: 分析対象のデータ
    x: 因子
    y_list: 目的変数（複数）
    cov_list: 共変量

  return: MANCOVAの分散分析表(DataFrame)
  """

  # Step 0: 各種パラメータを確認しよう
  N = len(df) # サンプルサイズ
  p = len(y_list) # 従属変数の数
  q = len(cov_list) # 共変量の数
  level_list = df[x].unique() # 水準数
  k = len(level_list) # グループ数
  df = df[[x] + cov_list + y_list] # データの並び替え

  # Step 1: 全体の変動行列(T)を計算し、分割する
  # 総合平均ベクトルを求める
  df_mean = df.copy()
  df_mean = df.groupby(x).mean()
  mean_all = np.array(df.drop(x, axis=1).mean())

  def calc_T(df, df_mean):
    """
    共変量を加えた全体変動行列Tを計算する。

    params:
    df: DataFrame
    df_mean: DataFrame

    return: np.array
    """
    # 結果を格納する変数を作る
    T = 0
    df_deviation = df.copy().drop(x, axis=1)
    df_deviation = df_deviation - mean_all # 各値から全体平均ベクトルを引く
    deviation_array = np.array(df_deviation) # 行列にする

    # 行列に対して、一行ずつ取り出し、変動行列を計算する。
    for i in range(deviation_array.shape[0]):
      T += deviation_array[i].reshape(1, len(deviation_array[i])).T * deviation_array[i]

    # Txx: 左上のブロック (共変量の分散)
    # 0行目からq行目まで、0列目からq列目までを抽出
    T_xx = T[0:q, 0:q]

    # Txy: 右上のブロック (共変量と従属変数の共分散)
    # 0行目からq行目まで、q列目から最後までを抽出
    T_xy = T[0:q, q:]

    # Wyx: 左下のブロック (Wxyの転置)
    # q行目から最後まで、0列目からq列目までを抽出
    T_yx = T[q:, 0:q]

    # Wyy: 右下のブロック (従属変数の分散共分散)
    # q行目から最後まで、q列目から最後までを抽出
    T_yy = T[q:, q:]

    return T, T_xx, T_xy, T_yx, T_yy

  # Step 2: 群内変動行列(W)を計算し、分割する
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

    # Wxx: 左上のブロック (共変量の分散)
    # 0行目からq行目まで、0列目からq列目までを抽出
    W_xx = W[0:q, 0:q]

    # Wxy: 右上のブロック (共変量と従属変数の共分散)
    # 0行目からq行目まで、q列目から最後までを抽出
    W_xy = W[0:q, q:]

    # Wyx: 左下のブロック (Wxyの転置)
    # q行目から最後まで、0列目からq列目までを抽出
    W_yx = W[q:, 0:q]

    # Wyy: 右下のブロック (従属変数の分散共分散)
    # q行目から最後まで、q列目から最後までを抽出
    W_yy = W[q:, q:]

    return W, W_xx, W_xy, W_yx, W_yy

  T, T_xx, T_xy, T_yx, T_yy = calc_T(df, df_mean)
  W, W_xx, W_xy, W_yx, W_yy = calc_W(df, df_mean)

  # Step 3: 共変量で「調整」した変動行列W_adjusted, T_adjustedを計算する
  W_adjusted = W_yy - W_yx @ np.linalg.inv(W_xx) @ W_xy
  T_adjusted = T_yy - T_yx @ np.linalg.inv(T_xx) @ T_xy

  # Step 4: 調整済みの群間変動行列(B_adjusted)を計算する
  B_adjusted = T_adjusted - W_adjusted

  # Step 5: 各種検定統計量を計算する
  # 固有値λを計算する
  eigenvalues = np.linalg.eigvals(np.linalg.inv(W_adjusted) @ B_adjusted)

  # ウィルクスのラムダ (Λ)
  Wilks_lambda = np.prod(1 / (1 + eigenvalues))

  # ピライのトレース (Pillai's Trace, V)
  pillais_trace_V = np.sum(eigenvalues / (1 + eigenvalues))

  # ホテリング・ローリーのトレース (Hotelling-Lawley Trace, T)
  hotelling_lawley_trace_T = np.sum(eigenvalues)

  # ロイの最大根 (Roy's Greatest Root, θ)
  roys_greatest_root_theta = np.max(eigenvalues)


  # Step 6: 有意性を確かめる
  def Rao_F_approximation_formula():
    s = np.sqrt((p**2 * (k-1)**2 - 4)/(p**2 + (k-1)**2 - 5)) if (p**2 + (k-1)**2 - 5)!=0 else 1
    m = (N - q) - 1 - (p+k)/2 # 共変量数qの分だけ自由度が小さくなることに注意
    df1_W = p * (k-1)
    df2_W = m * s - (p * (k-1))/2 + 1
    y = Wilks_lambda**(1/s)
    F_value_W = ((1-y)/y) * (df2_W/df1_W)
    return F_value_W, df1_W, df2_W

  def Pillai_F_approximation_formula():
    s = min(p, k-1)
    m = (abs(p - (k-1))-1) / 2
    n = (N - k - p - 1 - q) / 2 # 共変量数qの分だけ自由度が小さくなることに注意
    df1_P = s * (2*m + s + 1)
    df2_P = s * (2*n + s + 1)
    F_value_P = (pillais_trace_V / (s - pillais_trace_V)) * (df2_P/df1_P)
    return F_value_P, df1_P, df2_P

  def Hotelling_F_approximation_formula():
    s = min(p, k-1)
    m = (abs(p- (k-1))-1) / 2
    n = (N - k - p - 1 - q) / 2
    df1_H = s * (2*m + s + 1)
    df2_H = 2 * (s*n + 1)
    F_value_H = (hotelling_lawley_trace_T / s) * (df2_H/df1_H)
    return F_value_H, df1_H, df2_H

  def Roys_F_approximation_formula():
    df1_R = max(p, k-1)
    df2_R = N - max(p, k-1) - 1 - q
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

  # Step 7: 効果量を計算する
  s = np.sqrt((p**2 * (k-1)**2 - 4)/(p**2 + (k-1)**2 - 5)) if (p**2 + (k-1)**2 - 5)!=0 else 1
  effect_size_lambda = 1 - Wilks_lambda**(1/s)
  s = min(p, k-1)
  effect_size_V = pillais_trace_V / s
  effect_size_T = (hotelling_lawley_trace_T / s) / (1 + ((hotelling_lawley_trace_T / s)))
  effect_size_theta = roys_greatest_root_theta / (1 + roys_greatest_root_theta)

  # Step 8: 効果量の信頼区間を計算する
  def get_ci_for_eta_squared(F_obs, df1, df2, alpha=0.05):
      """
      観測されたF値から、多変量偏イータ二乗の効果量の信頼区間を計算する。

      params:
          F_obs: 観測されたF値
          df1: 分子の自由度
          df2: 分母の自由度
          alpha: 有意水準 (例: 0.05 -> 95%信頼区間)

      return:
          (効果量の下限値, 効果量の上限値)
      """
      
      # --- Step 2: 非心パラメータ(NCP)の信頼区間を求める ---
      
      # 信頼区間の下限となる非心パラメータ(ncp_l)を探すための関数
      # 観測されたF値が97.5パーセンタイル点となるような非心F分布を探す
      def lower_bound_func(ncp):
          return st.ncf.cdf(F_obs, df1, df2, nc=ncp) - (1 - alpha / 2)

      # 信頼区間の上限となる非心パラメータ(ncp_u)を探すための関数
      # 観測されたF値が2.5パーセンタイル点となるような非心F分布を探す
      def upper_bound_func(ncp):
          return st.ncf.cdf(F_obs, df1, df2, nc=ncp) - (alpha / 2)

      try:
          # F値が有意でない場合、下限は0になる
          if st.f.sf(F_obs, df1, df2) > alpha / 2:
              ncp_l = 0
          else:
              # brentqは、関数が0になる点を効率的に探すソルバー
              ncp_l = brentq(lower_bound_func, a=0, b=10000)
      except (ValueError, RuntimeError):
          ncp_l = 0 # エラーが発生した場合も下限は0とする

      try:
          # brentqを使って上限値を探す
          ncp_u = brentq(upper_bound_func, a=0, b=10000)
      except (ValueError, RuntimeError):
          ncp_u = 0 # エラーの場合は0

      # --- Step 3: 非心パラメータを効果量(偏イータ二乗)に変換する ---
      
      # 変換公式: η_p^2 = λ / (λ + df1 + df2 + 1)
      eta_sq_lower = ncp_l / (ncp_l + df1 + df2 + 1)
      eta_sq_upper = ncp_u / (ncp_u + df1 + df2 + 1)
      
      return (eta_sq_lower, eta_sq_upper)

  # 信頼区間を計算
  ci_W = get_ci_for_eta_squared(F_value_W, df1_W, df2_W)
  ci_P = get_ci_for_eta_squared(F_value_P, df1_P, df2_P)
  ci_H = get_ci_for_eta_squared(F_value_H, df1_H, df2_H)
  ci_R = get_ci_for_eta_squared(F_value_R, df1_R, df2_R)

  # Step 9: 分散分析表を作る
  mancova_results = {
      "Statistic": ["Wilks' Lambda", "Pillai's Trace", "Hotelling-Lawley Trace", "Roy's Greatest Root"],
      "Value": [Wilks_lambda, pillais_trace_V, hotelling_lawley_trace_T, roys_greatest_root_theta],
      "Num DF": [df1_W, df1_P, df1_H, df1_R], # 近似による自由度(分母)
      "Den DF": [df2_W, df2_P, df2_H, df2_R],  # 近似による自由度(分子)
      "F Value": [F_value_W, F_value_P, F_value_H, F_value_R],
      "p Value": [p_value_W, p_value_P, p_value_H, p_value_R],
      "Effect Size": [effect_size_lambda, effect_size_V, effect_size_T, effect_size_theta],
      "Effect Size 95%CL": [ci_W, ci_P, ci_H, ci_R]
  }
  mancova_df = pd.DataFrame(mancova_results)

  return mancova_df


# テスト1
import numpy as np
import pandas as pd
import scipy.stats as st

df = pd.DataFrame({
    '指導法': ['A', 'A', 'A', 'B', 'B', 'B'],
    '塾の時間': [2, 4, 6, 8, 10, 12],
    '国語': [8, 9, 10, 7, 8, 6],
    '理科': [7, 9, 8, 6, 5, 4]
})

print("入力：")
display(df)

test = MANCOVA(df, x="指導法", y_list=["国語", "理科"], cov_list=["塾の時間"])
print("出力：")
display(test)


# テスト2
import numpy as np
import pandas as pd
import scipy.stats as st


df = pd.DataFrame({
    '指導法': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
    '自習時間': [10, 12, 11, 13, 5, 7, 6, 4, 15, 18, 16, 17],
    '国語': [85, 88, 90, 86, 78, 82, 75, 72, 65, 70, 68, 72],
    '社会': [75, 72, 80, 77, 92, 88, 95, 90, 60, 68, 62, 65]
})

print("入力：")
display(df)

test = MANCOVA(df, x="指導法", y_list=["国語", "社会"], cov_list=["自習時間"])
print("出力：")
display(test)
