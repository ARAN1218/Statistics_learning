def cochran_armitage_test(data):
    """
    コクラン=アーミテージ検定
    順序尺度カテゴリと2値データからなるクロス集計表において、
    順序カテゴリの水準に伴う2値データの割合に線形的な傾向（トレンド）があるかを検定する。

    Args:
      data (np.ndarray): クロス集計表 (R x 2 のnumpy配列)
                         Rは順序カテゴリの数、2は2値の結果 (例: [結果1, 結果2])
                         サイトの例に合わせて、列は [効果なし, 効果あり] の順と仮定します。

    Return:
      results_df (pd.DataFrame): カイ二乗値、自由度、p値、検定結果を格納した分析表
    """
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    
    if data.shape[1] != 2:
        print("エラー: 入力データは R x 2 の配列である必要があります。")
        return

    # 各種パラメータの計算
    R = data.shape[0]  # 順序カテゴリの数 (例: 3)
    n = data.sum().sum()   # 全体の合計サンプルサイズ
    data["合計"] = data.iloc[:, 0] + data.iloc[:, 1]
    data["標本1の割合"] = data.iloc[:, 0] / data["合計"]
    p = data.iloc[:, 0].sum() / data["合計"].sum()
    df_slope = 1
    df_residual = R - 1
    df_total = R

    # 3つの平方和を求める
    w = 1 /(p * (1-p))
    x_mean = np.sum((np.sum(np.arange(1,R+1) * data["合計"])) / n)

    S_xx = np.sum(np.array(data["合計"]) * w * (np.arange(1,R+1) - x_mean)**2)
    S_yy = np.sum(data["合計"] * w * (data["標本1の割合"] - p)**2)
    S_xy = np.sum(data["合計"] * w * (np.arange(1,R+1) - x_mean)*(data["標本1の割合"] - p))

    # 検定統計量を求める
    Chi2_slope = S_xy**2 / S_xx
    Chi2_total = S_yy
    Chi2_residual = Chi2_total - Chi2_slope

    # p値を求める
    p_slope = st.chi2.sf(Chi2_slope, df_slope)
    p_residual = st.chi2.sf(Chi2_residual, df_residual)
    p_total = st.chi2.sf(Chi2_total, df_total)

    results_df = pd.DataFrame({
        'カイ二乗値': [Chi2_slope, Chi2_residual, Chi2_total],
        '自由度': [df_slope, df_residual, df_total],
        'p値': [p_slope, p_residual, p_total]
    }, index=['直線の傾き', '直線からのズレ', '合計'])

    return results_df


# テスト
data = {'お酒を飲まない': [10, 90],
        'お酒を少し飲む': [20, 60],
        'お酒をたくさん飲む': [30, 30]}
df = pd.DataFrame(data, index=['肥満である', '肥満ではない']).T

print("入力：")
display(df)

print("出力：")
cochran_armitage_test(df)
