def one_way_ANOVA(df):
    # 列と行の長さを取得
    f1 = len(df.columns)
    n = len(df.index)
    
    # 因子の効果を求めた後、因子変動S1を求める
    f_mean = df.mean().mean()
    f1_mean = df.mean()
    f1_effect = f1_mean - f_mean
    S1 = ((f1_effect**2) * n).sum()
    
    # 誤差変動Seを求める
    df_error = df.copy() - f1_mean
    Se = (df_error**2).sum().sum()
    
    # 総変動Stを求める
    St = ((df-f_mean)**2).sum().sum()
    
    # 自由度dfを求める
    df1 = f1 - 1
    dfe = f1*(n-1)
    dft = df1 + dfe
    
    # 不偏分散Vを求める
    V1 = S1 / df1
    Ve = Se / dfe
    
    # F値とp値を求める
    F = V1 / Ve
    p = 1 - st.f.cdf(F, dfn=df1, dfd=dfe)
    
    # 分散分析表を作成する
    df_S = pd.Series([S1, Se, St])
    df_df = pd.Series([df1, dfe, dft])
    df_V = pd.Series([V1, Ve])
    df_F = pd.Series([F])
    df_p = pd.DataFrame([p], columns=['p'])
    df_p['sign'] = df_p['p'].apply(lambda x : '**' if x < 0.01 else '*' if x < 0.05 else '')
    df_ANOVA = pd.concat([df_S, df_df, df_V, df_F, df_p], axis=1).set_axis(['S','df','V','F','p','sign'], axis=1).set_axis(['Columns','Error', 'Total']).fillna('')
    
    # 因子の効果をデータフレームにまとめる
    df_effect = pd.DataFrame(f1_effect).T.set_axis(['Effect'])
    
    # 結果を出力する
    return df_ANOVA, df_effect


# テスト
import numpy as np
import pandas as pd
import scipy.stats as st

df_1 = pd.DataFrame([[8,10,12], [4,8,12]]).set_axis(['b0', 'b1', 'b2'], axis=1)
print("入力：")
display(df_1)

df_ANOVA, df_effect = one_way_ANOVA(df_1)
print("出力：")
display(df_ANOVA, df_effect)
