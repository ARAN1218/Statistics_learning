def three_way_ANOVA(df_list):
    f3_len = len(df_list)
    f1_len, f2_len = len(df_list[0].columns), len(df_list[0].index)
    
    # それぞれの因子の効果を求める
    f1_mean = sum([df.mean(axis=1) for df in df_list]) / f3_len
    f2_mean = sum([df.mean() for df in df_list]) / f3_len
    f3_mean = pd.Series([df.mean().mean() for df in df_list])
    f_mean = sum([df.mean().mean() for df in df_list]) / f3_len
    f1_effect, f2_effect, f3_effect = f1_mean - f_mean, f2_mean - f_mean, f3_mean - f_mean
    
    # 因子変動S1,S2,S3を求める
    S1 = ((f1_effect**2) * (f1_len*f3_len)).sum()
    S2 = ((f2_effect**2) * (f2_len*f3_len)).sum()
    S3 = ((f3_effect**2) * (f1_len*f2_len)).sum()
    
    # 因子1,2の交互作用による変動S12を求める
    df_12 = (sum(df_list) / f3_len) - f_mean
    S1_2 = (df_12**2).sum().sum() * f3_len
    S12 = S1_2 - S1 - S2
    
    # 因子1,3の交互作用による変動S13を求める
    df_13 = pd.DataFrame([df.mean(axis=1) for df in df_list]) - f_mean
    S1_3 = (df_13**2).sum().sum() * f1_len
    S13 = S1_3 - S1 - S3
    
    # 因子2,3の交互作用による変動S23を求める
    df_23 = pd.DataFrame([df.mean() for df in df_list]) - f_mean
    S2_3 = (df_23**2).sum().sum() * f2_len
    S23 = S2_3 - S2 - S3
    
    # 誤差変動Seを求める
    S = sum([((df-f_mean)**2).sum().sum() for df in df_list])
    Se = S - S1 - S2 - S3 - S12 - S13 - S23
    
    # 自由度dfを求める
    df1 = f2_len - 1
    df2 = f1_len - 1
    df3 = f3_len - 1
    df12 = df1 * df2
    df13 = df1 * df3
    df23 = df2 * df3
    dfe = df1 * df2 * df3
    
    # 不偏分散Vを求める
    V1 = S1 / df1
    V2 = S2 / df2
    V3 = S3 / df3
    V12 = S12 / df12
    V13 = S13 / df13
    V23 = S23 / df23
    Ve = Se / dfe
    
    # F値を求める
    F1 = V1 / Ve
    F2 = V2 / Ve
    F3 = V3 / Ve
    F12 = V12 / Ve
    F13 = V13 / Ve
    F23 = V23 / Ve
    
    # p値を求める
    p1 = 1 - st.f.cdf(F1, dfn=df1, dfd=dfe)
    p2 = 1 - st.f.cdf(F2, dfn=df2, dfd=dfe)
    p3 = 1 - st.f.cdf(F3, dfn=df3, dfd=dfe)
    p12 = 1 - st.f.cdf(F12, dfn=df12, dfd=dfe)
    p13 = 1 - st.f.cdf(F13, dfn=df13, dfd=dfe)
    p23 = 1 - st.f.cdf(F23, dfn=df23, dfd=dfe)
    
    # 分散分析表を作成する
    df_S = pd.Series([S1, S2, S3, S12, S13, S23, Se])
    df_df = pd.Series([df1, df2, df3, df12, df13, df23, dfe])
    df_V = pd.Series([V1, V2, V3, V12, V13, V23, Ve])
    df_F = pd.Series([F1, F2, F3, F12, F13, F23])
    df_p = pd.DataFrame([p1, p2, p3, p12, p13, p23], columns=['p'])
    df_p['sign'] = df_p['p'].apply(lambda x : '**' if x < 0.01 else '*' if x < 0.05 else '')
    df_ANOVA = pd.concat([df_S, df_df, df_V, df_F, df_p], axis=1).set_axis(['S','df','V','F','p','sign'], axis=1).set_axis(['Index', 'Columns', 'Tables', 'Index*Columns', 'Index*Tables', 'Columns*Tables', 'Error']).fillna('')
    
    # 因子の効果をデータフレームにまとめる
    df_effect = pd.DataFrame(pd.concat([f1_effect, f2_effect, f3_effect])).T.set_axis(['Effect'])
    
    # 結果を出力する
    return df_ANOVA, df_effect


# テスト
import numpy as np
import pandas as pd
import scipy.stats as st

df_3_no = pd.DataFrame([[9,10,12], [4,8,10], [6,9,12], [2,6,13]]).set_axis(['b0', 'b1', 'b2'], axis=1).set_axis(['a0', 'a1', 'a0', 'a1'], axis=0)
df_3_no_upper = df_3_no.iloc[:2]
df_3_no_under = df_3_no.iloc[2:]
print("入力：")
display(df_3_no)
display(df_3_no_upper)
display(df_3_no_under)

df_3_no_list = [df_3_no_upper, df_3_no_under]
df_ANOVA, df_effect = three_way_ANOVA(df_3_no_list)
print("出力：")
display(df_ANOVA, df_effect)
