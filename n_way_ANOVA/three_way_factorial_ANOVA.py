def three_way_factorial_ANOVA(df_lists):
    # 繰返し数、因子1,2,3の長さを取得
    df_lists_len = len(df_lists)
    f1_len = len(df_lists[0][0].columns)
    f2_len = len(df_lists[0][0].index)
    f3_len = len(df_lists[0])
    
    # それぞれの因子の効果を求める
    f1_mean = sum([df[i].sum(axis=1) for df in df_lists for i in range(df_lists_len)]) / (f1_len*f3_len*df_lists_len)
    f2_mean = sum([df[i].sum() for df in df_lists for i in range(df_lists_len)]) / (f2_len*f3_len*df_lists_len)
    f3_mean = sum([pd.Series([df[i].mean().mean() for i in range(df_lists_len)]) for df in df_lists]) / f3_len
    f_mean = sum([df[i].sum().sum() for df in df_lists for i in range(df_lists_len)]) / (f1_len*f2_len*f3_len*df_lists_len)
    f1_effect, f2_effect, f3_effect = f1_mean - f_mean, f2_mean - f_mean, f3_mean - f_mean
    
    # 因子変動S1,S2,S3を求める
    S1 = ((f1_effect**2) * (f1_len*f3_len*df_lists_len)).sum()
    S2 = ((f2_effect**2) * (f2_len*f3_len*df_lists_len)).sum()
    S3 = ((f3_effect**2) * (f1_len*f2_len*df_lists_len)).sum()
    
    # 繰返し分を全て平均したテーブルを作成する
    df_ave = [0 for i in range(f3_len)]
    for i in range(f3_len):
        for j in range(df_lists_len):
            df_ave[i] += df_lists[j][i]
        df_ave[i] /= df_lists_len
    
    # 因子1,2の交互作用による変動S12を求める
    df_12 = (sum(df_ave) / f3_len) - f_mean
    S1_2 = (df_12**2).sum().sum() * (f3_len*df_lists_len)
    S12 = S1_2 - S1 - S2
    
    # 因子1,3の交互作用による変動S13を求める
    df_13 = pd.DataFrame([df.mean(axis=1) for df in df_ave]) - f_mean
    S1_3 = (df_13**2).sum().sum() * (f1_len*df_lists_len)
    S13 = S1_3 - S1 - S3
    
    # 因子2,3の交互作用による変動S23を求める
    df_23 = pd.DataFrame([df.mean() for df in df_ave]) - f_mean
    S2_3 = (df_23**2).sum().sum() * (f2_len*df_lists_len)
    S23 = S2_3 - S2 - S3
    
    # 因子1,2,3の交互作用による変動S123を求める
    df_123 = df_ave - f_mean
    S1_2_3 = (df_123**2).sum().sum() * df_lists_len
    S123 = S1_2_3 - S1 - S2 - S3 - S12 - S13 - S23
    
    # 誤差変動Seを求める
    St = sum([((df_lists[i][j]-f_mean)**2).sum().sum() for i in range(df_lists_len) for j in range(f3_len)])
    Se = St - S1 - S2 - S3 - S12 - S13 - S23 - S123
    
    # 自由度dfを求める
    df1 = f2_len - 1
    df2 = f1_len - 1
    df3 = f3_len - 1
    df12 = df1 * df2
    df13 = df1 * df3
    df23 = df2 * df3
    df123 = df1 * df2 * df3
    dfe = f1_len*f2_len*f3_len*(df_lists_len - 1)
    dft = df1 + df2 + df3 + df12 + df13 + df23 + df123 + dfe
    
    # 不偏分散Vを求める
    V1 = S1 / df1
    V2 = S2 / df2
    V3 = S3 / df3
    V12 = S12 / df12
    V13 = S13 / df13
    V23 = S23 / df23
    V123 = S123 / df123
    Ve = Se / dfe
    
    # F値を求める
    F1 = V1 / Ve
    F2 = V2 / Ve
    F3 = V3 / Ve
    F12 = V12 / Ve
    F13 = V13 / Ve
    F23 = V23 / Ve
    F123 = V123 / Ve
    
    # p値を求める
    p1 = 1 - st.f.cdf(F1, dfn=df1, dfd=dfe)
    p2 = 1 - st.f.cdf(F2, dfn=df2, dfd=dfe)
    p3 = 1 - st.f.cdf(F3, dfn=df3, dfd=dfe)
    p12 = 1 - st.f.cdf(F12, dfn=df12, dfd=dfe)
    p13 = 1 - st.f.cdf(F13, dfn=df13, dfd=dfe)
    p23 = 1 - st.f.cdf(F23, dfn=df23, dfd=dfe)
    p123 = 1 - st.f.cdf(F123, dfn=df123, dfd=dfe)
    
    # 分散分析表を作成する
    df_S = pd.Series([S1, S2, S3, S12, S13, S23, S123, Se, St])
    df_df = pd.Series([df1, df2, df3, df12, df13, df23, df123, dfe, dft])
    df_V = pd.Series([V1, V2, V3, V12, V13, V23, V123, Ve])
    df_F = pd.Series([F1, F2, F3, F12, F13, F23, F123])
    df_p = pd.DataFrame([p1, p2, p3, p12, p13, p23, p123], columns=['p'])
    df_p['sign'] = df_p['p'].apply(lambda x : '**' if x < 0.01 else '*' if x < 0.05 else '')
    df_ANOVA = pd.concat([df_S, df_df, df_V, df_F, df_p], axis=1).set_axis(['S','df','V','F','p','sign'], axis=1).set_axis(['Index', 'Columns', 'Tables', 'Index*Columns', 'Index*Tables', 'Columns*Tables', 'Index*Columns*Tables', 'Error', 'Total']).fillna('')
    
    # 因子の効果をデータフレームにまとめる
    df_effect = pd.DataFrame(pd.concat([f1_effect, f2_effect, f3_effect])).T.set_axis(['Effect'])
    
    # 結果を出力する
    return df_ANOVA, df_effect
  
  
# テスト
import numpy as np
import pandas as pd
import scipy.stats as st

df_3_yes_1 = pd.DataFrame([[8,10,12], [4,8,12], [6,6,12], [2,4,12]]).set_axis(['b0', 'b1', 'b2'], axis=1).set_axis(['a0', 'a1', 'a0', 'a1'], axis=0)
df_3_yes_2 = pd.DataFrame([[10,12,12], [6,10,10], [4,4,12], [0,2,12]]).set_axis(['b0', 'b1', 'b2'], axis=1).set_axis(['a0', 'a1', 'a0', 'a1'], axis=0)
df_3_yes_1_upper = df_3_yes_1.iloc[:2]
df_3_yes_1_under = df_3_yes_1.iloc[2:]
df_3_yes_2_upper = df_3_yes_2.iloc[:2]
df_3_yes_2_under = df_3_yes_2.iloc[2:]

print("入力：")
display(df_3_yes_1)
display(df_3_yes_1_upper)
display(df_3_yes_1_under)
print()
display(df_3_yes_2)
display(df_3_yes_2_upper)
display(df_3_yes_2_under)

df_3_yes_lists = [[df_3_yes_1_upper, df_3_yes_1_under], [df_3_yes_2_upper, df_3_yes_2_under]]
df_ANOVA, df_effect = three_way_factorial_ANOVA(df_3_yes_lists)
print("出力：")
display(df_ANOVA, df_effect)
