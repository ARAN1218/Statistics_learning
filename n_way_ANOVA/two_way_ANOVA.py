def two_way_ANOVA(df):
    f1_len, f2_len = len(df.columns), len(df.index)
    
    # 行平均/列平均/全体平均を求め、それぞれの水準の効果を求める
    f1_mean, f2_mean, f_mean = df.mean(axis=1), df.mean(), df.mean().mean()
    f1_effect, f2_effect = f1_mean - f_mean, f2_mean - f_mean
    
    # それぞれの因子変動S1, S2を求める。
    S1 = ((f1_effect**2) * f1_len).sum()
    S2 = ((f2_effect**2) * f2_len).sum()
    
    # 誤差変動Seを求めるため、総変動Stを求める
    df_total = df.copy() - f_mean
    St = (df_total**2).sum().sum()
    Se = St - S1 - S2
    
    # 自由度df(=それぞれの因子の水準数)を求める
    df1 = f2_len - 1
    df2 = f1_len - 1
    dfe = df1*df2
    dft = df1 + df2 + dfe
    
    # 不偏分散Vを求める
    V1 = S1 / df1
    V2 = S2 / df2
    Ve = Se / dfe
    
    # F値を求める
    F1 = V1 / Ve
    F2 = V2 / Ve
    
    # 求めたF値からp値を求める
    p1 = 1 - st.f.cdf(F1, dfn=df1, dfd=dfe)
    p2 = 1 - st.f.cdf(F2, dfn=df2, dfd=dfe)
    
    # 分散分析表を作成する
    df_S = pd.Series([S1, S2, Se, St])
    df_df = pd.Series([df1, df2, dfe, dft])
    df_V = pd.Series([V1, V2, Ve])
    df_F = pd.Series([F1, F2])
    df_p = pd.DataFrame([p1, p2], columns=['p'])
    df_p['sign'] = df_p['p'].apply(lambda x : '**' if x < 0.01 else '*' if x < 0.05 else '')
    df_ANOVA = pd.concat([df_S, df_df, df_V, df_F, df_p], axis=1).set_axis(['S','df','V','F','p','sign'], axis=1).set_axis(['Index', 'Columns','Error', 'Total']).fillna('')
    
    # 因子の効果をデータフレームにまとめる
    df_effect = pd.DataFrame(pd.concat([f1_effect, f2_effect])).T.set_axis(['Effect'])
    
    # 結果を出力する
    return df_ANOVA, df_effect


# テスト
import numpy as np
import pandas as pd
import scipy.stats as st

df_2_no = pd.DataFrame([[4,8,12], [8,10,12]]).set_axis(['b0', 'b1', 'b2'], axis=1).set_axis(['a0', 'a1'], axis=0)
print("入力：")
display(df_2_no)

df_ANOVA, df_effect = two_way_ANOVA(df_2_no)
print("出力：")
display(df_ANOVA, df_effect)
