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
    df_ANOVA = pd.concat([df_S, df_df, df_V, df_F, df_p], axis=1).set_axis(['S','df','V','F','p','sign'], axis=1).set_axis(['Indexes', 'Columns','Error', 'Total']).rename_axis('ANOVA_table', axis=1).fillna('')
    
    # 因子の効果をデータフレームにまとめる
    df_effect_indexes = pd.DataFrame(f1_effect).set_axis(['mean'], axis=1)
    df_effect_columns = pd.DataFrame(f2_effect).set_axis(['mean'], axis=1)
    df_effect_indexes['mean(95%CL)'] = df_effect_indexes['mean'].map(lambda x : st.t.interval(0.95, dfe, loc=x, scale=np.sqrt(Ve/f1_len)))
    df_effect_columns['mean(95%CL)'] = df_effect_columns['mean'].map(lambda x : st.t.interval(0.95, dfe, loc=x, scale=np.sqrt(Ve/f2_len)))
    df_effect = pd.concat([df_effect_indexes, df_effect_columns]).T.rename_axis('Effect', axis=1)
    
    # 各水準毎の取得データの予測値をデータフレームにまとめる
    df_prediction = pd.DataFrame([[f1 + f2 + f_mean for f2 in f2_effect] for f1 in f1_effect])
    df_prediction = df_prediction.applymap(lambda x : st.t.interval(0.95, dfe, loc=x, scale=np.sqrt(Ve/1 + Ve)))
    df_prediction = df_prediction.set_axis(df.index).set_axis(df.columns, axis=1).rename_axis('Prediction(95%CL)', axis=1)
    
    # 結果を出力する
    return df_ANOVA, df_effect, df_prediction


# テスト
import numpy as np
import pandas as pd
import scipy.stats as st

df_2_no = pd.DataFrame([[4,8,12], [8,10,12]]).set_axis(['column0', 'column1', 'column2'], axis=1).set_axis(['index0', 'index1'], axis=0)
print("入力：")
display(df_2_no)

df_ANOVA, df_effect, df_prediction = two_way_ANOVA(df_2_no)
print("出力：")
display(df_ANOVA, df_effect, df_prediction)
