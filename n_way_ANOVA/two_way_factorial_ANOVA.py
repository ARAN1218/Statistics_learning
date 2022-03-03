def two_way_factorial_ANOVA(df_list):
    # 繰返し数、因子1,2の長さを取得
    df_list_len = len(df_list)
    f1_len, f2_len = len(df_list[0].columns)*df_list_len, len(df_list[0].index)*df_list_len
    
    # それぞれの因子の効果を求める
    f1_mean = sum([df.mean(axis=1) for df in df_list]) / df_list_len
    f2_mean = sum([df.mean() for df in df_list]) / df_list_len
    f_mean = sum([df.mean().mean() for df in df_list]) / df_list_len
    f1_effect, f2_effect = f1_mean - f_mean, f2_mean - f_mean
    
    # 因子変動S1,S2を求める
    S1 = ((f1_effect**2) * f1_len).sum()
    S2 = ((f2_effect**2) * f2_len).sum()
    
    # 因子1,2の交互作用による変動S12を求める
    df_con = (sum(df_list) / df_list_len) - f_mean
    S1_2 = ((df_con**2) * df_list_len).sum().sum()
    S12 = S1_2 - S1 - S2
    
    # 誤差変動Seを求める
    St = sum([((df-f_mean)**2).sum().sum() for df in df_list])
    Se = St - S1 - S2 - S12
    
    # 自由度dfを求める
    df1 = (f2_len / df_list_len) - 1
    df2 = (f1_len / df_list_len) - 1
    df12 = df1 * df2
    dfe = (f1_len / df_list_len) * (f2_len / df_list_len) * (df_list_len - 1)
    dft = df1 + df2 + df12 + dfe
    
    # 不偏分散Vを求める
    V1 = S1 / df1
    V2 = S2 / df2
    V12 = S12 / df12
    Ve = Se / dfe
    
    # F値を求める
    F1 = V1 / Ve
    F2 = V2 / Ve
    F12 = V12 / Ve
    
    # p値を求める
    p1 = 1 - st.f.cdf(F1, dfn=df1, dfd=dfe)
    p2 = 1 - st.f.cdf(F2, dfn=df2, dfd=dfe)
    p12 = 1 - st.f.cdf(F12, dfn=df12, dfd=dfe)
    
    # 分散分析表を作成する
    df_S = pd.Series([S1, S2, S12, Se, St])
    df_df = pd.Series([df1, df2, df12, dfe, dft])
    df_V = pd.Series([V1, V2, V12, Ve])
    df_F = pd.Series([F1, F2, F12])
    df_p = pd.DataFrame([p1, p2, p12], columns=['p'])
    df_p['sign'] = df_p['p'].apply(lambda x : '**' if x < 0.01 else '*' if x < 0.05 else '')
    df_ANOVA = pd.concat([df_S, df_df, df_V, df_F, df_p], axis=1).set_axis(['S','df','V','F','p','sign'], axis=1).set_axis(['Indexes', 'Columns', 'Indexes*Columns', 'Error', 'Total']).rename_axis('ANOVA_table', axis=1).fillna('')
    
    # 因子の効果をデータフレームにまとめる
    df_effect_indexes = pd.DataFrame(f1_effect).set_axis(['mean'], axis=1)
    df_effect_columns = pd.DataFrame(f2_effect).set_axis(['mean'], axis=1)
    df_effect_indexes['mean(95%CL)'] = df_effect_indexes['mean'].map(lambda x : st.t.interval(0.95, dfe, loc=x, scale=np.sqrt(Ve/f1_len)))
    df_effect_columns['mean(95%CL)'] = df_effect_columns['mean'].map(lambda x : st.t.interval(0.95, dfe, loc=x, scale=np.sqrt(Ve/f2_len)))
    df_effect = pd.concat([df_effect_indexes, df_effect_columns]).T.rename_axis('Effect', axis=1)
    
    # 各水準毎の取得データの予測値をデータフレームにまとめる
    df_prediction = pd.DataFrame([[f1 + f2 + f_mean for f2 in f2_effect] for f1 in f1_effect])
    df_prediction = df_prediction.applymap(lambda x : st.t.interval(0.95, dfe, loc=x, scale=np.sqrt(Ve/df_list_len + Ve)))
    df_prediction = df_prediction.set_axis(df_list[0].index).set_axis(df_list[0].columns, axis=1).rename_axis('Prediction(95%CL)', axis=1)
    
    # 結果を出力する
    return df_ANOVA, df_effect, df_prediction


# テスト
import numpy as np
import pandas as pd
import scipy.stats as st

df_2_yes1 = pd.DataFrame([[8,10,12], [4,8,12]]).set_axis(['column0', 'column1', 'column2'], axis=1).set_axis(['index0', 'index1'], axis=0)
df_2_yes2 = pd.DataFrame([[6,6,12], [2,4,12]]).set_axis(['column0', 'column1', 'column2'], axis=1).set_axis(['index0', 'index1'], axis=0)
print("入力：")
display(df_2_yes1)
display(df_2_yes2)

df_2_yes_list = [df_2_yes1, df_2_yes2]
df_ANOVA, df_effect, df_prediction = two_way_factorial_ANOVA(df_2_yes_list)
print("出力：")
display(df_ANOVA, df_effect, df_prediction)
