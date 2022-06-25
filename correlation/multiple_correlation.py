import numpy as np
import pandas as pd

# 重相関係数
# 1:多の量的変数間の相関関係を表す
def multiple_correlation(response_name, explanatory_name_list, data, ddof=1):
    # 必要なデータだけを持ったデータフレームを作成する
    explanatory_name_num = len(explanatory_name_list)
    all_name_list = explanatory_name_list.copy()
    all_name_list.insert(0,response_name)
    df_multi = data[all_name_list]
    df_S = data[explanatory_name_list]
    display(df_multi)
    
    # 分散共分散行列等の準備
    vc_matrix = df_multi.cov(ddof=ddof)
    S22 = np.array(df_S.cov(ddof=ddof))
    S22_inv = np.linalg.inv(S22)
    s1 = np.array(vc_matrix['response'])[1:]
    s1_T = s1.T
    s11 = vc_matrix[response_name][response_name]
    
    # 重相関係数を計算する
    R = ((np.dot(np.dot(s1_T,S22_inv), s1)) / s11)**(1/2)
    
    # 結果を出力する
    display(pd.DataFrame([R, R**2], columns=['value'], index=['重相関係数', '決定係数']).T)
    

# テスト
df_test = pd.DataFrame([np.random.randint(0,100,10), np.random.randint(0,100,10), np.random.randint(0,100,10)]
                       , index=['response', 'ex1', 'ex2']).T

multiple_correlation('response', ['ex1', 'ex2'], df_test, ddof=1)
