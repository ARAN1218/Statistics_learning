def chi2_test(df_test, yates=False):
    # columnsとindexと全体の合計値を求める
    sum_columns, sum_index = list(df_test.sum()), list(df_test.sum(axis=1))
    sum_all = sum(sum_columns)
    
    # 期待度数を求める
    expected_frequency = []
    for index in sum_index:
        index_values = []
        for column in sum_columns:
            index_values.append(index*column / sum_all)
        expected_frequency.append(index_values)
        
    # 検定統計量とp値を求める
    yates = 0.5 if yates is True else 0
    df_expected = pd.DataFrame(expected_frequency, columns=df_test.columns, index=df_test.index)
    df_chi2 = (abs(df_test - df_expected) - yates)**2 / df_expected
    chi2 = df_chi2.sum().sum()
    df = (len(df_test.columns)-1) * (len(df_test.index)-1)
    p = st.chi2.sf(chi2, df)
    
    return chi2, p
  
  
# テスト1
import numpy as np
import pandas as pd
import scipy.stats as st
df_test = pd.DataFrame([[13, 7], [5, 15]], columns=['治った', '治らなかった'], index=['薬剤群', 'コントロール群'])
display(df_test)
chi2_test(df_test)

# テスト2
import numpy as np
import pandas as pd
import scipy.stats as st
df_test = pd.DataFrame([[35, 71], [52, 61]], columns=['草食系', '肉食系'], index=['女性', '男性'])
display(df_test)
print("イェーツの補正無し：", chi2_test(df_test, yates=False))
print("イェーツの補正有り：", chi2_test(df_test, yates=True))
