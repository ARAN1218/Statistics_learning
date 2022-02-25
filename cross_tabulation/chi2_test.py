def chi2_test(df_test):
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
    df_expected = pd.DataFrame(expected_frequency, columns=df_test.columns, index=df_test.index)
    chi2, p = [], []
    for yates in [0, 0.5]:
        df_chi2 = (abs(df_test - df_expected) - yates)**2 / df_expected
        chi2.append(df_chi2.sum().sum())
        df = (len(df_test.columns)-1) * (len(df_test.index)-1)
        p.append(st.chi2.sf(chi2, df))
    
    # 結果を出力する
    display(df_expected)
    # 期待度数の内、20%以上が5未満であった時、警告を出す
    check = [i for expected in expected_frequency for i in expected if i < 5]
    if (len(check) > len(expected_frequency)//5):
        print("注意：期待度数の内、5未満の値が全体の20%以上存在します！\nフィッシャーの正確確率検定の使用をオススメします！")
    return chi2[0], p[0][0], chi2[1], p[1][1]


# テスト1
import numpy as np
import pandas as pd
import scipy.stats as st
df_test = pd.DataFrame([[13, 7], [5, 15]], columns=['治った', '治らなかった'], index=['薬剤群', 'コントロール群'])

print('入力：')
display(df_test)
print('出力：')
answers = chi2_test(df_test)
print(answers)


# テスト2
import numpy as np
import pandas as pd
import scipy.stats as st
df_test = pd.DataFrame([[35, 71], [52, 61]], columns=['草食系', '肉食系'], index=['女性', '男性'])

print('入力：')
display(df_test)
print('出力：')
answers = chi2_test(df_test)
print(answers)


# テスト3
import numpy as np
import pandas as pd
import scipy.stats as st
df_test = pd.DataFrame([[24, 3, 999], [85, 1, 999]], columns=['痩せ', '標準', '肥満'], index=['女性', '男性'])

print('入力：')
display(df_test)
print('出力：')
answers = chi2_test(df_test)
print(answers)
