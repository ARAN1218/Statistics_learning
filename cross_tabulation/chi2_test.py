def chi2_test(df_test, df_theory=None):
    if df_theory is not None: # 適合度の検定
        chi2 = ((df_test - df_theory)**2 / df_theory).sum().sum()
        df = len(df_test.columns) - 1
        p = st.chi2.sf(chi2, df)
        return chi2, p
        
    else: # 独立性の検定
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
            p.append(st.chi2.sf(chi2, df)) # 片側検定(上側確率を参照)

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


# テスト4
import numpy as np
import pandas as pd
import scipy.stats as st
df_test = pd.DataFrame([55, 22, 16, 7], index=['A型', 'B型', 'C型', 'クワガタ'], columns=['人類']).T.rename_axis('test')
df_theory = pd.DataFrame([40, 30, 20, 10], index=['A型', 'B型', 'C型', 'クワガタ'], columns=['人類']).T.rename_axis('theory')

print('入力：')
display(df_test)
display(df_theory)
print('出力：')
answers = chi2_test(df_test, df_theory)
print(answers)


# テスト5
import numpy as np
import pandas as pd
import scipy.stats as st
df_test = pd.DataFrame([[12, 34, 56, 78], [87, 65, 43, 21]], columns=['A型', 'B型', 'C型', 'ニイガタ'], index=['男性', '女性']).rename_axis('test')
df_theory = pd.DataFrame([[19, 28 ,37, 46], [64, 73, 82, 91]], columns=['A型', 'B型', 'C型', 'ニイガタ'], index=['男性', '女性']).rename_axis('theory')

print('入力：')
display(df_test)
display(df_theory)
print('出力：')
answers = chi2_test(df_test, df_theory)
print(answers)
