def fisher_exact_test(df_test):
    # columnsとindexと全体の合計値を求める
    sum_columns, sum_index = list(df_test.sum()), list(df_test.sum(axis=1))
    sum_all = sum(sum_columns)
    
    # 分子を計算する
    factorial_columns = reduce(mul, [math.factorial(i) for i in sum_columns])
    factorial_index = reduce(mul, [math.factorial(i) for i in sum_index])
    numerator = factorial_columns * factorial_index
    
    # 分母を計算する
    factorial_list = []
    for index in df_test.index:
        for column in df_test.columns:
            factorial_list.append(math.factorial(df_test[column][index]))
    denominator = reduce(mul, factorial_list) * math.factorial(sum_all)
    
    # 結果を出力する
    p = numerator / denominator
    return p


# テスト
from functools import reduce
from operator import mul
import pandas as pd
import scipy.stats as st
df_test = pd.DataFrame([[3,2], [1,4]], columns=['肉が好き', '魚が好き'], index=['女性', '男性'])

print("入力：")
display(df_test)
print("出力：")
fisher_exact_test(df_test)


# python3.8以降の場合、以下のようにより簡潔に書ける(functools, operatorライブラリの代わりにmathライブラリをインポートする)
def fisher_exact_test(df_test):
    # columnsとindexと全体の合計値を求める
    sum_columns, sum_index = list(df_test.sum()), list(df_test.sum(axis=1))
    sum_all = sum(sum_columns)
    
    # 分子を計算する
    factorial_columns = math.prod([math.factorial(i) for i in sum_columns])
    factorial_index = math.prod([math.factorial(i) for i in sum_index])
    numerator = factorial_columns * factorial_index
    
    # 分母を計算する
    factorial_list = []
    for index in df_test.index:
        for column in df_test.columns:
            factorial_list.append(math.factorial(df_test[column][index]))
    denominator = math.prod(factorial_list) * math.factorial(sum_all)
    
    # 結果を出力する
    p = numerator / denominator
    return p
