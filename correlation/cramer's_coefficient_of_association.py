# クラメールの連関係数
# 名義尺度同士の相関係数を算出する
# あるデータフレームに対し、名義尺度のカラム名と対象のデータフレームを引数にし、相関係数等の情報を出力する
def cramer(column1, column2, df):
    # クロスタブを作成する
    df_crosstab = pd.crosstab(df[column1], df[column2])
    
    # 期待度数表を作成する
    sum_columns, sum_index = df_crosstab.sum(), df_crosstab.sum(axis=1)
    sum_all = sum(sum_index)
    expected_frequency = []
    for index in sum_index:
        multi_values = []
        for column in sum_columns:
            multi_values.append((index*column) / sum_all)
        expected_frequency.append(multi_values)
            
    df_expected = pd.DataFrame(expected_frequency, columns=df_crosstab.columns, index=df_crosstab.index)
    df_expected.index.name = ''
    df_expected.columns.name = 'expect'
    
    # カイ二乗値を算出する
    chisq = ((df_crosstab - df_expected)**2 / df_expected).sum().sum()
    
    # クラメールの連関係数を計算する
    n = df_crosstab.sum().sum()
    k = min([len(df_crosstab.columns), len(df_crosstab.index)])
    r = (chisq / (n*(k-1)))**(1/2)
    
    # 結果を出力する
    display(df_expected)
    display(pd.DataFrame([n,k,chisq,r, '***' if 0.5<=r else '**' if 0.25<=r and r<0.5 else '*' if 0.1<=r and r<0.25 else '']
            , index=['サンプルサイズ', '最小カテゴリー数', 'カイ二乗値', 'クラメールの連関係数', '相関関係の強度']
            , columns=['values']).T)
    
    
# テスト1
data = pd.DataFrame([np.random.randint(1,3,100), np.random.randint(5,7,100)], index=['data1', 'data2']).T

print('入力データ:')
display(data)
print('出力データ:')
cramer('data1', 'data2', data)


# テスト2
data = pd.DataFrame([np.random.randint(1,10,100), np.random.randint(5,7,100)], index=['data1', 'data2']).T

print('入力データ:')
display(data)
print('出力データ:')
cramer('data1', 'data2', data)
