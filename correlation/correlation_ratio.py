# 相関比
# 名義尺度と感覚尺度の相関係数を算出する
def correlation_ratio(class_name, value_name, data):
    # 水準に関する値を収集する
    class_columns = data[class_name].unique()
    all_class_num = len(class_columns)
    all_average = np.mean(data[value_name])
    
    # 級内変動を求める
    class_num_list, value_average_list, sum_of_squared_deviation_list = [], [], []
    for i, class_column in enumerate(class_columns):
        class_num_list.append(len(data[value_name][data[class_name] == class_column]))
        value_average_list.append(np.mean(data[value_name][data[class_name] == class_column]))
        sum_of_squared_deviation_list.append(sum((data[value_name][data[class_name] == class_column] - value_average_list[i])**2))
    Sw = sum(sum_of_squared_deviation_list)
    
    # 級間変動を求める
    Sb = sum([class_num * (value_average_list[i] - all_average)**2 for i, class_num in enumerate(class_num_list)])
    
    # 相関比を求める
    etasq = Sb / (Sw+Sb)
    
    # 結果を出力する
    display(pd.DataFrame([Sw, Sb, etasq]
                 ,index=['級内変動', '級間変動', '相関比']
                 ,columns=['value']
                ).T)
    
    
# テスト
df_test = pd.DataFrame([['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C']
                        , [29,32,35,36,38,40,41,43,48,20,22,24,29,35,38]]
                      ,index=['好きな商品', '年齢']).T
print('入力データ:')
display(df_test)
print('出力データ:')
correlation_ratio('好きな商品', '年齢', df_test)
