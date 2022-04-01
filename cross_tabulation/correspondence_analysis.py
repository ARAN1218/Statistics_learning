# 必要なライブラリをインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import mca
from adjustText import adjust_text


def correspondence(df_test):
    # 質的データ版の主成分分析を実行する
    mca_counts = mca.MCA(df_test)
    rows = mca_counts.fs_r(N=2) # 表側データ
    cols = mca_counts.fs_c(N=2) # 表頭データ
    
    # 表側・表頭データの第一成分と第二成分の値をデータフレームとして出力する
    df_rows = pd.DataFrame([rows[:,0], rows[:,1]], columns=df_test.index, index=['X', 'Y']).T
    df_cols = pd.DataFrame([cols[:,0], cols[:,1]], columns=df_test.columns, index=['X', 'Y']).T
    df_rows_cols = pd.concat([df_rows, df_cols])
    display(df_rows_cols)
    
    # 第一成分と第二成分の固有値と寄与率をデータフレームとして出力する
    df_factors = pd.DataFrame([mca_counts.L[:2], mca_counts.expl_var(greenacre=True, N=2)*100], columns=['X', 'Y'], index=['Eigen', 'Contribution']).rename_axis('Factors', axis=1).T
    display(df_factors)
    
    # 散布図の下地を作る
    plt.figure(figsize=(10,8))
    plt.axhline(y=0)
    plt.axvline(x=0)

    # 表側データのプロット
    plt.title('correspondence analysis', fontsize=20)
    plt.xlabel('Dim 1({:.2f}%)'.format(df_factors['Contribution']['X']), fontsize=13)
    plt.ylabel('Dim 2({:.2f}%)'.format(df_factors['Contribution']['Y']), fontsize=13)
    plt.scatter(rows[:,0], rows[:,1], c='b', marker='o')
    labels = df_test.index
    texts = [plt.text(rows[:,0][k], rows[:,1][k], labels[k], ha='center', va='center', fontsize=13) for k in range(0, len(labels))]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='green'))

    # 表頭データのプロット
    plt.scatter(cols[:,0], cols[:,1], c='r', marker='x')
    labels = df_test.columns
    texts = [plt.text(cols[:,0][k], cols[:,1][k], labels[k], ha='center', va='center', fontsize=13) for k in range(0, len(labels))]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='green'))

    plt.show()


# テスト
df_test = pd.read_csv("mca_data/burgundies.csv", sep=',', skiprows=1, index_col=0, header=0)

print('入力：')
display(df_test)
print('出力：')
correspondence(df_test)
