def correspondence(df_test):
    # 質的データ版の主成分分析を実行する
    mca_counts = mca.MCA(df_test)
    rows = mca_counts.fs_r(N=2) # 表側データ
    cols = mca_counts.fs_c(N=2) # 表頭データ
    
    # 第一成分と第二成分の固有値と寄与率をデータフレームとして出力する
    df_display = pd.DataFrame([mca_counts.L[:2], mca_counts.expl_var(greenacre=True, N=2)*100], columns=['1', '2'], index=['Eigen', 'Contribution']).T
    display(df_display)
    
    # 散布図の下地を作る
    plt.figure(figsize=(10,8))
    plt.axhline(y=0)
    plt.axvline(x=0)

    # 表側データのプロット
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
