def odds_risk_ratio(df_test):
    # オッズ比を計算する
    odds_list = []
    for df in df_test.iteritems():
        odds_list.append(df[1][0] / df[1][1])
    odds_ratio = odds_list[0] / odds_list[1]
    
    # 95%CLを計算する
    odds_95CL = [odds_ratio*np.exp(i * np.sqrt(1/df_test.iat[0, 0] + 1/df_test.iat[0, 1] + 1/df_test.iat[1, 0] + 1/df_test.iat[1, 1])) for i in [-1.96, 1.96]]
    
    
    # リスク比を計算する
    risk_list = []
    for df in df_test.iterrows():
        risk_list.append(df[1][0] / df[1].sum())
    risk_ratio = risk_list[0] / risk_list[1]
    
    # 95%CLを計算する
    risk_95CL = [risk_ratio*np.exp(i*np.sqrt(1/df_test.iat[0, 0]-1/(df_test.iat[0, 0]+df_test.iat[0, 1]) + 1/df_test.iat[1, 0]-1/(df_test.iat[1, 0]+df_test.iat[1, 1]))) for i in [-1.96, 1.96]]
    
    
    # 計算結果をデータフレームにして出力する
    df_display = pd.DataFrame([[odds_ratio, odds_95CL], [risk_ratio, risk_95CL]], columns=['点推定値', '95%信頼区間'], index=['オッズ比', 'リスク比'])
    display(df_display)
    
    # 95%信頼区間について、オッズ比・リスク比からの差分の絶対値に変換する
    odds_95CL = [abs(x-odds_ratio) for x in odds_95CL]
    risk_95CL = [abs(x-risk_ratio) for x in risk_95CL]
    
    # オッズ比についてフォレストプロットを行う
    plt.figure(figsize=(5,3))
    plt.errorbar(['odds_ratio', 'risk_ratio'], [odds_ratio, risk_ratio], fmt='o', ecolor='red', color='black', markersize=8)
    plt.errorbar(['odds_ratio'], [odds_ratio], yerr=odds_95CL[0], capsize=4, label="uplims", uplims=True, color='blue')
    plt.errorbar(['odds_ratio'], [odds_ratio], yerr=odds_95CL[1], capsize=4, label="lolims", lolims=True, color='blue')
    plt.errorbar(['risk_ratio'], [risk_ratio], yerr=risk_95CL[0], capsize=4, label="uplims", uplims=True, color='red')
    plt.errorbar(['risk_ratio'], [risk_ratio], yerr=risk_95CL[1], capsize=4, label="lolims", lolims=True, color='red')
    plt.xticks([-1, 0, 1, 2])
    plt.axhline(y=1)
    plt.show()
    
    
# テスト1
cross_taburation = [
    [20,40],
    [10,30]
]
df_test = pd.DataFrame(cross_taburation, columns=['肺がんである', '肺がんでない'], index=['喫煙者', '非喫煙者'])

print('入力：')
display(df_test)
print('出力：')
odds_risk_ratio(df_test)


# テスト2
cross_taburation = [
    [200,400],
    [100,300]
]
df_test = pd.DataFrame(cross_taburation, columns=['肺がんである', '肺がんでない'], index=['喫煙者', '非喫煙者'])

print('入力：')
display(df_test)
print('出力：')
odds_risk_ratio(df_test)
