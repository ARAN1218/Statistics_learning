import numpy as np
import pandas as pd
from scipy import stats as st


def ancova(data, y, x1, x2):
    df = data.copy()
    
    # 共分散分析のために、重回帰分析を実装する
    def mra(data, y, x1, x2, x3=False):
        df_mra = data.copy()
        # 共分散行列を作成する
        cov_x1x2 = np.linalg.inv(np.cov(df_mra[x1], df_mra[x2], ddof=1)) if x3==False else np.linalg.inv(np.cov([df_mra[x1], df_mra[x2]], df_mra[x1]*df_mra[x2], ddof=1))
        cov_yx1 = np.cov(df_mra[y], df_mra[x1], ddof=1)[0][1]
        cov_yx2 = np.cov(df_mra[y], df_mra[x2], ddof=1)[0][1]
        cov_yx3 = 0 if x3==False else np.cov(df_mra[y], df_mra[x1]*df_mra[x2], ddof=1)[0][1]
        cov_y = np.matrix([cov_yx1, cov_yx2]).T if x3==False else np.matrix([cov_yx1, cov_yx2, cov_yx3]).T

        # 偏回帰係数と切片を計算する
        a = np.dot(cov_x1x2, cov_y)
        b = np.mean(df_mra[y]) - a[0]*np.mean(df_mra[x1]) - a[1]*np.mean(df_mra[x2]) if x3==False else np.mean(df_mra[y]) - a[0]*np.mean(df_mra[x1]) - a[1]*np.mean(df_mra[x2]) - a[2]*np.mean(df_mra[x1]*df_mra[x2])

        # 不偏分散を求める
        df_mra['pred'] = df_mra.apply(lambda x : a[0]*x[x1] + a[1]*x[x2] + b, axis=1) if x3==False else df_mra.apply(lambda x : a[0]*x[x1] + a[1]*x[x2] + a[2]*x[x1]*x[x2] + b, axis=1)
        df_mra['residual'] = df_mra[y] - df_mra['pred'].astype(float)
        df_mra['RSoS'] = df_mra['residual']**2
        Ve = sum(df_mra['RSoS']) / (len(df_mra) - len(a) - 1)

        # 標準誤差を求める
        cov_x1x2_after = np.linalg.inv((len(df_mra)-1)*np.cov(df_mra[x1], df_mra[x2], ddof=1)) if x3==False else np.linalg.inv((len(df_mra)-1)*np.cov([df_mra[x1], df_mra[x2]], df_mra[x1]*df_mra[x2], ddof=1))
        std_e_X1 = np.sqrt(cov_x1x2_after[0][0] * Ve)
        std_e_X2 = np.sqrt(cov_x1x2_after[1][1] * Ve)
        std_e_X3 = 0 if x3==False else np.sqrt(cov_x1x2_after[2][2] * Ve)
        # 切片の標準誤差の算出方法がわからないが、共分散分析を行う上では問題ない
        # 参考：http://ifs.nog.cc/gucchi24.hp.infoseek.co.jp/MRA2.htm
        #std_e_b = np.sqrt(((1/len(df)) + X1_mean*X2_mean*cov_x1x2_after[0][0] + X1_mean*X2_mean*cov_x1x2_after[1][1]) * Ve)

        # t値を計算する
        t_X1 = a[0] / std_e_X1
        t_X2 = a[1] / std_e_X2
        t_X3 = 0 if x3==False else a[2] / std_e_X3

        # p値を求める(両側検定なので、p値を二乗する)
        p_X1 = st.t.sf(np.abs(t_X1), len(df_mra)-len(a)-1)*2
        p_X2 = st.t.sf(np.abs(t_X2), len(df_mra)-len(a)-1)*2
        p_X3 = 0 if x3==False else st.t.sf(np.abs(t_X3), len(df_mra)-len(a)-1)*2
        
        # p値を出力する
        return [float(p_X1), float(p_X2)] if x3==False else [float(p_X1), float(p_X2), float(p_X3)]
    
    # 重回帰分析を実行する
    p2 = mra(data=df, y=y, x1=x1, x2=x2)
    p3 = mra(data=df, y=y, x1=x1, x2=x2, x3=True)
    
    # 共分散分析の結果を出力する
    display(pd.DataFrame([p3[2], p2[1], p2[0]], columns=['p_value'], index=['回帰直線の平行性', '回帰直線の有意差', '共分散分析']))
    
    
# テスト
df = pd.DataFrame([[1,1,1,1,1,1,2,2,2,2,2,2],
             [12,24,32,30,48,50,11,9,22,38,42,49],
             [2,1,4,3,6,8,2,5,6,6,7,8]],
             index=['条件', '数学test', '計算test']).T
print('入力：')
display(df)
print('出力：')
ancova(data=df, y='計算test', x1='条件', x2='数学test')
