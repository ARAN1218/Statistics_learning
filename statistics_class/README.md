# statistics_class
## 概要
describe関数を作成している時にふとクラスバージョンも作りたくなったので作ってみました。  
クラスバージョンでは計算を一から実装するというよりは、既に存在しているオープンソースライブラリを使いこなす練習という意味で作成しています。
なので、NumpyやScipyといった強力な計算ライブラリをどんどん使って統計量を実装しています。ここで実装するために自分で調べてアウトプットし、
統計ライブラリの知識の引き出しを作っていこうという事です。  
作ったライブラリは今の所2つあり、思いつく限り機能やクラスを追加していく予定です。

## リファレンス
### Compareクラス
→ある平均値や分散との統計的検定を行う為の検定専門クラス。
#### インスタンス変数
- 入力データ(self.data)
- データ数(self.length)
- 平均値(self.mean)
- 母分散(self.var_p)
- 不偏分散(self.var_s)
- 母標準偏差(self.std_p)
- 標本標準偏差(self.std_s)

#### インスタンスメソッド
- ある比較したい平均値の検定(母分散既知)(mean_ztest(self, 比較値))
- ある比較したい平均値の検定(母分散未知)(mean_ttest(self, 比較値))
- ある比較したい分散の検定(母分散既知)(var_x2test(self, 比較値))


### Statisticsクラス
→データを引数にしたオブジェクトを作成し、そのデータの統計量の出力を幅広く取り扱う総合統計クラス。
#### インスタンス変数
- 入力データ(self.data)
- データ数(self.length)
- 平均値(self.mean)
- 母分散(self.var_p)
- 不偏分散(self.var_s)
- 母標準偏差(self.std_p)
- 標本標準偏差(self.std_s)
- 標準誤差(self.std_e)
- 最小値(self.min)
- 第一四分位数(self.per1)
- 中央値(self.median)
- 第三四分位数(self.per3)
- 最大値(self.max)
- 四分位偏差(self.quartile_range)

#### インスタンスメソッド
- ソートデータ(昇順)(sort_asc(self))
- ソートデータ(降順)(sort_desc(self))
- 共分散(cov_p(self, data))
- 不偏共分散(cov_s(self, data))


## 今後の展望
- Statisticsクラスに更に機能を盛り込んでいきたい。
