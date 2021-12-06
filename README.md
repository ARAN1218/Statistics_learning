# Descriptive_statistics_functions
## 概要
統計学の勉強も兼ねて、データを入力として記述統計情報を**一から計算して**出力するdescribe関数を作っています。  
まだ正確に計算できていない箇所があったり、追加できていない要素もありますので、これから随時改良していく予定です。


## 背景
データ分析を行う際、記述統計を求める作業が必ずと行って良い程あります。そのような時にはpandasライブラリのdescribeメソッドを用いるのですが、たった一行のプログラムで平均や標準偏差といった代表的な統計量を求められるという事に感動しました。  
ある時、ふとpandasのdescribeメソッドを見て「そういえば第一四分位数と第三四分位数はあるのに四分位範囲は出力されないのか」と気付きました。そこで私は四分位範囲も出力してくれる機能を追加したいと思い、実際にSeries型データを引数とした一行データdescribe関数を作ってみました。自分がプログラミングしたコードでpandasのdescribe要素+四分位範囲が出力されるという事象が想像以上に面白く、もっと統計量を出力する関数を作りたいと思いました。その好奇心が高じて、二行・三行バージョンのdescribe関数も作成に至りました。



## 機能
### 一行describe関数
- サンプルサイズ(count)
- 合計値(sum)
- 平均値(mean)
- 母分散(var.p)
- 不偏分散(var.s)
- 母標準偏差(std.p)
- 不偏標準偏差(std.s)
- 最小値(min)
- 第一四分位数(25%)
- 中央値(50%)
- 第三四分位数(75%)
- 最大値(max)
- 四分位範囲(25-75%)

#### グラフ
- 線グラフ
- 散布図
- ヒストグラム
- ヒストグラム(累積)
- 箱ひげ図
- バイオリンプロット
- イベントプロット


### 二行describe関数
#### それぞれ求めるもの
- サンプルサイズ(count)
- 合計値(sum)
- 平均値(mean)
- 母分散(var.p)
- 不偏分散(var.s)
- 母標準偏差(std.p)
- 不偏標準偏差(std.s)
- 最小値(min)
- 第一四分位数(25%)
- 中央値(50%)
- 第三四分位数(75%)
- 最大値(max)
- 四分位範囲(25-75%)

#### 2つのデータから求めるもの
- 母共分散(cov.p)
- 不偏共分散(cov.s)
- ピアソン相関係数(pearson_cor)
- ピアソン無相関検定(検定統計量t)(pearson_cor_test)
- スピアマン相関係数(spearman_cor)
- スピアマン無相関検定(検定統計量t)(spearman_cor_test)
- ケンドール相関係数(kendall_cor)
- ケンドール無相関検定(検定統計量z)(kendall_cor_test)
- F検定(検定統計量F)(f_test)
- 対応なし2標本t検定(検定統計量t)(indep_ttest.t)
- 効果量1(対応なしt検定ver)(Cohen_d)
- 対応あり2標本t検定(検定統計量t)(dep_ttest.t)
- 効果量2(対応ありt検定ver)(Cohen_d)
- ウェルチのt検定(検定統計量t)(welch.ttest)
- マン=ホイットニーのU検定(検定統計量U)(mw_utest.u)
- ウィルコクソンの順位和検定(検定統計量Tw)(w_rstest.tw)
- ウィルコクソンの符号順位検定(検定統計量Tw)(w_srtest.tw)
- 符号検定(検定統計量m)(sign_test)

#### グラフ(一標本)
- ヒストグラム
- 箱ひげ図
- バイオリンプロット
- イベントプロット

#### グラフ(二標本)
- 線グラフ
- 散布図
- ステムプロット
- ステッププロット
- hist2d


### 三行describe関数
#### それぞれ求めるもの
- サンプルサイズ(count)
- 合計値(sum)
- 平均値(mean)
- 母分散(var.p)
- 不偏分散(var.s)
- 母標準偏差(std.p)
- 不偏標準偏差(std.s)
- 最小値(min)
- 第一四分位数(25%)
- 中央値(50%)
- 第三四分位数(75%)
- 最大値(max)
- 四分位範囲(25-75%)

#### 2つのデータから求めるもの
- 母共分散(cov.p)
- 不偏共分散(cov.s)
- ピアソン相関係数(pearson_cor)
- ピアソン無相関検定(検定統計量t)(pearson_cor_test)
- スピアマン相関係数(spearman_cor)
- スピアマン無相関検定(検定統計量t)spearman_cor_test)
- ケンドール相関係数(kendall_cor)
- ケンドール無相関検定(検定統計量z)(kendall_cor_test)

#### 3つのデータから求めるもの
- 偏相関係数(partial_cor)
- バートレット検定(検定統計量x2)(bartlett_test)
- ルビーン検定(検定統計量F)(levene_test)
- 一元配置分散分析(検定統計量F)(anova)
- 反復測定分散分析(検定統計量F)(fm_test)
- クラスカル=ウォリス検定(検定統計量H)(kw_test)
- フリードマン検定(検定統計量x20)(fm_test)

Coming soon...


## 今後の展望
- 検定統計量の出力とセットで自由度も出力するように改善したい。
- 全てのdescribe関数における出力の表示を、見やすいように統一したい。
- 現在出力内容として組み込んでいるものの、計算式の間違い等の原因による出力ミスが起きている統計量を修正したい。
- 現在出力している統計量の他に何か計算できる統計量を調査し、それを新たに出力するプログラムを追加したい。
- ヒストグラムや散布図等、統計的なグラフを出力するプログラムも追加したい。
- 上記の出力内容の説明にて、その統計量の意味を正確にまとめていきたい。
- 三行describe関数を四行以上の引数にて正常に動作するプログラムに改善したい。(もしくは別の関数として定義するのもアリ)
