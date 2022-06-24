# Descriptive_statistics_functions
## 概要
様々な統計量をPythonによって実装し、分析の利便性や統計量理解を深めるためのプログラム集です。  
統計学の勉強を進めて新しい統計量を学び次第(あるいは自分のやる気が起き次第)、それらを順次実装していくつもりです。  
各フォルダにそれぞれのプログラムの説明が書かれているREADME.mdファイルがありますので、詳しくはそちらを参照してください。  
これらのプログラムを使用する際はMIT Licenseに従ってください。


## 背景
統計学の勉強も兼ねて、データを入力として記述統計情報を**一から計算して**出力するdescribe関数を作って**いました**。  
しかし、describe関数を作る過程で(統計学を勉強する過程も)で様々な統計量を知り、これらを全て実装したり、作った関数に利便性を持たせるためにはdescribe関数だけでは足りないことに気づき、他のものも作成しました。  
これからも未知なる統計量を勉強し、それらのPythonによる実装を目指していきたいです。


## 内容
大まかに**6つ**に分類しています。

### [describe_functions](https://github.com/ARAN1218/Descriptive_statistics_functions/tree/main/describe_functions)
- Pandasのdescribeメソッドに感動し、同時に失望したので思いついた関数。あらゆる統計量を網羅的に出力します。
- あくまで勉強が目的なので、便利なライブラリや関数、メソッドは極力使わず、自分の実装したプログラムによって**一から計算して**出力しています。

### [the_probability_distribution_tables](https://github.com/ARAN1218/Descriptive_statistics_functions/tree/main/the_probability_distribution_tables)
- describe_functionsによる検定で出力されるのは検定統計量であってp値ではないため、最終的には出力された検定統計量を基にご自分で対応する確率分布表を見て有意差を判定してもらいます。その時に簡単に調べられるようにExcelを用いて作っておいた種々の確率分布表です。
- せっかくなのでちょっと範囲広めに作っています。

### [statistics_class](https://github.com/ARAN1218/Descriptive_statistics_functions/tree/main/statistics_class)
- describe_functionsを作成している時にクラス版も欲しくなったので作りました。
- 比較値との検定に特化したCompareクラスと、総合的に統計量を計算するStatisticsクラスを作成しています。

### [n_way_ANOVA](https://github.com/ARAN1218/Descriptive_statistics_functions/tree/main/n_way_ANOVA)
- 実験計画法を勉強していく内に分散分析の面白さに気がつき、n元配置分散分析(nは3以下の自然数)をPythonで実装しました。
- 二元配置分散分析以降は繰返しの有無で関数を分けて作成しています。

### [cross_tabulation](https://github.com/ARAN1218/Descriptive_statistics_functions/tree/main/cross_tabulation)
- 質的変数を分析する手法としてクロス集計に関する統計的検定を自動化しました。
- 期待度数表の出力は勿論、イェーツの補正有無を両方とも出力します。

### [NLP(Natural Language Processing)](https://github.com/ARAN1218/Descriptive_statistics_functions/tree/main/NLP)
- アンケートデータ等のテキストマイニングの手法について、Pythonで実装できるものをまとめました。
- 環境構築が少々面倒かもしれません...

### [correlation](https://github.com/ARAN1218/Descriptive_statistics_functions/tree/main/correlation)
- ピアソン積率相関係数やスピアマン順位相関係数等の適用しやすい相関係数以外の相関係数を実装しました。
- 名義尺度同士でも相関を見れます。(クラメールの連関係数)
