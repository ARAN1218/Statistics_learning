# NLP(Natural Language Processing)
## 概要
アンケートの分析において、テキストデータを用いてワードクラウドを作成するという試みがなされ、思ったより簡単だったので個人でも実装してみました。
これを機に様々なテキストマイニング手法を勉強していく予定です。


## リファレンス
### wordcloud
- あるテキストデータ全体の「出現頻度」が多い単語について、「文字の大きさ」を大きくしてプロットすることで可視化する分析方法である。
  - 大きさだけでなく、文字の色や向き、フォント等を変えて表現することもある。
- よく使われている単語を知ることで得られる知見も当然存在するが、単に頻度を可視化したものであるため、過信は禁物である。


#### 使い方
<img width="95" alt="スクリーンショット 2022-03-30 12 57 48" src="https://user-images.githubusercontent.com/67265109/160748632-0f00c14e-cfdd-4e32-8801-d56965230a87.png">
<img width="565" alt="スクリーンショット 2022-03-30 12 57 57" src="https://user-images.githubusercontent.com/67265109/160748640-b2c2e9be-d39a-4bcc-b684-525c4ec52ba0.png">

1. アンケートの感想等、解析したいテキストデータの入ったSeries型のデータを引数に入れ、wordcloud関数を実行する。
2. 引数に入れた全ての文章に対してワードクラウドが作成され、出力される。
3. ワードクラウドを元に分析する。(このワードクラウドでは、文字の向き・色は関係ない)


### sentiment_analysis
- あるテキストデータが表現している内容がポジティブなのかネガティブなのかを二値分類する分析方法である。
  - ポジティブ判定の割合とネガティブ判定の割合を比較し、より大きい方を最終的な判定とする。
- ポジネガの二値分類以外にも、喜怒哀楽等の感情を分析する方法も存在する。
- この分析のためには評価表現辞書を用いて単語毎のポジネガのカウントをしたり、教師あり学習で判定モデルを作成したりする。(この関数は後者)


#### 使い方
<img width="661" alt="スクリーンショット 2022-03-31 21 48 26" src="https://user-images.githubusercontent.com/67265109/161058279-3b5ea85f-ae59-4621-8e1f-03de8cd78b77.png">


1. janome==0.3.7, scikit-learn==0.23.0にバージョンを合わせる。
2. 上記画像の様に分析対象のテキストデータを用意し、Series型データの引数としてsentiment_analysis関数を実行する。
3. 該当テキスト・ポジティブ割合・ネガティブ割合・ポジネガ判定の順に記載されているデータフレームが出力される。


### N-gram
- あるテキスト中の文字や単語を指定の語数ずつに連続的に分割し、それらの表記単位(gram)の出現頻度を集計する分析方法である。
  - テキスト中の任意の長さの表現の出現頻度パターンなどを知ることができる。
  - 長所：文字単位で分割する場合、辞書が必要ない。
  - 短所：分割後に思わぬ検索ノイズが入る可能性がある。(ex. 東京都→東京&京都)
- N-gramにおいて、N=1の場合をunigram、N=2の場合をbigram、N=3の場合をtrigramと呼ぶ。


#### 使い方
<img width="419" alt="スクリーンショット 2022-04-24 21 14 44" src="https://user-images.githubusercontent.com/67265109/164976006-7f3cfd31-48c0-439e-9e62-a507f0301642.png">


1. 上記画像の様に分析対象のSeries型テキストデータを用意する。
2. 下記の引数を設定し、n_gram関数を実行する。
3. 文字gram、単語gramの集計データフレームが出力される。

**引数の説明**  
引数select_partにより、特定の品詞のみを抽出可  
引数only_kanjiにより、ひらがなとカタカナを除去可  
引数display_cntにより、頻度表の表示数を変更可
