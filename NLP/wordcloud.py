def wordcloud(df_sentence, stopwords={}):
    # 形態素解析の準備
    t = Tokenizer()

    noun_list = []
    for sentence in list(df_sentence):
        for token in t.tokenize(sentence):
            split_token = token.part_of_speech.split(',')
            # 名詞・形容詞を抽出(好きな品詞に変更できる)
            if split_token[0] == '名詞' or split_token[0] == '形容詞':
                noun_list.append(token.surface)

    # 名詞リストの要素を空白区切りにする(word_cloudの仕様)
    noun_space = ' '.join(map(str, noun_list))
    # word cloudの設定(フォントの設定)
    wc = WordCloud(stopwords=stopwords, background_color="white", font_path="/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", width=300, height=300)
    wc.generate(noun_space)
    # 出力画像の大きさの指定
    plt.figure(figsize=(10,10))
    # 目盛りの削除
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False,
                   length=0)
    # word cloudの表示
    plt.imshow(wc)
    plt.show()


# テスト(Seriesを引数に取る)
wordcloud(pd.DataFrame([["Google"],["Apple"],["Facebook"],["Amazon"],['Microsoft']])[0])
