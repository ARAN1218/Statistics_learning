def wordcloud(df_sentence):  
    t = Tokenizer()
    noun_list = []
    
    # 分かち書き
    for sentence in list(df_sentence):
        for token in t.tokenize(sentence):
            split_token = token.part_of_speech.split(',')
            # 名詞・形容詞を抽出(好きな品詞に変更できる)
            if split_token[0] == '名詞' or split_token[0] == '形容詞':
                noun_list.append(token.surface)

    # 品詞リストの要素を半角スペース区切りにする
    noun_space = ' '.join(map(str, noun_list))
    # wordcloudの設定(フォントの設定も同時に行う)
    wc = WordCloud(background_color="white", font_path="/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", width=300, height=300)
    wc.generate(noun_space)
    
    # プロットの設定
    plt.figure(figsize=(10,10))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, length=0)
    
    # wordcloudの表示
    plt.imshow(wc)
    plt.show()


# テスト(Seriesを引数に取る)
wordcloud(pd.DataFrame([["Google"],["Apple"],["Facebook"],["Amazon"],['Microsoft']])[0])
