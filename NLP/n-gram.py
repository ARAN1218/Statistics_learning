import pandas as pd
import MeCab
from janome.tokenizer import Tokenizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


# N-gramの関数(全文字・単語対象)
def n_gram_all(sentence):
    # 分かち書きを行う関数
    def wakachi(text):
        mecab = MeCab.Tagger("-Owakati")
        return mecab.parse(text).strip().split(" ")
    
    # 文字gramを作成する関数
    def char_gram(text, char_num):
        return [text[i:i+char_num] for i in range(len(text)-char_num+1)]

    # 文字uni~tri-gramで分割
    char_grams = []
    for i in range(1,4):
        char_grams.append([char for char_list in list(sentence.map(lambda x : char_gram(x, i))) for char in char_list])

    # 単語uni~tri-gramで分割
    word_grams = []
    for i in range(1,4):
        vectrizer = CountVectorizer(tokenizer = wakachi, ngram_range = (i, i))
        vectrizer.fit(sentence)
        word_grams.append([vectrizer.get_feature_names(), vectrizer.transform(sentence)])

    # 文字uni~tri-gramの頻度表を作成する
    df_uni = pd.Series(char_grams[0]).value_counts().reset_index()
    df_bi = pd.Series(char_grams[1]).value_counts().reset_index()
    df_tri = pd.Series(char_grams[2]).value_counts().reset_index()
    
    df_char = pd.concat([df_uni, df_bi, df_tri], axis=1).set_axis(['uni_char', 'uni_cnt', 'bi_char', 'bi_cnt', 'tri_char', 'tri_cnt'], axis=1).rename_axis('char_gram')
    display(df_char.head(10))

    # 単語uni~tri-gramの頻度表を作成する
    df_uni = pd.DataFrame(word_grams[0][1].toarray(), columns=word_grams[0][0]).T
    df_bi = pd.DataFrame(word_grams[1][1].toarray(), columns=word_grams[1][0]).T
    df_tri = pd.DataFrame(word_grams[2][1].toarray(), columns=word_grams[2][0]).T
    
    df_uni['word_cnt'] = df_uni.sum(axis=1)
    df_bi['word_cnt'] = df_bi.sum(axis=1)
    df_tri['word_cnt'] = df_tri.sum(axis=1)
    
    df_uni = df_uni.sort_values('word_cnt', ascending=False).reset_index()[['index', 'word_cnt']]
    df_bi = df_bi.sort_values('word_cnt', ascending=False).reset_index()[['index', 'word_cnt']]
    df_tri = df_tri.sort_values('word_cnt', ascending=False).reset_index()[['index', 'word_cnt']]
    
    df_word = pd.concat([df_uni, df_bi, df_tri], axis=1).set_axis(['uni_word', 'uni_cnt', 'bi_word', 'bi_cnt', 'tri_word', 'tri_cnt'], axis=1).rename_axis('word_gram')
    display(df_word.head(10))

# テスト
sentence = pd.Series(["今日は雨が降っています。", "今日は雨が降っていません", "今日は雨が降っているとでも思っていたのか"])
n_gram_all(sentence)




# N-gramの関数(種々の引数による調整可)
# 引数select_partにより、特定の品詞のみを抽出可
# 引数only_kanjiにより、ひらがなとカタカナを除去可
# 引数display_cntにより、頻度表の表示数を変更可
def n_gram(sentence, select_part=[], only_kanji=False, display_cnt=10):
    # select_partが設定されていない場合、名詞と形容詞を抽出する
    if select_part == []:
        select_part = ['名詞', '形容詞']
    
    # 分かち書き関数
    def wakachi(text):
        # 形態素解析の準備
        t = Tokenizer()

        noun_list = []
        for sentence in list(text):
            for token in t.tokenize(sentence):
                split_token = token.part_of_speech.split(',')
                # 一般名詞を抽出
                if split_token[0] in select_part:
                    noun_list.append(token.surface)
        return noun_list
    
    # 文字gramを作成する関数
    def char_gram(text, char_num):
        return [text[i:i+char_num] for i in range(len(text)-char_num+1)]

    # 文字uni~tri-gramで分割
    char_grams = []
    tagger = MeCab.Tagger("-p")
    for i in range(1,4):
        temp_list = []
        for texts in list(sentence.map(lambda x : char_gram(x, i))):
            for text in texts:
                try:
                    temp_list.append(text if tagger.parse(f"{text}\n").split(',')[0].split('\t')[1] in select_part else None)
                except:
                    pass
        char_grams.append(temp_list)

    # 単語uni~tri-gramで分割
    word_grams = []
    for i in range(1,4):
        vectrizer = CountVectorizer(tokenizer = wakachi, ngram_range = (i, i))
        vectrizer.fit(sentence)
        word_grams.append([vectrizer.get_feature_names(), vectrizer.transform(sentence)])

    # 文字uni~tri-gramの頻度表を作成する
    df_uni = pd.Series(char_grams[0]).value_counts().reset_index()
    df_bi = pd.Series(char_grams[1]).value_counts().reset_index()
    df_tri = pd.Series(char_grams[2]).value_counts().reset_index()
    
    # only_kanjiがTrueの時、ひらがなとカタカナを削除する
    if only_kanji == True:
        df_uni = df_uni[~df_uni['index'].str.contains('[ぁ-んァ-ン]')].reset_index(drop=True)
        df_bi = df_bi[~df_bi['index'].str.contains('[ぁ-んァ-ン]')].reset_index(drop=True)
        df_tri = df_tri[~df_tri['index'].str.contains('[ぁ-んァ-ン]')].reset_index(drop=True)
    
    df_char = pd.concat([df_uni, df_bi, df_tri], axis=1).set_axis(['uni_char', 'uni_cnt', 'bi_char', 'bi_cnt', 'tri_char', 'tri_cnt'], axis=1).rename_axis('char_gram')
    display(df_char.head(display_cnt))

    # 単語uni~tri-gramの頻度表を作成する
    df_uni = pd.DataFrame(word_grams[0][1].toarray(), columns=word_grams[0][0]).T
    df_bi = pd.DataFrame(word_grams[1][1].toarray(), columns=word_grams[1][0]).T
    df_tri = pd.DataFrame(word_grams[2][1].toarray(), columns=word_grams[2][0]).T
    
    df_uni['word_cnt'] = df_uni.sum(axis=1)
    df_bi['word_cnt'] = df_bi.sum(axis=1)
    df_tri['word_cnt'] = df_tri.sum(axis=1)
    
    df_uni = df_uni.sort_values('word_cnt', ascending=False).reset_index()[['index', 'word_cnt']]
    df_bi = df_bi.sort_values('word_cnt', ascending=False).reset_index()[['index', 'word_cnt']]
    df_tri = df_tri.sort_values('word_cnt', ascending=False).reset_index()[['index', 'word_cnt']]
    
    df_word = pd.concat([df_uni, df_bi, df_tri], axis=1).set_axis(['uni_word', 'uni_cnt', 'bi_word', 'bi_cnt', 'tri_word', 'tri_cnt'], axis=1).rename_axis('word_gram')
    display(df_word.head(display_cnt))

# テスト
sentence = pd.Series(["今日は雨が降っています。", "今日は雨が降っていません ", "今日は雨が降っているとでも思っていたのか"])
n_gram(sentence, select_part=['名詞'], only_kanji=True, display_cnt=3)
