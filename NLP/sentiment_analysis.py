# janome==0.3.7, scikit-learn==0.23.0
#!pip install scikit-learn==0.23.0
#!pip install Janome==0.3.7


# 必要なライブラリをインポート
%%capture capt # # エラーの表示を消す
from asari.api import Sonar
import pandas as pd


def sentiment_analysis(df_texts):
    text_list, positive_list, negative_list, top_list = [], [], [], []
    for text in df_tests['texts']:
        result = sonar.ping(text=text)
        text_list.append(result['text'])
        positive_list.append(result['classes'][1]['confidence'])
        negative_list.append(result['classes'][0]['confidence'])
        top_list.append(result['top_class'])
    df_result = pd.DataFrame([text_list, positive_list, negative_list, top_list], index=['text','positive','negative','sentiment']).T
    display(df_result)


# テスト
sonar = Sonar()
df_tests = pd.DataFrame([
    ['とても良いパソコンですね。'],
    ['あなたって最低ですね。'],
    ['嫌いじゃないけど好きじゃない。'],
    ['好きじゃないけど嫌いだよ。'],
    ['私はこれが好きじゃなくなくなくなくなくなくなくなくなくなくない。'],
    ['私はこれが好きじゃなくなくなくなくなくなくなくなくなくなくなくない。']
], columns=['texts'])

print("入力：")
display(df_texts)
print("出力：")
sentiment_analysis(df_tests)
