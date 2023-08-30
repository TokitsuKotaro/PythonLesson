from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df_origin = pd.read_csv("news_min_jp.csv")

text_list = list(df_origin["text"])

# 品詞フィルターの設定
token_filters = [POSKeepFilter(['名詞', '動詞', '形容詞']), LowerCaseFilter()]

# アナライザーの初期化
a = Analyzer(tokenizer=Tokenizer(), token_filters=token_filters)

words_list = []
for text in text_list:
  # フィルター付きで形態素解析
  tokens = a.analyze(text)

  # 表層形のみを抽出する
  words = []
  for token in tokens:
    words.append(str(token).split("\t")[0])

  words_list.append(words)

# わかち書き（単語のスペース区切り）形式に変換
wakati_list = list(map(lambda x: " ".join(x), words_list))
print(wakati_list)

# tf-idfの計算
vec = TfidfVectorizer(token_pattern='\\b\\w+\\b')
vec.fit(wakati_list)
bow = vec.transform(wakati_list)

# データフレーム化
df = pd.DataFrame(bow.toarray(), columns=vec.get_feature_names())
df["group"] = df_origin["group"]
print(df.head())

# 10-NNで機械学習
X = df.drop("group", axis=1)
y = df["group"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)
score = model.score(X_train, y_train)
print("train score: {}".format(score))
score = model.score(X_test, y_test)
print("test score: {}".format(score))