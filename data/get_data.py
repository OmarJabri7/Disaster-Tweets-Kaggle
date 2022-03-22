import pandas as pd
import numpy as np
from nltk.corpus import words
import nltk
import re
import string
from data_processing import DisasterProcessor

X = pd.read_csv("emotion_data/tweet_emotions.csv")

stop_wrds = nltk.corpus.stopwords.words("english")
columns = X.columns
columns = ["content"]
preprocessor = DisasterProcessor()
eng_words = set(words.words())
for column in columns:
    X[column] = X[column].apply(
        lambda x: ' '.join([re.sub("[$@&#]","",w) for w in x.lower().split(" ") if w]))
    table = str.maketrans('', '', string.punctuation)
    X[column] = X[column].apply(
        lambda x: ' '.join([w.translate(table) for w in x.split(" ") if w.isalpha()]))
    X[column] = X[column].apply(
        lambda x: preprocessor.utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=stop_wrds))
    X[column] = X[column].apply(
        lambda x: ' '.join([w for w in x.split(" ") if len(w) >= 2]))

X["content"] = X["content"].apply(
    lambda x: ' '.join(([w for w in x.split(" ") if w in eng_words]))
)
unique_words = list(X['content'].str.split(' ', expand=True).stack().unique())
# X.Sentence = X.Sentence.apply(lambda x: x if len(x) > 2 else np.nan)

# X["clean_content"] = X["content"].str.replace('[#,@,&,=,[,http://]', '')

print(np.unique(X["sentiment"]))

X = X.loc[X['sentiment'].isin(['sadness','happiness','love','hate','fun','enthusiasm','relief','fear','anger',
                               'surprise', 'worry'])]

# X = X["sentiment" in ['sadness','happiness','love','hate','fun','enthusiasm','relief','fear','anger']]

X = X[['sentiment','content']]

# happy = X.loc[X['sentiment'].isin(['happiness','fun','enthusiasm','relief']), 'content'].values

happy = X.loc[X['sentiment'].isin(['happiness']), 'content'].values

love = X.loc[X['sentiment'].isin(['love']),'content'].values

# sadness = X.loc[X['sentiment'].isin(['sadness','worry']), 'content'].values

sadness = X.loc[X['sentiment'].isin(['sadness']), 'content'].values

# angry = X.loc[X['sentiment'].isin(['hate','anger']), 'content'].values

angry = X.loc[X['sentiment'].isin(['anger']), 'content'].values

surprise = X.loc[X['sentiment'].isin(['surprise']), 'content'].values

fear = X.loc[X['sentiment'].isin(['fear']),'content'].values

# emotions = dict(Emotion = ['happy','love','sadness','angry','surprise','fear'])
# data = {"Sentence" : [happy, love, sadness, angry, surprise, fear],
#         "Emotion" : ['joy','love','sadness','anger','surprise','fear'],}
#
data = {"Sentence" : [sadness, angry, fear],
        "Emotion" : ['sadness','anger','fear'],}

new_df = pd.DataFrame(data)

new_df = new_df.explode('Sentence', ignore_index=True)

new_df.to_csv('emotion_data/add_data.txt', header=None, index=None, sep=';')
