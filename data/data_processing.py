import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import words
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
nltk.download('wordnet')
nltk.download('omw-1.4')


class DisasterProcessor:

    def __init__(self):
        self.data = "data/disaster_data"

    def utils_preprocess_text(self, text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):

        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

        lst_text = text.split()
        if lst_stopwords is not None:
            lst_text = [word for word in lst_text if word not in lst_stopwords]

        if flg_stemm == True:
            ps = nltk.stem.porter.PorterStemmer()
            lst_text = [ps.stem(word) for word in lst_text]
        if flg_lemm == True:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            lst_text = [lem.lemmatize(word) for word in lst_text]

        text = " ".join(lst_text)
        return text

    def read_data(self, data_type):
        data = pd.read_csv(f"{data_type}.csv")
        if data_type == f"{self.data}/train":
            X = data.loc[:, "keyword":"text"].fillna("")
            y = data["target"].fillna("")
            return X, y
        else:
            X = data.loc[:, "keyword":"text"].fillna("")
            return X

    def preprocess_data(self, X):
        stop_wrds = nltk.corpus.stopwords.words("english")
        columns = X.columns
        eng_words = set(words.words())
        for column in columns:
            X[column] = X[column].apply(
                lambda x: ' '.join([re.sub("[$@&#]","",w) for w in x.lower().split(" ") if w]))
            table = str.maketrans('', '', string.punctuation)
            X[column] = X[column].apply(
                lambda x: ' '.join([w.translate(table) for w in x.split(" ") if w.isalpha()]))
            X[column] = X[column].apply(
                lambda x: self.utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, lst_stopwords=stop_wrds))
            X[column] = X[column].apply(
                lambda x: ' '.join([w for w in x.split(" ") if len(w) >= 2]))

        X["text"] = X["text"].apply(
            lambda x: ' '.join(([w for w in x.split(" ") if w in eng_words]))
        )
        unique_words = list(X['text'].str.split(' ', expand=True).stack().unique())
        X['sequences'] = X.text.str.split(" ")
        X["length"] = X.sequences.apply(lambda x: len(x))
        X.text = X.text.apply(lambda x: x if len(x) > 2 else np.nan)
        return X,unique_words

    def prepare_data(self):
        X_train, y_train = self.read_data(f"{self.data}/train")

        X_test = self.read_data(f"{self.data}/test")
        X_train, unique_words_train = self.preprocess_data(X_train)
        mean_sentences = np.mean(X_train.length)
        print("AVG Sentence count: " + str(mean_sentences))
        X_train["labels"] = y_train.values
        X_train = X_train[X_train['text'].notna()]
        X_train = X_train[X_train['length'] >= int(1)]
        y_train = pd.DataFrame(X_train["labels"], columns=["labels"])

        X_test, unique_words_test = self.preprocess_data(X_test)
        X_test = X_test[X_test['text'].notna()]

        train_data = [X_train, y_train]
        test_data = X_test
        unique_words = [unique_words_train, unique_words_test]
        return train_data, test_data, unique_words
