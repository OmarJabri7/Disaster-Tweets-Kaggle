import pandas as pd
from data.data_processing import DisasterProcessor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB, CategoricalNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
import pickle
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC

params_XGB = {
    'learning_rate': np.arange(0.01, 1.0, 0.01),
    'n_estimators': [1500],
}

params_MNB = {
    'fit_prior': [True],
    # 'alpha': np.arange(0.01, 1.0, 0.01),
    'alpha': [0.01]
}

params_SVC = {
    "kernel":'linear',
    # "C" : np.arange(0.01, 1.0, 0.01),
    # "class_weight": {0: 0.43, 1: 0.57},
    "C": 0.01,
    "probability" : False
}

params_SGD = {
    # "tol": np.arange(0.01, 1.0, 0.01),
    "tol": [1e-3],
    'loss': ["log"],
    # "alpha": np.arange(0.01, 1.0, 0.01),
    "alpha": [0.01],
    "penalty":["l1"],
    "n_iter_no_change": [100],
    "random_state":[42]
}

params_LR = {
"penalty": ["l1"],
# "tol": np.arange(0.001,0.01,0.001),
    "tol": [0.001],
 # "C": np.arange(0.01,1,0.01),
    "C": [0.01],
    # "solver": ["saga", "liblinear"]
    "solver": ["saga"]
}

params_RF = {
    "n_estimators":100,
    "criterion":"gini",
    "random_state":42,
    "verbose": 1,
    "n_jobs":-1
}

params_DT = {}

params_CNB = {
    'fit_prior': [True],
    'alpha': np.arange(0.01, 1.0, 0.01),
}

params_CNB_FIN = {
    'fit_prior': True,
    'alpha': 0.98
}

params_GBC = {"verbose":[2],
    "learning_rate":[0.01],
    "n_estimators":[100],
    "max_depth":[4],
    "min_samples_split":[6],
    "min_samples_leaf":[2],
    "max_features":[8],
    "subsample":[0.4]}

params_LSVC = {
    "C":  np.arange(0.001,1,0.001),
    "tol": [1e-2],
}

def create_w2vc(sentences, vector_size = 200):
    word_model = Word2Vec(sentences, vector_size=vector_size, min_count=1, window=5)
    pretrained_weights = word_model.wv.vectors
    vocab_size, embedding_size = pretrained_weights.shape
    return vocab_size, embedding_size

def tokenize_data(x, vocab, len_train_words):
    tokenizer = Tokenizer(num_words=len(vocab),
                          lower=True, char_level=False)
    tokenizer.fit_on_texts(x.tolist())
    training_sequences = tokenizer.texts_to_sequences(
        x.tolist())
    train_word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(train_word_index))
    x = pad_sequences(training_sequences,
                            maxlen=max(len_train_words))
    return x

def vectorize_data(x, vectorizer):
    if  vectorizer is CountVectorizer:
        vectorizer = CountVectorizer(decode_error='ignore', lowercase=True, min_df=2,
                                     analyzer = 'word', ngram_range=(1,2), stop_words='english')
    elif vectorizer is TfidfVectorizer:
        vectorizer = TfidfVectorizer(decode_error='ignore', lowercase=True, min_df=2,
                                     analyzer = 'word', ngram_range=(2,2), stop_words='english')
    elif vectorizer is TfidfTransformer:
        vectorizer = TfidfTransformer()
    if vectorizer:
        x = vectorizer.fit_transform(x.values.astype('U'))
    return x, vectorizer

def model_data(model_cls, params, hyperparam_search = None, x = None, y = None):
    if hyperparam_search and x is not None and y is not None:
        grid_search = hyperparam_search(model_cls(), params, n_jobs=-1, verbose=3)
        grid_search.fit(x, y)
        grid_params = set(grid_search.best_params_)
        known_params = set(params)
        common_params = known_params.intersection(grid_params)
        for param in common_params:
            params[param] = grid_search.best_params_[param]
        model = model_cls(**params)
    else:
        model = model_cls(**params)
    print(f"Best params: {params}")
    return model

def train_model(preprocessor):

    if preprocessor is DisasterProcessor:
        preprocessor = DisasterProcessor()
        feature = "text"

    train_data, test_data, unique_words = preprocessor.prepare_data()

    X_train, y_train = train_data[0], train_data[1]

    # y_train = y_train.astype(np.float).reshape((-1,1))


    X_train.to_csv("data/clean_train.csv")

    sentences = [[word for word in tokens.split(" ")]
                          for tokens in X_train[feature]]

    words = [word for tokens in X_train[feature] for word in tokens.split(" ")]

    len_train_words = [len(tokens) for tokens in X_train[feature]]

    TRAINING_VOCAB = sorted(list(set(words)))

    print("%s words total, with a vocabulary size of %s" %
          (len(words), len(TRAINING_VOCAB)))
    print("Max sentence length is %s" % max(len_train_words))
    if isinstance(preprocessor, DisasterProcessor):
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

    x_vec, vectorizer = vectorize_data(X_train[feature], CountVectorizer)

    x_vec = x_vec.toarray()

    x_vec_va = vectorizer.transform(X_test[feature]).toarray()

    model = model_data(MultinomialNB, params_MNB, RandomizedSearchCV, x_vec, y_train)

    model.fit(x_vec, y_train)

    print("train score:", model.score(x_vec, y_train))

    y_pred = model.predict(x_vec_va)

    print(accuracy_score(y_pred, y_test))

    if isinstance(preprocessor, DisasterProcessor):

        X_test = test_data[feature]

        X_vec_te = vectorizer.transform(X_test)

        X_vec_te = X_vec_te.toarray()

        y_pred = model.predict(X_vec_te)


        y_pred = ["disaster" if x == 1 else "NO disaster" for x in y_pred]

        res = pd.DataFrame(X_test, columns=[feature])

        res["result"] = y_pred

        res.to_csv("Final_disaster.csv")

    pickle.dump(model, open('outputs/model.sav', 'wb'))

    pickle.dump(vectorizer, open("outputs/vectorizer.sav", "wb"))
