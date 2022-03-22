import pandas as pd
from transformers import BertTokenizer, DistilBertTokenizerFast

from data.data_processing import DisasterProcessor
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from gensim import corpora
from transformers import glue_convert_examples_to_features
from chat import emotions
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
import pickle
from xgboost import XGBClassifier
from models.models import lstm_v1, lstm_v2, lstm_v3, lstm_v4, lstm_v5, lstm_v6, cnn_v1, cnn_v2, cnn_v3, cnn_v0, \
    best_cnn, bnn_v1, hybrid_v1, bert_v1, bert_v2
import tensorflow as tf

from sklearn.svm import SVC

params_XGB = {
    'learning_rate': np.arange(0.01, 1.0, 0.01),
    'n_estimators': 1500,
}

params_MNB = {
    'alpha': np.arange(0.01, 1.0, 0.01),
}

params_SVC = {
    "kernel":'rbf',
    "C" : np.arange(0.01, 1.0, 0.01)
}

params_SGD = {
    "max_iter": np.arange(1000,2000,1000),
    "tol": np.arange(0.01, 1, 0.01)
}

def create_w2vc(sentences, vector_size = 200):

    word_model = Word2Vec(sentences, vector_size=vector_size, min_count=1, window=5)
    pretrained_weights = word_model.wv.vectors
    vocab_size, embedding_size = pretrained_weights.shape
    return vocab_size, embedding_size, pretrained_weights

def load_w2vc():
    word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    weights = word2vec.vectors
    vocab_size, embedding_size = word2vec.vectors.shape
    return vocab_size, embedding_size, weights

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
        vectorizer = CountVectorizer(decode_error='ignore', lowercase=True, min_df=2)
    elif vectorizer is TfidfVectorizer:
        vectorizer = TfidfVectorizer(decode_error='ignore', lowercase=True, min_df=2)
    elif vectorizer is TfidfTransformer:
        vectorizer = TfidfTransformer()
    if vectorizer:
        x = vectorizer.fit_transform(x.values.astype('U'))
    return x, vectorizer

def model_data(model_cls, params, grid_cv = False, x = None, y = None):
    model = model_cls(**params)
    if grid_cv and x is not None and y is not None:
        grid_search = GridSearchCV(model, param_grid=params, n_jobs=-1, verbose=1)
        grid_search.fit(x, y)
        grid_params = set(grid_search.best_params_)
        known_params = set(params)
        common_params = known_params.intersection(grid_params)
        for param in common_params:
            params[param] = grid_search.best_params_[param]
        model = model_cls(**params)
    return model

def train_net(preprocessor, reshape=False, split = False, model="normal"):
    
    feature = "text"

    train_data, test_data, unique_words = preprocessor.prepare_data()

    X_train, y_train = train_data[0], train_data[1]

    X_train.to_csv("outputs/clean_data.csv")

    sentences = [[word for word in tokens.split(" ")]
                          for tokens in X_train[feature]]

    words = [word for tokens in X_train[feature] for word in tokens.split(" ")]

    len_train_words = [len(tokens) for tokens in X_train[feature]]

    TRAINING_VOCAB = sorted(list(set(words)))

    print("%s words total, with a vocabulary size of %s" %
          (len(words), len(TRAINING_VOCAB)))
    print("Max sentence length is %s" % max(len_train_words))

    if(model != "bert"):
        tok = Tokenizer(num_words=len(TRAINING_VOCAB),
                        lower=True, char_level=False)

        tok.fit_on_texts(X_train[feature].tolist())

        train_word_index = tok.word_index

        sequences = tok.texts_to_sequences(X_train[feature].tolist(),)

        X = pad_sequences(sequences, maxlen=max(len_train_words))

        X = np.asarray(X).astype('float64')
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", return_dict=False)
        # X = tokenizer(X_train[feature].tolist(), pad_to_max_length=max(len_train_words), max_length=max(len_train_words))
        X = glue_convert_examples_to_features(X_train, tokenizer, 128, 'mrpc')


    if split:
        X_train, X_val, y_train, y_val = train_test_split(X, y_train, test_size = 0.1, random_state = 42)

        print(f"Numbers of NO disasters {(y_train==0).sum()/len(y_train)*100}%")

        print(f"Numbers of disasters {(y_train==1).sum()/len(y_train)*100}%")

    else:
        X_train = X #Comment to use with test set

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',) #min_delta= 1e-1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="outputs/", histogram_freq=1)

    batch_size = 64
    epochs = 2

    if reshape:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]) #Comment to not reshape

    # y_train = y_train.astype(np.float).reshape((-1,1))

    embedding_size, vocab_size, weights = create_w2vc(sentences)
    # model = hybrid_v1(max(len_train_words), len(words), embedding_size, X_train, weights)
    # model = cnn_v1(max(len_train_words), len(words), embedding_size, X_train, weights)
    # model = cnn_v0(max(len_train_words), len(words), embedding_size, X_train, weights)
    # model = lstm_v6(max(len_train_words), len(words), embedding_size, X_train, weights)
    model = best_cnn(max(len_train_words), len(words), embedding_size, X_train, weights)
    # model = bnn_v1(X_train.shape[0]*0.9, max(len_train_words), len(words), embedding_size, X_train, weights)
    # model = bert_v1(max(len_train_words), len(words), embedding_size, X_train, weights)
    # model = bert_v2()

    
    model.fit([X_train], y_train, batch_size=batch_size,epochs=epochs,
          validation_split=0.2,)
              # callbacks=[tensorboard_callback])
    if split:
        score, acc = model.evaluate(X_val, y_val,
                                    batch_size=batch_size)
        print('Test loss:', score)
        print('Test accuracy:', acc)

    model.save('outputs/network.h5')
    # Save Tokenizer i.e. Vocabulary
    with open('outputs/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_test = test_data[feature]
    sequences = tok.texts_to_sequences(X_test.tolist())
    X_test = pad_sequences(sequences, maxlen=max(len_train_words))

    if reshape:
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    X_test = X_test.astype(np.float)

    y_pred = model.predict([X_test])

    y_preds = []

    for preds in y_pred:
        y_preds.append(preds[0])

    y_preds = np.array(y_preds)

    y_preds = list(map(lambda x: 0 if x else 1, (y_preds < 0.5)))

    y_preds = ["disaster" if x == 1 else "NO disaster" for x in y_preds]

    res = pd.DataFrame(test_data[feature], columns=[feature])

    res["result"] = y_preds

    res.to_csv("data/FinalDisaster.csv")

