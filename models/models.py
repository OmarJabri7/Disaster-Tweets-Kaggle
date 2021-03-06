from keras import Input, regularizers
from keras.models import Sequential, Model
from keras.layers import Dropout, LSTM, Bidirectional, Dense, Embedding, BatchNormalization, SpatialDropout1D, GRU, \
    Activation, SimpleRNN, Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, concatenate, GlobalAveragePooling1D
import tensorflow as tf

from keras import backend as K

import tensorflow_probability as tfp
from transformers import BertModel, BertTokenizer, DistilBertTokenizerFast, DistilBertModel, \
    TFBertForSequenceClassification, TFDistilBertForSequenceClassification


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

# from tensorflow.keras import regularizers
from keras.utils.vis_utils import plot_model


def lstm_v1(max_len, max_words, embedding_size, optimizer='RMSprop', dropout=0.2, units=512):
    model = Sequential()
    model.add(Embedding(max_words, 100, input_length=max_len))
                        #weights=[weights]))
    model.add(Bidirectional(LSTM(units, return_sequences=True)))

    model.add(Dropout(dropout))

    model.add(Bidirectional(LSTM(units)))

    model.add(Dropout(dropout))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['acc'])
    return model

def lstm_v2(max_len, max_words, embedding_size,  X = None, weights = None):
    model = Sequential()
    model.add(Embedding(max_words, 100, input_length=max_len))
    model.add(Bidirectional(GRU(128)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='RMSprop', metrics=['acc'])
    return model

def lstm_v3(max_len, max_words, embedding_size = None, X = None, weights = None):
    model = Sequential()
    optimizer = tf.keras.optimizers.Adam(0.001)
    model.add(Embedding(embedding_size, 200, input_length=max_len, weights=[weights]))
    model.add(Bidirectional(LSTM(256,
                                 dropout = 0.3,
                                 recurrent_dropout=0.3,
                                 kernel_regularizer=regularizers.l2(l2=1e-2),
                                 bias_regularizer=regularizers.l2(l2=1e-2),
                                 return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Flatten())
    # model.add(Bidirectional(LSTM(256,
    #                              dropout=0.2,
    #                              recurrent_dropout=0.2,
    #                              kernel_regularizer=regularizers.l2(l2=1e-3),
    #                              bias_regularizer=regularizers.l2(l2=1e-3)
    #                              )))
    # model.add(BatchNormalization())
    # model.add(Dense(512, activation="relu"))
    # model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model

def lstm_v4(a,b,c):
    model = Sequential()
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics=['acc'])
    return model

def lstm_v5(max_len, max_words, embedding_size, X = None, weights = None):
    model = Sequential()
    model.add(Embedding(embedding_size, 100, input_length=max_len))
    model.add(LSTM(256, batch_input_shape=(128, X.shape[0], X.shape[1]),dropout=0.3, recurrent_dropout=0.1))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def lstm_v6(max_len, max_words, embedding_size = None, X = None, weights = None):
    model = Sequential()
    model.add(Embedding(max_words, 300, input_length=max_len))
    # model.add()
    model.add(Bidirectional(GRU(256,
                   # batch_input_shape=(64, X.shape[0], X.shape[1]),
                   dropout=0.3,
                   recurrent_dropout=0.3,
                   # return_sequences=True,
                   # stateful=True
                   )))
    # model.add(Bidirectional(LSTM(256)))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def best_cnn(max_len, max_words, embedding_size, weights = None, X = None, filters = 250, kernel_size = 3, hidden_dims = 250):
    inputs1 = Input(shape=(max_len,))
    model = Sequential()
    model.add(Embedding(embedding_size, 300, input_length=max_len))#, weights=[weights]))
    model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1,
                     input_shape=(max_len, embedding_size),
                     ))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    # model.add(Conv1D(128, 3, padding='valid', activation='relu', strides=1,
    #                  input_shape=(max_len, embedding_size)))
    # model.add(Dropout(0.5))
    # model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # model.add(BatchNormalization())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

def cnn_v0(max_len, max_words, embedding_size, weights = None, X = None, filters = 250, kernel_size = 3, hidden_dims = 250):
    inputs1 = Input(shape=(max_len,))
    embedding1 = Embedding(embedding_size, 300, input_length=max_len)(inputs1)
    conv1 = Conv1D(256, 3, padding='valid', activation='relu', strides=1,
                 input_shape=(max_len, embedding_size),
                    kernel_regularizer=regularizers.l2(l2=1e-2),
                   # bias_regularizer=regularizers.l2(l2=1e-2)
                   )(embedding1)
    drop1 = SpatialDropout1D(0.5)(conv1)
    max_pool1 = MaxPooling1D()(drop1)
    conv2 = Conv1D(128, 5, padding='valid', activation='relu', strides=1,
                   # kernel_regularizer=regularizers.l2(l2=1e-2),
                   # bias_regularizer=regularizers.l2(l2=1e-2)
                   )(max_pool1)
    drop2 = SpatialDropout1D(0.5)(conv2)
    max_pool2 = MaxPooling1D()(drop2)
    flatten1 = Flatten()(max_pool2)
    dense1 = Dense(256, activation="relu")(flatten1)
    drop2 = Dropout(0.5)(dense1)
    outputs = Dense(1, activation="sigmoid")(drop2)
    model = Model(inputs=inputs1, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def cnn_v1(max_len, max_words, embedding_size, weights = None, X = None, filters = 250, kernel_size = 3, hidden_dims = 250):
    model = Sequential()
    model.add(Embedding(embedding_size, 300, input_length=max_len))#, weights=[weights]))
    model.add(Conv1D(64, 3, padding='valid', activation='relu', strides=2,
                     input_shape=(max_len, embedding_size),
                     kernel_regularizer=regularizers.l2(l2=1e-3),
                     ))
    model.add(SpatialDropout1D(0.3))
    model.add(MaxPooling1D())
    # model.add(Conv1D(512, 7, padding='valid', activation='relu', strides=1,
    #                  input_shape=(max_len, embedding_size)))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D())
    # model.add((GRU(256,
    #              # dropout=0.3,
    #              # recurrent_dropout=0.3,
    #              # kernel_regularizer=regularizers.l2(l2=1e-2),
    #              # bias_regularizer=regularizers.l2(l2=1e-2),
    #              # return_sequences=True
    #                              )))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn_v2(max_len, max_words, embedding_size, X = None, weights = None, filters = 250, kernel_size = 3, hidden_dims = 250):
    inputs1 = Input(shape=(max_len,))
    embedding1 = Embedding(embedding_size, 200, weights=[weights])(inputs1)
    conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(l2=1e-3))(embedding1)
    drop1 = SpatialDropout1D(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    inputs2 = Input(shape=(max_len,))
    embedding2 = Embedding(embedding_size, 200, weights=[weights])(inputs2)
    conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(l2=1e-3))(embedding2)
    drop2 = SpatialDropout1D(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    inputs3 = Input(shape=(max_len,))
    embedding3 = Embedding(embedding_size, 200, weights=[weights])(inputs3)
    conv3 = Conv1D(filters=128, kernel_size=7, activation='relu', kernel_regularizer=regularizers.l2(l2=1e-3))(embedding3)
    drop3 = SpatialDropout1D(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    merged = concatenate([flat1, flat2, flat3])
    dense1 = Dense(64, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

def cnn_v3(max_len, max_words, embedding_size, weights = None, X = None, filters = 250, kernel_size = 3, hidden_dims = 512):
    model = Sequential()
    model.add(Embedding(embedding_size, 500, input_length=max_len,))# weights=[weights]))
    model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1,
                     input_shape=(max_len, embedding_size)))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def bnn_v1(train_size,max_len, max_words, embedding_size, weights = None, X = None, filters = 250, kernel_size = 3, hidden_units = [32]):
    inputs1 = Input(shape=(max_len,))
    # embedding1 = Embedding(embedding_size, 300, input_length=max_len)(inputs1)
    # conv1 = Conv1D(64, 3, padding='valid', activation='relu', strides=2,
    #                  input_shape=(max_len, embedding_size),
    #                  kernel_regularizer=regularizers.l2(l2=1e-3),
    #                  )(embedding1)
    # space_drop1 = SpatialDropout1D(0.3)(conv1)
    # max_pool1 = MaxPooling1D()(space_drop1)
    # flatten1 = Flatten()(max_pool1)
    features = BatchNormalization()(inputs1)
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight= 1 / train_size,
            activation="sigmoid",
        )(features)
    # drop1 = Dropout(0.5)(features)
    # dense1 = Dense(256)(drop1)
    drop1 = Dropout(0.5)(features)
    outputs = Dense(1, activation="sigmoid")(drop1)
    model = Model(inputs=inputs1, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def hybrid_v1(max_len, max_words, embedding_size, weights = None, X = None, filters = 250, kernel_size = 3, hidden_dims = 250):
    inputs1 = Input(shape=(max_len,))
    embedding1 = (Embedding(embedding_size, 300, input_length=max_len))(inputs1) # , weights=[weights]))
    conv1 = Conv1D(256, 3, padding='valid', activation='relu', strides=1,
                     input_shape=(max_len, embedding_size),)(embedding1)
    drop1 = Dropout(0.5)(conv1)
    max_pool1 = GlobalMaxPooling1D()(drop1)
    flatten1 = Flatten()(max_pool1)
    inputs2 = Input(shape=(max_len,))
    embedding2 = Embedding(max_words, 300, input_length=max_len)(inputs2)
    gru1 = Bidirectional(GRU(256,
                                dropout=0.3,
                                recurrent_dropout=0.3,
                                ))(embedding2)
    merged = concatenate([flatten1, gru1])
    dense1 = Dense(256, activation="relu")(merged)
    drop2 = Dropout(0.5)(dense1)
    output = Dense(1, activation="sigmoid")(drop2)
    model = Model(inputs=[inputs1, inputs2], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

def bert_v1(max_len, max_words, embedding_size, weights = None, X = None, filters = 250, kernel_size = 3, hidden_dims = 250):
    from transformers import AutoTokenizer, TFAutoModel
    import tensorflow_addons as tfa  # Adam with weight decay
    # optimizer = tfa.optimizers.AdamW(0.005, learning_rate=0.01)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    bert = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    tokenizer.encode_plus("sentence", max_length=max_len, truncation=True,
                          pad_to_max_length=True, add_special_tokens=True,
                          return_attention_mask=True, return_token_type_ids=False,
                          return_tensors='tf')
    input_ids = tf.keras.layers.Input(shape=(max_len,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(max_len,), name='attention_mask', dtype='int32')
    embeddings = bert(input_ids, attention_mask=mask)[0]  # we only keep tensor 0 (last_hidden_state)
    X = GlobalAveragePooling1D()(embeddings)  # reduce tensor dimensionality
    X = BatchNormalization()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.1)(X)
    y = Dense(1, activation='softmax', name='outputs')(X)
    model = Model(inputs=[input_ids, mask], outputs=y)
    model.layers[2].trainable = False
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    return model

def bert_v2():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model
