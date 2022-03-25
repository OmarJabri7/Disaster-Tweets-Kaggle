from keras import Input, regularizers
from keras.models import Sequential, Model
from keras.layers import Dropout, LSTM, Bidirectional, Dense, Embedding, BatchNormalization, SpatialDropout1D, GRU, \
    Activation, SimpleRNN, Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, concatenate, GlobalAveragePooling1D
import tensorflow as tf

from keras import backend as K

import tensorflow_probability as tfp
from transformers import BertModel, BertTokenizer, DistilBertTokenizerFast, DistilBertModel, \
    TFBertForSequenceClassification, TFDistilBertForSequenceClassification, TFBertModel, TFRobertaModel, TFAutoModel

import tensorflow_hub as hub

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

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="mean",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, #segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


# Build model
def build_model(max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    dense = tf.keras.layers.Dense(256, activation="relu")(bert_output)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model



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
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    bert = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    # tokenizer.encode_plus("sentence", max_length=max_len, truncation=True,
    #                       pad_to_max_length=True, add_special_tokens=True,
    #                       return_attention_mask=True, return_token_type_ids=False,
    #                       return_tensors='tf')
    input_ids = tf.keras.layers.Input(shape=(max_len,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(max_len,), name='attention_mask', dtype='int32')
    embeddings = bert(input_ids, attention_mask=mask) # we only keep tensor 0 (last_hidden_state)
    X = GlobalAveragePooling1D()(embeddings)  # reduce tensor dimensionality
    X = BatchNormalization()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.1)(X)
    y = Dense(1, activation='softmax', name='outputs')(X)
    model = Model(inputs=[input_ids, mask], outputs=y)
    model.layers[2].trainable = False
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    print(model.summary())
    return model

def bert_v2():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["acc"])
    return model

def bert_v3(transformer, max_len):
    for layer in transformer.layers:
        layer.trainable = True
        # Input layers
    input_ids_layer = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    input_attention_layer = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]
    cls_token = last_hidden_state[:, 0, :]
    # Hidden layers
    #     output = keras.layers.Dense(8,kernel_initializer=keras.initializers.GlorotUniform(seed=1),  kernel_constraint=None,
    #                                 bias_initializer='zeros',
    #                                 activation='relu')(cls_token)
    #     output = keras.layers.Dropout(0.2)(output)
    #     #output = keras.layers.Dense(8, activation = 'relu')(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(cls_token)
    # Define the model
    model = tf.keras.Model([input_ids_layer, input_attention_layer], output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    return model

def bert_v4(max_len):
    input_ids_layer = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    input_attention_layer = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    # input_seg = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_segments')
    model = TFBertModel.from_pretrained("bert-base-uncased")
    for layer in model.layers:
        layer.trainable = False
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    seq_output = model([input_ids_layer, input_attention_layer])[0]
    X = GlobalAveragePooling1D()(seq_output)  # reduce tensor dimensionality
    X = BatchNormalization()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.1)(X)
    y = Dense(2, activation='softmax', name='outputs')(X)
    model = tf.keras.Model([input_ids_layer, input_attention_layer], y)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model

def bert_v5(max_len):
    input_ids_layer = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    input_attention_layer = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    # input_seg = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_segments')
    model = TFAutoModel.from_pretrained("bert-base-uncased")
    # for param in model.roberta.parameters():
    #     param.requires_grad = False
    for layer in model.layers:
        layer.trainable = False
    # for param in model.bert.parameters():
    #     param.requires_grad = False
    seq_output = model([input_ids_layer, input_attention_layer])[0]
    # X = GlobalAveragePooling1D()(seq_output)  # reduce tensor dimensionality
    X = BatchNormalization()(seq_output)
    X = GRU(64)(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.1)(X)
    y = Dense(2, activation='softmax', name='outputs')(X)
    model = tf.keras.Model([input_ids_layer, input_attention_layer], y)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model

def bert_v6(max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    # in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask]#, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    dense = tf.keras.layers.Dense(256, activation="relu")(bert_output)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model
