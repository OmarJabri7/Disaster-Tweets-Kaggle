from keras_preprocessing.sequence import pad_sequences

from audio.audio_record import record
import pyttsx3
import pickle
import tensorflow as tf
import time
import numpy as np

# emotions = {0: "angry", 1:"afraid", 2:"happy",3:"in love",4:"sad",5:"surprised"}

emotions = {0: "angry", 1:"afraid", 2:"sad"}

def converse():
    bot_name = "Jabri"
    engine = pyttsx3.init()
    while True:
        time.sleep(3)
        sentence = record()
        if sentence == "quit" or sentence == "":
            break
        vectorizer = pickle.load(open("outputs/vectorizer.sav", "rb"))
        with open('outputs/tokenizer.pickle', 'rb') as handle:
            tok = pickle.load(handle)

        # X = vectorizer.transform(sentence).toarray()

        X = tok.texts_to_sequences([sentence])

        X = pad_sequences(X, maxlen=200)

        X = X.astype(np.float)

        # loaded_model = pickle.load(open('outputs/model.sav', 'rb'))

        model = tf.keras.models.load_model('outputs/network_v0.h5')

        y_pred = model.predict(X)

        y_hat = np.argmax(y_pred, axis=1)

        # result = "You are in danger" if y_pred == 1 else "You are safe"
        result = f"Why are you {emotions[max(y_hat)]}?"

        print(result)

if __name__ == "__main__":
    converse()
