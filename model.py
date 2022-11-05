import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import text
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import tensorflow
from sklearn.svm import SVC
from gensim.models.fasttext import FastText
from nltk.tokenize import word_tokenize


def get_fasttext_vectors(corpus, model):
    def get_vector(sentence):
        word_tokens = word_tokenize(sentence)
        vector = 0
        for w in word_tokens:
            try:
                vector += model.wv[w]
            except:
                continue
        return vector

    vectorized = corpus.apply(lambda x: get_vector(x)).to_numpy()

    target = vectorized[0].shape[0]
    mismatches = {}
    for i in range(len(vectorized)):
        try:
            if vectorized[i].shape[0] != target:
                mismatches[i] = vectorized[i].shape[0]
        except Exception as e:
            mismatches[i] = str(e)
            continue
    print(mismatches)

    for i in mismatches.keys():
        vectorized = np.delete(vectorized, i)

    out_corpus = np.stack(vectorized)

    return out_corpus


def fast_text_svm(train, test, target):
    fasttext_train_samples = train['str'].to_list()
    fasttext_train_samples = [x.split(' ') for x in fasttext_train_samples]

    fasttext_test_samples = test['str'].to_list()
    fasttext_test_samples = [x.split(' ') for x in fasttext_test_samples]

    fasttext_sentences = fasttext_train_samples + fasttext_test_samples

    our_FastText = FastText(size=300)
    our_FastText.build_vocab(sentences=fasttext_sentences)
    our_FastText.train(
        sentences=fasttext_sentences, epochs=our_FastText.epochs,
        total_examples=our_FastText.corpus_count,
        total_words=our_FastText.corpus_total_words
    )

    own_ft_train_vectors = get_fasttext_vectors(train['str'],
                                                our_FastText)

    own_ft_test_vectors = get_fasttext_vectors(test['str'],
                                               our_FastText)

    X_train, X_test, y_train, y_test = train_test_split(
        own_ft_train_vectors,
        target,
        test_size=0.25,
        random_state=42)

    clf = SVC()
    clf.fit(X_train, y_train)

    acc_train = np.mean(clf.predict(X_train) == y_train)
    print(f"Train Accuracy {acc_train}")
    acc_test = np.mean(clf.predict(X_test) == y_test)
    print(f"Test Accuracy {acc_test}")


def lstm_model(train, test, target):
    x_train, x_test, y_train, y_test = train_test_split(train.str.values, target,
                                                        stratify=target,
                                                        random_state=42,
                                                        test_size=0.1, shuffle=True)

    tokenizer = text.Tokenizer(num_words=None)
    max_len = 70
    tokenizer.fit_on_texts(list(x_train) + list(x_test))
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    x_train_pad = pad_sequences(x_train_seq, maxlen=max_len)
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)
    word_index = tokenizer.word_index

    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        300,
                        input_length=max_len,
                        trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    y_train_enc = np_utils.to_categorical(y_train)
    y_test_enc = np_utils.to_categorical(y_test)

    es = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.01, patience=5, verbose=0, mode='min',
        baseline=None, restore_best_weights=True
    )
    callbacks = [es]

    model.fit(x_train_pad, y_train_enc, batch_size=512, epochs=40, verbose=1,
              validation_data=(x_test_pad, y_test_enc), callbacks=callbacks)

    validate_seq = tokenizer.texts_to_sequences(test.str.values)
    validate_pad = pad_sequences(validate_seq, maxlen=max_len)
    predictions = model.predict(validate_pad)
    predictions = predictions.argmax(axis=1)
    test['Predicted'] = predictions
    frame = test.drop(['Body', 'str', "code_available"], axis=1)
    frame.to_csv('test.csv')
