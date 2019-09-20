from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from gensim.models import Word2Vec
import pandas as pd
import settings
import json
import logging
import os
import re
import pickle


def get_files_names(dir_name):
    result = []
    i = 0
    for root, sub_folder, files in os.walk(dir_name):
        i += 1
        if i == 1:
            continue

        result.extend([os.path.join(root, file) for file in files])

    return result


def load_tokens(string):
    _string = " ".join(string.splitlines())
    pattern = r"[{}()\[\]\'\":.*\s,#=_/\\><;?\-|+]"
    result = re.split(pattern, _string)
    tokens = []
    for word in result:
        if word.strip() != "" and word not in tokens:
            tokens.append(word)
    return [word for word in result if word.strip() != ""]


def load_file_tokens_or_log_error(file_name):
    try:
        with open(file_name, "r", encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        logging.warning(f"Erro ao ler o arquivo {file_name}")
        return []
    return load_tokens(content)


def get_language_from_filename(file_name):
    index = -1
    while file_name[index] not in ['\\', '/']:
        index -= 1
    language = file_name[index+1:]
    if language.lower().endswith('java'):
        language = 0
    elif language.lower().endswith('py'):
        language = 1
    return language


def vocabulary_label_builder(train_dir, min_count):
    vocabulary = []
    languages = []
    files_names = get_files_names(train_dir)
    for file_name in files_names:
        tokens = load_file_tokens_or_log_error(file_name)
        if tokens:
            languages.append(get_language_from_filename(file_name))
            vocabulary.append(tokens)

    # vocabulary = [word for word, count in vocabulary.items() if count >= min_count]
    return vocabulary, languages


def tokenized_vocabulary_builder(vocabulary):
    tokenizer = Tokenizer(lower=False, filters="")
    tokenizer.fit_on_texts(vocabulary)
    return tokenizer


def save_dictionary(location, dictionary):
    with open(location, 'w', encoding='utf-8') as dic_file:
        json.dump(dictionary, dic_file)


def save_tokenized_vocabulary(location, tok_voc):
    with open(location, 'wb', encoding='utf-8') as file:
        pickle.dump(tok_voc, file, protocol=pickle.HIGHEST_PROTOCOL)


def is_in_vocab(word, tok_voc):
    return word in tok_voc.word_counts.keys()


def word2vec_builder(training_dir, tok_voc):
    all_words = []
    files_names = get_files_names(training_dir)
    for file_name in files_names:
        words = load_file_tokens_or_log_error(file_name)
        all_words.append([word for word in words if is_in_vocab(word, tok_voc)])
    model = Word2Vec(all_words, size=settings.word2vec_dimension, window=5, workers=8, min_count=1)
    return {word: model[word] for word in model.wv.index2word}


def model_builder(tok_voc, vocabulary, labels, test_tok_voc, test_voc, test_labels, max_features):
    X = tok_voc.texts_to_sequences(vocabulary)
    X = pad_sequences(X, 100)
    Y = pd.get_dummies(labels)

    X_test = test_tok_voc.texts_to_sequences(test_voc)
    X_test = pad_sequences(X_test, 100)
    Y_test = pd.get_dummies(test_labels)

    embed_dimension = 128
    lstm_out = 64
    embedding = Embedding(max_features, embed_dimension, input_length=100)

    model = Sequential()
    model.add(embedding)
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', dilation_rate=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', dilation_rate=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(LSTM(lstm_out))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dense(len(Y.columns), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    batch_size = 10
    history = model.fit(X, Y, epochs=400, batch_size=batch_size)

    model_json = model.to_json()
    with open(settings.model_file_location, "w", encoding='utf-8') as file:
        file.write(model_json)

    # model.save(settings.model_file_location)
    model.save_weights(settings.model_weights_file_location)

    score, accuracy = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)

    print(model.metrics_names)
    print('Loss: %f' % score)
    print('Accuracy: %f' % accuracy)

    return model




