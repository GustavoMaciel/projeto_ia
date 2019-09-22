import trainer_lib as lib
import settings

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
import numpy as np

known_languages = ["Java", "Python"]

vocabulary, labels = lib.vocabulary_label_builder(settings.training_data_dir, settings.minimal_amount_of_repetitions)
tokenized_vocabulary = lib.tokenized_vocabulary_builder(vocabulary)

test_voc, test_labels = lib.vocabulary_label_builder(settings.testing_data_dir, settings.minimal_amount_of_repetitions)


lib.save_tokenized_vocabulary(settings.tokenized_vocabulary_location, tokenized_vocabulary)

# model_max_features = 10000
# model = lib.model_builder(tokenized_vocabulary, vocabulary, labels, tokenized_test_voc, test_voc, test_labels,
                       # model_max_features)


tokens = tokenized_vocabulary.texts_to_sequences(vocabulary)
input_dim = lib.get_input_dim(tokens)

X = pad_sequences(tokens, input_dim)

labels = keras.utils.to_categorical(labels, num_classes=2)
Y = np.asarray(labels)

#print(f"{len(tokens)}  {len(tokens)}")
#print(tokens[40])
#print(labels[40])


test_tokens = tokenized_vocabulary.texts_to_sequences(test_voc)

X_test = pad_sequences(test_tokens, input_dim)

test_labels = keras.utils.to_categorical(test_labels, num_classes=2)
Y_test = np.asarray(test_labels)
#print(f"{len(test_tokens)} {len(test_labels)}")
#print(test_tokens[0])
#print(test_labels[0])


# BUILD MODEL

input_dim = lib.get_input_dim(tokens)

model = Sequential()

# model.add(keras.layers.Embedding(input_dim, 16))
# model.add(keras.layers.GlobalAveragePooling1D())

# Dense é uma camada, primeiro parametro é numero de neuronios, input_dim é a qtd de entradas dos neuronios
model.add(Dense(12, activation='relu', input_dim=input_dim))
model.add(Dense(12, activation='relu'))
# Ultima camada é o número de classificações que queremos, softmax é pra considerar cada classificação de forma separada
model.add(Dense(2, activation='softmax'))

# Definir que será usado a descida gradiente
sgd = SGD(lr=0.01)

# Optimizador é o descida gradiente, varias classes com softmax a medição de erro melhor é categorical_crossentropy
# metrics é medir a acurácia, taxa de acerto, do model
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Epochs são o número de repetições pra cada exemplo
# Batch é o número de iterações que precisam ser feitos antes de atualizar o gradiente
model.fit(X, Y, epochs=500, batch_size=5)
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=2)

model.save(settings.model_file_location)
model.save_weights(settings.model_weights_file_location)

print(f"\nInput dim: {input_dim}")
print(model.metrics_names)
print('Loss: %f' % score)
print('Accuracy: %f' % acc)

