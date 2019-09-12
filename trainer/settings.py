import os

current_dir = os.path.dirname(os.path.abspath(__file__))

training_data_dir = os.path.join(current_dir, "../resources/training_data")
testing_data_dir = os.path.join(current_dir, "../resources/testing_data")

vocabulary_location = os.path.join(current_dir, "../trainer/vocabulary.txt")
tokenized_vocabulary_location = os.path.join(current_dir, "../trainer/tokenized_vocabulary")
dictionary_location = os.path.join(current_dir, "../trainer/word_index.json")

word2vec_location = os.path.join(current_dir, "../trainer/word2vec.txt")
model_file_location = os.path.join(current_dir, "../resources/models/model.json")
weights_file_location = os.path.join(current_dir, "../resources/models/model.h5")

minimal_amount_of_repetitions = 5
input_length = 500
word2vec_dimension = 100
