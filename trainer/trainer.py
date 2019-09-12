import trainer_lib as lib
import settings

known_languages = ["Java", "Python"]

vocabulary, labels = lib.vocabulary_label_builder(settings.training_data_dir, settings.minimal_amount_of_repetitions)
tokenized_vocabulary = lib.tokenized_vocabulary_builder(vocabulary)

test_voc, test_labels = lib.vocabulary_label_builder(settings.testing_data_dir, settings.minimal_amount_of_repetitions)
tokenized_test_voc = lib.tokenized_vocabulary_builder(test_voc)

dictionary = tokenized_vocabulary.word_index
test_dictionary = tokenized_test_voc.word_index

# lib.save_dictionary(settings.dictionary_location, dictionary)
# lib.save_tokenized_vocabulary(settings.tokenized_vocabulary_location, tokenized_vocabulary)

# word2vec_vocabulary = lib.word2vec_builder(settings.training_data_dir, tokenized_vocabulary)

model_max_features = 10000
model = lib.model_builder(tokenized_vocabulary, vocabulary, labels, tokenized_test_voc, test_voc, test_labels,
                          model_max_features)


