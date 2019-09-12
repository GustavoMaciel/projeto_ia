import trainer_lib as lib
import settings

python = """
def python(re):
    return re
"""

java = """
public class Main(){
    public static void main(String[] args){
        System.out.println("Hello world!");
    }
}
"""

known_languages = ["Java", "Python"]

vocabulary, labels = lib.vocabulary_label_builder(settings.training_data_dir, settings.minimal_amount_of_repetitions)

tokenized_vocabulary = lib.tokenized_vocabulary_builder(vocabulary)
dictionary = tokenized_vocabulary.word_index
# lib.save_dictionary(settings.dictionary_location, dictionary)
# lib.save_tokenized_vocabulary(settings.tokenized_vocabulary_location, tokenized_vocabulary)

word2vec_vocabulary = lib.word2vec_builder(settings.training_data_dir, tokenized_vocabulary)

print(labels)

