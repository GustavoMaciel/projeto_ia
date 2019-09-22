import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import settings
import trainer_lib as lib


def evaluate_code(code, _model, tok_voc):
    tokens = lib.load_tokens(code)
    predict = tok_voc.texts_to_sequences(tokens)
    return _model.predict(predict)


with open(settings.model_file_location, encoding='latin1') as file:
    model_json = file.read()


model = model_from_json(settings.model_file_location)
model.load_weights(settings.model_weights_file_location)

tokenized_vocabulary = lib.load_tokenized_vocabulary(settings.tokenized_vocabulary_location)


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

print(evaluate_code(java, model, tokenized_vocabulary))

