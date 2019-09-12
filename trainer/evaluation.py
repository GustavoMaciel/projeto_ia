import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import settings
import trainer_lib as lib


def evaluate_code(code, _model):
    tokens = lib.load_tokens(code)




with open(settings.model_file_location) as file:
    model_json = file.read()

model = model_from_json(model_json)
model.load_weights(settings.model_weights_file_location)


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


