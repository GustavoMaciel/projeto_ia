import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import settings
import trainer_lib as lib


def evaluate_code(code, _model, tok_voc):
    tokens = lib.load_tokens(code)
    predict = tok_voc.texts_to_sequences(tokens)

    tokens = [tok[0] for tok in predict if len(tok) > 0]
    predict = pad_sequences([tokens], 79)

    return _model.predict(predict)


with open(settings.model_file_location) as file:
    model_json = file.read()


model = model_from_json(model_json)
model.load_weights(settings.model_weights_file_location)

tokenized_vocabulary = lib.load_tokenized_vocabulary(settings.tokenized_vocabulary_location)


python = """
class DeclarativeFieldsMetaclass(MediaDefiningClass):
    def __new__(mcs, name, bases, attrs):
        # Collect fields from current class.
        current_fields = []
        for key, value in list(attrs.items()):
            if isinstance(value, Field):
                current_fields.append((key, value))
                attrs.pop(key)
        attrs['declared_fields'] = dict(current_fields)

        new_class = super(DeclarativeFieldsMetaclass, mcs).__new__(mcs, name, bases, attrs)

        # Walk through the MRO.
        declared_fields = {}
        for base in reversed(new_class.__mro__):
            # Collect fields from base class.
            if hasattr(base, 'declared_fields'):
                declared_fields.update(base.declared_fields)
"""

java = """
import javax.swing.*
public class Main(){
    public static void main(String[] args){
        int i = 0 + 2
        System.out.println(i);
        System.out.println("Hello world!");
    }
}
"""


i = evaluate_code(java, model, tokenized_vocabulary)
j = evaluate_code(python, model, tokenized_vocabulary)
#i = np.argmax(i)

print(i, j)
