from keras.preprocessing.text import Tokenizer
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
    pattern = r"[{}()\[\]\'\":.*\s,#=_\/\\><;?\-|+]"
    result = re.split(pattern, _string)
    tokens = []
    #for word in result:
     #   if word.strip() != "" and word not in tokens:
      #      tokens.append(word)
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


def get_input_dim(vocabulary):
    index = 0
    min_index = 0
    for i in vocabulary:
        if len(i) < len(vocabulary[index]):
            min_index = index
        index += 1
    return len(vocabulary[min_index])


def tokenized_vocabulary_builder(vocabulary):
    tokenizer = Tokenizer(lower=False, filters="")
    tokenizer.fit_on_texts(vocabulary)
    return tokenizer


def save_tokenized_vocabulary(location, tok_voc):
    with open(location, 'wb', encoding='utf-8') as file:
        pickle.dump(tok_voc, file, protocol=pickle.HIGHEST_PROTOCOL)






