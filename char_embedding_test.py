from pre_processing_lib import *
from model import character_network
from relation_format_extraction import double_format, get_tokenized_sentences
from constants import word_model, tag_model
from data_model import SequencePair
import os
from post_processing import joint_plot
from random import randint


def generative_missing_labels(missing: List):
    labels = list()
    for id1, id2, class_val in missing:
        if class_val == 'unrelated':
            labels.append(0)
        if class_val == 'effect':
            labels.append(1)
        if class_val == 'mechanism':
            labels.append(2)
        if class_val == 'advise':
            labels.append(3)
        if class_val == 'int':
            labels.append(4)
    labels_array = np.asarray(labels, dtype='int32')
    return labels_array


def matrix_composition(doc_list: List[Doc], max_length: int):
    word_matrix = np.zeros((len(doc_list), max_length, word_model.vector_size))
    pos_matrix = np.zeros((len(doc_list), max_length, tag_model.vector_size))
    for i, sent in enumerate(doc_list):
        for j in range(len(sent)):
            try:
                word_matrix[i, j, :] = word_model.wv[sent[j].text.lower()]
                pos_matrix[i, j, :] = tag_model.wv[sent[j].pos_]
            except KeyError:
                pass
    return word_matrix, pos_matrix


def generate_predictions(model, test_set: List[np.array], test_labels, test_negative):
    predictions = model.predict(test_set)
    numeric_predictions = np.argmax(predictions, axis=1)
    numeric_labels = np.argmax(test_labels, axis=1)

    numeric_labels = np.concatenate((numeric_labels, test_negative))
    numeric_predictions = np.concatenate((numeric_predictions, np.zeros(len(test_negative), dtype=np.int64)))
    return numeric_labels, numeric_predictions


def char_matrix_composition(sents: List[Doc], max_length: int, max_word_length: int) -> np.ndarray:
    char_dict = get_character_dictionary()
    char_matrix = np.zeros(shape=(len(sents), max_length, max_word_length), dtype='int32')
    for i in range(len(sents)):
        sent = sents[i]
        for j in range(len(sent)):
            t = sent[j]
            token_text = t.text
            for k in range(len(token_text)):
                s = token_text[k]
                if s not in char_dict.keys():
                    continue
                char_matrix[i][j][k] = char_dict.get(s)
    return char_matrix


def generate_second_labels_dict(train_pairs: List[SequencePair]) -> dict:
    labels_dict = dict()
    index = 0
    train_labels = [s.second_labels for s in train_pairs]
    for i in range(len(train_labels)):
        for j in range(len(train_labels[i])):
            label = train_labels[i][j]
            if label not in labels_dict.keys():
                labels_dict.__setitem__(label, index)
                index += 1
    return labels_dict


def first_label_vectors(labels: List[List[str]], max_length: int) -> np.ndarray:
    label_matrix = np.zeros(shape=(len(labels), max_length, 3), dtype='int32')
    for i in range(len(labels)):
        for j in range(max_length):
            if j >= len(labels[i]):
                label_matrix[i][j] = np.array([1, 0, 0])
            else:
                label = labels[i][j]
                if label == 'O' or j >= len(labels[i]):
                    label_matrix[i][j] = np.array([1, 0, 0])
                if label.startswith('B'):
                    label_matrix[i][j] = np.array([0, 1, 0])
                if label.startswith('I'):
                    label_matrix[i][j] = np.array([0, 0, 1])
    return label_matrix


def second_label_vectors(labels: List[List[str]], max_length: int, labels_dict: dict) -> np.ndarray:
    label_matrix = np.zeros(shape=(len(labels), max_length, len(labels_dict)), dtype='int32')
    for i in range(len(labels)):
        for j in range(max_length):
            label_vector = np.zeros(shape=len(labels_dict), dtype='int32')
            if j >= len(labels[i]):
                index = labels_dict.get('N')
                label_vector[index] = 1
            else:
                label = labels[i][j]
                index = labels_dict.get(label)
                label_vector[index] = 1
            label_matrix[i][j] = label_vector
    return label_matrix


def get_dataset(pairs: List[SequencePair], max_length, max_char_length, labels_dict):
    first_labels = [p.first_labels for p in pairs]
    out_1 = first_label_vectors(first_labels, max_length)
    second_labels = [p.second_labels for p in pairs]
    out_2 = second_label_vectors(second_labels, max_length, labels_dict)
    docs = [p.doc for p in pairs]
    word_input, pos_input = matrix_composition(docs, max_length)
    char_input = char_matrix_composition(docs, max_length, max_char_length)
    return word_input, pos_input, char_input, out_1, out_2


# Pre-processing
char_dict = get_character_dictionary()
train_pairs = double_format(test=False)
values_first = 3
test_pairs = double_format(test=True)
labels_dict = generate_second_labels_dict(test_pairs)
values_second = len(labels_dict.keys())
lengths = [len(p) for p in train_pairs+test_pairs]
max_length = max(lengths)
char_lenghts = [len(t) for p in train_pairs+test_pairs for t in p.word_list]
char_max = max(char_lenghts)
word, pos, char, o1, o2 = get_dataset(train_pairs, max_length, char_max, labels_dict)
t_word, t_pos, t_char, t_o1, t_o2 = get_dataset(test_pairs, max_length, char_max, labels_dict)
lstm_layers = [1, 2, 3]
lstm_dimensions = [48, 72, 96, 120, 144, 168, 200, 224, 248, 272, 296]
char_lstm_dimensions = [5, 10, 15, 20, 25]
character_bool = [True, False]
attention_bool = [True, False]
custom_layer_bool = [True, False]
date_path = '2020_03_06'
if not os.path.exists(date_path):
    os.mkdir(date_path)
for i in range(20):
    layers = lstm_layers[randint(0, len(lstm_layers)-1)]
    lstm_dim = lstm_dimensions[randint(0, len(lstm_dimensions)-1)]
    char_lstm = char_lstm_dimensions[randint(0, len(char_lstm_dimensions)-1)]
    character = character_bool[randint(0, 1)]
    attention = attention_bool[randint(0, 1)]
    custom = custom_layer_bool[randint(0, 1)]
    path = date_path + '/L{}_D{}'.format(layers, lstm_dim)
    model = character_network(layers, lstm_dim, char_lstm, values_first, values_second, 25, max_length, char_max,
                              True, character, attention, custom)
    input = [word, pos]
    if character:
        input += [char]
        path += '_C{}_char'.format(char_lstm)
    if attention:
        path += '_att'
    if custom:
        path += '_custom'
    if not os.path.exists(path):
        os.mkdir(path)
    history = model.fit(x=input, y={'first_crf': o1, 'second_crf': o2}, validation_split=0.2,
                        batch_size=128, epochs=1, verbose=2)
    joint_plot(path, 'plot', history)