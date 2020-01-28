import datetime
import pickle
from os import mkdir
from os.path import exists
from gensim.models import Word2Vec
from pre_processing_lib import *
from post_processing import plot, error_analysis, metrics
from model import neural_network
from relation_format_extraction import instances_from_prediction, joint_negative_filtering, joint_labelled_instances, JointInstance


def matrix_composition(doc_list):
    word_matrix = np.zeros((len(doc_list), dim, 200))
    pos_matrix = np.zeros((len(doc_list), dim, tag_model.vector_size))
    d1_matrix = np.zeros((len(doc_list), dim, 1))
    d2_matrix = np.zeros((len(doc_list), dim, 1))
    for i, sent in enumerate(doc_list):
        index1 = -1
        index2 = -1
        for j in range(len(sent)):
            if sent[j].text == 'PairDrug1':
                index1 = j
            if sent[j].text == 'PairDrug2':
                index2 = j
        for j in range(len(sent)):
            try:
                word_matrix[i, j, :] = word_model.wv[sent[j].text]
                pos_matrix[i, j, :] = tag_model.wv[sent[j].pos_]
                d1_matrix[i, j, :] = (j - index1) / len(sent)
                d2_matrix[i, j, :] = (j - index2) / len(sent)
            except KeyError:
                pass
    return word_matrix, pos_matrix, d1_matrix, d2_matrix


def generate_negative_labels(negatives: List[JointInstance]):
    length = len(negatives)
    matrix = np.zeros((length, 5))
    for i in range(length):
        matrix[i] = np.array([1, 0, 0, 0, 0])
    return matrix

def prediction(model, test_set: List[np.array], test_labels, test_negative):
    predictions = model.predict(test_set)
    numeric_predictions = np.argmax(predictions, axis=1)
    numeric_labels = np.argmax(test_labels, axis=1)

    numeric_labels = np.concatenate((numeric_labels, np.argmax(test_negative, axis=1)))
    numeric_predictions = np.concatenate((numeric_predictions, np.zeros(len(test_negative), dtype=np.int64)))

    # Metrics
    matrix, report, overall_precision, overall_recall, overall_f_score = metrics(numeric_labels,
                                                                                 numeric_predictions)
    f = open(combination_folder + '/metrics.txt', 'w')
    text = 'Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nOverall precision\n\n{}' \
           + '\n\nOverall recall\n\n{}\n\nOverall F-score\n\n{}\n'
    f.write(text.format(report, matrix, overall_precision, overall_recall, overall_f_score))
    f.close()

    # Model to JSON
    model_json = model.to_json()
    with open(combination_folder + '/model.json', "w") as json_file:
        json_file.write(model_json)

    # Model pickle
    with open(combination_folder + '/metrics.pickle', 'wb') as pickle_file:
        pickle.dump([matrix, report, overall_precision, overall_recall, overall_f_score], pickle_file)


# Pre-processing
sents = get_sentences('Train/Sample')
instances = get_instances(sents)
instances = [x for x in instances if x is not None]
instances = negative_filtering(instances)
instances = [x for x in instances if x.get_dependency_path() is not None and len(x.get_dependency_path()) > 0]
t_right, t_approximate, t_wrong, t_missing = instances_from_prediction()
right_selected, right_negative = joint_negative_filtering(t_right)
approximate_selected, approximate_negative = joint_negative_filtering(t_approximate)
wrong_selected, wrong_negative = joint_negative_filtering(t_wrong)
right_selected = [x for x in right_selected if x.dependency_path is not None and len(x.dependency_path) > 0]
approximate_selected = [x for x in approximate_selected if x.dependency_path and len(x.dependency_path) > 0]
wrong_selected = [x for x in wrong_selected if x.dependency_path and len(x.dependency_path) > 0]

sents, Y_train = get_labelled_instances(instances)
right_sents, right_labels = joint_labelled_instances(right_selected)
approximate_sents, approximate_labels = joint_labelled_instances(approximate_selected)
wrong_sents, wrong_labels = joint_labelled_instances(wrong_selected)
# sents, labels = get_labelled_instances(instances)
word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
tag_model = Word2Vec.load('ddi_pos_embedding.model')
lengths = [len(x) for x in sents+t_right+t_approximate+t_wrong]
dim = max(lengths)

X_word, X_pos, X_d1, X_d2 = matrix_composition(sents)
R_word, R_pos, R_d1, R_d2 = matrix_composition(right_sents)
A_word, A_pos, A_d1, A_d2 = matrix_composition(approximate_sents)
W_word, W_pos, W_d1, W_d2 = matrix_composition(wrong_sents)

folder = 'prova'
if not exists(folder):
    mkdir(folder)
for i in range(5):
    lstm_units = np.random.randint(6, 13)*10
    dropout = np.random.rand() * 0.2 + 0.3
    r_dropout = np.random.rand() * 0.2 + 0.4
    batch_size = 128
    epochs = 5
    name = "LSTM_%d_DROP_%.2f_RDROP_%.2f" % (lstm_units, dropout, r_dropout)
    parameters_folder = folder+'/'+name
    if not exists(parameters_folder):
        mkdir(parameters_folder)
    combinations = [(False, False), (True, False), (True, True)]
    for pos_tag, offset in combinations:
        combination_name = 'word'
        if pos_tag:
            combination_name += '_pos'
        if offset:
            combination_name += '_offset'
        combination_folder = parameters_folder + '/' + combination_name
        if not exists(combination_folder):
            mkdir(combination_folder)
        training_set = [X_word]
        right_set = [R_word]
        approximate_set = [A_word]
        wrong_set = [W_word]
        if pos_tag:
            training_set += [X_pos]
            right_set += [R_pos]
            approximate_set += [A_pos]
            wrong_set += [W_pos]
        if offset:
            training_set += [X_d1, X_d2]
            right_set += [R_d1, R_d2]
            approximate_set += [A_d1, A_d2]
            wrong_set += [W_d1, W_d2]
        model = neural_network(dim, lstm_units, dropout, r_dropout,
                               pos_tag, offset)
        history = model.fit(training_set, Y_train,
                            validation_split=0.2,
                            batch_size=batch_size,
                            epochs=epochs, verbose=2)
        plot(combination_folder, 'loss_accuracy_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), history)
        prediction(model, right_set, right_labels, generate_negative_labels(right_negative))
        prediction(model, approximate_set, approximate_labels, generate_negative_labels(approximate_negative))
        prediction(model, wrong_set, wrong_labels, generate_negative_labels(wrong_negative))