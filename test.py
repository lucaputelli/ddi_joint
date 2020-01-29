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


def generate_negative_labels(instance_list):
    length = len(instance_list)
    matrix = np.zeros(length)
    return matrix


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


def generate_predictions(model, test_set: List[np.array], test_labels, test_negative):
    predictions = model.predict(test_set)
    numeric_predictions = np.argmax(predictions, axis=1)
    numeric_labels = np.argmax(test_labels, axis=1)

    numeric_labels = np.concatenate((numeric_labels, test_negative))
    numeric_predictions = np.concatenate((numeric_predictions, np.zeros(len(test_negative), dtype=np.int64)))
    return numeric_labels, numeric_predictions


def results(labels, predictions, test_name, folder):
    # Metrics
    matrix, report, overall_precision, overall_recall, overall_f_score = metrics(labels,
                                                                                 predictions)
    f = open(folder + '/metrics_'+test_name+'.txt', 'w')
    text = 'Classification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\nOverall precision\n\n{}' \
           + '\n\nOverall recall\n\n{}\n\nOverall F-score\n\n{}\n'
    f.write(text.format(report, matrix, overall_precision, overall_recall, overall_f_score))
    f.close()
    # Model to JSON
    model_json = model.to_json()
    with open(folder + '/model_'+test_name+'.json', "w") as json_file:
        json_file.write(model_json)
    # Model pickle
    with open(folder + '/metrics_'+test_name+'.pickle', 'wb') as pickle_file:
        pickle.dump([matrix, report, overall_precision, overall_recall, overall_f_score], pickle_file)
    return matrix


# Pre-processing
sents = get_sentences('Dataset/Train/Onlytrain')
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
missing_labels = generative_missing_labels(t_missing)
# sents, labels = get_labelled_instances(instances)
word_model = Word2Vec.load('pub_med_retrained_ddi_word_embedding_200.model')
tag_model = Word2Vec.load('ddi_pos_embedding.model')
lengths = [len(x) for x in sents+t_right+t_approximate+t_wrong]
dim = max(lengths)

X_word, X_pos, X_d1, X_d2 = matrix_composition(sents)
R_word, R_pos, R_d1, R_d2 = matrix_composition(right_sents)
A_word, A_pos, A_d1, A_d2 = matrix_composition(approximate_sents)
W_word, W_pos, W_d1, W_d2 = matrix_composition(wrong_sents)

folder = '2020_01_29_complete'
if not exists(folder):
    mkdir(folder)
for i in range(10):
    lstm_units = np.random.randint(6, 13)*10
    dropout = np.random.rand() * 0.2 + 0.3
    r_dropout = np.random.rand() * 0.2 + 0.4
    batch_size = 128
    epochs = 65
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
                            validation_split=0.15,
                            batch_size=batch_size,
                            epochs=epochs, verbose=2)
        plot(combination_folder, 'loss_accuracy_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), history)
        total_right_labels, right_predictions = generate_predictions(model, right_set, right_labels,
                                                               generate_negative_labels(right_negative))
        total_approximate_labels, approximate_predictions = generate_predictions(model, approximate_set, approximate_labels,
                                                                           generate_negative_labels(
                                                                               approximate_negative))
        total_wrong_labels, wrong_predictions = generate_predictions(model, wrong_set, wrong_labels,
                                                               generate_negative_labels(wrong_negative))
        missing_predictions = generate_negative_labels(t_missing)
        complete_labels = np.concatenate([total_right_labels, total_approximate_labels, total_wrong_labels, missing_labels])
        complete_predictions = np.concatenate(
            [right_predictions, approximate_predictions, wrong_predictions, missing_predictions])
        results(complete_labels, complete_predictions, 'complete', combination_folder)