import numpy as np
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import History


def plot(folder: str, name: str, history: History):
    pyplot.clf()
    pyplot.figure(1, figsize=(13, 6))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.subplot(1, 2, 2)
    try:
        pyplot.plot(history.history['accuracy'])
    except KeyError:
        pyplot.plot(history.history['acc'])
    try:
        pyplot.plot(history.history['val_accuracy'])
    except KeyError:
        pyplot.plot(history.history['val_acc'])
    pyplot.title('model train vs validation accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.tight_layout()
    pyplot.savefig(folder + '/' + name + '.png')


def joint_plot(folder: str, name: str, history: History):
    pyplot.clf()
    fig, axs = pyplot.subplots(2, 2)
    axs[0, 0].plot(history.history['first_crf_loss'])
    axs[0, 0].plot(history.history['val_first_crf_loss'])
    # axs[0, 0].set_title('first loss')
    axs[0, 1].plot(history.history['first_crf_acc'])
    axs[0, 1].plot(history.history['val_first_crf_acc'])
    # axs[0, 1].set_title('first acc')
    axs[1, 0].plot(history.history['second_crf_loss'])
    axs[1, 0].plot(history.history['val_second_crf_loss'])
    # axs[1, 0].set_title('second loss')
    axs[1, 1].plot(history.history['second_crf_acc'])
    axs[1, 1].plot(history.history['val_second_crf_acc'])
    # axs[1, 1].set_title('second acc')
    pyplot.savefig(folder + '/' + name + '.png')


def metrics(numeric_labels, t_predictions):
    target_names = ['unrelated', 'effect', 'mechanism', 'advise', 'int']
    matrix = confusion_matrix(numeric_labels, t_predictions)
    FP = (matrix.sum(axis=0) - np.diag(matrix))[1:]
    FN = (matrix.sum(axis=1) - np.diag(matrix))[1:]
    TP = (np.diag(matrix))[1:]
    overall_fp = np.sum(FP)
    overall_fn = np.sum(FN)
    overall_tp = np.sum(TP)
    overall_precision = overall_tp / (overall_tp + overall_fp)
    overall_recall = overall_tp / (overall_tp + overall_fn)
    overall_f_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
    report = classification_report(numeric_labels, t_predictions, labels=[0, 1, 2, 3, 4], target_names=target_names)
    return matrix, report, overall_precision, overall_recall, overall_f_score


def error_analysis(folder, t_selected_sents, t_negative_sents, numeric_labels, numeric_predictions):
    # Maximum length
    lengths = [len(x) for x in t_selected_sents + t_negative_sents]
    max_length = max(lengths)

    # Error plots
    errors = np.zeros(max_length)
    errors0 = np.zeros(max_length)
    errors1 = np.zeros(max_length)
    errors2 = np.zeros(max_length)
    errors3 = np.zeros(max_length)
    errors4 = np.zeros(max_length)

    for i in range(max_length):
        indices = [len(sent) == i + 1 for sent in t_selected_sents + t_negative_sents]
        if len(indices) > 0:
            errors[i] = np.sum(numeric_labels[indices] != numeric_predictions[indices]) / len(indices)

        indices0 = np.logical_and(indices, [numeric_labels == 0])[0]
        if len(indices0) > 0:
            errors0[i] = np.sum(numeric_labels[indices0] != numeric_predictions[indices0]) / len(indices0)

        indices1 = np.logical_and(indices, [numeric_labels == 1])[0]
        if len(indices1) > 0:
            errors1[i] = np.sum(numeric_labels[indices1] != numeric_predictions[indices1]) / len(indices1)

        indices2 = np.logical_and(indices, [numeric_labels == 2])[0]
        if len(indices2) > 0:
            errors2[i] = np.sum(numeric_labels[indices2] != numeric_predictions[indices2]) / len(indices2)

        indices3 = np.logical_and(indices, [numeric_labels == 3])[0]
        if len(indices3) > 0:
            errors3[i] = np.sum(numeric_labels[indices3] != numeric_predictions[indices3]) / len(indices3)

        indices4 = np.logical_and(indices, [numeric_labels == 4])[0]
        if len(indices4) > 0:
            errors4[i] = np.sum(numeric_labels[indices4] != numeric_predictions[indices4]) / len(indices4)

    pyplot.clf()
    pyplot.plot(range(1, max_length + 1), errors)
    pyplot.xlabel('Sentence length')
    pyplot.ylabel('% of errors (total)')
    pyplot.savefig(folder + '/errors_total.png')

    pyplot.clf()
    pyplot.plot(range(1, max_length + 1), errors0)
    pyplot.xlabel('Sentence length')
    pyplot.ylabel('% of errors (Class 0: unrelated)')
    pyplot.savefig(folder + '/errors0.png')

    pyplot.clf()
    pyplot.plot(range(1, max_length + 1), errors1)
    pyplot.xlabel('Sentence length')
    pyplot.ylabel('% of errors (Class 1: effect)')
    pyplot.savefig(folder + '/errors1.png')

    pyplot.clf()
    pyplot.plot(range(1, max_length + 1), errors2)
    pyplot.xlabel('Sentence length')
    pyplot.ylabel('Number of errors (Class 2: mechanism)')
    pyplot.savefig(folder + '/errors2.png')

    pyplot.clf()
    pyplot.plot(range(1, max_length + 1), errors3)
    pyplot.xlabel('Sentence length')
    pyplot.ylabel('% of errors (Class 3: advice)')
    pyplot.savefig(folder + '/errors3.png')

    pyplot.clf()
    pyplot.plot(range(1, max_length + 1), errors4)
    pyplot.xlabel('Sentence length')
    pyplot.ylabel('% of errors (Class 4: int)')
    pyplot.savefig(folder + '/errors4.png')

    return errors, errors0, errors1, errors2, errors3, errors4