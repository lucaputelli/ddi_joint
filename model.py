from keras.layers import Input, Concatenate, Bidirectional, LSTM, Dense, TimeDistributed, Embedding, Lambda, Layer, RepeatVector
from keras.optimizers import Adam
from keras.models import Model
from AttentionMechanism import AttentionL
from ChainCRF import ChainCRF
import keras.backend as K
import tensorflow as tf
from keras_multi_head import MultiHeadAttention
from constants import number_of_charachters


class EntityAwareDecodingLayer(Layer):

    def __init__(self):
        super(EntityAwareDecodingLayer, self).__init__()

    def call(self, inputs, **kwargs):
        assert len(inputs) == 3
        lstm_out = inputs[0]
        crf_argmax = inputs[1]
        label_embedding = inputs[2]
        zero = tf.constant(0, dtype='int64')
        mask = tf.cast(tf.not_equal(crf_argmax, zero), dtype='float32')
        mask = tf.expand_dims(mask, 2)
        product = tf.multiply(lstm_out, mask)
        label_product = tf.multiply(label_embedding, mask)
        lstm_sum = K.sum(product, axis=1)
        label_sum = K.sum(label_product, axis=1)
        concatenate = K.concatenate([lstm_sum, label_sum], axis=1)
        return concatenate


class MyRepeatVector(Layer):

    def __init__(self, n, output_dim, **kwargs):
        super(MyRepeatVector, self).__init__(**kwargs)
        self.n = n
        self.output_dim = output_dim

    def call(self, inputs, **kwargs):
        return K.repeat(inputs, self.n)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n, self.output_dim


def neural_network(input_length: int, lstm_units: int, dropout: float, recurrent_dropout: float,
                   pos_tag: bool, offset: bool):
    input1 = Input(shape=(input_length, 200))
    lstm_input = input1
    complete_input = input1
    if pos_tag:
        pos_input1 = Input(shape=(input_length, 20))
        if offset:
            d1_input = Input(shape=(input_length, 1))
            d2_input = Input(shape=(input_length, 1))
            lstm_input = Concatenate()([input1, pos_input1, d1_input, d2_input])
            complete_input = [input1, pos_input1, d1_input, d2_input]
        else:
            lstm_input = Concatenate()([input1, pos_input1])
            complete_input = [input1, pos_input1]
    seq_sentence = Bidirectional(LSTM(lstm_units,
                                      dropout=dropout, return_sequences=True, return_state=False,
                                      recurrent_dropout=recurrent_dropout))(lstm_input)
    sentence_out = AttentionL(input_length)(seq_sentence)
    main_output = Dense(5, activation='softmax', name='main_output')(sentence_out)
    model = Model(inputs=complete_input, outputs=[main_output])
    algorithm = Adam(lr=0.0001, decay=0, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy',
                  optimizer=algorithm,
                  metrics=['accuracy'])
    model.summary()
    return model


def character_network(lstm_layers: int, lstm_units: int, char_lstm_units: int, value_first: int, value_second: int, label_embedding_dim: int,
                      max_length, max_word_length, pos_tag: bool, character: bool,
                      attention, custom_layer: bool) -> Model:
    word_input = Input(shape=(max_length, 200), name='word_input')
    input_list = [word_input]
    lstm_list = [word_input]
    if pos_tag:
        pos_input = Input(shape=(max_length, 20), name='pos_input')
        input_list += [pos_input]
        lstm_list += [pos_input]
    if character:
        char_input = Input(shape=(max_length, max_word_length), name='char_input')
        char_embedding = TimeDistributed(Embedding(input_dim=number_of_charachters, output_dim=25))(char_input)
        char_lstm = TimeDistributed(Bidirectional(LSTM(char_lstm_units, return_sequences=False)))(char_embedding)
        lstm_list = input_list + [char_lstm]
        input_list += [char_input]
    if len(input_list) == 1:
        lstm_input = word_input
    else:
        lstm_input = Concatenate()(lstm_list)
    word_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True,
                                       dropout=0.2, recurrent_dropout=0.2))(lstm_input)
    for i in range(lstm_layers-1):
            word_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True,
                                           dropout=0.2, recurrent_dropout=0.2))(word_lstm)
    if attention:
        attention = MultiHeadAttention(head_num=8)(word_lstm)
        dense_first = TimeDistributed(Dense(value_first, activation=None))(attention)
    else:
        dense_first = TimeDistributed(Dense(value_first, activation=None))(word_lstm)
    crf_layer = ChainCRF(name='first_crf')
    first_output = crf_layer(dense_first)
    argmax = Lambda(lambda x: K.argmax(x))(first_output)
    label_embedding = Embedding(input_dim=value_first+1, output_dim=label_embedding_dim, trainable=True)(argmax)
        # print(entity_aware_matrix.shape)
    final_input = Concatenate(axis=2)([word_lstm, label_embedding])
    # print(second_input.shape)
    if custom_layer:
        entity_aware = EntityAwareDecodingLayer()([word_lstm, argmax, label_embedding])
        entity_aware_matrix = MyRepeatVector(max_length, 2*lstm_units+label_embedding_dim)(entity_aware)
        final_input = Concatenate(axis=2)([final_input, entity_aware_matrix])
    # print(final_input.shape)
    dense_second = TimeDistributed(Dense(value_second, activation=None))(final_input)
    second_crf = ChainCRF(name='second_crf')
    second_output = second_crf(dense_second)
    model = Model(inputs=input_list, outputs=[first_output, second_output])
    algorithm = Adam(lr=0.0001, decay=0, beta_1=0.9, beta_2=0.999)
    losses = {
        "first_crf": crf_layer.loss,
        "second_crf": second_crf.loss,
    }
    model.compile(loss=losses,
                  optimizer=algorithm,
                  metrics=['accuracy'])
    model.summary()
    return model
