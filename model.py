from keras.layers import Input, Concatenate, Bidirectional, LSTM, Dense, TimeDistributed, Embedding
from keras.optimizers import Adam
from keras.models import Model
from AttentionMechanism import AttentionL
from ChainCRF import ChainCRF


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


def character_network(lstm_units: int, value_first: int, value_second: int,
                      max_length, max_word_length, pos_tag: bool, character: bool) -> Model:
    word_input = Input(shape=(max_length, 200), name='word_input')
    input_list = [word_input]
    lstm_list = []
    if pos_tag:
        pos_input = Input(shape=(max_length, 20), name='pos_input')
        input_list += [pos_input]
    if character:
        char_input = Input(shape=(max_length, max_word_length), name='char_input')
        input_list += [char_input]
        char_embedding = TimeDistributed(Embedding(input_dim=max_word_length, output_dim=25))(char_input)
        char_lstm = TimeDistributed(Bidirectional(LSTM(25)))(char_embedding)
        lstm_list = input_list + [char_lstm]
    if len(input_list) == 1:
        lstm_input = word_input
    else:
        lstm_input = Concatenate()(lstm_list)
    word_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_input)
    dense_first = TimeDistributed(Dense(value_first, activation=None))(word_lstm)
    crf_layer = ChainCRF(name='first_crf')
    first_output = crf_layer(dense_first)
    second_input = Concatenate(axis=2)([word_lstm, first_output])
    dense_second = TimeDistributed(Dense(value_second, activation=None))(second_input)
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
