from keras.layers import Input, Concatenate, Bidirectional, LSTM, Dense
from keras.optimizers import Adam
from keras.models import Model
from AttentionMechanism import AttentionL


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
