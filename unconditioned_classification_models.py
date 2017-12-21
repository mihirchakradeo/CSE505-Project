"""Collection of benchmark models for classifying proof steps without
conditioning on the conjecture.
"""
from keras import layers
from keras.models import Model


def cnn_2x(voc_size, max_len, dropout=0.5):
    """One branch embedding a statement. Binary classifier on top.
    Args:
      voc_size: size of the vocabulary for the input statements.
      max_len: maximum length for the input statements.
    Returns:
      A Keras model instance.
    """
    statement_input = layers.Input(shape=(max_len,), dtype='int32')

    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(statement_input)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    embedded_statement = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(256, activation='relu')(embedded_statement)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model(statement_input, prediction)
    return model


def cnn_2x_lstm(voc_size, max_len, dropout=0.5):
    """One branch embedding a statement. Binary classifier on top.
    Args:
      voc_size: size of the vocabulary for the input statements.
      max_len: maximum length for the input statements.
    Returns:
      A Keras model instance.
    """
    statement_input = layers.Input(shape=(max_len,), dtype='int32')
    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(statement_input)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    embedded_statement = layers.LSTM(256)(x)

    x = layers.Dense(256, activation='relu')(embedded_statement)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model(statement_input, prediction)
    return model


def embedding_logreg(voc_size, max_len, dropout=0.5):
    """One branch embedding a statement. Binary classifier on top.
    Args:
      voc_size: size of the vocabulary for the input statements.
      max_len: maximum length for the input statements.
    Returns:
      A Keras model instance.
    """
    statement_input = layers.Input(shape=(max_len,), dtype='int32')

    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(statement_input)
    x = layers.Flatten()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model(statement_input, prediction)
    return model


#contribution
def cnn_2x_rnn(voc_size, max_len, dropout=0.5):
    statement_input = layers.Input(shape=(max_len,), dtype='int32')
    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(statement_input)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    embedded_statement = layers.SimpleRNN(256)(x)

    x = layers.Dense(256, activation='relu')(embedded_statement)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model(statement_input, prediction)
    return model

def cnn_2x_gru(voc_size, max_len, dropout=0.5):
    statement_input = layers.Input(shape=(max_len,), dtype='int32')
    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(statement_input)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    embedded_statement = layers.GRU(256)(x)

    x = layers.Dense(256, activation='tanh', inner_activation='hard_sigmoid')(embedded_statement)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model(statement_input, prediction)
    return model

def cnn_2x_encdec(voc_size, max_len, dropout=0.5):
    statement_input = layers.Input(shape=(max_len,), dtype='int32')
    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(statement_input)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.GRU(256)(x) #this acts as the encoder
    x = layers.RepeatVector(256)(x) #take the last output of GRU and feed it to the decoder
    embedded_statement = layers.GRU(256)(x) #this acts as the decoder

    x = layers.Dense(256, activation='tanh')(embedded_statement)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model(statement_input, prediction)
    return model

# Contains both the model defition function and the type of encoding needed.
MODELS = {
    'cnn_2x': (cnn_2x, 'integer'),
    'cnn_2x_lstm': (cnn_2x_lstm, 'integer'),
    'embedding_logreg': (embedding_logreg, 'integer'),
    'cnn_2x_rnn': (cnn_2x_rnn, 'integer'),
    'cnn_2x_gru': (cnn_2x_rnn, 'integer'),
    'cnn_2x_encdec': (cnn_2x_encdec, 'integer')
}
