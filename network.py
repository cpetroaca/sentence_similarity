import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def create_network(n_hidden, batch_size, n_epoch, max_seq_length, embeddings, embedding_dim):
    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')
    
    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)
    
    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)
    
    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = LSTM(n_hidden)
    
    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)
    
    # Calculates the distance as defined by the MaLSTM model
    malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    
    # Pack it all up into a model
    malstm = Model([left_input, right_input], [malstm_distance])
    
    return malstm;