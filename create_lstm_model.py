import tensorflow as tf
import keras
from keras.layers import LSTM, Dropout, Dense



# Creates LSTM model
def create_lstm_model(num_nodes, dropout, unrollings):
    model = keras.models.Sequential() # The type of model. Groups a linear stack of layers onto the model

    # Adds the LSTM first layer (200 neurons)
    model.add(LSTM(units=num_nodes[0], # Number of LSTM cells in each layer
                   dropout=dropout, # Dropout rate
                   return_sequences=True, # True means output each time step for stacking
                   input_shape =(unrollings, 1),
                   kernel_initializer='glorot_uniform',)) # Initializes weights
    model.add(Dropout(dropout))

    # Second LSTM layer
    model.add(LSTM(units=num_nodes[1]
                   , return_sequences=True
                   , kernel_initializer='glorot_uniform')) # Initializes weights
    model.add(Dropout(dropout))

    # Third LSTM layer
    model.add(LSTM(units=num_nodes[2]
                   , return_sequences=False # Only want the last output
                   , kernel_initializer='glorot_uniform')) # Initializes weights
    model.add(Dropout(dropout))

    model.add(Dense(units=1,
                    kernel_initializer='glorot_uniform', # Initializes weights
                    bias_initializer=keras.initializers.RandomUniform(-0.1, 0.1)
                    ))

    return model


