import tensorflow as tf
import keras
from keras.layers import LSTM, Dropout, Dense
from tensorflow.python.keras import activations


# Creates LSTM model
def create_lstm_model(num_nodes, dropout, unrollings):
    model = keras.models.Sequential() # The type of model. Groups a linear stack of layers onto the model

    # Adds the LSTM Hidden first layer (200 neurons). Forget Gate
    model.add(LSTM(units=num_nodes[0], # Number of LSTM Neurons in each layer. Outputs 200 values per timestep (Hidden & cell state)
                   return_sequences=True, # True means each output (hidden h) from each time step will be passed onto the next LSTM layer. (c & h) go onto the next timestep until its sequence is over then resets once a new sequence begins
                   input_shape =(unrollings, 1),
                   kernel_initializer='glorot_uniform',)) # Initializes weights. Updates after every batch
    model.add(Dropout(dropout))

    # Second LSTM Hidden layer
    model.add(LSTM(units=num_nodes[1]
                   , return_sequences=True
                   , kernel_initializer='glorot_uniform')) # Initializes weights
    model.add(Dropout(dropout))

    # Third LSTM Hidden layer
    model.add(LSTM(units=num_nodes[2]
                   , return_sequences=False # Same process as the other layers, however after going through timesteps 1-49 this only sends the last 150 values from the very last timestep to the dense layer for prediction
                   , kernel_initializer='glorot_uniform')) # Initializes weights
    model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(units=1, # Final output. one weighted sum and gives you the predicted price.
                    kernel_initializer='glorot_uniform', # Initializes weights
                    bias_initializer=keras.initializers.RandomUniform(-0.1, 0.1),
                    ))

    return model

"""
Timeline of model
"""


