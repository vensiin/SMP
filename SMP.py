import keras.layers
from fontTools.misc.arrayTools import scaleRect
from pandas.io.xml import preprocess_data # This is related to handling XML data with Pandas. https://pandas.pydata.org/docs/
import matplotlib.pyplot as plt # Lets you make charts and graphs. # https://matplotlib.org/stable/users/explain/quick_start.html#a-simple-example
import pandas as pd # Pandas is used for data manipulation and analysis (working with tables).
import datetime as dt # Lets you work with dates and times (e.g., parsing strings into dates). import datetime as dt # https://docs.python.org/3/library/datetime.html#
import urllib.request, json # Imports urllib.request (for making web requests, like downloading from URLs). Imports json (to handle JSON data — text data formatted like dictionaries).
import os # Lets you interact with the operating system (like checking if a file exists). https://docs.python.org/3/library/urllib.request.html#module-urllib.request
import numpy as np # A math library for arrays, vectors, and numerical computing.
import tensorflow as tf # Imports TensorFlow, a machine learning library.
import keras
from keras import layers
from keras.layers import LSTM, Dense, Dropout
from rich import color
from sklearn.preprocessing import MinMaxScaler # Imports a tool to scale (normalize) values into a range (like 0 → 1).

import LRS
from user import *
from forecast_Predictions import  *
from dataGenerator import *
from create_lstm_model import *
from multi_Step_auto_aggressive import *
from LRS import *
import os
from dotenv import load_dotenv

load_dotenv()
# 1. Obtaining the data

data_source = "kaggle".lower()

api_key = os.getenv("API_KEY")

user1 = my_user()
user1.user_enter_ticker()

ticker = "AAL"

# Conditional that checks which method we are using
if data_source == "alphavantage":


    url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={user1.return_ticker()}&outputsize=full&apikey={api_key}" # Website we are pulling data from

    file_to_save = f"stock_market_data-{user1.return_ticker()}.csv" # Name of the file we are saving the data to

    # Checks to see if the path is made
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url: # Opens the alpha advantage website API
            data = json.loads(url.read().decode()) # Loads the data in JSON

            data = data["Time Series (Daily)"] # Obtain the category key (dataset name) which you will be going through
            df = pd.DataFrame(columns=["Date", "Low", "High", "Close",  "Open"]) # Create a dataframe with said columns
            for k,v in data.items(): # Iterate through the key,value pair where k is the date and v is the key:values inside the date (It is a dictionary inside a dictionary)
                date = dt.datetime.strptime(k, "%Y-%m-%d") # Format the outer key (date)
                data_row = [date.date(), float(v['3. low']), float(v['2. high']), # Format the key, value pairs inside of date
                            float(v['4. close']), float(v['1. open'])] # Obtains the date from date by using date.date(), and all the values from the key/value pairs
                df.loc[0,:] = data_row # Sets values for the entire row for each row
                df.index = False # Assigns an index for every data row/date
            df.to_csv(file_to_save)  # Saves the data as a CSV file since it is neater than JSON
            print(f"Data saved to: {file_to_save}") # Output that we saved the file


    # Else conditional on if the file is already saved
    else:
        print("File already exists. Loading data from CSV")
        df = pd.read_csv(file_to_save)



# Kaggle method
else:
    df = pd.read_csv(os.path.join('Stocks', f'{user1.return_ticker().lower()}.us.txt'), delimiter=',',
                     usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    print('Loaded data from the Kaggle repository')


# Sort and output the data by date
df.sort_values('Date', inplace=True)
print(f"the first 5 dates: \n{df.head()}")
print(f"shape:{range(df.shape[0])}")


#2. Plotting the data

fig,ax = plt.subplots(figsize=(25,16)) # Creates the graph
ax.plot(range(df.shape[0]), (df["Low"] + df["High"]) / 2) # Parameters are x and y. X is just all the rows in the dataframe and y is the average of the low and high.
# This gives us our x-ticks. We iterate through n elements in our dataframe in steps of 500. We then apply the dates for every 500 steps
ax.set_xticks(range(0, df.shape[0], round(df.shape[0] // 24)), df["Date"].loc[::df.shape[0] // 24 ], rotation = 45, fontsize = 50) # Takes 2 arguments; the number position of the amount of ticks & the label for each tick which would be every 500 dates.
ax.set_title(f"Times Series Daily: {user1.return_ticker().upper()}", fontsize = 25) # Sets the title
ax.set_xlabel("Date", fontsize = 20) # Sets the x_label
ax.set_ylabel("Mid Price", fontsize = 30) # Sets the y_label
ax.tick_params(axis = 'both', length = 20, width = 6, color = "blue" ,labelsize = 25) # Function to change the ticks properties
# ax.set_title("Alphavantage Inplace", fontsize = 25)
plt.show(block= False)

high_prices = df.loc[:, "High"] # all the prices in the high column. This is now a numpy array after using to_numpy(). array[row_start:row_end, column_start:column_end]
low_prices = df.loc[:, "Low"] # all the prices in the low column. This is now a numpy array after using to_numpy().
high_prices.to_numpy() #as_matrix() and .values() does not work anymore. Modern way is to_numpy()
print(f"high prices: \n{high_prices}")
low_prices.to_numpy() # as_matrix() and .values() does not work anymore. Modern way is to_numpy()
print(f"low prices: \n{low_prices}")

mid_prices = (high_prices + low_prices) / 2.0 # Used because we need the average of both to give us a good estimate on our predictions
elements_of_mp = mid_prices.shape # Tuple
size_of_mp = int(elements_of_mp[0]) # Integer
print(f"size_of_mp divided by 2: {size_of_mp // 2}")

print(f"total # of data points: {int(elements_of_mp[0])} \n") # Outputs the total number of data points we have. np.prod(mid_prices.shape also works
print(f" here are the mid: {mid_prices}")

#3. Separating the data into training and test data

# train_data = mid_prices[:size_of_mp // 2] # First half of data points, so this is going to be the data we use to train. This is now a numpy array
# test_data = mid_prices[size_of_mp // 2:] # Last half data points, test if the trained model learned. Also a numpy array

train_data = mid_prices[:user1.training_vs_test_dp(size_of_mp)]
test_data = mid_prices[user1.training_vs_test_dp(size_of_mp):]

Scaler = MinMaxScaler() # Scaler to make all values between 0 and 1. The reason for this is to let the machine learn smoother and faster. (Formula: x-min/max-min)
# 2513 rows, 1 column for this specific ticker
train_data = train_data.values.reshape(-1, 1).copy() # After slicing the data, we now have a numpy array. We must reshape it into a 2d array. (11,000, 1) [[10],[20][30], etc.]. We do this because MinMaxScaler expects 2d arrays.
print(f"train data: {train_data}")
# 2513 rows, 1 column for the last bit of this ticker
test_data = test_data.values.reshape(-1, 1).copy() # After slicing the data, we now have a numpy array. We must reshape it into a 2d array. (11,000, 1) [[10],[20][30], etc.] Think of it vertically instead of horizontally. We do this because MinMaxScaler expects 2d arrays.
print(f"test data: {test_data}")

# 4. Converting the training data into scaled ones

# Number of chunks
now = 4 # How many chunks we want to separate the training_data. 4 is usually in the middle and gets rid of the most moderate noise
bounds = np.linspace(0, train_data.shape[0], (now + 1), dtype=int) # Creates values from 0 up to the training data shape evenly.
print(f"bounds: {bounds}") # Outputs the evenly distributed values between 0 & our training space

# Iterates through 2 arrays and creates a tuple which is a sliding window
for start, end in zip(bounds[:-1], bounds[1:]): # Start is every element except the last and last is every element except the first
    chunk = train_data[start:end, :] # Same logic as the comment made about 2D array slicing
    Scaler.fit(chunk) # Scales that window chunk of data
    train_data[start:end, :] = Scaler.transform(chunk) # Replaces the original value with the scaled ones


train_data = train_data.reshape(-1) # Reshapes the data back into an array
print(f"Viewing what the train_data looks like: {train_data}")
test_data = Scaler.transform(test_data).reshape(-1) # Reshape and transform the test data.  We only fit the training data on the training data because we don't want the test data to know the min/max which would
print(f"Viewing what the test_data looks like: {test_data}")

# 5.Smoothing the training data using EMA. EMA is a way to make old data points weigh less than the new ones.

EMA = 0.0 # Estimated moving average
gamma = 0.1 # Controls how the EMA reacts. 0.1 for old data to have more weight (10% new, 90% old) and 0.9 for new data to have more weight (10% old, 90% new)

# for ti in range(size_of_mp // 2)
for ti in range(train_data.shape[0]):
    EMA = gamma * train_data[ti] + (1-gamma) * EMA # Formula: (gamma * xt) + (1-gamma) * EMA, where t is an integer/index
    train_data[ti] = EMA # Swap the old data points for the new EMA ones
all_mid_data = np.concatenate([train_data, test_data], axis = 0) # Puts all the data in one array
print(f"mid data: {all_mid_data}") # Prints out the smoothed data

# 6. Standard average
window_size = 100 # Initiated variable 100; this is going to be how we get the updated prices by using the previous 100
# N = np.prod(train_data.shape) # Amount of elements in train_data/Size.
N = np.size(train_data, axis = 0)
std_avg_predictions = [] # Empty array to be used to store our predictions
std_avg_x = [] # Corresponding dates for each prediction ?
std_mse_errors = [] # Stores our losses for each loop

print(f"this is n:{N}")

# Calculating the mean squared error for the Standard Average. We are using previous data points to predict the next. Each prediction uses the last 100 DP
# Loop through all the elements in the train_data data set starting at 100 because we need the first 100 to predict the next one. (We can't predict anything if we don't have a headstart you could say)
for pred_idx in range(window_size, N):
    if pred_idx >= N:
        date = dt.datetime.strptime(k, "%Y-%m-%d").date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx, "Date"] # Assigns the corresponding index with the date. Ex. if we are at index 102 and the date is 11/23/2025 then we assign that date to the variable "date"

   # Obtaining the mean is our best chance at guessing.
    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx])) # Sliding window. Grabs the last 100 values. If we are at index 100, we would go from 0 (100 - 100) to 99 (numpy excludes the last index). We would then use those values to predict the index we are looking for.
                                                                                   # Ex. my_array = [10, 12, 11 ,15, 20] my_array[0:3] = (10+12+11/3) <- Our mean would be our prediction for train_data[3] but the true value is 15
    std_mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2) # Appends the errors to mse array. It grabs the most recent value from the prediction (our current estimate from the previous 100 points)
                                                                         # and applies the MSE error function with the actual true value
    std_avg_x.append(date) # Appends the date to the corresponding prediction
print(f"MSE error for standard averaging: {0.5*(np.mean(std_mse_errors)):.5f}") # Finds the MSE for all the values in mse_errors and multiply it by .5


fig, ax = plt.subplots(figsize=(25,16))
ax.plot(range(df.shape[0]), all_mid_data, color = 'r', label = "True") # Plot all the rows in the dataframe as x and all the mid_data as our y.
ax.plot(range(window_size, N), std_avg_predictions, color = 'b', label = "Predictions") # Plot x from 100 to the size of the rows in our dataframe
ax.set_xticks(range(0, df.shape[0], 500), df["Date"].loc[::500], rotation = 45)
ax.set_title(f"Standard Average: {user1.return_ticker().upper()}", fontsize = 25)
ax.set_xlabel("Data Point", fontsize = 20)
ax.set_ylabel("Mid Price", fontsize = 20)
ax.legend(fontsize = 20, loc = 'upper left')
ax.tick_params(axis = 'both', length = 20, width = 6, color = "blue" ,labelsize = 25)
plt.show(block=False)



# Calculating MSE for EMA (Exponential Moving Average). EMA takes into account all the values but decays the older values as we get new ones
ema_avg_predictions = []
ema_avg_x = []
ema_mse_errors = []

running_mean = 0.0
ema_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):

    # Each prediction incorporates data from all previous data points/values.
    running_mean = running_mean * decay + (1 - decay) * train_data[pred_idx - 1] # Allows the older values to "decay" more to have less impact on the newer values
    ema_avg_predictions.append(running_mean) # Appends the estimated ema average
    ema_mse_errors.append((train_data[pred_idx] - ema_avg_predictions[-1]) ** 2) # Appends the error
    ema_avg_x.append(df.loc[pred_idx, "Date"])

    '''
    Example of how EMA works:
    
    Decay = 0.5
    my_array = [10, 20, 15, 25, 30]
    running_mean = 0.0
    
    -1st iteration: running_mean * decay + ( 1-decay) * my_array[1 - 1] aka (my_array[0]) = (0.0 * 0.5) + (0.5 * 10) = 5.0
    *5.0 would be our first prediction aka our prediction for my_array[1] (20), because we use the previous value to guess the next.
    *There is no previous value before 10, so we initialize running_mean to 0 and use 10 as our initial value to predict 20 array[1]. 
    *We would use 5.0 that for the second iteration.
    
    -2nd iteration: Now we are going to use the previous running_mean along with the n - 1 data point to find the n + 1 data point
    *running_mean * decay + ( 1-decay)my_array[2 - 1] aka (my_array[1]) = (5.0 * 0.5) + (0.5 * 20) = 12.5 (This is the prediction for my_array[2] aka 15).
    
    We keep going until we go through all the elements in the array. This is how we take into account all values
    
    '''

# print(f"ema avg pred: {ema_avg_predictions}")
print(f"MSE for EMA: {0.5 * np.mean(ema_mse_errors):.5f}")

# Graphs EMA
fig, ax = plt.subplots(figsize=(25,16))
ax.plot(range(df.shape[0]), all_mid_data, color = 'orchid', label = "True")
ax.plot(range(0, N), ema_avg_predictions, color = 'gold', label = "Predictions")
# ax.set_xticks(range(0, df.shape[0], 500), df["Date"].loc[::500], rotation = 45) # If we do not set x_ticks, it will average the data out for us
ax.set_title(f"Exponential Moving Average: {user1.return_ticker().upper()}", fontsize = 25)
ax.set_xlabel("Date", fontsize = 20)
ax.set_ylabel("Mid Price", fontsize = 20)
ax.legend(fontsize = 20, loc = 'upper left')
ax.tick_params(axis = 'both', length = 20, width = 6, color = "blue" ,labelsize = 25)
plt.show(block=False)

D = 1 # Dimensionality. Only viewing one thing price.
num_unrollings = 50 # How many time steps we are looking back to predict the next value
num_of_sequences = 500 # 500 Sequences (Samples)
num_nodes = [200, 200, 150] # Hidden nodes in each LSTM layer
n_layers = len(num_nodes)
dropout_rate = 0.2 # Dropout amount to avoid memorizing instead of learning

dg = DataGenerator(train_data, batch_size=num_of_sequences, num_unroll=num_unrollings)
u_data, u_labels = dg.unroll_batches() # Stores the arrays of data & labels from the function

# For loop iterating through both the index with data & labels
for ui,(data,label) in enumerate(zip(u_data, u_labels)): # Zip and enumerate basically create 2 tuples. Zip creates a tuple with u_data & u_labels and enumerate creates an index with the tuple u_data & u_labels
                                                         # EX. (0, ([10, 12, 14, 16], [11, 13, 16, 18])) is what it would look like
                                                         #    Index        Data             Label
    print(f"Unrolled index: {ui}")
    print(f"tInputs: {data}")
    print(f"tOutputs: {label}\n")



model = create_lstm_model(num_nodes, dropout_rate, num_unrollings) # Creates LSTM Model
model.summary() # Outputs a summary of the model

# Hyperparameters for callback function
initial_learning_rate = 0.001
min_learning_rate = 0.00001
num_epochs = 50
train_num_of_sequences = 32

# Assign the wrapper function to a variable
# lr_function = LRS.wrapper(initial_learning_rate,min_learning_rate)

# Used to update the models learning rate. Takes in a callable function that returns the updated learning rate
# lr_callback = keras.callbacks.LearningRateScheduler(lr_function, verbose=0) # During training (.fit), the epoch argument will be automatically fulfilled by keras

# In your main code, replace the lr_callback line with. Works the same way as the function method. Takes epoch as an argument and will be filled by keras
lr_callback = keras.callbacks.LearningRateScheduler(
    lambda epoch: max(
        initial_learning_rate * (0.5 ** epoch), # Calculated decayed learning rate
        min_learning_rate
    ),
    verbose=0
)

'''
In the beginning of the training, provided your model has enough capacity (if it doesn't, that's another problem),
both training loss and validation loss will decrease as epochs occur.
After some time, the validation loss will stop decreasing and will start increasing,
while the training loss continues to decrease forever (by definition, because that's what you are optimizing for).
That's exactly the point when you must STOP the training - at that moment the model started overfitting.
'''
# Monitors the training and stops when monitored metric stops progressing
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', # Watches the validation loss to prevent overfitting
    patience=5,
    restore_best_weights=True, # Uses the best weights from the best epoch model
    verbose=1,
)

# Saves the best epoch model
checkpoint = keras.callbacks.ModelCheckpoint(
    "best_lstm_model.keras", # Saves the best model during each epoch
    monitor='val_loss', # Watches the validation loss to prevent overfitting
    save_best_only=True,
    verbose=1,
)

# Used to update the loss / prediction error. Used an advanced SGD algorithm
optimizer = keras.optimizers.Adam(learning_rate = initial_learning_rate,# Default learning rate
                                  clipnorm=5.0)

# Configurations to train the mode
model.compile(optimizer = optimizer,
              loss = 'mse', # Using the mse as the loss function
              metrics = ['mae']) # Calculates the MAE to view. Whatever metrics we want to be viewed must be included or keras won't include it as a key

X_train = np.array(u_data).T.reshape(num_of_sequences, num_unrollings, D) # Training data. Reformatted into (500, 50, 1) 500 samples, 50 unrollings, 1 feature being looked at
y_train = np.array(u_labels).T.reshape(num_of_sequences, num_unrollings, 1) # Testing data. Reformatted into (500, 50, 1) 500 samples, 50 unrollings, 1 feature being looked at

# print(f"\nX_train shape: {X_train.shape}")  # (500, 50, 1) 500 samples (sequences), each example takes 50 days of history, with 1 feature being looked at
# print(f"y_train shape: {y_train.shape}")    # (500, 50, 1)
print(f"Training on LAST time step predictions only")


# How the model gets trained. Returns a history object with an attribute named history to view a report
history = model.fit(
    X_train, # Training data
    y_train[:, -1, :], # Target data/ Actual values
    epochs=num_epochs, # Number of epochs
    batch_size=num_of_sequences, # Default batch size
    validation_split=0.2, # Sets aside 20% of the data to be used for validation. Uses the weights & bias from the 80% and uses it on the other 20% of the set aside data. Prevents overfitting (Memorizing instead of learning)
    callbacks=[lr_callback, early_stop, checkpoint], # Automatically calls lr_callback(lr_decay_schedule(epoch)), early_stop, and, checkpoint
    verbose=1
)

print(f"available metrics to be viewed: {history.history.keys()}")

# best_model = keras.models.load_model("best_lstm_model.keras")
# best_model = keras.models.load_model("best_lstm_model.keras")


print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Final training loss: {history.history['loss'][-1]:.6f}") # Outputs the final MSE/Loss
print(f"Best training loss: {min(history.history['loss']):.6f}") # Outputs the best MSE/Loss
print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
print(f"Best validation loss: {min(history.history['val_loss']):.6f}")


#  Visual Training History

fig, axes = plt.subplots(1, 3, figsize=(20,5))
# 1.1: MSE/Loss
axes[0].plot(history.history["loss"], label = "Training Loss", marker="o", linewidth=2) # Graphs the MSE/Loss on the first diagram. How the model is learning. Penalize error way harder
axes[0].plot(history.history["val_loss"], label = "Validation Loss", marker="s", linewidth=2) # Graphs the validation loss. How the model is predicting on the validation set based on the previous data it was trained on.
axes[0].set_xlabel("Epoch", fontsize = 12)
axes[0].set_ylabel("Loss (MSE)", fontsize = 12)
axes[0].set_title("Training & Validation loss", fontsize = 14, fontweight="bold")
axes[0].legend(fontsize = 11)
axes[0].grid(True, alpha = .3)
axes[0].set_yscale("log")

# 1.2: MAE
axes[1].plot(history.history["mae"], label = "Training MAE", marker="o", linewidth=2) # Graphs the MAE. Shows how well the model is doing to the human eye
axes[1].plot(history.history["val_mae"], label = "Validation MAE", marker="s", linewidth=2)  # Graphs the validation_mae. Uses the adjusted weights per epoch on the validation section of the training_data 
axes[1].set_xlabel("Epoch", fontsize = 12)
axes[1].set_ylabel("Mean Absolute Error", fontsize = 12)
axes[1].set_title("Training & Validation MAE", fontsize = 14, fontweight="bold")
axes[1].legend(fontsize = 11)
axes[1].grid(True, alpha = .3)


# 1.3: Learning Rate

# Function method
# lr_values = [lr_function(num_epochs)
#              for epoch in range(len(history.history["loss"]))]
# # axes[2].plot(lr_values, marker = "o", color="orange", linewidth=2)

# lambda method
# lr_values = [max((initial_learning_rate * (0.5 ** epoch)), min_learning_rate)
#              for epoch in range(len(history.history["loss"]))] # Creates the learning rate manually
# axes[2].plot(lr_values, marker = "o", color="orange", linewidth=2)

axes[2].plot((history.history["learning_rate"]), marker = "o", color="orange", linewidth=2) # Graphs the learning rate automatically using the history learning_rate metric

# 2: MAE
axes[2].set_xlabel("Epoch", fontsize = 12)
axes[2].set_ylabel("Learning Rate ", fontsize = 12)
axes[2].set_title("LR Schedule", fontsize = 14, fontweight="bold")
axes[2].legend(fontsize = 11)
axes[2].grid(True, alpha = .3)

plt.tight_layout()
plt.show(block=False)

# Making Predictions

print("\n" + "="*60)
print("GENERATING PREDICTIONS")
print("="*60)

predictions = model.predict(X_train, verbose=0) # Returns numpy array of predictions. Predicts at the end of every sequence
actual = y_train[:, -1, :]  # Last time step (what we trained to predict)
print(f"actual: {actual}")
print(f"shape of predictions: {predictions.shape}")
print(f"Min actual Value: {actual.min()}") # Min val in actual values array
print(f"Max actual Value: {actual.max()}") # Min val in actual values array
# print(f"MSE: {np.mean((predictions - actual) ** 2)}")
# print(f"MAE: {np.mean(np.abs(predictions - actual))}")
# print(f"Predictions shape: {predictions.shape}")  # (500, 1)
# print(f"Actual values shape: {actual.shape}")     # (500, 1)

# mse = np.mean((predictions - actual) ** 2) # Mean Standard Error
# mae = np.mean(np.abs(predictions - actual) ** 2) # Mean Absolute Error

# outputs the first 10 index, prediction, actual, and the error
for i in range(10):
    pred_val = predictions[i, 0]
    actual_val = actual[i, 0]
    error = pred_val - actual_val
    mse = np.mean((actual_val - pred_val) ** 2)
    mae =  np.mean(np.abs(actual_val - pred_val))
    print(f"Index: {i:<10} Prediction: {pred_val:<15.6f} Actual: {actual_val:<15.6f} Error: {error:<15.6f} MSE: {mse:<15.6f} MAE: {mae:<15.6f}")

# Visualizing first 100 predictions

plt.figure(figsize=(12, 6))
plt.plot(actual[:100], label='Actual', marker='o') # Plots the first 100 values from the actual data set
plt.plot(predictions[:100], label='Predicted', marker='x') # Plots the first 100 values from the prediction data set
plt.legend()
plt.title('Stock Price Predictions vs Actual')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.show(block=False)

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Plot 1: First 100 predictions
axes[0].plot(actual[:100], label='Actual', marker='o', linewidth=2, markersize=4, alpha=0.7) # Plots the first 100 actual values. The x-axis is generated by matplot
axes[0].plot(predictions[:100], label='Predicted', marker='x', linewidth=2, markersize=4, alpha=0.7) # Plots the first 100 predictions. The x-axis is generated by matplot
axes[0].set_xlabel('Sample Index', fontsize=12)
axes[0].set_ylabel('Normalized Price', fontsize=12)
axes[0].set_title('LSTM Predictions vs Actual (First 100 Samples) lambda method', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot 2: Scatter plot of predictions vs actual
axes[1].scatter(actual, predictions, alpha=0.5, s=20) # Scatter plots the actual as x and pred as y
# Creates the dotted dash line to represent the accuracy of the predictions
axes[1].plot([actual.min(), actual.max()],
             [actual.min(), actual.max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Values', fontsize=12)
axes[1].set_ylabel('Predicted Values', fontsize=12)
axes[1].set_title('Prediction Accuracy Scatter Plot', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=False)

print("\n" + "="*60)
print("GENERATING PREDICTIONS INTO THE FUTURE")
print("="*60)


ior = np.linspace(train_data.shape[0], test_data[0], 1, )
# Predicting n amount of times into the future
n_predict_once = 50  # Predict 50 steps at a time
test_points_seq = np.arange(train_data.shape[0], math.floor((train_data.shape[0] + (train_data.shape[0] * .091))), 50).tolist()  # Creates a list of values from 11000 to 11999 with an increment of 50

all_predictions = [] # Where all the predictions will be stored
x_axis_seq = [] # Where the x-axis will be stored

# Goes through each element in the test_points_seqn
for w_i in test_points_seq:
    # Get the sequence leading up to this point
    start_idx = w_i - num_unrollings # Starting index is 10950. Uses 50 timestep prices to predict 11000. 11000 is not included in the [10950, 11000]. goes up to 10999.
    initial_seq = all_mid_data[start_idx:w_i].reshape(1, num_unrollings, 1) # Reshapes the data into shape the model takes in

    # Predict 50 steps ahead
    multi_step_predictions = predict_sequence(model, initial_seq, n_predict_once) # Runs the function with the trained model, the sequence of data formatted for model, and the #
    all_predictions.append(multi_step_predictions)

    # Track x-axis positions
    x_axis = list(range(w_i, w_i + n_predict_once)) # Creates a list of (11000, 11000 + 50) and so on
    x_axis_seq.append(x_axis)

    actual_values = all_mid_data[w_i:w_i + n_predict_once] # Actual prices/values
    mse = np.mean((np.array(multi_step_predictions) - actual_values) ** 2) # Calculates the MSE
    print(f"Prediction window at {w_i}: MSE = {mse:.5f}") # Outputs the prediction window and the mse for said prediction window

print(f"predictions: {len(all_predictions)}") # Amount of elements in prediction
print(f"x_axis: {len(x_axis_seq)}") # Amount of elements in x-axis

print("\n" + "="*60)
print("VISUALIZING PREDICTIONS INTO THE FUTURE")
print("="*60)

plt.figure(figsize=(18, 9))
# Plot actual data
plt.plot(range(len(all_mid_data)), all_mid_data, color='black',
         linewidth=2, label='True', alpha=0.7)

# Plot each prediction window
for i, (predictions, x_axis) in enumerate(zip(all_predictions, x_axis_seq)):
    plt.plot(x_axis, predictions, linewidth=2, alpha=0.6,
             label=f'Prediction {i+1}' if i < 3 else None)

plt.title('Multi-Step Ahead Predictions', fontsize=18)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Normalized Price', fontsize=14)
plt.xlim(11000, 12500)  # Focus on prediction region
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()