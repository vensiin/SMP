from fontTools.misc.arrayTools import scaleRect
from pandas.io.xml import preprocess_data # This is related to handling XML data with Pandas. https://pandas.pydata.org/docs/
from pandas_datareader import data # This can fetch stock/financial data from sources like Yahoo, Google, etc. # https://pandas-datareader.readthedocs.io/en/latest/index.html
import matplotlib.pyplot as plt # Lets you make charts and graphs. # https://matplotlib.org/stable/users/explain/quick_start.html#a-simple-example
import pandas as pd # Pandas is used for data manipulation and analysis (working with tables).
import datetime as dt # Lets you work with dates and times (e.g., parsing strings into dates). import datetime as dt # https://docs.python.org/3/library/datetime.html#
import urllib.request, json # Imports urllib.request (for making web requests, like downloading from URLs). Imports json (to handle JSON data — text data formatted like dictionaries).
import os # Lets you interact with the operating system (like checking if a file exists). https://docs.python.org/3/library/urllib.request.html#module-urllib.request
import numpy as np # A math library for arrays, vectors, and numerical computing.
import tensorflow as tf # Imports TensorFlow, a machine learning library.
from sklearn.preprocessing import MinMaxScaler # Imports a tool to scale (normalize) values into a range (like 0 → 1).
import os
from dotenv import load_dotenv


# 1. Obtaining the data

data_source = "kaggle"
lower_case = data_source.lower()

api_key = "0NSF2RAF00LQ4N9K"
ticker = "AAL"

# Conditional that checks which method we are using
if lower_case == "alphavantage":


    url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}" # Website we are pulling data from

    file_to_save = f"stock_market_data-{ticker}.csv" # Name of the file we are saving the data to

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
            df.to_csv(file_to_save)  # Saves the data as a CSV file since it is more neater than JSON
            print(f"Data saved to: {file_to_save}") # Output that we saved the file


    # Else conditional on if the file is already saved
    else:
        print("File already exists. Loading data from CSV")
        df = pd.read_csv(file_to_save)



# Kaggle method
else:
    df = pd.read_csv(os.path.join('Stocks', 'hpq.us.txt'), delimiter=',',
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
ax.set_xticks(range(0, df.shape[0], 500), df["Date"].loc[::500], rotation = 45, fontsize = 50) # Takes 2 arguments; the number position of the amount of ticks & the label for each tick which would be every 500 dates.
ax.set_title("Times Series Daily", fontsize = 25) # Sets the title
ax.set_xlabel("Date", fontsize = 20) # Sets the x_label
ax.set_ylabel("Mid Price", fontsize = 30) # Sets the y_label
ax.tick_params(axis = 'both', length = 20, width = 6, color = "blue" ,labelsize = 25) # Function to change the ticks properties
# ax.set_title("Alphavantage Inplace", fontsize = 25)
plt.show()

high_prices = df.loc[:, "High"] # all the prices in the high column. This is now a numpy array after using to_numpy(). array[row_start:row_end, column_start:column_end]
low_prices = df.loc[:, "Low"] # all the prices in the low column. This is now a numpy array after using to_numpy().
high_prices.to_numpy() #as_matrix() and .values() does not work anymore. Modern way is to_numpy()
print(f"high prices: \n{high_prices}")
low_prices.to_numpy() # as_matrix() and .values() does not work anymore. Modern way is to_numpy()
print(f"low prices: \n{low_prices}")

mid_prices = (high_prices + low_prices) / 2.0
elements_of_mp = mid_prices.shape # Tuple
size_of_mp = int(elements_of_mp[0]) # Integer
print(f"size_of_mp divided by 2: {size_of_mp // 2}")

print(f"total # of data points: {int(elements_of_mp[0])} \n") # Outputs the total number of data points we have. np.prod(mid_prices.shape also works
# print(f" here are the mid: {mid_prices}")

#3. Separating the data into training and test data

# train_data = mid_prices[:size_of_mp // 2] # First half of data points, so this is going to be the data we use to train. This is now a numpy array
# test_data = mid_prices[size_of_mp // 2:] # Last half data points, test if the trained model learned. Also a numpy array

train_data = mid_prices[:11000]
test_data = mid_prices[11000:]

Scaler = MinMaxScaler() # Scaler to make all values between 0 and 1. The reason for this is to let the machine learn smoother and faster. (Formula: x-min/max-min)
# 2513 rows, 1 column for this specific ticker
train_data = train_data.values.reshape(-1, 1) # After slicing the data, we now have a numpy array. We must reshape it into a 2d array. (11,000, 1) [[10],[20][30], etc.]. We do this because MinMaxScaler expects 2d arrays.
print(f"train data: {train_data}")
# 2513 rows, 1 column for the last bit of this ticker
test_data = test_data.values.reshape(-1, 1) # After slicing the data, we now have a numpy array. We must reshape it into a 2d array. (11,000, 1) [[10],[20][30], etc.] Think of it vertically instead of horizontally. We do this because MinMaxScaler expects 2d arrays.
print(f"test data: {test_data}")

# 4. Converting the training data into scaled ones
smoothing_wndw_size = 2500 # chunk of data we will be processing per loop
print(f"smoothing_wndw_size: {smoothing_wndw_size}")
stop = size_of_mp // 2

di = 0

# For loop that goes over each chunk of data. (Must automate because right now it only works for AAL )
for di in range(di, 10000, smoothing_wndw_size):
    '''
    The reason we are splitting apart the data is because we do not want the older data to be negligible compared to the new data.
    So to counter that we just split the data set into different chunks so that the old stocks do not get compared to near 0 decimals compared to the larger stock prices today.
    Also if we go over the amount of data we have, we will get an indexing error

    Ex. [10, 12, 14, 300, 320, 350]
    If we scale all at once, we'll get [0.0, 0.006, 0.011, 0.85, 0.91, 1.0]. All the early values are negligible and hold no weight.
    When we scale the values, now all the prices hold some form of weight which could influence the training.

     Simple analogy:
    Imagine you’re tracking a kid’s growth:
    At age 5: height changes from 3’0 → 3’5
    At age 15: height changes from 5’0 → 5’5
    If you only look at raw inches, the later growth (60" to 65") looks “bigger.”
    But relative to their size, both are +5 inches.

    Scaling makes both periods equally important so the model can see “this stock went up relative to its normal range” instead of “this number is bigger.”
    '''

    # We are using numpy 2d array slicing, where the syntax is array[row_slice, column_slice]. The format is [x,:], where x is the row = full row; [:,x], where x is the column, = full column. array[row_start:row_end, column_start:column_end]
    Scaler.fit(train_data[di:di+smoothing_wndw_size, :]) # So our first iteration would be from rows 0 to 2500, then our next would be from rows 2500:5000, and so on. It stores the (min,max) pairs per each loop
    print(Scaler.fit(train_data[di:di+smoothing_wndw_size, :]))
    train_data[di:di + smoothing_wndw_size, :] = Scaler.transform(train_data[di:di + smoothing_wndw_size, :]) # Replaces the original values with the scaled ones (0-1) using the scaling formula which uses the (min,max) values from scalar.fit

    '''
    Example of how the 2d array looks
    [[1],
     [2],
     [3],] where there is one column and many rows
    '''

left_over = di + smoothing_wndw_size
Scaler.fit(train_data[left_over:, :]) # All the rows starting from left_over and all the columns(there is only 1 column.)
train_data[left_over:, :] = Scaler.transform(train_data[left_over:, :])
# Scaler.fit(train_data[di+smoothing_wndw_size:,:]) # Do the same but for the rest of the data
# print(Scaler.fit(train_data[di+smoothing_wndw_size:,:]))
# train_data[di:di + smoothing_wndw_size, :] = Scaler.transform(train_data[di:di + smoothing_wndw_size, :]) # Do the same for the last bit of data

train_data = train_data.reshape(-1) # Reshapes the data back into an array
test_data = Scaler.transform(test_data).reshape(-1) # Reshape and transform the test data.  We only fit the training data on the training data because we don't want the test data to know the min/max which would
                                                                                            # lead the machine to not actually learn.

# 5.Smoothing the training data using EMA. EMA is a way to make old data points weigh less than the new ones.

EMA = 0.0 # Estimated moving average
gamma = 0.1 # Controls how the EMA reacts. 0.1 for old data to have more weight (10% new, 90% old) and 0.9 for new data to have more weight (10% old, 90% new)

# for ti in range(size_of_mp // 2)
for ti in range(11000):
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
ax.set_title("Standard Average", fontsize = 25)
ax.set_xlabel("Data Point", fontsize = 20)
ax.set_ylabel("Mid Price", fontsize = 20)
ax.legend(fontsize = 20, loc = 'upper left')
ax.tick_params(axis = 'both', length = 20, width = 6, color = "blue" ,labelsize = 25)
plt.show()



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


fig, ax = plt.subplots(figsize=(25,16))
ax.plot(range(df.shape[0]), all_mid_data, color = 'orchid', label = "True")
ax.plot(range(0, N), ema_avg_predictions, color = 'gold', label = "Predictions")
# ax.set_xticks(range(0, df.shape[0], 500), df["Date"].loc[::500], rotation = 45) # If we do not set x_ticks, it will average the data out for us
ax.set_title("Exponential Moving Average", fontsize = 25)
ax.set_xlabel("Date", fontsize = 20)
ax.set_ylabel("Mid Price", fontsize = 20)
ax.legend(fontsize = 20, loc = 'upper left')
ax.tick_params(axis = 'both', length = 20, width = 6, color = "blue" ,labelsize = 25)
plt.show()