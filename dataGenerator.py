import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch

# Used to assign the data for the tensorflow model can read & hold it
class DataGenerator(object):
    def __init__(self, prices, batch_size, num_unroll):
        self.prices = prices # Stores our array of stock prices
        # Used so we do not go out of bounds.
        self.prices_length = len(self.prices) - num_unroll # prices initialized with precaution of going out of bounds.
        self.batch_size = batch_size # of sections/sequences (Think of it as the columns) (Also known as the sequences)
        self.num_unroll = num_unroll # of elements per sequence/batch_size (Think of it as the rows) (Also known as the time steps)
        self.segments = self.prices_length // self.batch_size # Splits data so sequences do not overlap. Set amount of data points to read

        self.cursor = [offset * self.segments for offset in range(self.batch_size)] # Keeps track of the current position of each segment.

        """
        Ex.
        data_series = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ...]
        
        batch_size = 4. We create 4 Separate sequences/samples: A,B,C,D. Each sequence is a pointer into the data set.
        Example: A = 0, B = 5, C = 10, D = 15
        At one time step, we take one value from each sequence. So it would be [0 5 10 15] (Batch_size = 4). Then the next sequence would be [1, 6, 11, 16], however 0 -> 1 is a time step and 0 -> 5 is changing sequences.
                                                                               [1 6 11 16] easier to understand like this. Rows = time steps, columns = sequences.
                                                                               [2 7 12 17] Each row is ONE time step, Each column is ONE sequence
        Sample/Sequence A could be: [0,1,2,3]
        Sample/Sequence B could be: [1,2,3,4] 
        and so on until it reaches Sample/Sequence D. While each Sample has x amount of time steps (values they are looking at to predict the next)                                                                      
                                                                               
        *prices_length = What we initialized our price to so it does not go down.
        For example:
        my_arr = [1 2 3 4 5]
        *num_unrolling = 3 (we need this because this is how many values we are going to be time stepping at once) essentially they are clues the model uses to predict
        i = 2(the element 3 in the array)
        If we try unrolling, we will get an out of bounds error
        
        *segments = basically sliding window all over again so that the sequences do not overlap with the same data
        my_arr = [99 values]. A = 0:24, B = 24:49 & so on
        
       *self.cursor = creates an array of elements that point to their respective sequences. Does this by multiplying each generated element in the array by segments
       Ex.
       batch_size = 5
       segments = 2415
       cursor is tracking the progress. generates a list of [0 1 2 3 4] then multiply each generated element by segments
       0 * 2415, 1 * 2415, 2 * 2415, 3 * 2415, 4 * 2415. These would be our cursors.
       We update the cursors when we call next_batch 
        
        Way to think about it:
        Data Set = Long Book
        Sequence = n # of people reading different chapters (depends on batch_size)
        Batch  = current word each person is reading (current element/data point)
        Batch_size = # of people reading (columns)
        num_unroll = # of time steps per sequence or number of rows (rows)
        princes_length = threshold of pages to read
        Segment = each person gets their own chapter
        cursor = like a bookmark for each chapter
        """

    # Returns one time step worth of data
    def next_batch(self):
        batch_data = np.zeros((self.batch_size), dtype=np.float32) # Initializes a numpy array of 0's for inputs 
        batch_labels = np.zeros((self.batch_size), dtype=np.float32) # Initializes a numpy array of 0's for targets

        # Iterates over the sequences
        for i in range(self.batch_size): # i = 0 is column 0, i = 1 is column 1, and so on.
            if self.cursor[i] + 1 >= self.prices_length: # Condition for if the sequence is about to go over the data set. We need at least one future value for the label
                # Resets the cursor to safe random position inside its segment
                self.cursor[i] = np.random.randint(0, (i + 1) * self.segments) # (i + 1) * self.segments ensures it is inside its segment

            # This is the inputs. What we have/know. Current price
            batch_data[i] = self.prices[self.cursor[i]] # Takes the current segment index and store it inside the array
            # This is the outputs. What we want to predict/Future price
            batch_labels[i] = self.prices[self.cursor[i] + np.random.randint(0,5)] # Instead of using cursor + 1, we use cursor with a random int from 0 to 5 for data augmentation

            # Gets the pointer/position of the element/timestep we are at for each sequence and moves it one time step. This is how the rows advance
            self.cursor[i] = (self.cursor[i] + 1) % self.prices_length # Ex. B4 nb: Cursor[0] = 0, After nb: (2415 + 1) % 12075 = 2416 because if the first # is smaller than the divisor, it returns the first number;

        return batch_data, batch_labels # Returns arrays

    # Returns multiple time steps of data
    def unroll_batches(self):

        unroll_data, unroll_labels = [], [] # Holding the data. These are 2 separate arrays just in the same line

        # Loops depending on how much unroll is. these are basically the rows
        for ui in range(self.num_unroll):
            data, labels = self.next_batch() # Assigns next_batch function to variable called data & labels
                                             # Calls next_batch num_unroll times

            unroll_data.append(data) # Appends the data to the list
            unroll_labels.append(labels) # Appends the output we want to predict

        return unroll_data, unroll_labels # Returns the arrays

    def reset_indices(self):
        for i in range(self.batch_size):
            self.cursor[i] = np.random.randint(0, min((i + 1) * self.segments),self.prices_length - 1)




