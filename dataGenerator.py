import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch


class DataGenerator(object):
    def __init__(self, prices, batch_size, num_unroll):
        self.prices = prices # Stores our array of stock prices
        self.prices_length = len(self.prices) - num_unroll #
        self.batch_size = batch_size # of sections/sequences (Think of it as the columns) (Also known as the sequences)
        self.num_unroll = num_unroll # of elements per sequence/batch_size (Think of it as the rows) (Also known as the time steps)
        self.segments = self.prices_length // self.batch_size

        self.cursor = [offset * self.segments for offset in range(self.batch_size)]

        """
        Ex.
        data_series = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ...]
        
        batch_size = We create 4 Separate sequences: A,B,C,D. Each sequence is a pointer into the data set.
        Example: A = 0, B = 5, C = 10, D = 15
        At one time step, we take one value from each sequence. So it would be [0 5 10 15] (Batch_size = 4). Then the next time step would be [1, 6, 11, 16], however 0 -> 1 is a time step and 0 -> 5 is changing sequences.
                                                                               [1 6 11 16] easier to understand like this. Rows = time steps, columns = sequences.
                                                                               [2 7 12 17] Each row is ONE time step, 📌 Each column is ONE sequence
        
        Way to think about it:
        Data Set = Long Book
        Sequence = n # of people reading different chapters (depends on batch_size)
        Batch  = current word each person is reading (current element/data point)
        Batch_size = # of people reading;
        """


    def next_batch(self):
        batch_data = np.zeros((self.batch_size), dtype=np.float32)
        batch_labels = np.zeros((self.batch_size), dtype=np.float32)

        for i in range(self.batch_size):
            if self.cursor[i] + 1 >= self.prices_length:
                self.cursor[i] = np.random.randint(0, (i + 1) * self.segments)

            batch_data[i] = self.prices[self.cursor[i]]
            batch_labels[i] = self.prices[self.cursor[i] + np.random.randint(0,5)]

            self.cursor[i] = (self.cursor[i] + 1) % self.prices_length

        return batch_data, batch_labels

    def unroll_batches(self):

        unroll_data, unroll_labels = [], []
        init_data, init_label = None, None

        for ui in range(self.num_unroll):
            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for i in range(self.batch_size):
            self.cursor[i] = np.random.randint(0, min((i + 1) * self.segments),self.prices_length - 1)




