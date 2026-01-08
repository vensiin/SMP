import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch


class DataGenerator(object):
    def __init__(self, prices, batch_size, num_unroll):
        self.prices = prices
        self.prices_length = len(self.prices) - num_unroll
        self.batch_size = batch_size
        self.num_unroll = num_unroll
        self.segments = self.prices_length // self.batch_size

        self.cursor = [offset * self.segments for offset in range(self.batch_size)]


    def next_batch(self):
        batch_data = np.zeros((self.batch_size), dtype=np.float32)
        batch_labels = np.zeros((self.batch_size), dtype=np.float32)

        for i in range(self.batch_size):
            if self.cursor[i] + 1 >= self.prices_length:
                self.cursor[i] = np.random.randint(0, (i + 1) * self.segments)

            batch_data[i] = self.prices[self.cursor[i]]


