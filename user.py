import math
import numpy as np
from keras.src.ops import numpy


class my_user():
    def __init__(self):
        self.selected_ticker = ""

    def user_enter_ticker(self):
        self.selected_ticker = input("Please enter your ticker to predict: ")

    def return_ticker(self):
        return self.selected_ticker.upper()

    def num_data_points(self, data_points):
        return int(data_points[0])

    def training_vs_test_dp(self, total_points):
        return round(total_points - (total_points * .089))



# user1 = my_user()
# user1.user_enter_ticker()
# # print(user1.return_ticker().lower())
# print(user1.selected_ticker.upper())

# i = 12075
# print(math.ceil(12075 - (12075 * .172)))

# train_data = np.zeros(11000)
# test_data = np.zeros(1075)
#
# ior = np.linspace(train_data.shape[0], 12074, 3, )
# print(ior)