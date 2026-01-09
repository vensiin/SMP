import matplotlib.pyplot as plt # https://matplotlib.org/stable/users/explain/quick_start.html#a-simple-example
import numpy as np

#
# fig = plt.figure()             # an empty figure with no Axes
fig, ax = plt.subplots()       # a figure with a single Axes
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
# fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
# # a figure with one Axes on the left, and two on the right:
# fig, axs = plt.subplot_mosaic([['left', 'right_top'],
#                                ['left', 'right_bottom']])
plt.show()

