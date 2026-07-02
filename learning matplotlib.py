import matplotlib.pyplot as plt # https://matplotlib.org/stable/users/explain/quick_start.html#a-simple-example
import numpy as np

#
# fig = plt.figure()             # an empty figure with no Axes
fig, ax = plt.subplots()       # a figure with a single Axes
ax.plot([1, 2, 3, 4]) # This will plot them on the y-axis but will auto generate the x-axis using the min & max values (0-3)
# fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
# # a figure with one Axes on the left, and two on the right:
# fig, axs = plt.subplot_mosaic([['left', 'right_top'],
#                                ['left', 'right_bottom']])
plt.show()

# Generate sample data
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# Create basic scatter plot
plt.scatter(x, y)

# Add titles and labels
plt.title("Basic Scatter Plot")
plt.xlabel("X-Axis Label")
plt.ylabel("Y-Axis Label")

# Display the plot
plt.show()


