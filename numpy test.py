import numpy as np

test_arr = np.array([1,2,3,4,5,6,7,8,9,10])
evens = np.array([])

for i in range(test_arr.size):
    if test_arr[i] % 2 == 0:
        evens = np.append(evens, test_arr[i])

    test_arr[i] = test_arr[i] * 2


print(test_arr.size)
print(test_arr)
print(evens)



