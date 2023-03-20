#! /usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt

arr_base = np.random.randn(100000)

plt.hist(arr_base, bins=100)
plt.show()
