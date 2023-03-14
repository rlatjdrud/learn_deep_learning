#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt


def step_func1(x) : 
	return np.array(x>0,dtype=np.int)
	


x = np.arange(-5.0,5.0,0.1)
y = step_func1(x)

plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


