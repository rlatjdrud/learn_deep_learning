#! /usr/bin/python3

import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient



def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x 
    x_history = []
    v=[0.0,0.0]

    for i in range(step_num):
        x_history.append( x.copy() ) ##경사법이 적용된 뒤의 x값을 배열에 계속 쌓아둔다.
        grad = numerical_gradient(f, x) ##기울기를 구한다.
        for idx in range(len(grad)):
            
            v[idx]=0.9*v[idx]-lr*grad[idx]
            print(f"v{idx} : {v[idx]}")
            x[idx] += v[idx] ##경사법을 수행한다. 
          
    return x, np.array(x_history)


def function_2(x):
    return x[0]**2/20 + x[1]**2 

init_x = np.array([-3.0, 4.0])    

lr = 0.05
step_num = 100
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
