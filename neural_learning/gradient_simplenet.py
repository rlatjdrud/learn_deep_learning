#! /usr/bin/python3

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

def f(W) : 
	return net.loss(x,t)

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t): ##이미 numerical_gradient함수 때문에 net.loss함수를 실행하기전에 w에서 W+h 또는 W-h로 바뀌어있다
        print(f"xloss:{x}")
        print(f"w1{self.W}")
        z = self.predict(x)
        print(f"w2{self.W}")
        y = softmax(z)
        print(f"y:{y}")
        loss = cross_entropy_error(y, t)
        print(f"loss{loss}")
        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()
print(f"net.W:{net.W}")
##f = lambda w: net.loss(x, t) # w는 매개변수(predict함수에 쓰임), net.loss(x,t)는 리턴 값 => def f(w)함수와 같다.

dW = numerical_gradient(f, net.W)
print(dW)
