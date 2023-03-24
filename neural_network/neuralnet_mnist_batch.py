#! /usr/bin/python3

#! /usr/bin/python3
import sys,os
sys.path.append(os.pardir) ##부모디렉터리의 파일을 가져오도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from softmax import softmax
from sigmoid import sigmoid
 

def get_data():
	(x_train,t_train),(x_test,t_test) = load_mnist(flatten = True,normalize=True,one_hot_label=False)
	return x_test,t_test

def init_network():
	with open("sample_weight.pkl","rb") as f : 
		network = pickle.load(f)

	return network

def predict(network,x):
	w1,w2,w3 = network['W1'],network['W2'],network['W3']
	b1,b2,b3 = network['b1'],network['b2'],network['b3']
	
	a1 = np.dot(x,w1)+b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1,w2)+b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2,w3)+b3
	y = softmax(a3)
	return y

x,t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size) : 
	x_batch = x[i:i+batch_size]
	y_batch = predict(network,x_batch) ##100x784 -> 100x50 -> 100x100 -> 100x10
	p = np.argmax(y_batch,axis=1)
	accuracy_cnt += np.sum(p == t[i:i+batch_size]) 

##(p == t[i:i+batch_size])에서 100(batch_size)개 중 라벨값과 예측한 인덱스가 일치한 인덱스는 True로 바뀐다. 
##그러므로 총 100열의 True,false로 구성된 1차원행렬이된다. 
##그리고 이 행렬을 np.sum을 사용하면 true 갯수가 나온다

print(f"Accuracy : {accuracy_cnt/len(x)}")
#print("Accuracy :"+str(float(accuracy_cnt)/len(x)))	
