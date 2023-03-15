#! /usr/bin/python3

import numpy as np

def cross_entropy_error(y,t) : 
	if y.ndim==1 : ## 모델의 출력데이터가 1차원, 즉 배치모드아 아닐경우
		t = t.reshape(1,t.size) # 행이 1이고 열은 데이터셋의 클래스의 수 로 이로어진 행렬로 바꿈
		y = y.reshape(1,y.size) # 행이 1이고 열은 데이터셋의 클래스의 수 로 이로어진 행렬로 바꿈
	
	batch_size = y.shape[0] # 1x클래스의 수 행렬 이므로 batch_size=1임.
	return -np.sum(t*np.log(y+1e-7)/batch_size)



