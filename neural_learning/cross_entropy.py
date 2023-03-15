#! /usr/bin/python3
import numpy as np

def cross_entropy_error(y,t) : 
	if y.ndim==1 : ## 모델의 출력데이터가 1차원, 즉 배치모드아 아닐경우
		t = t.reshape(1,t.size) # 행이 1이고 열은 데이터셋의 클래스의 수 로 이로어진 행렬로 바꿈
		y = y.reshape(1,y.size) # 행이 1이고 열은 데이터셋의 클래스의 수 로 이로어진 행렬로 바꿈
	
	batch_size = y.shape[0] # 만약 배치모드가 아니면 (1x클래스의 수) 행렬 이므로 y.shape[0]=1임.
	return -np.sum(np.log(y[np.arange(batch_size),t]/batch_size))
	## 만약 batch_size=100 이면 np.arange에서 [0,1,2,---99]행렬 생성한다.
	## t에는 [2,4,7,----9] 100개의 라벨값이 들어있다.   
	## y[np.arange(batch_size),t]는 y[0,2]의 값, y[1,4]의 값 y[2,7]의 값 ----- y[99,9]값을 의미한다.
	## y[0,2]의 의미는 0행 2열의 값 즉 첫번째 이미지가 클래스 2로 예측될 확률을 의미한다.
	## y는 배치사이즈가 100개이면 1개마다 10개의 클래스에 대한 확률이 있는 (100x10)행렬 일 것이다.     

