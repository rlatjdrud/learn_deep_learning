# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h ##=>여기서 x는 w이다. 여기서 W를 W+h로 바꾼다.
        print(f"x:{x}")
        fxh1 = f(x) # f(w+h)=>loss(x+h,t) => x+h가 모델로 들어가고 예측확률인 y로 바뀜 => -log1(y)    
        print(f"fxh1{idx}:{fxh1}")
        
        x[idx] = tmp_val - h 
        print(f"x:{x}")
        fxh2 = f(x) # f(w-h)=>loss(x-h,t) => x-h가 모델로 들어가고 예측확률인 y로 바뀜 => -log2(y) 
        print(f"fxh2{idx}:{fxh2}")
        grad[idx] = (fxh1 - fxh2) / (2*h) ##grad[idx]=-log1+log2/2h가 계산됨 
        print(f"grad[idx] :{grad[idx]}")
	##첫번째 루프를 돌면 w11의 미분값이 나옴
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad
