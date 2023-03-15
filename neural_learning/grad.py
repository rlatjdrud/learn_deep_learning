#! /usr/bin/python3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    ## x=[x0,x1]
    for idx in range(x.size): ##첫번째루프 : idx=0 두번째루프 idx=1
        tmp_val = x[idx] ##첫번째루프 : tmp_val = x0 두번째 루프 tmp_val = x1
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h ##첫번째루프 : x0 = x0+h 두번째 루프 : x1 = x1+h
        fxh1 = f(x) ##첫번째루프 : fx1 = [f(x0+h),f(x1)} 두번째 루프 : fxh1 = [f(x0),f(x1+h)]
        
        # f(x-h) 계산
        x[idx] = tmp_val - h ##첫번째루프 : x0 = x0-h 두번째 루프: x1 = x1-h 
        fxh2 = f(x)  ##첫번째루프 : fx2 =[f(x0-h),f(x1)] 두번째 루프 : fx2=[f(x0),f(x1-h)]
        
        grad[idx] = (fxh1 - fxh2) / (2*h) ##첫번째루프 : fxh1-fxh2 = f(x0+h)-f(x0-h) 두번째 루프 fxh1-fxh2 = f(x1+h)-f(x1-h)
        x[idx] = tmp_val # 값 복원
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else: ##x = 100x2 행렬
        grad = np.zeros_like(X) ## 100x2 크기의 영행렬        
        for idx, x in enumerate(X): ## idx에 현재 루프의 수가 들어가고 x에 [x0,x2]가 들어감
            	grad[idx] = _numerical_gradient_no_batch(f, x)
        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2) 	##첫번째루프 : fx1 =[f(x0+h)+f(x1)} 두번째 루프 : fxh1=[f(x0)+f(x1+h)]
				##첫번째루프 : fx2 =[f(x0-h)+f(x1)] 두번째 루프 : fx2=[f(x0)+f(x1-h)]
    else:
        return np.sum(x**2, axis=1)

a=numerical_gradient(function_2,np.array([[3.0,4.0],[5.0,6.0]]))
print(a)
