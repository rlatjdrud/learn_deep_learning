#! /usr/bin/python3
import numpy as np

def NAND(x1,x2) : 
	x = np.array([x1,x2])
	w = np.array([-0.5,-0.5])
	b = 0.7
	tmp = np.sum(x*w)+b
	if(tmp<=0) : 
		return 0;
	else : 
		return 1;

def OR(x1,x2) : 
	x=np.array([x1,x2])
	w = np.array([0.5,0.5])
	b = -0.3
	tmp = np.sum(x*w)+b
	if(tmp<=0) :
		return 0;
	else : 
		return 1;


def AND(x1,x2) : 
	x = np.array([x1,x2])
	w = np.array([0.5,0.5])
	b = -0.7
	tmp = np.sum(x*w)+b
	if(tmp<=0) :
		return 0
	else : 
		return 1


def XOR(x1,x2) : 
	s1 = NAND(x1,x2)
	s2 = OR(x1,x2)
	result = AND(s1,s2)
	return result

print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))
