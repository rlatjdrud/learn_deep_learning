#! /usr/bin/python3
import os
import sys
sys.path.append(os.pardir) 
import numpy as np
import matplotlib.pyplot as plt
from common.util import *

a = np.random.randn(1,3,4,4)
col = im2col(a,2,2,2,0)
col2 = col.reshape(-1,2*2) 
out = np.max(col2,axis=1)
out2 = out.reshape(1,3,2,2)
print(col.shape)
print(col2.shape)
print(out.shape)
print(out2.shape)
