#! /usr/bin/python3
import sys,os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
	pil_img = Image.fromarray(np.uint8(img))
	pil_img.show()

(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False,one_hot_label=True)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28) ##flatten으로 인해 1차원 784열이기때문에 다시 28x28 2차원으로 변경
print(img.shape)

img_show(img)

