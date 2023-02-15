import numpy as np
from network import *
import time

X = np.random.randn(100, 1, 28, 28)

conv1 = Conv2D(1, 32, 3)
conv2 = Conv2D(32, 32, 3)


input_channels = 1
kernel_size = 3
output_channels = 32

new_h = (28 - kernel_size) // 1+ 1

new_h = (new_h - kernel_size) // 1+ 1



flat = Flatten()
l1 = DenseLayer(32 * 24 * 24, 1)

out = conv1.forward(X)
out = conv2.forward(out)

out = flat.forward(out)
out = l1.forward(out)

mse = MSE()

grad = mse.backward(y_pred=out, y_true=1)
grad = l1.backward(grad)
grad = flat.backward(grad)

grad = conv1.backward(grad)
grad = conv2.backward(grad)


optim = SGD(0.001)

optim.step([l1, conv1, conv2])

