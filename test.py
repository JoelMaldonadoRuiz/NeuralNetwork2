import numpy as np
import matplotlib.pyplot as plt
import pickle
from datasets import get_mnist
from network import *
from mnist import MNIST_Model

def one_hot_encode(scalar, num_classes):
    return np.eye(num_classes)[scalar]

X_train, y_train, X_test, y_test = get_mnist()

X_train = X_train.reshape(-1, 28 * 28) / 255.0
X_test = X_test.reshape(-1, 28 * 28) / 255.0

model = MNIST_Model()
model.load("MNIST_Model_Acc_98.54.npz")

pickle.dump(model, open('MNIST_MODEL_FULL_98.54.mdl', 'wb'))

# for i in range(1000):
i = 0
img = X_test[i]
y_pred = model.forward(img).argmax()

plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f"Pred: {y_pred} | Actual: {y_test[i]}")
plt.show()
