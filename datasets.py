import requests
import os
import gzip
import numpy as np


def get_mnist():
    if not os.path.exists("data/MNIST"):
        os.makedirs("data/MNIST/", exist_ok=True)

        X_train_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        y_train_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
        X_test_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
        y_test_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

        with open("data/MNIST/X_train.gz", "wb") as f:
            f.write(requests.get(X_train_url).content)

        with open("data/MNIST/y_train.gz", "wb") as f:
            f.write(requests.get(y_train_url).content)

        with open("data/MNIST/X_test.gz", "wb") as f:
            f.write(requests.get(X_test_url).content)

        with open("data/MNIST/y_test.gz", "wb") as f:
            f.write(requests.get(y_test_url).content)
    
    with gzip.open('data/MNIST/X_train.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')

        image_data = f.read()
        X_train = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))

    with gzip.open('data/MNIST/y_train.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')

        label_data = f.read()
        y_train = np.frombuffer(label_data, dtype=np.uint8)

    with gzip.open('data/MNIST/X_test.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')

        image_data = f.read()
        X_test = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))

    with gzip.open('data/MNIST/y_test.gz', 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')

        label_data = f.read()
        y_test = np.frombuffer(label_data, dtype=np.uint8)

    return X_train, y_train, X_test, y_test

