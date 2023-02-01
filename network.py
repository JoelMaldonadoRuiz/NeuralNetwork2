import numpy as np
import os

class BaseModel:
    def __init__(self):
        self.layers = []
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def add(self, layer):
        self.layers.append(layer)

    def state_dict(self):
        data = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                data[f"layer_{i}_weights"] = layer.weights
                data[f"layer_{i}_biases"] = layer.biases

        return data
    
    def load_state_dict(self, state_dict):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                layer.weights = state_dict[f"layer_{i}_weights"]
                layer.biases = state_dict[f"layer_{i}_biases"]

    def save(self, filename):
        data = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                data[f"layer_{i}_weights"] = layer.weights
                data[f"layer_{i}_biases"] = layer.biases
        np.savez(filename, **data)
        
    def load(self, filename):
        if os.path.exists(filename):
            data = np.load(filename)
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'weights'):
                    layer.weights = data[f"layer_{i}_weights"]
                    layer.biases = data[f"layer_{i}_biases"]
        else:
            raise FileNotFoundError(f"No file found at {filename}")




class Layer:
    def __init__(self):
        self.params = []
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    

class DenseLayer(Layer):
    def __init__(self, n_inputs, n_outputs, he_normal=True):
        self.weights = 0.1 * np.random.randn(n_inputs, n_outputs)
        self.biases = np.zeros((1, n_outputs))

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

        if he_normal:
            self.weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        return np.dot(dvalues, self.weights.T)


class Loss:
    def __init__(self):
        pass
    
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
    def backward(self, y_pred, y_true):
        raise NotImplementedError


class MSE(Loss):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return np.mean(np.square(diff))

    def backward(self, y_pred, y_true):
        self.dout = 2 * (y_pred - y_true) / y_pred.shape[0]
        return self.dout

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - np.sum(y_true * np.log(y_pred), axis=-1)

    def backward(self, y_pred, y_true):
        self.dout = y_pred - y_true
        return self.dout




class Optimizer:
    def __init__(self):
        pass
    
    def step(self, layers):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            layer.weights -= self.lr * layer.dweights
            layer.biases -= self.lr * layer.dbiases


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = dict()
        self.v = dict()

    def step(self, layers):
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights'):
                # Initialize mean and variance with zeros
                if i not in self.m:
                    self.m[i] = dict()
                    self.v[i] = dict()
                    self.m[i]['weights'] = np.zeros_like(layer.weights)
                    self.m[i]['biases'] = np.zeros_like(layer.biases)
                    self.v[i]['weights'] = np.zeros_like(layer.weights)
                    self.v[i]['biases'] = np.zeros_like(layer.biases)

                # Calculate the mean and variance
                self.m[i]['weights'] = self.beta_1 * self.m[i]['weights'] + (1 - self.beta_1) * layer.dweights
                self.v[i]['weights'] = self.beta_2 * self.v[i]['weights'] + (1 - self.beta_2) * np.square(layer.dweights)
                m_hat = self.m[i]['weights'] / (1 - self.beta_1**(i + 1))
                v_hat = self.v[i]['weights'] / (1 - self.beta_2**(i + 1))

                # Update the layer's weights and biases
                layer.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                self.m[i]['biases'] = self.beta_1 * self.m[i]['biases'] + (1 - self.beta_1) * layer.dbiases
                self.v[i]['biases'] = self.beta_2 * self.v[i]['biases'] + (1 - self.beta_2) * np.square(layer.dbiases)
                m_hat = self.m[i]['biases'] / (1 - self.beta_1**(i + 1))
                v_hat = self.v[i]['biases'] / (1 - self.beta_2**(i + 1))
                layer.biases -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)



class Activation:
    def __init__(self):
        pass
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dout[self.x <= 0] = 0
        return dout


class GeLU(Activation):
    def forward(self, x):
        self.x = x
        return x * 0.5 * (1 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
    
    def backward(self, dout):
        return dout * (0.5 * (1 + np.tanh((np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))))) * (1 - 0.5 * np.power(np.tanh((np.sqrt(2 / np.pi) * (self.x + 0.044715 * np.power(self.x, 3)))), 2))



class Dataset:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X))
        self.current_index = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= len(self.X):
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.indexes)
            raise StopIteration
        else:
            start = self.current_index
            end = self.current_index + self.batch_size
            self.current_index += self.batch_size
            return self.X[self.indexes[start:end]], self.y[self.indexes[start:end]]