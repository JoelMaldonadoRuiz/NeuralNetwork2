import numpy as np
import pickle


class Module:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self) -> str:
        return str(self.__class__.__name__)


class Sequential(Module):
    def __init__(self, *modules):
        self.modules = [module for module in modules]
    
    def forward(self, x):
        out = x
        for module in self.modules:
            out = module(out)
        return out
    
    def backward(self, x):
        out = x
        for module in self.modules:
            out = module.backward(out)
        return out

class Model(Module):
    def __init__(self):
        self.layers = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            layer_name = name
            self.layers[layer_name] = value

        super().__setattr__(name, value)

    def state_dict(self):
        state_dict = {}
        for layer_key, layer in self.layers.items():
            if hasattr(layer, "weights"):
                state_dict[f"{layer_key}_weights"] = layer.weights
                state_dict[f"{layer_key}_biases"] = layer.biases
        return state_dict

    def load_state_dict(self, state_dict):
        for name, layer in self.layers.items():
            if hasattr(layer, 'weights'):
                layer.weights = state_dict[name + '_weights']
            if hasattr(layer, 'biases'):
                layer.biases = state_dict[name + '_biases']


class Loss(Module):
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)


class MSE(Loss):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return np.mean(diff ** 2)

    def backward(self, y_pred, y_true):
        self.dout = 2 * (y_pred - y_true)



class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Scalar
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # One Hot
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dout = (-y_true / dvalues) / samples


# class DenseLayer(Module):
#     def __init__(self, n_inputs, n_outputs):
#         self.weights = 0.1 * np.random.randn(n_inputs, n_outputs)
#         self.biases = np.zeros((1, n_outputs))

#         self.dweights = np.zeros_like(self.weights)
#         self.dbiases = np.zeros_like(self.biases)

#     def forward(self, inputs):
#         self.inputs = inputs
#         return np.dot(inputs, self.weights) + self.biases

#     def backward(self, dvalues):
#         self.dweights = np.dot(self.inputs.T, dvalues)
#         self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
#         return np.dot(dvalues, self.weights.T)

class DenseLayer(Module):
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.biases = np.zeros((1, out_features))
        self.dweights = None
        self.dbiases = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.weights) + self.biases
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.weights.T)
        self.dweights = np.dot(self.x.T, dout)
        self.dbiases = np.sum(dout, axis=0, keepdims=True)
        return dx



class ReLU(Module):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dout):
        self.dout = dout.copy()
        self.dout[self.x < 0] = 0
        return self.dout

class SGD_Optimizer():
    def __init__(self, layers, lr=0.1, clip_threshold=1.0):
        self.layers = layers
        self.lr = lr
        self.clip_threshold = clip_threshold

    def zero_grad(self):
        for layer in self.layers.values():
            if hasattr(layer, "weights"):
                layer.dweights = np.zeros_like(layer.dweights)
                layer.dbiases = np.zeros_like(layer.dbiases)

    def step(self):
        for layer in self.layers.values():
            if hasattr(layer, "weights"):
                weight_updates = -self.lr * layer.dweights
                bias_updates = -self.lr * layer.dbiases

                weight_updates = np.clip(
                    weight_updates, -self.clip_threshold, self.clip_threshold)
                bias_updates = np.clip(
                    bias_updates, -self.clip_threshold, self.clip_threshold)

                layer.weights += weight_updates
                layer.biases += bias_updates


class Optimizer_Adam:
    def __init__(self, layers, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.layers = layers
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def zero_grad(self):
        for layer in self.layers.values():
            if hasattr(layer, "weights"):
                layer.dweights = np.zeros_like(layer.dweights)
                layer.dbiases = np.zeros_like(layer.dbiases)


                

    def step(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

        for layer in self.layers.values():
            if hasattr(layer, "weights"):
                if not hasattr(layer, 'weight_cache'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.weight_cache = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)
                    layer.bias_cache = np.zeros_like(layer.biases)

                # Update momentum with current gradients
                layer.weight_momentums = self.beta_1 * \
                    layer.weight_momentums + (1 - self.beta_1) * layer.dweights
                layer.bias_momentums = self.beta_1 * \
                    layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

                # Get corrected momentum self.iteration is 0 at first pass and need to start with 1 here
                weight_momentums_corrected = layer.weight_momentums / \
                    (1 - self.beta_1 ** (self.iterations + 1))
                bias_momentums_corrected = layer.bias_momentums / \
                    (1 - self.beta_1 ** (self.iterations + 1))

                # Update cache with squared current gradients
                layer.weight_cache = self.beta_2 * layer.weight_cache + \
                    (1 - self.beta_2) * layer.dweights**2
                layer.bias_cache = self.beta_2 * layer.bias_cache + \
                    (1 - self.beta_2) * layer.dbiases**2

                # Get corrected cache
                weight_cache_corrected = layer.weight_cache / \
                    (1 - self.beta_2 ** (self.iterations + 1))
                bias_cache_corrected = layer.bias_cache / \
                    (1 - self.beta_2 ** (self.iterations + 1))

                # Vanilla SGD parameter update + normalization with square rooted cache
                layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
                    (np.sqrt(weight_cache_corrected) + self.epsilon)
                layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
                    (np.sqrt(bias_cache_corrected) + self.epsilon)
        
        self.iterations += 1

