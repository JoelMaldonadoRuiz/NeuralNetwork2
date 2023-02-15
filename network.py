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

    def save(self, filename):
        data = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                data[f"layer_{i}_weights"] = layer.weights
                data[f"layer_{i}_biases"] = layer.biases
               
        data['summary'] = self.summary()
        np.savez(filename, **data)
       
    def load(self, filename):
        if os.path.exists(filename):
            data = np.load(filename)
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'weights'):
                    layer.weights = data[f"layer_{i}_weights"]
                    layer.biases = data[f"layer_{i}_biases"]
            return data['summary']
        else:
            raise FileNotFoundError(f"No file found at {filename}")
   
    def state_dict(self):
        data = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                data[f"layer_{i}_weights"] = layer.weights
                data[f"layer_{i}_biases"] = layer.biases
        return data
   
    def load_state_dict(self, data):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                layer.weights = data[f"layer_{i}_weights"]
                layer.biases = data[f"layer_{i}_biases"]
        return data['summary']
       
    def summary(self):
        return "\n".join([str(layer) for layer in self.layers])



class Layer:
    def __init__(self):
        pass
   
    def forward(self, x):
        raise NotImplementedError
   
    def backward(self, grad):
        raise NotImplementedError
   
    def __repr__(self):
        raise NotImplementedError
   

class DenseLayer(Layer):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
       
        self.weights = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_outputs))

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)


    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
       
        return np.dot(dvalues, self.weights.T)
       
    def __repr__(self):
        return f"Dense({self.n_inputs}, {self.n_outputs})"


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





class Flatten(Layer):
    def __init__(self):
        pass
       
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
   
    def backward(self, dvalues):
        self.dout = dvalues.reshape(self.input_shape)
        return self.dout


class Debug(Layer):
    def __init__(self):
        pass
       
    def forward(self, x):
        print(f"Input: {x.shape}")
        return x
   
    def backward(self, dvalues):
        return dvalues
    


class Conv2D():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size) * np.sqrt(1. / (self.kernel_size))
        self.biases = np.random.randn(self.out_channels) * np.sqrt(1. / self.out_channels)

        self.dweights = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.dbiases = np.zeros((self.out_channels))

        self.cache = None

    def forward(self, X):
        in_n, in_c, in_h, in_w = X.shape

        n_C = self.out_channels
        n_H = int((in_h + 2 * self.padding - self.kernel_size)/ self.stride) + 1
        n_W = int((in_w + 2 * self.padding - self.kernel_size)/ self.stride) + 1
        
        X_col = self._im2col(X, self.kernel_size, self.kernel_size, self.stride, self.padding)
        w_col = self.weights.reshape((self.out_channels, -1))
        b_col = self.biases.reshape(-1, 1)
        
        # Perform matrix multiplication.
        out = np.dot(w_col, X_col) + b_col
        
        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, in_n)).reshape((in_n, n_C, n_H, n_W))
        
        self.cache = X, X_col, w_col
        return out

    def backward(self, dout):
        X, X_col, w_col = self.cache
        in_n, _, _, _ = X.shape

        self.dbiases = np.sum(dout, axis=(0,2,3))

        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, in_n))
        dout = np.concatenate(dout, axis=-1)

        dX_col = np.dot(w_col.T, dout)
        dw_col = np.dot(dout, X_col.T)
        dX = self._col2im(dX_col, X.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)

        self.dweights = dw_col.reshape((dw_col.shape[0], self.in_channels, self.kernel_size, self.kernel_size))
                
        return dX
    
    def _get_indices(self, X_shape, h, w, stride, pad):
        n, in_c, in_h, in_w = X_shape

        out_h = int((in_h + 2 * pad - h) / stride) + 1
        out_w = int((in_w + 2 * pad - w) / stride) + 1
    
        level1 = np.repeat(np.arange(h), w)
        level1 = np.tile(level1, in_c)
        everyLevels = stride * np.repeat(np.arange(out_h), out_w)
        i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

        slide1 = np.tile(np.arange(w), h)
        slide1 = np.tile(slide1, in_c)
        everySlides = stride * np.tile(np.arange(out_w), out_h)
        j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

        d = np.repeat(np.arange(in_c), h * w).reshape(-1, 1)

        return i, j, d

    def _im2col(self, X, h, w, stride, pad):
        X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
        i, j, d = self._get_indices(X.shape, h, w, stride, pad)

        # Multi-dimensional arrays indexing.
        cols = X_padded[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)
        return cols

    def _col2im(self, dX_col, X_shape, h, w, stride, pad):
        N, C, H, W = X_shape

        # Add padding if needed.
        H_padded, W_padded = H + 2 * pad, W + 2 * pad
        X_padded = np.zeros((N, C, H_padded, W_padded))
        
        i, j, d = self._get_indices(X_shape, h, w, stride, pad)
        dX_col_reshaped = np.array(np.hsplit(dX_col, N))
        np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)

        # Remove padding from new image if needed.
        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[pad:-pad, pad:-pad, :, :]




class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.biases = np.zeros((out_channels, 1, 1)) if bias else None
   
    def forward(self, dvalues):
        self.last_input = dvalues
   
        n, c, h, w = dvalues.shape
        new_img_h = (h - self.kernel_size + 2*self.padding) // self.stride + 1
        new_img_w = (w - self.kernel_size + 2*self.padding) // self.stride + 1
   
        # Pad input
        input_padded = np.pad(dvalues, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
   
        # Reshape filters to perform dot product
        filters_reshaped = self.weights.reshape(self.out_channels, -1)
   
        # Initialize output
        outputs = np.zeros((n, self.out_channels, new_img_h, new_img_w))
   
        # Perform convolution using dot product
        for h in range(new_img_h):
            for w in range(new_img_w):
                h_shift, w_shift = h * self.stride, w * self.stride
                input_patch = input_padded[:, :, h_shift:h_shift+self.kernel_size, w_shift:w_shift+self.kernel_size].reshape(n, -1)
                outputs[:, :, h, w] = np.dot(input_patch, filters_reshaped.T)
   
        # Add bias term
        if self.bias is not None:
            outputs += self.biases.reshape(1, -1, 1, 1)
   
        return outputs

    def backward(self, grad_output):
        n, c_out, h_out, w_out = grad_output.shape
        _, c_in, h, w = self.last_input.shape
    
        # Initialize gradients
        self.dinputs = np.zeros((n, c_in, h, w))
        self.dweights = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.dbiases = np.zeros((self.out_channels, 1, 1)) if self.bias else None
    
        # Pad input
        input_padded = np.pad(self.last_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        grad_input_padded = np.zeros_like(input_padded)
    
        # Reshape filters to perform dot product
        filters_reshaped = self.weights.reshape(self.out_channels, -1)
    
        # Compute gradients
        for h in range(h_out):
            for w in range(w_out):
                h_shift, w_shift = h * self.stride, w * self.stride
                input_patch = input_padded[:, :, h_shift:h_shift+self.kernel_size, w_shift:w_shift+self.kernel_size].reshape(n, -1)
                grad_output_patch = grad_output[:, :, h, w].reshape(n, -1)
                grad_input_patch = np.dot(grad_output_patch, filters_reshaped).reshape(n, self.in_channels, self.kernel_size, self.kernel_size)
                self.dweights += np.dot(grad_output_patch.T, input_patch).reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
                grad_input_padded[:, :, h_shift:h_shift+self.kernel_size, w_shift:w_shift+self.kernel_size] += grad_input_patch
    
        # Remove padding from gradients
        self.dinputs = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    
        # Compute bias gradients
        if self.bias is not None:
            # grad_biases = np.sum(grad_output, axis=(0, 2, 3)).reshape(self.out_channels, 1, 1)
            grad_output = grad_output.reshape(n, self.out_channels, -1)
            self.dbiases = np.sum(grad_output, axis=(0, 2)).reshape(self.out_channels, 1, 1)

        return self.dinputs



class MaxPool2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.cache = None

    def forward(self, X):
        self.cache = X
        n, c, h, w = X.shape
        ph, pw = self.pool_size
        oh, ow = h//ph, w//pw
        out = np.zeros((n, c, oh, ow))
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = np.amax(X[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw], axis=(2, 3))
        return out

    def backward(self, dout):
        n, c, oh, ow = dout.shape
        ph, pw = self.pool_size
        _, _, h, w = self.cache.shape
        dx = np.zeros_like(self.cache)
        for i in range(oh):
            for j in range(ow):
                window = self.cache[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                m = np.amax(window, axis=(2, 3), keepdims=True)
                mask = (window == m)
                dx[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw] += mask * (dout[:, :, i, j])[:, :, None, None]
        return dx





# class MaxPooling2D(Layer):
#     def __init__(self, pool_size, stride=None):
#         self.pool_size = pool_size
#         self.stride = stride or pool_size
#         self.last_input = None

#     def forward(self, x):
#         n, c, h, w = x.shape
#         new_h = (h - self.pool_size) // self.stride + 1
#         new_w = (w - self.pool_size) // self.stride + 1
#         out = np.zeros((n, c, new_h, new_w))
#         for i in range(new_h):
#             for j in range(new_w):
#                 window = x[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
#                 out[:, :, i, j] = np.max(window, axis=(2, 3))
#         self.last_input = x
#         return out

#     def backward(self, grad):
#         n, c, h, w = self.last_input.shape
#         new_h = (h - self.pool_size) // self.stride + 1
#         new_w = (w - self.pool_size) // self.stride + 1
#         out = np.zeros_like(self.last_input)
#         for i in range(new_h):
#             for j in range(new_w):
#                 window = self.last_input[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
#                 max_vals = np.max(window, axis=(2, 3), keepdims=True)
#                 mask = (window == max_vals)
#                 out[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size] += mask * grad[:, :, i, j, np.newaxis, np.newaxis]
#         return out

class MaxPooling2D(Layer):
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride or pool_size
   
    def forward(self, input):
        self.last_input = input
        n, c, h, w = input.shape
        new_h = (h - self.pool_size) // self.stride + 1
        new_w = (w - self.pool_size) // self.stride + 1
        self.mask = np.zeros_like(input)
        output = np.zeros((n, c, new_h, new_w))
        for i in range(new_h):
            for j in range(new_w):
                h_start, w_start = i*self.stride, j*self.stride
                h_end, w_end = h_start+self.pool_size, w_start+self.pool_size
                input_slice = input[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.amax(input_slice, axis=(2, 3))
                max_mask = (input_slice == output[:, :, i, j].reshape(n, c, 1, 1))
                self.mask[:, :, h_start:h_end, w_start:w_end] = max_mask
        return output
   
    def backward(self, output_error):
        input_error = np.zeros_like(self.last_input)
        n, c, new_h, new_w = output_error.shape
        for i in range(new_h):
            for j in range(new_w):
                h_start, w_start = i*self.stride, j*self.stride
                h_end, w_end = h_start+self.pool_size, w_start+self.pool_size
                input_slice = self.last_input[:, :, h_start:h_end, w_start:w_end]
                max_mask = self.mask[:, :, h_start:h_end, w_start:w_end]
                grad = output_error[:, :, i, j].reshape(n, c, 1, 1)
                input_error[:, :, h_start:h_end, w_start:w_end] += grad * max_mask
        return input_error


class AveragePooling2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.cache = None
    
    def forward(self, X):
        N, C, self.H, self.W = X.shape
        kh, kw = self.pool_size, self.pool_size
        new_h = int(self.H / kh)
        new_w = int(self.W / kw)
        X_reshaped = X.reshape(N, C, kh, kw, new_h, new_w)
        out = X_reshaped.mean(axis=(2, 3))
        self.cache = (X_reshaped, out)
        return out

    def backward(self, dout):
        X_reshaped, out = self.cache
        N, C, kh, kw, new_h, new_w = X_reshaped.shape
        dX_reshaped = np.zeros_like(X_reshaped)
        dX_reshaped.reshape(N, C, kh*kw, new_h, new_w)[range(N), :, :, :, :] = \
            dout[:, :, np.newaxis, :, :] / (kh*kw)
        dX = dX_reshaped.reshape(N, C, self.H, self.W)
        return dX




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
            if hasattr(layer, 'weights'):
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
    
    def __len__(self):
        return len(self.X) // self.batch_size