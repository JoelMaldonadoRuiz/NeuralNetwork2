import numpy as np

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
        """
            Performs a forward convolution.
           
            Parameters:
            - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
            Returns:
            - out: previous layer convolved.
        """
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
        """
            Distributes error from previous layer to convolutional layer and
            compute error for the current convolutional layer.
            Parameters:
            - dout: error from previous layer.
            
            Returns:
            - dX: error of the current convolutional layer.
            - self.dweights: weights gradient.
            - self.dbiases: bias gradient.
        """
        X, X_col, w_col = self.cache
        in_n, _, _, _ = X.shape

        self.dbiases = np.sum(dout, axis=(0,2,3))

        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, in_n))
        dout = np.concatenate(dout, axis=-1)

        dX_col = w_col.T @ dout
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dw_col = dout @ X_col.T
        # Reshape back to image (col2im).
        dX = self._col2im(dX_col, X.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
        # Reshape dw_col into dw.
        self.dweights = dw_col.reshape((dw_col.shape[0], self.in_channels, self.kernel_size, self.kernel_size))
                
        return dX
    
    def _get_indices(self, X_shape, HF, WF, stride, pad):
        """
            Returns index matrices in order to transform our input image into a matrix.
            Parameters:
            -X_shape: Input image shape.
            -HF: filter height.
            -WF: filter width.
            -stride: stride value.
            -pad: padding value.
            Returns:
            -i: matrix of index i.
            -j: matrix of index j.
            -d: matrix of index d. 
                (Use to mark delimitation for each channel
                during multi-dimensional arrays indexing).
        """
        n, in_c, in_h, in_w = X_shape

        out_h = int((in_h + 2 * pad - HF) / stride) + 1
        out_w = int((in_w + 2 * pad - WF) / stride) + 1
    
        level1 = np.repeat(np.arange(HF), WF)
        level1 = np.tile(level1, in_c)
        everyLevels = stride * np.repeat(np.arange(out_h), out_w)
        i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

        slide1 = np.tile(np.arange(WF), HF)
        slide1 = np.tile(slide1, in_c)
        everySlides = stride * np.tile(np.arange(out_w), out_h)
        j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

        d = np.repeat(np.arange(in_c), HF * WF).reshape(-1, 1)

        return i, j, d

    def _im2col(self, X, HF, WF, stride, pad):
        """
            Transforms our input image into a matrix.
            Parameters:
            - X: input image.
            - HF: filter height.
            - WF: filter width.
            - stride: stride value.
            - pad: padding value.
            Returns:
            -cols: output matrix.
        """
        X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
        i, j, d = self._get_indices(X.shape, HF, WF, stride, pad)

        # Multi-dimensional arrays indexing.
        cols = X_padded[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)
        return cols

    def _col2im(self, dX_col, X_shape, HF, WF, stride, pad):
        """
            Transform our matrix back to the input image.
            Parameters:
            - dX_col: matrix with error.
            - X_shape: input image shape.
            - HF: filter height.
            - WF: filter width.
            - stride: stride value.
            - pad: padding value.
            Returns:
            -x_padded: input image with error.
        """

        N, C, H, W = X_shape

        # Add padding if needed.
        H_padded, W_padded = H + 2 * pad, W + 2 * pad
        X_padded = np.zeros((N, C, H_padded, W_padded))
        
        i, j, d = self._get_indices(X_shape, HF, WF, stride, pad)
        dX_col_reshaped = np.array(np.hsplit(dX_col, N))
        np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)

        # Remove padding from new image if needed.
        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[pad:-pad, pad:-pad, :, :]


X = np.random.randn(100, 1, 28, 28)

conv1 = Conv2D(1, 32, 3)

out = conv1.forward(X)

print(out.shape)
