import numpy as np


class MultiLayerNet:
    def __init__(self, input_dim, num_nodes_arr, batch_size):
        self.num_layer = len(num_nodes_arr)
        self.layers = []
        self.batch_size = batch_size
        self.last_layer = SoftmaxWithLoss(num_nodes_arr[self.num_layer - 1], batch_size)

        earlier_dim = input_dim
        for i in range(self.num_layer):
            self.layers += [Affine(earlier_dim, num_nodes_arr[i]), Relu(num_nodes_arr[i])]
            earlier_dim = num_nodes_arr[i]
        self.layers.append(Affine(earlier_dim, self.num_layer - 1))  # output layer

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def loss(self, x, t):
        return self.last_layer.loss(self.forward(x), t)


class Affine:
    def __init__(self, input_dim, num_node):
        self.num_node = num_node
        self.x = None
        self.W = np.random.randn(input_dim, num_node)
        self.b = np.zeros(num_node)
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class Relu:
    def __init__(self, num_node):
        self.num_node = num_node
        self.mask = None

    def forward(self, x):
        mask = x <= 0
        self.mask = mask

        ret = x.copy()
        ret[mask] = 0
        return ret

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class SoftmaxWithLoss:
    def __init__(self, output_dim, batch_size):
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.x = None
        self.y = None
        self.t = None

    def softmax(self, x):
        if x.ndim == 1:
            x = x.reshape(1, x.size)
        max = np.max(x, axis=1)            # prevent overflow
        max = max.reshape(max.size, 1)

        sum = np.sum(np.exp(x - max), axis=1)
        sum = sum.reshape(sum.size, 1)
        return np.exp(x - max) / sum

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            y = y.reshape(1, y.size)
        delta = 1e-7
        log_x = np.log(y + delta)               # prevent log(0)
        return -np.sum(t * log_x) / y.shape[0]

    def forward(self, x, t):
        self.x = x
        self.t = t
        y = self.softmax(x)
        self.y = y
        return self.cross_entropy_error(y, t)

    def backward(self):
        return (self.y - self.t) / self.batch_size


a = MultiLayerNet(2, [3], 10)
print(a.forward([[1, 2]]))