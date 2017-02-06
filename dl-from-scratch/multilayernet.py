import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist


class MultiLayerNet:
    def __init__(self, input_dim, hidden_dims, output_dim, batch_size):
        self.x = None
        self.p = None
        self.num_layer = len(hidden_dims)
        self.layers = []
        self.batch_size = batch_size

        earlier_dim = input_dim
        for i in range(self.num_layer):
            self.layers += [Affine(earlier_dim, hidden_dims[i]), Relu(hidden_dims[i])]
            earlier_dim = hidden_dims[i]

        self.layers.append(Affine(earlier_dim, output_dim))  # output layer
        self.last_layer = SoftmaxWithLoss(output_dim, batch_size)
    def accuracy(self, p, t):
        return np.sum(p.argmax(axis=1) == t.argmax(axis=1)) / p.shape[0]

    def predict(self, x):
        self.x = x
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        self.p = out

        return out

    def gradient_update(self):
        dout = self.last_layer.backward()

        for layer in reversed(self.layers):
            dout = layer.backward(dout, True)


    def loss(self, t):
        return self.last_layer.forward(self.p, t)


class Affine:
    def __init__(self, input_dim, num_node):
        self.num_node = num_node
        self.x = None
        self.W = 0.01 * np.random.randn(input_dim, num_node)
        self.b = np.zeros(num_node)
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout, update=False, learning_rate=0.1):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        if update:
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db

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

    def backward(self, dout, update=False):
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

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)
batch_size = 100
net = MultiLayerNet(784, [50], 10, batch_size)
loss_list = []
for i in range(10000):
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    p = net.predict(x_batch)
    loss_list.append(net.loss(t_batch))
    print(net.accuracy(p, t_batch))
    net.gradient_update()

x = np.arange(0, 10000)
plt.plot(x, loss_list)
plt.show()