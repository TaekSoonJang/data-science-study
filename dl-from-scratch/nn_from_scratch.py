import numpy as np
from dataset.mnist import load_mnist
from functions import sigmoid, softmax, cross_entropy_error

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

print(x_train.shape)    # 60,000 * 784
print(t_train.shape)    # 60,000 * 10

def create_neural_net(input_dim, hidden_dim, output_dim, weight_init_std=0.01):
    # TODO: weight_init_std
    # if weight_init_std is not used, differentiation cannot be worked because value can be under delta
    b1 = np.zeros(hidden_dim)
    w1 = weight_init_std * np.random.randn(input_dim, hidden_dim)
    b2 = np.zeros(output_dim)
    w2 = weight_init_std * np.random.randn(hidden_dim, output_dim)

    return {
        'b1': b1,
        'w1': w1,
        'b2': b2,
        'w2': w2
    }

def numerical_gradient(f, x):
    d = 1e-4

    grad = {}
    grad['b1'] = np.zeros_like(x['b1'])
    grad['w1'] = np.zeros_like(x['w1'])
    grad['b2'] = np.zeros_like(x['b2'])
    grad['w2'] = np.zeros_like(x['w2'])

    for param in ('b1', 'b2', 'w1', 'w2'):
        it = np.nditer(x[param], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            i = it.multi_index
            tmp = x[param][i]
            x[param][i] = tmp + d
            fx1 = f(x)
            x[param][i] = tmp - d
            fx2 = f(x)
            x[param][i] = tmp

            grad[param][i] = (fx1 - fx2) / (2 * d)
            it.iternext()

    return grad


def predict(net, x):
    a1 = np.dot(x, net['w1']) + net['b1']
    z1 = sigmoid(a1)

    a2 = np.dot(z1, net['w2']) + net['b2']
    y = softmax(a2)

    return y

def accuracy(x, t):
    return np.sum(x.argmax(axis=1) == t.argmax(axis=1)) / x.shape[0]


net = create_neural_net(784, 50, 10)
lr = 0.2
for i in range(100):
    sampling_idx = np.random.choice(x_train.shape[0], 100)
    x_batch = x_train[sampling_idx]
    t_batch = t_train[sampling_idx]
    f = lambda net: cross_entropy_error(predict(net, x_batch), t_batch)

    grad = numerical_gradient(f, net)
    net['b1'] -= grad['b1'] * lr
    net['b2'] -= grad['b2'] * lr
    net['w1'] -= grad['w1'] * lr
    net['w2'] -= grad['w2'] * lr

    p = predict(net, x_train)
    print(cross_entropy_error(p, t_train))

