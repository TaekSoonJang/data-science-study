import sys, os
sys.path.append(os.curdir)

import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pylab as plt
from PIL import Image
import pickle


# activation function

def step_function(x):
    """
    :param x: numpy array type
    :return:
    """
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# sigma function

def identity_function(x):
    return x


def softmax(a):
    """
    1) monotone increasing function
        - If a > b, then f(a) > f(b)
        - to reduce calculation, generally omitted in inference level (final output)
    2) sum of outputs is 1
        - It can be translated to percentage.
    """
    c = np.max(a)
    exp_a = np.exp(a - c)       # c : to prevent overflow
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 =  np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

    return x_test, t_test

predict = forward

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))