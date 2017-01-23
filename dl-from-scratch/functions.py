import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    """
    1) monotone increasing function
        - If a > b, then f(a) > f(b)
        - to reduce calculation, generally omitted in inference level (final output)
    2) sum of outputs is 1
        - It can be translated to percentage.
    """
    if a.ndim == 2:
        a = a.T
        a = a - np.max(a, axis=0)
        y = np.exp(a) / np.sum(np.exp(a), axis=0)

        return y.T

    c = np.max(a)
    exp_a = np.exp(a - c)       # c : to prevent overflow
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def cross_entropy_error(y, t):
    # t should be formatted as one-hot encoding. If you want single label, refer to the book.
    if y.ndim == 1:
        # convert array to matrix
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size