import numpy as np
from functools import reduce

mnist = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

flatten_list_of_arrays = lambda list_of_arrays: reduce (
    lambda acc, v: np.array([*acc.flatten(), *v.flatten()]),
    list_of_arrays
)

def inflate_matrixes(flat_thetas, shapes):
    layers = len(shapes) + 1
    sizes = [shape[0] * shape[1] for shape in shapes]
    steps = np.zeros(layers, dtype=int)

    for i in range(layers - 1):
        steps[i + 1] = steps[i] + sizes[i]

    return [
        flat_thetas[steps[i]: steps[i + 1]].reshape(*shapes[i])
        for i in range(layers - 1)
    ]

def feed_forward(thetas, X):
    a = [np.asarray(X)]

    for i in range(len(thetas)):
        a.append(
            sigmoid(
                np.matmul(
                    np.hstack((
                        np.ones(len(X)).reshape(len(X), 1),
                        a[i]
                    )), thetas[i].T
                )
            )            
        )
    return a

def sigmoid(z):
    a = [(1 / (1 + np.exp(-x))) for x in z]
    return np.asarray(a).reshape(z.shape)

def cost_function(flat_thetas, shapes, X, Y):
    a = feed_forward(
        inflate_matrixes(flat_thetas, shapes),
        X
    )
    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len(X)

def back_propagation(flat_thetas, shapes, X, Y):
    m, layers = len(X), len(shapes) + 1
    thetas = inflate_matrixes(flat_thetas, shapes)
    a = feed_forward(thetas, X) # 2.2
    deltas = [*range(layers - 1), a[-1] - Y]

    # 2.4
    for i in range(layers - 2, 0, -1):
        deltas[i] = (deltas[i + 1] @ np.delete(thetas[i], 0, 1)) * (a[i] * (1 - a[i]))

    Deltas = []
    for i in range(layers - 1):
        Deltas.append(
            (deltas[i + 1].T
            @
            np.hstack((
                np.ones(len(a[i])).reshape(len(a[i]), 1),
                a[i]
            ))) / m
        )
    Deltas = np.asarray(Deltas)

    return flatten_list_of_arrays(
        Deltas
    )
    





