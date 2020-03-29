import numpy as np

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feed_forward(thetas, X):
    a = [X]

    for i in range(len(thetas)):
        a.append(
            sigmoid(
                np.hstack((
                    np.ones(len(X)).reshape(len(X), 1),
                    a[i]
                )) @ thetas[i].T
            )            
        )
    return a

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
        deltas.insert(
            0,
            (thetas[i] @ deltas[i + 1])
            *
            (a[i] * (1 - a[i]))
        )
    
def inflate_matrixes(flat_thetas, shapes):
    thetas = []
    for shape in shapes:
        temp = flat_thetas[: shape[0] * shape[1]]
        flat_thetas = flat_thetas[shape[0] * shape[1]:]

        temp = temp.reshape(shape[0], shape[1])
        thetas.append(temp)
    return thetas





