from RedesNeuronales import *
from utilities import *
import numpy as np
import pandas as pd
from scipy import optimize as op

print("Lectura de mnist_train")
dataset_train = pd.read_csv('fashion-mnist_train.csv')

print("Lectura de mnist_test")
dataset_test = pd.read_csv('fashion-mnist_test.csv')

# Normalizar los datasets de X (train y test)
x_1 = dataset_train.iloc[:, 1:] / 1000.0
m1, n1 = x_1.shape

x_2 = dataset_test.iloc[:, 1:] / 1000.0
m2, n2 = x_2.shape

# Union de ambos datasets
X = np.vstack((
    x_1,
    x_2
))
m, n = X.shape

# Datasets de Y (train y test)
y_1 = np.asarray(dataset_train.iloc[:, 0])
y_1 = y_1.reshape(m1, 1)

y_2 = np.asarray(dataset_test.iloc[:, 0])
y_2 = y_2.reshape(m2, 1)

# Union de ambos datasets
y = np.vstack((
    y_1,
    y_2
))
y = y.reshape(m, 1)

# Vector -> Matriz de categorizacion
Y = (y == np.array(range(10))).astype(int)

# Estructura de la red neuronal
NEURAL_NET = np.array([
    n,
    130,
    10
])

# Se obtienen las shapes de las thetas
theta_shapes = np.hstack((
    NEURAL_NET[1:].reshape(len(NEURAL_NET) - 1, 1),
    (NEURAL_NET[:-1] + 1).reshape(len(NEURAL_NET) - 1, 1)
))

# Matrix -> array de thetas
flat_thetas = flatten_list_of_arrays([
    np.random.rand(*theta_shape)
    for theta_shape in theta_shapes
])

# Optimizacion para encontrar thetas
print("\n---------------- OPTIMIZING ----------------\n")
result = op.minimize(
    fun = cost_function,
    x0 = flat_thetas,
    args = (theta_shapes, X, Y),
    method = 'L-BFGS-B',
    jac = back_propagation,
    options = {'disp': True, 'maxiter': 3000}
)
print("\n---------------- OPTIMIZED ----------------\n")

# Se guarda el resultado de thetas optimizadas
np.savetxt('final.txt', result.x)
