import random
import numpy as np

def getPartitions(dataset, trainingNumber, cvNumber):
    random.shuffle(dataset)

    a = int(len(dataset) * trainingNumber)
    b = int(len(dataset) * cvNumber)

    training = dataset[:a]
    cv = dataset[a: a + b]
    test = dataset[a + b:]

    return training, cv, test

def ecm(predicciones, reales):
    return np.sum((predicciones - reales)**2) / len(predicciones)

def readFile (filename):
    return np.loadtxt(filename)

def listToMatrix(pixels, size):
    pixelsTuple = [(x, x, x) for x in pixels]
    return np.asarray(pixelsTuple).reshape(size, size, 3)
