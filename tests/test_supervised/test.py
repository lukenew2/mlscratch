import numpy as np 
from mlscratch.supervised.regression import LinearRegression


X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X ** 2 + X + 2

reg = LinearRegression(degree=2)

reg.fit(X, y)


