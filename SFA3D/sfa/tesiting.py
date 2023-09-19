import numpy as np

A = np.array([[1,1,1], [1,1,1]])

w = np.array([1,2])
w = w[:, np.newaxis]

A = A - w

pass
