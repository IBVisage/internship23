import numpy as np
import matplotlib.pyplot as plt
from copy import copy




# State model matrix
dt = 0.01
F = np.array([[1,dt,    0.5*dt**2],
              [0,1,     dt],
              [0,0,     1]])

# Process noise matrix
Q = np.array([[dt**4/4, dt**3/2, dt**2/2],
              [dt**3/2, dt**2,   dt     ],
              [dt**2/2, dt,      1      ]])

# Covariance matrix of measurement
# Copromised of sigmas who define measurement uncertanty
sigma_x, sigma_y, sigma_z = 2, 2, 2
R = np.array([[sigma_x, 0, 0],
              [0, sigma_y, 0],
              [0, 0, sigma_z]])


def ukf_predict(x_, sigma_p, F, Q, sigma_q, w, wc, P_nn):
    # Sigma point computation

    # N: dimension of measure space
    N = x_.shape[0]
    # number of sigma points = 2N + 1
    # lam: weight parameter
    # k: scaling paramater; usually 0
    # alpha: determines spread of distribution
    # beta: prior knowledge of distribution - 2 for gaussian

    # x_: mean value
    # X_0: mean sigma point
    X_0 = x_

    # P matrix

    # square root matrix (N + k)Pnn
    P_nn_sqrt = np.linalg.cholesky((N+k) * P_nn)

    # Adding +/- sigma points around mean
    X_nn = [X_0]
    for ii in range(N):
        # Taking 
        X_nn.append(x_ + P_nn_sqrt[:, ii])
    for ii in range(N):
        X_nn.append(x_ - P_nn_sqrt[:, ii])
    
    X_nn = np.array(X_nn).transpose()
    #X_nn = np.reshape(X_nn, ())

    # Sigma point propagation
    # Nonlinear funciton F

    X_apriori = np.matmul(F,X_nn)

    # Mean computation
    x_apriori = np.matmul(X_apriori, w)
    
    # Reshape to column vector
    x_apriori = x_apriori[:, np.newaxis]

    # Covariance computation   A * diag(w) * A'

    P_apriori = np.matmul(np.matmul((X_apriori - x_apriori), wc),    (X_apriori - x_apriori).transpose())  + sigma_q *  Q 


    return X_apriori, x_apriori, P_apriori

def ukf_update(z, X_apriori, x_apriori, P_apriori, w, wc, R, measure_func):

    # Measurement space of state sigma point
    Z = measure_func(X_apriori)

    # Mean of measurement space
    z_mean = np.matmul(Z, w)
    # Reshape to column vector
    z_mean = z_mean[:, np.newaxis]

    # Calculation covariance of measurement space
    P_z = np.matmul(np.matmul((Z - z_mean), np.diag(w)),    (Z - z_mean).transpose())  + R

    # Calculating cross-covariance of state and measurement
    P_xz = np.matmul(np.matmul((X_apriori - x_apriori), wc), (Z - z_mean).transpose())

    # Kalman gain
    P_zinv = np.linalg.inv(P_z)
    K = np.matmul(P_xz, P_zinv)

    x_aposteriori = x_apriori + np.matmul(K, z - z_mean)
    P_aposteriori = P_apriori - np.matmul(np.matmul(K, P_z), K.transpose())
    return x_aposteriori, P_aposteriori
    pass

def meas_func(X):
    z = X
    return z

x_ = np.array([0])
# New measure


N = x_.shape[0]

beta = 2
k = 0
alpha = 0.1
lam = alpha**2 * (N + k) - N

# Sigma point weights
w0 = lam / (N+ k)
w0c = lam / (N+ lam) + (1-alpha**2) + beta
wi = 1 / (2 * (N + lam))

w = np.array([w0])
w = np.append(w, wi*np.ones(2 * N))

wc = np.array([w0c])
wc = np.append(wc, wi * np.ones(2 * N))
wc = wc * np.eye(1 + 2 * N)
    

sigma_p = 100
sigma_q = 10



x_true = np.linspace(0,3,300)
y_true = x_true**np.sinc(x_true)

x_predict = []


P_nn = sigma_p * np.eye(N)

for ii in range(len(x_true)):
    if ii == 0:
        X_apriori, x_apriori, P_apriori = ukf_predict(x_, sigma_p, F, Q, sigma_q, w, wc, P_nn)
    else:
        X_apriori, x_apriori, P_apriori = ukf_predict(x_apriori, sigma_p, F, Q, sigma_q, w, wc, P_apriori)

    x_apriori, P_apriori = ukf_update(z, X_apriori, x_apriori, P_apriori, w, wc, R, measure_func= meas_func)

plt.figure()
plt.plot(x_true, y_true)
plt.show()

# Model procesnog šuma Q i model matrice F i mjerna funkcija h



