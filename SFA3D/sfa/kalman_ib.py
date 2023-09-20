import numpy as np
import matplotlib.pyplot as plt
from copy import copy




# State model matrix
dt = 1
F = np.array([[1,dt,    0.5*dt**2],
              [0,1,     dt],
              [0,0,     1]])

# Process noise matrix
Q = np.array([[dt**4/4, dt**3/2, dt**2/2],
              [dt**3/2, dt**2,   dt     ],
              [dt**2/2, dt,      1      ]])

# Covariance matrix of measurement
# Copromised of sigmas who define measurement uncertanty
sigma_r, sigma_phi = 5, 0.0087
R = np.array([[sigma_r**2, 0],
              [0, sigma_phi**2]])



F = np.block([[F, np.zeros((3,3))],
              [np.zeros((3,3)), F]])

Q = np.block([[Q, np.zeros((3,3))],
              [np.zeros((3,3)), Q]]) 

# R = np.block([[R, np.zeros((3,3))],
#               [np.zeros((3,3)), R]]) 


def compute_mean_vector(X, w):
    L = X.shape[1]
    X_w = np.zeros(X.shape[0])


    for ii in range(L):
        X_w +=  (w[ii] / L) * X[:, ii] 

    return X_w

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
    # P_nn_sqrt = np.linalg.cholesky((N+k) * P_nn)
    P_nn_sqrt = np.linalg.cholesky(3 * P_nn)

    # Adding +/- sigma points around mean
    X_nn = [X_0]
    for ii in range(N):
        # Taking 
        X_nn.append(x_ + P_nn_sqrt[:, ii])
    for ii in range(N):
        X_nn.append(x_ - P_nn_sqrt[:, ii])
    
    X_nn = np.array(X_nn).transpose()

    # Sigma point propagation
    # Nonlinear funciton F

    X_apriori = F @ X_nn

    # Mean computation
    # x_apriori = compute_mean_vector(X_apriori, w)
    x_apriori = X_apriori @ w
    # x_apriori = np.matmul(X_apriori, w)
    
    # Reshape to column vector
    # x_apriori = x_apriori[:, np.newaxis]

    # Covariance computation   A * diag(w) * A'

    P_apriori = (X_apriori - x_apriori[:, np.newaxis]) @ wc @ (X_apriori - x_apriori[:, np.newaxis]).transpose()  + sigma_q *  Q 


    return X_apriori, x_apriori, P_apriori

def ukf_update(z, X_apriori, x_apriori, P_apriori, w, wc, R, measure_func):

    # Measurement space of state sigma point
    Z = measure_func(X_apriori)

    # Mean of measurement space
    # z_mean = np.matmul(Z, w)S
    # z_mean = compute_mean_vector(Z, w)
    z_mean = np.mean(Z * w, axis=1)
    
    # Reshape to column vector

    # Calculation covariance of measurement space
    z_mean = z_mean[:, np.newaxis]
    P_z = (Z - z_mean) @ np.diag(w) @ (Z - z_mean).transpose()  + R

    # Calculating cross-covariance of state and measurement
    P_xz = (X_apriori - x_apriori[:, np.newaxis]) @ wc @ (Z - z_mean).transpose()

    # Kalman gain
    P_zinv = np.linalg.inv(P_z)
    K = np.matmul(P_xz, P_zinv)

    x_aposteriori = x_apriori.transpose() + np.matmul(K, (z - z_mean.transpose())[0,:])
    P_aposteriori = P_apriori - np.matmul(np.matmul(K, P_z), K.transpose())
    return x_aposteriori, P_aposteriori

def meas_func(X):
    x, y = X[0, :], X[3, :]
    # z = X
    # z = np.array([X[0,:], X[2,:]])

    # Calculate magnitude (r) element-wise
    z_1 = np.sqrt(x**2 + y**2)

    # Calculate angle (θ) element-wise using arctan2
    z_2 = np.arctan2(y, x)

    # Stack z_1 and z_2 horizontally to create a 2x(m) numpy array
    z = np.vstack((z_1, z_2))

    return z

# Initial state estimate
x_ = np.array([400,0,0,-300,0,0])


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

# Test purposes
w = np.array([-1, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
wc = np.diag(w)

sigma_p = 500
sigma_q = 0.2



x_true = np.linspace(0,3.28,100)
y_true = np.sinc(x_true)

data = np.array([
    [502.55, 477.34, 457.21, 442.94, 427.27, 406.05, 400.73, 377.32, 360.27, 345.93, 333.34, 328.07, 315.48, 301.41,
      302.87, 304.25, 294.29, 299.38, 299.37, 300.68, 304.1, 301.96, 300.3, 301.9, 297.07, 295.29, 296.31, 300.62, 292.3, 298.11, 298.07, 298.92],
    [-0.9316, -0.8977, -0.8512, -0.8114, -0.7853, -0.7392, -0.7052, -0.6478, -0.59, -0.5183, -0.4698, -0.3952, -0.3026, -0.2445, -0.1626, -0.0937,
     0.0856, 0.1675, 0.2467, 0.329, 0.4149, 0.504, 0.5934, 0.667, 0.8354, 0.9195, 1.0039, 1.0923, 1.1546, 1.2564, 1.3274, 1.409]

])



x_true = data[0,:]
y_true = data[1, :]


x_predict = []

# plt.figure()
# plt.plot(x_true, y_true)
# plt.show()


P_nn = sigma_p * np.eye(N)

x_apriori = None
P_apriori = None
P_apriori = None
for ii in range(len(x_true)):
    if ii == 0:
        X_apriori, x_apriori, P_apriori = ukf_predict(x_, sigma_p, F, Q, sigma_q, w, wc, P_nn)
    else:
        X_apriori, x_apriori, P_apriori = ukf_predict(x_apriori, sigma_p, F, Q, sigma_q, w, wc, P_apriori)

    z = np.array([x_true[ii], y_true[ii]])

    x_apriori, P_apriori = ukf_update(z, X_apriori, x_apriori, P_apriori, w, wc, R, measure_func= meas_func)

    x_predict.append([x_apriori[0], x_apriori[2]])
    
    pass



pass



# Model procesnog šuma Q i model matrice F i mjerna funkcija h



