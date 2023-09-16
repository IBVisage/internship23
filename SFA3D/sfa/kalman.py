import numpy as np

def ukf_init():

    pass


x_ = np.array([1,2,3])
sigma_p = 100
sigma_q = 10
F = np.array([[1,1,1],[1,1,1],[1,1,1]])
Q = np.array([[1,1,1],[1,1,1],[1,1,1]])


def ukf_predict(x_, sigma_p, F, Q, sigma_q):
    # Sigma point computation

    # N: dimension of measure space
    N = x_.shape[0]
    # number of sigma points = 2N + 1
    # lam: weight parameter
    # k: scaling paramater; usually 0
    # alpha: determines spread of distribution
    # beta: prior knowledge of distribution - 2 for gaussian
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
    
    # x_: mean value
    # X_0: mean sigma point
    X_0 = x_

    # P matrix
    P_nn = sigma_p * np.eye(N)
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

    P_apriori = np.matmul(np.matmul((X_apriori - x_apriori), np.diag(w)),    (X_apriori - x_apriori).transpose())  + sigma_q *  Q 


    return X_apriori, x_apriori, P_apriori, w

def ukf_update(X_apriori, x_apriori, P_apriori, w, measure_func):

    # Measurement space of state sigma point
    Z = measure_func(X_apriori)
    pass

def meas_func(X):
    z = X
    return z


X_apriori, x_apriori, P_apriori, w = ukf_predict(x_, sigma_p, F, Q, sigma_q)

ukf_update(X_apriori, x_apriori, P_apriori, w, meas_func)

# Model procesnog šuma Q i model matrice F i mjerna funkcija h



