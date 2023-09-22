import numpy as np
from scipy.linalg import sqrtm
from tracking_utils.kalman_utils import compute_sigma_points_using_schur, non_linear_h, predict, update

# def compute_sigma_points(dim, no_sigma_p, cov_matrix, current_x):
#     sigma_matrix = np.zeros((dim, no_sigma_p))
#     sigma_matrix[:, 0] = current_x
#
#     lower_matrix_l = np.linalg.cholesky(3 * cov_matrix)
#     for i in range(1, dim + 1):
#         sigma_matrix[:, i] = current_x + lower_matrix_l[:, i-1]
#         sigma_matrix[:, 6+i] = current_x - lower_matrix_l[:, i-1]
#
#     return sigma_matrix
#
#
# def compute_sigma_points_using_schur(dim, no_sigma_p, cov_matrix, current_x):
#     sigma_matrix = np.zeros((dim, no_sigma_p))
#     sigma_matrix[:, 0] = current_x
#
#     sqrt_3p = sqrtm(3 * cov_matrix)
#     for i in range(1, dim + 1):
#         sigma_matrix[:, i] = current_x + sqrt_3p[:, i-1]
#         sigma_matrix[:, 6+i] = current_x - sqrt_3p[:, i-1]
#
#     return sigma_matrix
#
#
# def non_linear_h(x, y):
#     z_1 = np.sqrt(x**2 + y**2)
#     z_2 = np.arctan2(y, x)
#
#     z = np.vstack((z_1, z_2))
#
#     return z
#
#
# def predict(f, p, q, x, w, w_mat):
#     n = f.shape[0]
#     no_sigma_points = 2*n + 1
#
#     eigenvalues = np.linalg.eigvals(p)
#
#     epsilon = 1e-6
#     p = p + epsilon * np.identity(p.shape[0])
#
#     sigma_matrix = compute_sigma_points_using_schur(n, no_sigma_points, p, x.flatten())
#
#     sigma_propagated = f @ sigma_matrix
#
#     x_pred = sigma_propagated @ w
#     p_pred = (sigma_propagated - x_pred) @ w_mat @ (sigma_propagated - x_pred).T + q
#
#     p_pred[np.abs(p_pred < 0.000005)] = 0
#
#     return x_pred, p_pred, sigma_propagated
#
#
# def update(f, p, q, r, x, w, w_mat, sigma_propagated, measurement):
#     z = non_linear_h(sigma_propagated[0, :], sigma_propagated[3, :])
#
#     z_bar = z @ w
#
#     p_z = (z - z_bar) @ w_mat @ (z - z_bar).T + r
#     p_xz = (sigma_propagated - x) @ w_mat @ (z - z_bar).T
#
#     k_gain = p_xz @ np.linalg.pinv(p_z)
#
#     x_updated = x + k_gain @ (measurement - z_bar)
#     p_updated = p - k_gain @ p_z @ k_gain.T
#
#     return x_updated, p_updated


dt = 1
sigma_a = 0.2
F = np.array([[1, dt, 0.5*(dt**2), 0, 0, 0],
              [0, 1, dt, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, dt, 0.5*(dt**2)],
              [0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 1]])

Q = (sigma_a**2) * np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                  [dt**3/2, dt**2, dt, 0, 0, 0],
                  [dt**2/2, dt, 1, 0, 0, 0],
                  [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                  [0, 0, 0, dt**3/2, dt**2, dt],
                  [0, 0, 0, dt**2/2, dt, 1]])

sigma_r = 5
sigma_phi = 0.0087
R = np.array([[sigma_r**2, 0],
              [0, sigma_phi**2]])

W = np.array([[-1], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6], [1/6]])
W_for_diag = [-1, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
W_mat = np.diag(W_for_diag)

X = np.array([400, 0, 0, -300, 0, 0])
X = X.T
P = np.array([[500, 0, 0, 0, 0, 0],
              [0, 500, 0, 0, 0, 0],
              [0, 0, 500, 0, 0, 0],
              [0, 0, 0, 500, 0, 0],
              [0, 0, 0, 0, 500, 0],
              [0, 0, 0, 0, 0, 500]])

measurements_data = [
    [502.55, -0.9316],
    [477.34, -0.8977],
    [457.21, -0.8512],
    [442.94, -0.8114],
    [427.27, -0.7853],
    [406.05, -0.7392],
    [400.73, -0.7052],
    [377.32, -0.6478],
    [360.27, -0.59],
    [345.93, -0.5183],
    [333.34, -0.4698],
    [328.07, -0.3952],
    [315.48, -0.3026],
    [301.41, -0.2445],
    [302.87, -0.1626],
    [304.25, -0.0937],
    [294.46, 0.0085],
    [294.29, 0.0856],
    [299.38, 0.1675],
    [299.37, 0.2467],
    [300.68, 0.329],
    [304.1, 0.4149],
    [301.96, 0.504],
    [300.3, 0.5934],
    [301.9, 0.667],
    [296.7, 0.7537],
    [297.07, 0.8354],
    [295.29, 0.9195],
    [296.31, 1.0039],
    [300.62, 1.0923],
    [292.3, 1.1546],
    [298.11, 1.2564],
    [298.07, 1.3274],
    [298.92, 1.409],
    [298.04, 1.5011]
]

z_measurements = np.array(measurements_data)

for i in range(35):
    z_measurement = z_measurements[i, :].T.reshape(-1, 1)

    X_pred, P_pred, sigma_prop = predict(F, P, Q, X, W, W_mat)

    X_updated, P_updated = update(F, P_pred, Q, R, X_pred, W, W_mat, sigma_prop, z_measurement, 11)

    X = X_updated
    P = P_updated

print(f"\nFinal X = {X}\n")
print(f"\nFinal P = {P}\n")
