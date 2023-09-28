import numpy as np
from scipy.linalg import sqrtm


def compute_sigma_points(dim, no_sigma_p, cov_matrix, current_x):
    sigma_matrix = np.zeros((dim, no_sigma_p)) # 6x13
    # print(f"\nSize of x is?? {current_x.shape}\n")
    # print(f"\nKaj je s current_x?? {current_x}\n")
    sigma_matrix[:, 0] = current_x

    lower_matrix_l = np.linalg.cholesky(3 * cov_matrix)
    for i in range(1, dim + 1):
        sigma_matrix[:, i] = current_x + lower_matrix_l[:, i-1]
        sigma_matrix[:, 6+i] = current_x - lower_matrix_l[:, i-1]

    return sigma_matrix


def compute_sigma_points_using_schur(dim, no_sigma_p, cov_matrix, current_x):
    sigma_matrix = np.zeros((dim, no_sigma_p)) # 6x13
    sigma_matrix[:, 0] = current_x

    # lower_matrix_l = np.linalg.cholesky(3 * cov_matrix)
    sqrt_3p = np.sqrt(dim) * sqrtm(cov_matrix)
    # print(sqrt_3p)
    for i in range(1, dim + 1):
        sigma_matrix[:, i] = current_x + sqrt_3p[:, i-1]
        sigma_matrix[:, 6+i] = current_x - sqrt_3p[:, i-1]

    return sigma_matrix


def non_linear_h(x, y):
    z_1 = np.sqrt(x**2 + y**2)
    z_2 = np.arctan2(y, x)

    z = np.vstack((z_1, z_2))

    return z


def predict(f, p, q, x, w, w_mat):
    n = f.shape[0]
    no_sigma_points = 2*n + 1

    # eigenvalues = np.linalg.eigvals(p)
    # print("\nEigenvalues (Spectrum) before regularization with 0.2*I:\n")
    # print(eigenvalues)

    # epsilon = min(eigenvalues)
    # epsilon = 1e-6
    # p = p + epsilon * np.identity(p.shape[0])

    # print(p)
    # sigma_matrix = compute_sigma_points(n, no_sigma_points, p, x.flatten())
    sigma_matrix = compute_sigma_points_using_schur(n, no_sigma_points, p, x.flatten())

    sigma_propagated = f @ sigma_matrix

    x_pred = sigma_propagated @ w
    p_pred = (sigma_propagated - x_pred) @ w_mat @ (sigma_propagated - x_pred).T + q

    p_pred[np.abs(p_pred < 0.000005)] = 0

    # print(f"\nOvakvi mi x_pred izlaze {x_pred}")

    return x_pred, p_pred, sigma_propagated


def update(f, p, q, r, x, w, w_mat, sigma_propagated, measurement, act_id):
    if measurement.shape != (2, 1):
        measurement = measurement.reshape(-1, 1)

    z = non_linear_h(sigma_propagated[0, :], sigma_propagated[3, :])

    z_bar = z @ w

    p_z = (z - z_bar) @ w_mat @ (z - z_bar).T + r
    p_xz = (sigma_propagated - x) @ w_mat @ (z-z_bar).T

    k_gain = p_xz @ np.linalg.pinv(p_z)
    # if act_id == 1:
    #     print(f"\n z je {measurement}")
    #     print(f"\n z_bar je {z_bar}")
    #     print(f"\n k_gain je {k_gain}")

    x_updated = x + k_gain @ (measurement - z_bar)
    p_updated = p - k_gain @ p_z @ k_gain.T

    return x_updated, p_updated
