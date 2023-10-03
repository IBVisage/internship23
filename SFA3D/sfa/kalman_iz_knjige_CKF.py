"""
# Authors: Ivan Bukač, Ante Ćubela
# DoC: 2023.10.06.
-----------------------------------------------------------------------------------
# Description: Example 13 from Section 14.7. from "Kalman Filter from the Ground Up", by Alex Becker, using CKF
"""

import numpy as np
from tracking_utils.kalman_utils_CKF import predict, update

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
