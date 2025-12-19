import numpy as np

class KalmanFilter:
    def __init__(self):
        self.dt = 1.0
        self.A = np.array([[1, self.dt],
                           [0, 1]])
        self.C = np.array([[1, 0]])
        self.Q = np.array([[0.0001, 0],
                           [0, 0.0001]])
        self.R = np.array([[0.01]])
        self.P = np.eye(2)
        self.x = np.array([[0],
                           [0]])
        self.ticks = 0

    def update(self, z):
        self.ticks += 1
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)

        y = np.array([[z]]) - self.C @ x_pred

        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.C) @ P_pred

        est_price = round(self.x[0, 0], 6)
        velocity = round(self.x[1, 0], 6)
        return est_price, velocity
