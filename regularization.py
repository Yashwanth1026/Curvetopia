import numpy as np
from scipy.optimize import minimize

def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def fit_circle(points):
    # Improved circle fitting using non-linear optimization
    def calc_R(xc, yc):
        """ Calculate the distance of each 2D point from the center (xc, yc) """
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)

    def f_2(c):
        """ Calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return np.sum((Ri - Ri.mean()) ** 2)

    # Initial guess for circle center
    center_estimate = np.mean(points, axis=0)
    center = minimize(f_2, center_estimate).x
    Ri = calc_R(*center)
    R = Ri.mean()
    return center[0], center[1], R
