# coding utf:8
"""
Simple N^2 implementation of Lipschitz pool adjacent violators based on solving
the corresponding quadratic program. Note that
there is an N log(N) possible, but it seems more complicated.
"""

__author__ = "Timo Klock"

import numpy as np
import inspect
import os

from sklearn.base import BaseEstimator, RegressorMixin
from cvxopt import matrix, solvers
from cvxopt import spmatrix
from cvxopt.blas import dot
from cvxopt.solvers import qp


class LPAV(BaseEstimator, RegressorMixin):
    """
    Self-made Isotron implementation. Following the reference:

    [1] Kalai, Adam Tauman, and Ravi Sastry. "The Isotron Algorithm: High-Dimensional Isotonic Regression." COLT. 2009.
    """

    def __init__(self, sorted = False, solver = None):
        """
        sorted : Boolean
            If True, input is assumed to be sorted according to X.
        """
        # Set attributes of object to the same name as given in the argument list.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        solvers.options['show_progress'] = False
        for arg, val in values.items():
            setattr(self, arg, val)


    def fit(self, X, y=None):
        """
        Paramters
        ------------------------------------------------------------
        X : np.array of size (N, 1)
            Nodes in the x-domain

        y : np.array of size N
            Responses corresponding to X
        """
        # First step: sorting the data
        if sorted is False:
            order = np.argsort(X)
            X = X[order]
            y = y[order]
        N = X.shape[0]
        PP = spmatrix(1.0, range(N), range(N)) # N x N identity
        qq = matrix(-y, (N, 1)) # vector q
        # Bounds for the constraints
        hh = matrix([0.0 for i in range(N * (N-1))], (N * (N-1), 1))
        # Only need to set upper bound, lower bound is zero.
        ctr = 0
        for i in range(N):
            for j in range(i+1,N):
                hh[ctr] = X[j] - X[i]
                ctr += 1
        # Upper bound for constraints
        rows = [0 for i in range(2 * N * (N-1))]
        cols = [0 for i in range(2 * N * (N-1))]
        vals = [0.0 for i in range(2 * N * (N-1))]
        ctr = 0
        row_ctr = 0
        # First upper bounds, then lower boundary
        for sgn in [-1.0, 1.0]:
            # We repeat the procedure to get the matrix also for the lower boudaries
            for i in range(N):
                for j in range(i+1,N):
                    rows[ctr] = row_ctr
                    cols[ctr] = i
                    vals[ctr] = sgn
                    ctr += 1
                    rows[ctr] = row_ctr
                    cols[ctr] = j
                    vals[ctr] = -sgn
                    ctr += 1
                    row_ctr += 1
        G_mat = spmatrix(vals, rows, cols)
        result = qp(PP, qq, G_mat, hh, solver = self.solver)
        self.X_ = np.reshape(X, (N))
        self.newy_ = np.array(result['x'])[:,0]
        return self


    def predict(self, X, y=None):
        try:
            getattr(self, "X_")
            getattr(self, "newy_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0]))
        return np.interp(X, self.X_, self.newy_)


# Testing the implementation
if __name__ == "__main__":
    N = 1000
    X = np.random.uniform(low = 1.0, high = 2.0, size = N)
    X = np.sort(X)
    X = np.reshape(X, (N, 1))
    X_test = np.random.uniform(low = 1.0, high = 2.0, size = N)
    X_test = np.sort(X_test)
    X_test = np.reshape(X_test, (N, 1))
    Y = X[:,0] + np.random.normal(scale = 0.1, size = N)
    lpav = LPAV()
    lpav.fit(X, Y)
    prediction = lpav.predict(X_test)
    plt.plot(X_test, prediction, label = 'Prediction')
    plt.legend()
    plt.show()
