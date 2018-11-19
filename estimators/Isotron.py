# coding utf:8


# outputs a list of (x,y) pairs
import random
import bisect
import numpy as np
from numpy import *
from scipy.linalg import decomp
import inspect
import os
import sys

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression # This is the PAV algorithm used in Isotron


class Isotron(BaseEstimator, RegressorMixin):
    """
    Self-made Isotron implementation. Following the reference:

    [1] Kalai, Adam Tauman, and Ravi Sastry. "The Isotron Algorithm: High-Dimensional Isotonic Regression." COLT. 2009.
    """

    def __init__(self, n_iter = 100):
        """

        """
        # Set attributes of object to the same name as given in the argument list.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)


    def fit(self, X, y=None):
        """
        """
        isreg = IsotonicRegression(out_of_bounds = 'clip')
        n_samples, n_features = X.shape
        index = np.zeros(n_features) # Initial index vector
        iter_ctr = 0
        while iter_ctr < self.n_iter:
            isreg = isreg.fit(index.dot(X.T), y)
            ynew = isreg.predict(index.dot(X.T))
            index += np.mean((X.T * (y - ynew)).T, axis = 0)
            iter_ctr += 1
        # Final fitting of the isotonic regression
        isreg = isreg.fit(index.dot(X.T), y)
        self.isreg_ = isreg # Assign as class attribute to use in prediction
        self.index_ = index # Save SIM index vector


    def predict(self, X, y=None):
        try:
            getattr(self, "isreg_")
            getattr(self, "index_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        return self.isreg_.predict(X.dot(self.index_))
