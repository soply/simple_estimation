# coding utf:8


# outputs a list of (x,y) pairs
import numpy as np
import inspect
import os
import sys

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from LPAV import LPAV
from sklearn.model_selection import train_test_split


class SlisotronCV(BaseEstimator, RegressorMixin):
    """
    Self-made Slisotron implementation with on the fly cross validation. Following the reference:

    [1] Kakade, Sham M., et al. "Efficient learning of generalized linear and single
        index models with isotonic regression." Advances in Neural Information Processing Systems. 2011.
    """
    def __init__(self, max_iter = 1000, cv_split = 0.15, qp_solver = None):
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
        # Split into training and cv set first
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.cv_split)
        self.hold_out_error_ = np.zeros(self.max_iter) # Saving cv'ed errors
        isreg = LPAV(solver = self.qp_solver)
        n_features = X.shape[1]
        index = np.zeros(n_features) # Initial index vector
        isreg = isreg.fit(index.dot(X_train.T), y_train)
        # Compute error of current estimator on hold out set
        self.hold_out_error_[0] = np.linalg.norm(isreg.predict(index.dot(X_test.T)) - y_test) ** 2 # Square doesn't really matter
        best_index = index # Keep track of the best index according to CV
        best_err = self.hold_out_error_[0]
        iter_ctr = 1
        while iter_ctr < self.max_iter:
            y_train_new = isreg.predict(index.dot(X_train.T))
            index = index + np.mean((X_train.T * (y_train - y_train_new)).T, axis = 0)
            isreg = isreg.fit(index.dot(X_train.T), y_train)
            # Compute error of current estimator on hold out set
            self.hold_out_error_[iter_ctr] = np.linalg.norm(isreg.predict(index.dot(X_test.T)) - y_test) ** 2 # Square doesn't really matter
            if self.hold_out_error_[iter_ctr] < best_err:
                best_index = index
                best_err = self.hold_out_error_[iter_ctr]
            iter_ctr += 1
        # Final fitting of the isotonic regression
        isreg = isreg.fit(best_index.dot(X.T), y)
        self.isreg_ = isreg # Assign as class attribute to use in prediction
        self.index_ = best_index # Save SIM index vector
        return self


    def predict(self, X, y=None):
        try:
            getattr(self, "isreg_")
            getattr(self, "index_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        return self.isreg_.predict(X.dot(self.index_))


    def n_iter_cv(self):
        return np.argmin(self.hold_out_error_)
