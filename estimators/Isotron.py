# coding utf:8
import numpy as np
import inspect
import os

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression # This is the PAV algorithm used in Isotron
from sklearn.model_selection import train_test_split

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
        return self


    def predict(self, X, y=None):
        try:
            getattr(self, "isreg_")
            getattr(self, "index_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        return self.isreg_.predict(X.dot(self.index_))


class IsotronCV(BaseEstimator, RegressorMixin):
    """
    Self-made Isotron implementation with on the fly cross validation. Following the reference:

    [1] Kalai, Adam Tauman, and Ravi Sastry. "The Isotron Algorithm: High-Dimensional Isotonic Regression." COLT. 2009.
    """

    def __init__(self, max_iter = 1000, cv_split = 0.15):
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
        isreg = IsotonicRegression(out_of_bounds = 'clip')
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
