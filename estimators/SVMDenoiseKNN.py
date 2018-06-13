# coding: utf8
import os
import sys
import inspect

sys.path.insert(0, '../../sdr_toolbox/')
sys.path.insert(0, '../../svm_denoise/')

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from svm_denoise.fixed_splits import SVMFixedSplits


class SVMKnn(BaseEstimator, RegressorMixin):
    """
    Implementing SVM denoising plus kNN estimation on top.
    """

    def __init__(self, n_neighbors = 1, n_splits = 2, split_by = 'Y_equivalent',
                 class_weight = 'balanced', max_iter = 100, n_jobs = 1):
        """

        """
        # Set attributes of object to the same name as given in the argument
        # list.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y=None):
        """
        """
        tf = SVMFixedSplits(n_splits = self.n_splits, split_by = self.split_by,
                            class_weight = self.class_weight, max_iter = self.max_iter,
                            n_jobs = self.n_jobs)
        X, modY = tf.transform(X, y)
        self.knn_ = KNeighborsRegressor(n_neighbors = self.n_neighbors)
        self.knn_ = self.knn_.fit(X, modY)
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "knn_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        return self.knn_.predict(X)
