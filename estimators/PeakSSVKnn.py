# coding: utf8
import inspect
import os
import sys

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor

# Full path
dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, dir_path + '/../../../sdr_toolbox/')

from sdr_toolbox.sdr_estimators.peakSSV import peakSSV



class PeakSSVKnn(BaseEstimator, RegressorMixin):
    """
    Implementing PeakSSV to get a reduced subspace, and puts kNN estimation on
    top of the reduced variable.
    """

    def __init__(self, n_neighbors = 1, n_components = 1, n_sdr_samples = 10,
                 n_jobs = 1, rescale = False):
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
        n_samples, n_features = X.shape
        self.peakSSV_space_ = peakSSV(X.T, y, d = self.n_components,
                              n_samples = self.n_sdr_samples,
                              rescale = self.rescale)
        self.XT_ = (self.peakSSV_space_.T.dot(X.T)).T
        self.knn_ = KNeighborsRegressor(n_neighbors = self.n_neighbors,
                                        n_jobs = self.n_jobs)
        self.knn_ = self.knn_.fit(self.XT_, y)
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "knn_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")

        return self.knn_.predict(self.peakSSV_space_.T.dot(X.T).T)
