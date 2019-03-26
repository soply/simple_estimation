# coding: utf8
import inspect
import os
import sys

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

# Full path
sys.path.insert(0, '../../sdr_toolbox/')

from sdr_toolbox.sdr_estimators.sir import sir





class SIRKnn(BaseEstimator, RegressorMixin):
    """
    Implementing SIR for dimension reduction + kNN on reduced features as
    regressor.
    """

    def __init__(self, n_neighbors = 1, n_components = 1,
                 n_levelsets = 1, n_jobs = 1, rescale = True):
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
        if self.n_components > self.n_levelsets:
            raise RuntimeError("n_components = {0} > {1} = n_levelsets".format(self.n_components, self.n_levelsets))
        n_samples, n_features = X.shape
        self.SIR_space_ = sir(X.T, y, d = self.n_components,
                              n_levelsets = self.n_levelsets,
                              rescale = self.rescale)
        self.XT_ = (self.SIR_space_.T.dot(X.T)).T
        self.knn_ = KNeighborsRegressor(n_neighbors = self.n_neighbors,
                                        n_jobs = self.n_jobs)
        self.knn_ = self.knn_.fit(self.XT_, y)
        return self


    def predict(self, X, y=None):
        try:
            getattr(self, "knn_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")

        return self.knn_.predict(self.SIR_space_.T.dot(X.T).T)
