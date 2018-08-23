# coding: utf8
import inspect
import os
import sys

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

from sdr_toolbox.sdr_estimators.save import save

sys.path.insert(0, '../../sdr_toolbox/')




class SAVEKnn(BaseEstimator, RegressorMixin):
    """
    Implementing SAVE for dimension reduction + kNN on reduced features as
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
        n_samples, n_features = X.shape
        self.SAVE_space_ = save(X.T, y, d = self.n_components,
                              n_levelsets = self.n_levelsets,
                              rescale = self.rescale)
        self.XT_ = (self.SAVE_space_.T.dot(X.T)).T
        self.knn_ = KNeighborsRegressor(n_neighbors = self.n_neighbors,
                                        n_jobs = self.n_jobs)
        self.knn_ = self.knn_.fit(self.XT_, y)
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "knn_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")

        return self.knn_.predict(self.SAVE_space_.T.dot(X.T).T)
