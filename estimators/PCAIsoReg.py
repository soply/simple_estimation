# coding: utf8
import inspect
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression


class PCAIsoReg(BaseEstimator, RegressorMixin):
    """
    Implementing PCA + Isotonin regression on transformed variables regressor.
    """

    def __init__(self):
        """
        n_dim : int
            The number of components to extract in PCA and to project on.
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
        self.pca_ = PCA(n_components = 1) # Isotonic regression works only on 1D features
        self.XT_ = self.pca_.fit_transform(X)[:,0]
        self.isoReg_ = IsotonicRegression(increasing = 'auto',
                                          out_of_bounds = 'clip')
        order = np.argsort(self.XT_)
        self.isoReg_ = self.isoReg_.fit(self.XT_[order],y[order])
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "isoReg_")
            getattr(self, "pca_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        XT = self.pca_.transform(X)[:,0]
        return self.isoReg_.predict(XT)
