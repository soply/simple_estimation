# coding: utf8
import os
import sys
import inspect

sys.path.insert(0, '../../sdr_toolbox/')


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sdr_toolbox.sdr_estimators.save import save


class SAVEIsoReg(BaseEstimator, RegressorMixin):
    """
    Implementing SAVE for dimension reduction + isotonic regression on
    transformed variables.
    """

    def __init__(self, n_levelsets = 1, rescale = True):
        """
        'n_levelsets' : int
            Number of level set slices to use in save procedure

        'rescale' : Rescale/Whiten data when applying save
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
        self.save_space_ = save(X.T, y, d = 1, n_levelsets = self.n_levelsets,
                              rescale = self.rescale, return_mat = False)
        self.XT_ = (self.save_space_.T.dot(X.T)).T[:,0]
        self.isoReg_ = IsotonicRegression(increasing = 'auto',
                                          out_of_bounds = 'clip')
        self.isoReg_ = self.isoReg_.fit(self.XT_,y)
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "isoReg_")
            getattr(self, "save_space_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        XT = (self.save_space_.T.dot(X.T)).T[:,0]
        return self.isoReg_.predict(XT)
