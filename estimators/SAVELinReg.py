# coding: utf8
import os
import sys
import inspect



from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Full path
dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, dir_path + '/../../../sdr_toolbox/')

from sdr_toolbox.sdr_estimators.save import save


class SAVELinReg(BaseEstimator, RegressorMixin):
    """
    Implementing SAVE for dimension reduction + linear regression on
    transformed variables.
    """

    def __init__(self, n_dim = 1, n_levelsets = 1, rescale = True):
        """
        'n_dim' : int
            Number of components to select in save procedure

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
        assert (self.n_dim <= n_features), "Number of SAVE components must be < number of features: {0} < {1}".format(self.n_dim, n_features)
        self.save_space_ = save(X.T, y, d = self.n_dim, n_levelsets = self.n_levelsets,
                              rescale = self.rescale, return_mat = False)
        self.XT_ = (self.save_space_.T.dot(X.T)).T
        self.linReg_ = LinearRegression()
        self.linReg_ = self.linReg_.fit(self.XT_,y)
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "linReg_")
            getattr(self, "save_space_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        XT = (self.save_space_.T.dot(X.T)).T
        return self.linReg_.predict(XT)
