# coding: utf8
import inspect


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class PCALinReg(BaseEstimator, RegressorMixin):
    """
    Implementing PCA + Linear regression on transformed variables regressor.
    """

    def __init__(self, n_dim = 1):
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
        assert (self.n_dim <= n_features), "Number of PCA components must be < number of features: {0} < {1}".format(self.n_dim, n_features)
        self.pca_ = PCA(n_components = self.n_dim)
        self.XT_ = self.pca_.fit_transform(X)
        self.linReg_ = LinearRegression()
        self.linReg_ = self.linReg_.fit(self.XT_,y)
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "linReg_")
            getattr(self, "pca_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        XT = self.pca_.transform(X)
        return self.linReg_.predict(XT)
