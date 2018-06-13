# coding: utf8
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import mean_squared_error

__available_methods__ = ['dyadic_svm', 'median_svm']


class CrossvalidatedInverseRegressionTree(BaseEstimator, RegressorMixin):
    """ Crossvalidates on all tree levels without rebuilding the whole tree. Faster
    then using RegressionTree in combination with GridSearch CV (needs to rebuild
    tree). """


    def __init__(self, method = 'pca_tree', height = 0,
                 minLeafSize = 5, n_cv_splits = 10, test_levels = [0]):
        self.height = height
        self.method = method
        self.minLeafSize = minLeafSize
        self.n_cv_splits = n_cv_splits
        self.test_levels = test_levels
        self.Ymean_ = None
        self.left_ = None
        self.right_ = None
        self.v_classifier_ = None
        self.t_classifier_ = None
        self.tree_ = InverseRegressionTree(method = method, predict_level = 0,
            height = height, minLeafSize = minLeafSize)
        self.cved_level_ = 0


    def fit(self, X, Y):
        # Build tree and crossvalidate for the best level
        kf = ShuffleSplit(n_splits = self.n_cv_splits, test_size = 0.1)
        errors = np.zeros((len(self.test_levels), self.n_cv_splits))
        split_iter = 0
        for idx_train, idx_cv in kf.split(X):
            X_train, Y_train = X[idx_train,:], Y[idx_train]
            X_cv, Y_cv = X[idx_cv,:], Y[idx_cv]
            self.tree_.fit(X_train, Y_train)
            for j, level in enumerate(self.test_levels):
                # Adjust prediction level w/o refitting the tree
                self.tree_._change_prediction_level(level)
                # Compare to true values2
                errors[j, split_iter] =  mean_squared_error(self.tree_.predict(X_cv), Y_cv)
            split_iter += 1
        avg_errors = np.mean(errors, axis = 1)
        self.cved_level_ = np.argmin(avg_errors)
        # Refit tree on the whole data set with cv'ed level
        self.tree_.fit(X, Y)
        self.tree_._change_prediction_level(level)
        import pdb; pdb.set_trace()
        return self

    def predict(self, Z):
        return self.tree_.predict(Z)


class InverseRegressionTree(BaseEstimator, RegressorMixin):


    def __init__(self, method = 'dyadic_svm', predict_level = 0, height = 0,
                 minLeafSize = 5):
        self.predict_level = predict_level
        self.height = height
        self.method = method
        self.minLeafSize = minLeafSize
        self.Ymean_ = None
        self.left_ = None
        self.right_ = None
        self.classifier_ = None


    def fit(self, X, Y):
        N, D = X.shape
        self.Ymean_ = np.mean(Y)
        # Sepparate data by chosen method
        if N <= self.minLeafSize:
            return
        else:
            success, labels, self.classifier_ = split(X, Y, self.method)
            if not success:
                return
            self.left_ = InverseRegressionTree(method = self.method,
                                               height = self.height + 1,
                                               predict_level = self.predict_level,
                                               minLeafSize = self.minLeafSize)
            self.right_ = InverseRegressionTree(method = self.method,
                                                height = self.height + 1,
                                                predict_level = self.predict_level,
                                                minLeafSize = self.minLeafSize)
            self.left_.fit(X[labels == 0, :], Y[labels == 0])
            self.right_.fit(X[labels == 1, :], Y[labels == 1])
        return self


    def predict(self, Z):
        if self.height == self.predict_level or self.left_ is None or self.right_ is None:
            try:
                getattr(self, "Ymean_")
            except AttributeError:
                raise RuntimeError("You must train estimator before predicting data!")
            # decide here if correct height, or leaf does not have children
            return np.ones(Z.shape[0]) * self.Ymean_
        elif self.height < self.predict_level:
            # pass to left or right node, depending on classifier
            label = self._child(Z)
            retr = np.zeros(Z.shape[0])
            if len(np.where(label == 0)[0]) > 0:
                retr[label == 0] = self.left_.predict(Z[label == 0,:])
            if len(np.where(label == 1)[0]) > 0:
                retr[label == 1] = self.right_.predict(Z[label == 1,:])
            return retr
        else:
            raise RuntimeError("Something went wrong")


    def _child(self, Z):
        try:
            getattr(self, "classifier_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        if len(Z.shape) == 1:
            Z = Z.reshape((1,-1))
        return self.classifier_.predict(Z)


    def _change_prediction_level(self, new_level):
        self.predict_level = new_level
        if self.left_ is not None:
            self.left_._change_prediction_level(new_level)
        if self.right_ is not None:
            self.right_._change_prediction_level(new_level)


def split(X, Y, method):
    if method == 'dyadic_svm':
        return split_dyadic_svm(X, Y)
    elif method == 'median_svm':
        return split_median_svm(X, Y)
    elif method == 'median_svm_repart':
        return split_median_svm_repart(X, Y)
    elif method == 'dyadic_svm_repart':
        return split_median_svm_repart(X, Y)
    else:
        raise NotImplementedError('Methods: ')


def split_dyadic_svm(X, Y):
    N, D = X.shape
    labels = np.zeros(N).astype('int')
    Y_mid = 0.5 * (np.max(Y) + np.min(Y))
    idx1 = np.where(Y < Y_mid)[0]
    idx2 = np.where(Y >= Y_mid)[0]
    labels[idx2] = 1
    if len(idx1) > 0 and len(idx2) > 0:
        svm = LinearSVC(class_weight = 'balanced')
        svm = svm.fit(X, labels)
        return True, labels, svm
    else:
        return False, labels, None



def split_median_svm(X, Y):
    N, D = X.shape
    labels = np.zeros(N).astype('int')
    Y_mid = np.median(Y)
    idx1 = np.where(Y < Y_mid)[0]
    idx2 = np.where(Y >= Y_mid)[0]
    labels[idx2] = 1
    if len(idx1) > 0 and len(idx2) > 0:
        svm = LinearSVC(class_weight = 'balanced')
        svm = svm.fit(X, labels)
        return True, labels, svm
    else:
        return False, labels, None


def split_dyadic_svm_repart(X, Y):
    N, D = X.shape
    labels = np.zeros(N).astype('int')
    Y_mid = 0.5 * (np.max(Y) + np.min(Y))
    idx1 = np.where(Y < Y_mid)[0]
    idx2 = np.where(Y >= Y_mid)[0]
    labels[idx2] = 1
    if len(idx1) > 0 and len(idx2) > 0:
        svm = LinearSVC(class_weight = 'balanced')
        svm = svm.fit(X, labels)
        new_labels = svm.predict(X)
        # Change function values for next steps of assignment
        increase_label = np.where(np.logical_and(new_labels == 1, labels == 0))[0]
        decrease_label = np.where(np.logical_and(new_labels == 0, labels == 1))[0]
        labels[increase_label] = 1
        labels[decrease_label] = 0
        idx1 = np.where(labels == 1)[0]
        idx2 = np.where(labels == 0)[0]
        if len(idx1) > 0 and len(idx2) > 0:
            return True, labels, svm
        else:
            return False, labels, None
    else:
        return False, labels, None



def split_median_svm_repart(X, Y):
    N, D = X.shape
    labels = np.zeros(N).astype('int')
    Y_mid = np.median(Y)
    idx1 = np.where(Y < Y_mid)[0]
    idx2 = np.where(Y >= Y_mid)[0]
    labels[idx2] = 1
    if len(idx1) > 0 and len(idx2) > 0:
        svm = LinearSVC(class_weight = 'balanced')
        svm = svm.fit(X, labels)
        new_labels = svm.predict(X)
        # Change function values for next steps of assignment
        increase_label = np.where(np.logical_and(new_labels == 1, labels == 0))[0]
        decrease_label = np.where(np.logical_and(new_labels == 0, labels == 1))[0]
        labels[increase_label] = 1
        labels[decrease_label] = 0
        idx1 = np.where(labels == 1)[0]
        idx2 = np.where(labels == 0)[0]
        if len(idx1) > 0 and len(idx2) > 0:
            return True, labels, svm
        else:
            return False, labels, None
    else:
        return False, labels, None







if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt
    sys.path.insert(0, '../../../DataSets/')
    from handler_UCI_Concrete import read_all
    data = read_all(scaling = 'MeanVar')
    X, Y = data[:,:-1], data[:,-1]
    tree = CrossvalidatedInverseRegressionTree(method = 'median_svm', height = 0,
                                 minLeafSize = 5, test_levels = [0,1,2,3,4,5,6,7,8,9,10,11])
    tree = tree.fit(X, Y)
    print tree.predict(X[0:10,:])
    import pdb; pdb.set_trace()
    print Y[0:10]
