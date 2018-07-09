# coding: utf8
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.svm import LinearSVC

__available_methods__ = ['dyadic_svm', 'median_svm',
                         'dyadic_svm_repart', 'median_svm_repart',
                         'dyadic_svm_overlap', 'median_svm_overlap']


class CrossvalidatedInverseRegressionTree(BaseEstimator, RegressorMixin):
    """ Crossvalidates on all tree levels without rebuilding the whole tree. Faster
    then using RegressionTree in combination with GridSearch CV (needs to rebuild
    tree). """


    def __init__(self, method = 'pca_tree', height = 0,
                 minLeafSize = 5, n_cv_splits = 10, test_levels = [0],
                 undecided_prop = 0.0):
        self.height = height
        self.method = method
        self.minLeafSize = minLeafSize
        self.undecided_prop = undecided_prop
        self.n_cv_splits = n_cv_splits
        self.test_levels = test_levels
        self.tree_ = InverseRegressionTree(method = method, predict_level = 0,
            height = height, minLeafSize = minLeafSize,
            undecided_prop = undecided_prop)
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
        self.cved_level_ = self.test_levels[np.argmin(avg_errors)] # FIXED THIS
        # Refit tree on the whole data set with cv'ed level
        self.tree_.fit(X, Y)
        self.tree_._change_prediction_level(self.cved_level_)
        return self

    def predict(self, Z, level = None):
        return self.tree_.predict(Z, level)


class InverseRegressionTree(BaseEstimator, RegressorMixin):


    def __init__(self, method = 'dyadic_svm', predict_level = 0, height = 0,
                 minLeafSize = 5, undecided_prop = 0.0):
        self.predict_level = predict_level
        self.height = height
        self.method = method
        self.minLeafSize = minLeafSize
        self.undecided_prop = undecided_prop
        self.Ymean_ = None
        self.left_ = None
        self.right_ = None
        self.classifier_ = None
        # Auxiliary information
        self.Nused_ = None
        self.Ymin_ = None
        self.Ymax_ = None



    def fit(self, X, Y):
        N, D = X.shape
        self.Nused_ = N
        self.Ymean_ = np.mean(Y)
        self.Ymin_ = np.min(Y)
        self.Ymax_ = np.max(Y)
        # Sepparate data by chosen method
        if N <= self.minLeafSize:
            return
        else:
            success, labels, self.classifier_ = split(X, Y, self.method,
                                                      self.undecided_prop)
            if not success:
                return
            self.left_ = InverseRegressionTree(method = self.method,
                                               height = self.height + 1,
                                               predict_level = self.predict_level,
                                               minLeafSize = self.minLeafSize,
                                               undecided_prop = self.undecided_prop)
            self.right_ = InverseRegressionTree(method = self.method,
                                                height = self.height + 1,
                                                predict_level = self.predict_level,
                                                minLeafSize = self.minLeafSize,
                                                undecided_prop = self.undecided_prop)
            self.left_.fit(X[labels[0,:]], Y[labels[0,:]])
            self.right_.fit(X[labels[1,:]], Y[labels[1,:]])
        return self


    def predict(self, Z, level = None):
        if level is None:
            level = self.predict_level
        if self.height == level or self.left_ is None or self.right_ is None:
            try:
                getattr(self, "Ymean_")
            except AttributeError:
                raise RuntimeError("You must train estimator before predicting data!")
            # decide here if correct height, or leaf does not have children
            return np.ones(Z.shape[0]) * self.Ymean_
        elif self.height < level:
            # pass to left or right node, depending on classifier
            label = self._child(Z)
            retr = np.zeros(Z.shape[0])
            if len(np.where(~label)[0]) > 0:
                retr[~label] = self.left_.predict(Z[~label,:], level)
            if len(np.where(label)[0]) > 0:
                retr[label] = self.right_.predict(Z[label,:], level)
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


    def depth(self):
        if self.left_ is not None and self.right_ is not None:
            return 1 + np.maximum(self.left_.depth(), self.right_.depth())
        elif self.left_ is not None:
            return 1 + self.left_.depth()
        elif self.right_ is not None:
            return 1 + self.right_.depth()
        else:
            return 1


    def bds(self, stack = []):
        if len(stack) == 0:
            stack.append(self)
            if self.left_ is not None:
                stack.append(self.left_)
            if self.right_ is not None:
                stack.append(self.left_)


    def print_Nused(self):
        nodes = self.get_breadth_first_nodes()
        for node in nodes:
            print "Node: Height = {0}   N = {1}     Ymin = {2}      Ymax = {3}".format(
                                                                node.height,
                                                                node.Nused_,
                                                                node.Ymin_,
                                                                node.Ymax_)


    def get_breadth_first_nodes(self):
        nodes = []
        stack = [self]
        while stack:
            cur_node = stack[0]
            stack = stack[1:]
            nodes.append(cur_node)
            if cur_node.left_ is not None:
                stack.append(cur_node.left_)
            if cur_node.right_ is not None:
                stack.append(cur_node.right_)
        return nodes


class Classifier(object):

    def __init__(self, direction, svm):
        self.direction_ = direction
        self.svm_ = svm

    def predict(self, X):
        return self.svm_.predict(np.reshape(X.dot(self.direction_), (-1,1)))


def split(X, Y, method, undecided_prop):
    if method == 'dyadic_svm':
        return split_svm(X, Y, 'dyadic', undecided_prop)
    elif method == 'median_svm':
        return split_svm(X, Y, 'median', undecided_prop)
    elif method == 'dyadic_svm_repart':
        return split_svm_repart(X, Y, 'dyadic')
    elif method == 'median_svm_repart':
        return split_svm_repart(X, Y, 'median')
    elif method == 'dyadic_svm_overlap':
        return split_svm_overlap(X, Y, 'dyadic')
    elif method == 'median_svm_overlap':
        return split_svm_overlap(X, Y, 'median')
    elif method == 'median_naive':
        return split_naive(X, Y, 'median', undecided_prop)
    elif method == 'dyadic_naive':
        return split_naive(X, Y, 'dyadic', undecided_prop)
    elif method == 'median_naive_repart':
        return split_naive_repart(X, Y, 'median')
    elif method == 'dyadic_naive_repart':
        return split_naive_repart(X, Y, 'dyadic')
    elif method == 'median_naive_overlap':
        return split_naive_overlap(X, Y, 'median')
    elif method == 'dyadic_naive_overlap':
        return split_naive_overlap(X, Y, 'dyadic')
    else:
        raise NotImplementedError('Available Methods: ' + str(__available_methods__))


def split_svm(X, Y, method, undecided_prop):
    N, D = X.shape
    labels = np.zeros(N).astype('bool')
    Y_offset = 0
    if method == 'dyadic':
        Y_lower = (1.0 - undecided_prop) * 0.5 * (np.max(Y) + np.min(Y))
        Y_upper = (1.0 + undecided_prop) * 0.5 * (np.max(Y) + np.min(Y))
    elif method == 'median':
        Y_lower, Y_upper = np.percentile(Y, np.array([100.0 * (1-undecided_prop) * 0.5,
                                                    100.0 * (1+undecided_prop) * 0.5]))
    idx1 = np.where(Y < Y_lower)[0]
    idx2 = np.where(Y >= Y_upper)[0]
    idxMid = np.where(np.logical_and(Y >= Y_lower, Y < Y_upper))[0]
    labels[idx2] = 1
    if len(idx1) > 0 and len(idx2) > 0:
        svm = LinearSVC(class_weight = 'balanced')
        svm = svm.fit(X, labels)
        # Modify labels so that second column contains the logical complement of the first
        labels = np.vstack((~labels, labels))
        labels[0,idxMid], labels[1, idxMid] = True, True
        return True, labels, svm
    else:
        return False, labels, None


def split_svm_repart(X, Y, method, undecided_prop):
    N, D = X.shape
    labels = np.zeros(N).astype('bool')
    if method == 'dyadic':
        Y_mid = 0.5 * (np.max(Y) + np.min(Y))
    elif method == 'median':
        Y_mid = np.median(Y)
    idx1 = np.where(Y < Y_mid)[0]
    idx2 = np.where(Y >= Y_mid)[0]
    labels[idx2] = 1
    if len(idx1) > 0 and len(idx2) > 0:
        svm = LinearSVC(class_weight = 'balanced')
        svm = svm.fit(X, labels)
        new_labels = svm.predict(X)
        increase_label = np.where(np.logical_and(new_labels, ~labels))[0]
        decrease_label = np.where(np.logical_and(~new_labels, labels))[0]
        labels[increase_label] = True
        labels[decrease_label] = False
        idx1 = np.where(~labels)[0]
        idx2 = np.where(labels)[0]
        labels = np.vstack((~labels, labels))
        if len(idx1) > 0 and len(idx2) > 0:
            return True, labels, svm
        else:
            return False, labels, None
    else:
        return False, labels, None


def split_svm_overlap(X, Y, method):
    N, D = X.shape
    labels = np.zeros(N).astype('bool')
    if method == 'dyadic':
        Y_mid = 0.5 * (np.max(Y) + np.min(Y))
    elif method == 'median':
        Y_mid = np.median(Y)
    idx1 = np.where(Y < Y_mid)[0]
    idx2 = np.where(Y >= Y_mid)[0]
    labels[idx2] = 1
    if len(idx1) > 0 and len(idx2) > 0:
        svm = LinearSVC(class_weight = 'balanced')
        svm = svm.fit(X, labels)
        new_labels = svm.predict(X)
        increase_label = np.where(np.logical_and(new_labels, ~labels))[0]
        decrease_label = np.where(np.logical_and(~new_labels, labels))[0]
        # Modify labels vector
        labels = np.vstack((~labels, labels))
        labels[0,decrease_label] = True
        labels[1,increase_label] = True
        if any(~labels[0,:]) and any(~labels[1,:]):
            # Checking for whether or not we decrease the cells at all anymore
            return True, labels, svm
        else:
            return False, labels, None
    else:
        return False, labels, None


def split_naive(X, Y, method, undecided_prop):
    N, D = X.shape
    labels = np.zeros(N).astype('bool')
    if method == 'dyadic':
        Y_lower = (1.0 - undecided_prop) * 0.5 * (np.max(Y) + np.min(Y))
        Y_upper = (1.0 + undecided_prop) * 0.5 * (np.max(Y) + np.min(Y))
    elif method == 'median':
        Y_lower, Y_upper = np.quantile(Y, np.array([(1.0-undecided_prop) * 0.5,
                                                    (1.0+undecided_prop) * 0.5]))
    idx1 = np.where(Y < Y_lower)[0]
    idx2 = np.where(Y >= Y_upper)[0]
    idxMid = np.where(np.logical_and(Y >= Y_lower, Y < Y_upper))[0]
    labels[idx2] = 1
    if len(idx1) > 0 and len(idx2) > 0:
        c1, c2 = np.mean(X[idx1, :], axis = 0), np.mean(X[idx2,:], axis = 0)
        direction = (c2-c1)/np.linalg.norm(c2-c1)
        svm = LinearSVC(class_weight = 'balanced')
        svm = svm.fit(np.reshape(X.dot(direction),(-1,1)), labels)
        # offset = 0.5 * (np.max(X.dot(direction)[labels == 0]) + np.min(X.dot(direction)[labels == 1]))
        classifier = Classifier(direction, svm)
        # Modify labels so that second column contains the logical complement of the first
        labels = np.vstack((~labels, labels))
        labels[0,idxMid], labels[1, idxMid] = True, True
        return True, labels, classifier
    else:
        return False, labels, None


def split_naive_repart(X, Y, method):
    N, D = X.shape
    labels = np.zeros(N).astype('bool')
    if method == 'dyadic':
        Y_mid = 0.5 * (np.max(Y) + np.min(Y))
    elif method == 'median':
        Y_mid = np.median(Y)
    idx1 = np.where(Y < Y_mid)[0]
    idx2 = np.where(Y >= Y_mid)[0]
    labels[idx2] = 1
    if len(idx1) > 0 and len(idx2) > 0:
        c1, c2 = np.mean(X[idx1, :], axis = 0), np.mean(X[idx2,:], axis = 0)
        direction = (c2-c1)/np.linalg.norm(c2-c1)
        svm = LinearSVC(class_weight = 'balanced')
        svm = svm.fit(np.reshape(X.dot(direction),(-1,1)), labels)
        # offset = 0.5 * (np.max(X.dot(direction)[labels == 0]) + np.min(X.dot(direction)[labels == 1]))
        classifier = Classifier(direction, svm)
        new_labels = classifier.predict(X)
        increase_label = np.where(np.logical_and(new_labels, ~labels))[0]
        decrease_label = np.where(np.logical_and(~new_labels, labels))[0]
        labels[increase_label] = True
        labels[decrease_label] = False
        idx1 = np.where(~labels)[0]
        idx2 = np.where(labels)[0]
        labels = np.vstack((~labels, labels))
        if len(idx1) > 0 and len(idx2) > 0:
            return True, labels, classifier
        else:
            return False, labels, None
    return False, labels, None


def split_naive_overlap(X, Y, method):
    N, D = X.shape
    labels = np.zeros(N).astype('bool')
    if method == 'dyadic':
        Y_mid = 0.5 * (np.max(Y) + np.min(Y))
    elif method == 'median':
        Y_mid = np.median(Y)
    idx1 = np.where(Y < Y_mid)[0]
    idx2 = np.where(Y >= Y_mid)[0]
    labels[idx2] = 1
    if len(idx1) > 0 and len(idx2) > 0:
        c1, c2 = np.mean(X[idx1, :], axis = 0), np.mean(X[idx2,:], axis = 0)
        direction = (c2-c1)/np.linalg.norm(c2-c1)
        svm = LinearSVC(class_weight = 'balanced')
        svm = svm.fit(np.reshape(X.dot(direction),(-1,1)), labels)
        # offset = np.median(X.dot(direction))
        classifier = Classifier(direction, svm)
        new_labels = classifier.predict(X)
        increase_label = np.where(np.logical_and(new_labels, ~labels))[0]
        decrease_label = np.where(np.logical_and(~new_labels, labels))[0]
        # Modify labels vector
        labels = np.vstack((~labels, labels))
        labels[0,decrease_label] = True
        labels[1,increase_label] = True
        if any(~labels[0,:]) and any(~labels[1,:]):
            # Checking for whether or not we decrease the cells at all anymore
            return True, labels, classifier
        else:
            return False, labels, None
    return False, labels, None





if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt
    sys.path.insert(0, '../../../DataSets/')
    from handler_UCI_Concrete import read_all
    data = read_all(scaling = 'MeanVar')#, feature_adjustment = False)
    X, Y = data[:,:-1], data[:,-1]
    Y = np.log(Y) # Using Logarithmic Data
    tree = InverseRegressionTree(method = 'median_svm', height = 0,
                                 minLeafSize = 5, predict_level = 10,
                                 undecided_prop = 0.5)
    tree.fit(X, Y)
    tree.print_Nused()

    import pdb; pdb.set_trace()
    # print X.shape
    # kf = KFold(n_splits = 10, shuffle = True)
    # for idx_train, idx_test in kf.split(X):
    #     tree = tree.fit(X[idx_train,:], Y[idx_train])
    #     import pdb; pdb.set_trace()
    # print tree.predict(X[0:10,:])
    # print Y[0:10]
