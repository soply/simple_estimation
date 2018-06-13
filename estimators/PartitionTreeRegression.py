# coding: utf8
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import mean_squared_error


__available_methods__ = ['pca_tree', 'kd_tree', 'rp_tree', '2M_tree']


class CrossvalidatedRegressionTree(BaseEstimator, RegressorMixin):
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
        self.tree_ = RegressionTree(method = method, predict_level = 0,
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
        return self

    def predict(self, Z):
        return self.tree_.predict(Z)



class RegressionTree(BaseEstimator, RegressorMixin):


    def __init__(self, method = 'pca_tree', predict_level = 0, height = 0,
                 minLeafSize = 5):
        self.predict_level = predict_level
        self.height = height
        self.method = method
        self.minLeafSize = minLeafSize
        self.Ymean_ = None
        self.left_ = None
        self.right_ = None
        self.v_classifier_ = None
        self.t_classifier_ = None


    def fit(self, X, Y):
        N, D = X.shape
        self.Ymean_ = np.mean(Y)
        # Sepparate data by chosen method
        if N <= self.minLeafSize:
            return
        else:
            labels, v, t = split(X, Y, self.method)
            if len(np.where(labels == 0)[0]) == 0 or \
                    len(np.where(labels == 1)[0]) == 0:
                return
            self.v_classifier_ = v
            self.t_classifier_ = t
            self.left_ = RegressionTree(method = self.method,
                                        height = self.height + 1,
                                        predict_level = self.predict_level,
                                        minLeafSize = self.minLeafSize)
            self.right_ = RegressionTree(method = self.method,
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
            getattr(self, "v_classifier_")
            getattr(self, "t_classifier_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        if len(Z.shape) == 1:
            Z = Z.reshape((1,-1))
        projections = self.v_classifier_.dot(Z.T)
        labels = np.zeros(Z.shape[0]).astype('int')
        labels[projections > self.t_classifier_] = 1
        return labels

    def _change_prediction_level(self, new_level):
        self.predict_level = new_level
        if self.left_ is not None:
            self.left_._change_prediction_level(new_level)
        if self.right_ is not None:
            self.right_._change_prediction_level(new_level)



def split(X, Y, method):
    if method == 'pca_tree':
        return split_pca_tree(X, Y)
    elif method == 'kd_tree':
        return split_kd_tree(X, Y)
    elif method == 'rp_tree':
        return split_rp_tree(X, Y)
    elif method == '2M_tree':
        return split_2M_tree(X, Y)
    else:
        raise NotImplementedError('Methods: ')


def split_rp_tree(X, Y, dirs = 20):
    N, D = X.shape
    # Pick 20 random directions
    random_projections = np.random.normal(size = (dirs, D))
    dist_mat = pairwise_distances(X)
    min_diameter = np.max(dist_mat)
    dir = 0
    for j in range(dirs):
        random_projections[j,:] = random_projections[j,:]/np.linalg.norm(random_projections[j,:])
        projections = random_projections[j,:].dot(X.T)
        med_proj = np.median(projections)
        labels = np.zeros(N).astype('int')
        labels[projections >= med_proj] = 1
        if len(np.where(labels == 0)[0]) > 0:
            diam0 = np.max(dist_mat[labels == 0,:][:,labels == 0])
        else:
            # Maybe there are no points with that label -> diameter 0
            diam0 = 0
        if len(np.where(labels == 1)[0]) > 0:
            diam1 = np.max(dist_mat[labels == 1,:][:,labels == 1])
        else:
            diam1 = 0
        if np.maximum(diam0, diam1) < min_diameter:
            dir = j
            min_diameter = np.maximum(diam0, diam1)
    best_direction = random_projections[dir, :]
    projections = best_direction.dot(X.T)
    med_proj = np.median(projections)
    labels = np.zeros(N).astype('int')
    labels[projections >= med_proj] = 1
    return labels, best_direction, med_proj


def split_pca_tree(X, Y):
    N, D = X.shape
    pca = PCA()
    pca = pca.fit(X)
    largest_sv = pca.components_[0,:]
    projections = largest_sv.dot(X.T)
    med_proj = np.median(projections)
    labels = np.zeros(N).astype('int')
    labels[projections >= med_proj] = 1
    return labels, largest_sv, med_proj


def split_kd_tree(X, Y):
    N, D = X.shape
    max_spread = 0
    dim = 0
    for j in range(D):
        spread = np.max(X[:,j]) - np.min(X[:,j])
        if spread > max_spread:
            dim = j
            max_spread = spread
    coordinate_direction = np.zeros(D)
    coordinate_direction[dim] = 1.0
    projections = coordinate_direction.dot(X.T)
    med_proj = np.median(projections)
    labels = np.zeros(N).astype('int')
    labels[projections >= med_proj] = 1
    return labels, coordinate_direction, med_proj


def split_2M_tree(X, Y):
    N, D = X.shape
    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(X)
    labels = kmeans.labels_
    direction = kmeans.cluster_centers_[1,:] - kmeans.cluster_centers_[0,:]
    direction = direction/np.linalg.norm(direction)
    projections = direction.dot(X.T)
    med_proj = 0.5 * (kmeans.cluster_centers_[1,:] + \
                        kmeans.cluster_centers_[0,:]).dot(direction)
    # labels2 = np.zeros(N).astype('int')
    # labels2[projections >= med_proj] = 1
    return labels, direction, med_proj


if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt
    sys.path.insert(0, '../../../DataSets/')
    from handler_UCI_Concrete import read_all
    data = read_all(scaling = 'MeanVar')
    X, Y = data[:,:-1], data[:,-1]
    tree = CrossvalidatedRegressionTree(method = 'rp_tree', height = 0,
                          minLeafSize = 5, test_levels = [0,1,2,3,4,5,6,7,8,9,10,11])
    tree = tree.fit(X, Y)
    print tree.predict(X[0:1,:])
