# coding: utf8

"""
This implements a simple and easy to use (shallow) feed forward neural network.
This file serves just functions as a wrapper for scikit-neuralnetwork.
"""
import numpy as np
import inspect
import os

from sklearn.base import BaseEstimator, RegressorMixin

from sknn.mlp import Regressor, Layer

class FeedForwardNetwork(BaseEstimator, RegressorMixin):
    """
    Wrapper for sknn learn implementation for the special case of a feed forward
    neural network with 1 hidden layer. Output layer is linear (as normal for
    regression tasks).
    """

    def __init__(self, n_hidden = 10, activation = "Sigmoid", hidden_options = {}, output_options = {}, general_options = {}):
        """
        Parameters
        ------------------------------------------------------------------------
        n_hidden: int
            Number of neurons in the hidden layer

        activation : String
            Identifier for the activation function of the hidden layern. Can be
            'Rectifier', 'Sigmoid', 'Tanh', 'ExpLin', 'Linear', 'Softmax'.

        hidden_options : dict
            Additional arguments that are passed to the hidden layer (e.g. dropout, normalization).

        output_options : dict
            Additional arguments that are passed to the output layer (e.g. dropout, normalization).

        general_options : dict
            Additional arguments that are passed to the neural network (e.g. learning rate, n_iter).
        """
        # Set attributes of object to the same name as given in the argument list.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        self._errors = []
        for arg, val in values.items():
            setattr(self, arg, val)


    def fit(self, X, y=None):
        """
        """
        self.nn_ = Regressor(
            layers=[
                Layer(self.activation, units=self.n_hidden, **self.hidden_options),
                Layer("Linear", **self.output_options)],
            # callback={'on_epoch_finish': store_stats},
            **self.general_options)
        self.nn_.fit(X, y)
        return self


    def predict(self, X, y=None):
        try:
            getattr(self, "nn_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")
        return self.nn_.predict(X)

# def store_stats(avg_valid_error, avg_train_error, **kwargs):
#     print kwargs['best_params']
        # self._errors.append((avg_valid_error, avg_train_error))
