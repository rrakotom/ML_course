# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

def compute_loss_mse(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    error = y - np.dot(tx, w)
    return np.dot(error.T, error)/(2*len(error))

def compute_loss_mae(y, tx, w):
    """Calculate the loss using either MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    error = y - np.dot(tx, w)
    return np.sum(np.abs(error))/len(error)

def compute_loss(y, tx, w, cost = 'mse'):
    if(cost == 'mse'):
        return compute_loss_mse(y, tx, w)
    else:
        return compute_loss_mae(y, tx, w)