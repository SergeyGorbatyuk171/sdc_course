# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import inv


def kalman_transit_covariance(S, A, R):
    """
    :param S: Current covariance matrix
    :param A: Either transition matrix or jacobian matrix
    :param R: Current noise covariance matrix
    """
    new_S = A.dot(S).dot(A.T) + R
    return new_S


def kalman_process_observation(mu, S, observation, C, Q):
    """
    Performs processing of an observation coming from the model: z = C * x + noise
    :param mu: Current mean
    :param S: Current covariance matrix
    :param observation: Vector z
    :param C: Observation matrix
    :param Q: Noise covariance matrix (with zero mean)
    """
    K = S.dot(C.T).dot(inv(C.dot(S).dot(C.T) + Q))
    new_mu = mu + K.dot(observation - C.dot(mu))
    new_S = (np.eye(K.shape[0]) - K.dot(C)).dot(S)
    return new_mu, new_S
