"""
Abstract base class for the classifiers.

ABC: https://www.geeksforgeeks.org/abstract-classes-in-python/
"""

from abc import ABC, abstractmethod

import configparser

class ClassifierBase(ABC, object):
  def __init__(self):
    pass

  @abstractmethod
  def fit(self, X, y):
    """[Fit a predictor based on X and y.]

    Args:
        X ([TODO]): [Input]
        y ([TODO]): [Label]
    """
    pass

  @abstractmethod
  def predict(self, X):
    """[Predict based on X, return a prediction.]

    Args:
        X ([TODO]): [Input]
    """
    pass