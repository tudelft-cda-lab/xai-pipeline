"""
Abstract base class for the classifiers.

ABC: https://www.geeksforgeeks.org/abstract-classes-in-python/
"""

from abc import ABC, abstractmethod

import configparser
import os

class ClassifierBase(ABC, object):
  def __init__(self):
    self.is_fitted = False
    self.read_ini()

  
  def test_is_fitted(self):
    if not self.is_fitted:
      raise Exception("Error: Classifier not fitted yet.")


  def print_wrong_predictions(self, X, y, output_path):
    self.test_is_fitted()
    pred = self.predict(X)
    outf = open(os.path.join(output_path, "pred_vs_true.csv"), "wt")
    for i, (y_pred, y_true) in enumerate(zip(pred, y)):
      outf.write((", ".join([str(i), "Pred: {}".format(y_pred), "True value: {}".format(y_true), "correct" if y_pred == y_true else "wrong"]) + "\n"))


  def fit(self, X, y):
    """[Fit a predictor based on X and y.]

    Args:
        X ([TODO]): [Input]
        y ([TODO]): [Label]
    """
    self.classifier.fit(X, y)
    self.is_fitted = True


  def predict(self, X):
    """[Predict based on X, return a prediction.]

    Args:
        X ([TODO]): [Input]
    """
    y_pred = self.classifier.predict(X)
    return y_pred

  @abstractmethod
  def read_ini(self):
    """
    Read the .ini file of a predifined filepath.
    See the randomforest example.
    """
    pass