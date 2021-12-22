"""
Abstract base class for the explainers.

ABC: https://www.geeksforgeeks.org/abstract-classes-in-python/
"""

from abc import ABC, abstractmethod

import configparser

class ExplainerBase(ABC, object):
  def __init__(self):
    pass


  @abstractmethod
  def explain(self, classifier, X, y):
    """[Print an explanation of the classifier based on X and y.]

    Args:
        classifier ([ClassifierBase]): [The polymorphic classifier.]
        X ([TODO]): [Input]
        y ([TODO]): [Label]
    """
    pass

  @abstractmethod
  def save_results(self, output_path):
    """[Saves the results in the specified output path]

    Args:
        output_path ([str]): [Output path]
    """