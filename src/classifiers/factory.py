"""
Factory design pattern for the classifiers
"""

from classifiers.classifiers.randomforest import RandomForest
from classifiers.classifiers.groot_randomforest import GrootRandomForest

class ClassifierFactory:
  
  def __init__(self):
    pass # nothing to do here atm
  
  def create_classifier(self, classifier):
    if classifier == "randomforest":
      return RandomForest()
    elif classifier == "grootrandomforest":
      return GrootRandomForest()
