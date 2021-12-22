"""
Factory design pattern for the classifiers
"""

from classifiers.classifiers.randomforest import RandomForest

class ClassifierFactory:
  
  def __init__(self):
    pass # nothing to do here atm
  
  def create_classifier(self, classifier):
    if classifier == "randomforest":
      return RandomForest()