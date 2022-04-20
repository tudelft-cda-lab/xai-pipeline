"""
Factory design pattern for the classifiers
"""

from classifiers.classifiers.randomforest import RandomForest
from classifiers.classifiers.decisiontree import DecisionTree
from classifiers.classifiers.svm import SupportVectorMachine
from classifiers.classifiers.ebmclassifier import EBMClassifier

class ClassifierFactory:
  
  def __init__(self):
    pass # nothing to do here atm
  
  def create_classifier(self, classifier):
    if classifier == "randomforest":
      return RandomForest()
    elif classifier == "decisiontree":
      return DecisionTree()
    elif classifier == "svm":
      return SupportVectorMachine()
    elif classifier == "ebmclassifier":
      return EBMClassifier()
    else:
      raise Exception("Unknown classifier: ", classifier)