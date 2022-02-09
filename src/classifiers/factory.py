"""
Factory design pattern for the classifiers
"""

class ClassifierFactory:
  
  def __init__(self):
    pass # nothing to do here atm
  
  def create_classifier(self, classifier):
    if classifier == "randomforest":
      from classifiers.classifiers.randomforest import RandomForest
      return RandomForest()
    elif classifier == "grootrandomforest":
      from classifiers.classifiers.grootrandomforest import GrootRandomForest
      return GrootRandomForest()
    elif classifier == "pgdneuralnetwork":
      from classifiers.classifiers.pgdneuralnetwork import PgdNeuralNetwork
      return PgdNeuralNetwork()
    else:
      raise Exception("Unknown classifier: " + classifier)
