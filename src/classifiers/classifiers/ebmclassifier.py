"""
Implementation of an example classifier. Wraps the sklearn RandomForest to 
have a generic interface for own, homegrown classifiers as well. 

Read up configparser here: https://docs.python.org/3/library/configparser.html

Interpret ML: https://github.com/interpretml/interpret
EBM: https://interpret.ml/docs/ebm.html
"""

from classifierbase import ClassifierBase

from interpret.glassbox import ExplainableBoostingClassifier

import configparser


class EBMClassifier(ClassifierBase):
  def __init__(self):
    super().__init__()

    self.classifier = ExplainableBoostingClassifier() # TODO: give parameters from .ini file

  def explain_global(self):
    return self.classifier.explain_global()

  def explain_local(self, X, y):
    return self.classifier.explain_local(X, y)

  def read_ini(self):
    self.params = dict()

    config = configparser.ConfigParser()

    # TODO: uncomment and set path straight
    #if not os.path.isfile("../parameters/limeexplainer.ini"):
    #  raise Exception("Problem reading limeexplainer.ini file in LimeExplainer. Is the file existent?")

    try:
      config.read("../parameters/explainableboostingmachine.ini")
    except:
      raise Exception("Problem reading explainableboostingmachine.ini file. Is the file existent and clean?")

    for section in config:
      for param in config[section]:
        value = config[section][param]
        if param == "n_estimators" or param == "max_depth":
          self.params[param] = int(value) # TODO: is there a more elegant way to do this?
          # TODO: smash the params into the random forest before training