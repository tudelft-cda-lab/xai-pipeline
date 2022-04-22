"""
Implementation of an example classifier. Wraps the sklearn RandomForest to 
have a generic interface for own, homegrown classifiers as well. 

Read up configparser here: https://docs.python.org/3/library/configparser.html
"""

from src.classifiers.classifierbase import ClassifierBase

from sklearn.tree import DecisionTreeClassifier

import configparser


class DecisionTree(ClassifierBase):
  def __init__(self):
    super().__init__()

    self.classifier = DecisionTreeClassifier() # TODO: give parameters from .ini file


  def read_ini(self):
    self.params = dict()

    config = configparser.ConfigParser()

    # TODO: uncomment and set path straight
    #if not os.path.isfile("../parameters/limeexplainer.ini"):
    #  raise Exception("Problem reading limeexplainer.ini file in LimeExplainer. Is the file existent?")

    try:
      config.read("../parameters/decisiontree.ini")
    except:
      raise Exception("Problem reading randomforest.ini file in RandomForest. Is the file existent and clean?")

    for section in config:
      for param in config[section]:
        value = config[section][param]
        if param == "max_depth":
          self.params[param] = int(value) # TODO: is there a more elegant way to do this?
          # TODO: smash the params into the random forest before training