"""
Implementation of an example classifier. Wraps the sklearn RandomForest to 
have a generic interface for own, homegrown classifiers as well. 

Read up configparser here: https://docs.python.org/3/library/configparser.html
"""

from classifierbase import ClassifierBase

from sklearn.ensemble import RandomForestClassifier

import configparser


class RandomForest(ClassifierBase):
  def __init__(self):
    super().__init__()

    self.params = dict()

    config = configparser.ConfigParser()
    config.read("src/classifiers/parameters/randomforest.ini")
    
    for section in config:
      for param in config[section]:
        value = config[section][param]
        self.params[param] = eval(value)

    self.classifier = RandomForestClassifier()
    self.classifier.set_params(**self.params)


  def fit(self, X, y):
    self.classifier.fit(X, y) 


  def predict(self, X):
    y_pred = self.classifier.predict(X)
    return y_pred