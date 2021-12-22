from classifierbase import ClassifierBase

from groot.model import GrootRandomForestClassifier

import configparser


class GrootRandomForest(ClassifierBase):
  def __init__(self):
    super().__init__()

    self.params = dict()

    config = configparser.ConfigParser()

    try:
      config.read("../parameters/groot_randomforest.ini")
    except:
      raise Exception("Problem reading groot_randomforest.ini file in GrootRandomForest. Is the file existent and clean?")

    for section in config:
      for param in config[section]:
        value = config[section][param]
        self.params[param] = eval(value)

    self.classifier = GrootRandomForestClassifier()
    self.classifier.set_params(**self.params)

  def fit(self, X, y):
    self.classifier.fit(X, y) 


  def predict(self, X):
    y_pred = self.classifier.predict(X)
    return y_pred
