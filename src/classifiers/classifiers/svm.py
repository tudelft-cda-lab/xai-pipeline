"""
Read up configparser here: https://docs.python.org/3/library/configparser.html
"""

from classifierbase import ClassifierBase

from sklearn.svm import SVC, LinearSVC

import configparser


class SupportVectorMachine(ClassifierBase):
  def __init__(self):
    super().__init__()

    #self.classifier = SVC(kernel="rbf") # TODO: give parameters from .ini file
    self.classifier = LinearSVC() # TODO: give parameters from .ini file


  def read_ini(self):
    self.params = dict()

    config = configparser.ConfigParser()

    # TODO: uncomment and set path straight
    #if not os.path.isfile("../parameters/limeexplainer.ini"):
    #  raise Exception("Problem reading limeexplainer.ini file in LimeExplainer. Is the file existent?")

    try:
      config.read("../parameters/svm.ini")
    except:
      raise Exception("Problem reading svm.ini file. Is the file existent and clean?")

    for section in config:
      for param in config[section]:
        value = config[section][param]
        if param == "kernel":
          self.params[param] = str(value) # TODO: is there a more elegant way to do this?
          # TODO: smash the params into the random forest before training