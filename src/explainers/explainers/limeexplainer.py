"""
An example of an explainer. Implements the LIME explainer.
"""

import lime
from lime import lime_tabular

import configparser
import pickle
import os

import matplotlib.pyplot as plt

from explainers.explainerbase import ExplainerBase


class LimeExplainer(ExplainerBase):
  def __init__(self):
    super().__init__()


  def explain(self, classifier, X, y):
    """[Runs a lime explanation, and saves results as object attribute.]

    Args:
        classifier ([ClassifierBase]): [The used classifier]
        X ([np.array]): [X]
        y ([np.array]): [y]

    Raises:
        Exception: [.ini file does not exist]
    """
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(scriptdir, "../parameters/limeexplainer.ini")
    if not os.path.isfile(os.path.join(filedir)):
      raise Exception("Problem reading limeexplainer.ini file in LimeExplainer. Is the file existent?")

    config = configparser.ConfigParser()
    config.read(filedir)
    
    #params = config["DEFAULT"]

    feature_names = config.get("DEFAULT" , "feature_names").split()
    try:
      categorical_feature_names = config.get("DEFAULT" , "categorical_features").split()
    except: 
      categorical_feature_names = None
      print("No categorical feature names provided to limeexplainer. Adjustable in corresponding .ini")
    categorical_feature_indices = [i for i in range(len(feature_names)) if feature_names[i] in categorical_feature_names] # a bit inefficient, but ok for our small arrays

    explainer = lime_tabular.LimeTabularExplainer(X, 
                                                  feature_names=feature_names,
                                                  categorical_features=categorical_feature_indices,
                                                  categorical_names=categorical_feature_names,
                                                  class_names=["benign", "malicious"],
                                                  mode="classification", 
                                                  # discretize_continuous=False,
                                                  verbose=False
                                                )

    exps = list()
    for index in range(X.shape[0]):
      exp = explainer.explain_instance(data_row=X[index, :], predict_fn=classifier.classifier.predict_proba) # TODO: have a better predict function interface here
      exps.append(exp)

    self.explanations = exps

  def save_results(self, output_path):
    pickle.dump(self.explanations, open(os.path.join(output_path, "lime_explanations.pk"), "wb"))

    for i, exp in enumerate(self.explanations):
      #fig = exp.as_pyplot_figure(show_table=True, show_all=True)
      #plt.savefig("explanation_{}.jpg".format(i))
      exp.save_to_file(os.path.join(output_path, "explanation_{}.html".format(i)))