"""
An example of an explainer. Implements the SHAP explainer.
"""
import shap

import configparser
import pickle
import os

import matplotlib.pyplot as plt

from explainers.explainerbase import ExplainerBase


class ShapleyExplainer(ExplainerBase):
  def __init__(self):
    super().__init__()


  def explain(self, classifier, X, y):
    """[Runs a shap explanation, and saves results as object attribute.]

    Args:
        classifier ([ClassifierBase]): [The used classifier]
        X ([np.array]): [X]
        y ([np.array]): [y]
    """
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(scriptdir, "../parameters/shapexplainer.ini")
    if not os.path.isfile(os.path.join(filedir)):
      raise Exception("Problem reading shapexplainer.ini file in ShapExplainer. Is the file existent?")

    config = configparser.ConfigParser()
    config.read(filedir)

    feature_names = config.get("DEFAULT" , "feature_names").split()
    explainer = shap.Explainer(classifier.predict, X, feature_names=feature_names)
    self.shap_values = explainer(X)
    self.classifier = classifier
    self.X = X
    

  def save_results(self, output_path):
    pickle.dump(self.shap_values, open(os.path.join(output_path, "shapley_explanations.pk"), "wb"))

    
    for i in range(len(self.shap_values)):
      shap.plots.force(self.shap_values[i, :], matplotlib=True, show=False)

      plt.savefig(os.path.join(output_path, f"shap_explanation_{i}.png"))
      plt.close()
    
    shap.summary_plot(self.shap_values, show=False, plot_size=(10,8))
    plt.savefig(os.path.join(output_path, "shap_summary_plot.png"))
    plt.close()
