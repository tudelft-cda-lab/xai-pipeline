"""
An example of an explainer. Implements the ELI5 explainer.
"""
import eli5

from IPython import display
import configparser
import pickle
import os

import matplotlib.pyplot as plt

from explainers.explainerbase import ExplainerBase


class ELI5Explainer(ExplainerBase):
  def __init__(self):
    super().__init__()


  def explain(self, classifier, X, y):
    """[Runs a ELI5 explanation, and saves results as object attribute.]

    Args:
        classifier ([ClassifierBase]): [The used classifier]
        X ([np.array]): [X]
        y ([np.array]): [y]
    """
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    filedir = os.path.join(scriptdir, "../parameters/eli5explainer.ini")
    if not os.path.isfile(os.path.join(filedir)):
      raise Exception("Problem reading eli5explainer.ini file in Eli5Explainer. Is the file existent?")

    config = configparser.ConfigParser()
    config.read(filedir)

    feature_names = config.get("DEFAULT" , "feature_names").split()
    self.explanation = eli5.explain_weights(classifier.classifier, feature_names=feature_names)
    

  def save_results(self, output_path):
    html_file = eli5.formatters.html.format_as_html(explanation=self.explanation)
    with open(output_path + 'ELI5_weights_explanations.html', 'w') as output:
      output.write(html_file)
