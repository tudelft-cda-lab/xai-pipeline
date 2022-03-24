"""
Explainer to explainable boosting machine. WARNING: Only works with ExplainableBoostingRegressor() from interpret package.

EBM: https://interpret.ml/docs/ebm.html
"""

import configparser
import pickle
import os

from explainers.explainerbase import ExplainerBase

class EBMExplainer(ExplainerBase):
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
    
    self.global_explanation = classifier.explain_global()
    self.local_explanations = classifier.explain_local(X, y)
    

  def save_results(self, output_path):
    pickle.dump(self.global_explanation, open(os.path.join(output_path, "ebm_global_exps.pk"), "wb"))
    pickle.dump(self.local_explanations, open(os.path.join(output_path, "ebm_local_exps.pk"), "wb"))