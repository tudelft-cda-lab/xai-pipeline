"""
Factory design pattern for the explainers.
"""

from src.explainers.explainers.limeexplainer import LimeExplainer
from src.explainers.explainers.shapexplainer import ShapleyExplainer
from src.explainers.explainers.eli5explainer import ELI5Explainer
from src.explainers.explainers.explainableboostingexplainer import EBMExplainer

class ExplainerFactory:
  
  def __init__(self):
    pass # nothing to do here atm
  
  def create_explainer(self, type):
    if type == "lime":
      return LimeExplainer()
    elif type == "shap":
      return ShapleyExplainer()
    elif type == "eli5":
      return ELI5Explainer()
    elif type == "ebm":
      return EBMExplainer()
    else:
      raise Exception("Unknown explainer: ", type)
