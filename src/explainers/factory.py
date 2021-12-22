"""
Factory design pattern for the explainers.
"""

from explainers.explainers.limeexplainer import LimeExplainer
from explainers.explainers.shapexplainer import ShapleyExplainer

class ExplainerFactory:
  
  def __init__(self):
    pass # nothing to do here atm
  
  def create_explainer(self, type):
    if type == "lime":
      return LimeExplainer()
    elif type == "shap":
      return ShapleyExplainer()