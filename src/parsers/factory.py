"""
Factory design pattern for the parsers.
"""

from src.parsers.parsers.csvparser import CSVParser
from src.parsers.parsers.numpyparser import NumpyParser

class ParserFactory:
  
  def __init__(self):
    pass # nothing to do here atm
  
  def create_parser(self, type):
    if type == "csv":
      return CSVParser()
    elif type == "npy":
      return NumpyParser()