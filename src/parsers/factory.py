"""
Factory design pattern for the parsers.
"""

from parsers.csvparser import CSVParser
from parsers.numpyparser import NumpyParser

class ParserFactory:
  
  def __init__(self):
    pass # nothing to do here atm
  
  def create_parser(self, type):
    if type == "csv":
      return CSVParser()
    elif type == "npy":
      return NumpyParser()