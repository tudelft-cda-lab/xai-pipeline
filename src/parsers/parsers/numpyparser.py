"""
Simple numpy-parser. Takes in the path a .np file and returns the numpy array.
"""

import numpy as np

from src.parsers.parserbase import ParserBase


class NumpyParser(ParserBase):
  def __init__(self):
    super().__init__()


  def parse(self, infile):
    """[Abstract base method to parse the infile.]

    Args:
        infile ([str]): [The path to the infile.]
    """
    assert(infile.endswith(".npy"))
    
    return np.load(infile)