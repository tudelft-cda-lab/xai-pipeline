"""
Abstract base class for the parsers.

ABC: https://www.geeksforgeeks.org/abstract-classes-in-python/
"""

from abc import ABC, abstractmethod

import configparser

class ParserBase(ABC, object):
  def __init__(self):
    pass


  @abstractmethod
  def parse(self, infile, **kwargs):
    """[Abstract base method to parse the infile.]

    Args:
        infile ([str]): [The path to the infile.]
    """
    pass