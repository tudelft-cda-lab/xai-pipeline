"""
The simplest fileparser. Takes in the path a .csv file and returns the opened file as a numpy array.
"""

import pandas as pd

from src.parsers.parserbase import ParserBase


class CSVParser(ParserBase):
  def __init__(self):
    super().__init__()

  def parse(self, infile, **kwargs):
    """[Abstract base method to parse the infile.]

    Args:
        infile ([str]): [The path to the infile.]
    """
    assert(infile.endswith(".csv"))

    header = 1 if "has_header" in kwargs else None
    df = pd.read_csv(infile, header=header)
    arr = df.to_numpy()
    return arr