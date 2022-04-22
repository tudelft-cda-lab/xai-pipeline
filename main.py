"""
Main script, taking in input parameters and starting subroutines.
"""

import os
import sys

workdir = os.path.dirname(os.path.realpath(__file__))

import argparse
import pickle

import src.classifiers
import src.explainers
import src.parsers

from src.parsers.factory import ParserFactory
from src.classifiers.factory import ClassifierFactory
from src.explainers.factory import ExplainerFactory


def test_filepaths(args):
  """[Tests the filepaths set by arguments]

  Args:
      args ([args]): [The args as returned by argumentparser]

  Raises:
      Exception: [Path to inputfile not provided or wrong]
  """
  X_train_f = args.xtrain
  if not os.path.isfile(X_train_f):
    raise Exception("Invalid path to xtrain-file: {}".format(X_train_f))

  y_train_f = args.ytrain
  if not os.path.isfile(y_train_f):
    raise Exception("Invalid path to ytrain-file: {}".format(y_train_f))

  X_test_f = args.xtest
  if X_test_f and not os.path.isfile(X_test_f):
    raise Exception("Invalid path to xtest-file: {}".format(X_test_f))

  y_test_f = args.ytest
  if y_test_f and not os.path.isfile(y_test_f):
    raise Exception("Invalid path to ytest-file: {}".format(y_test_f))

  X_explain_f = args.xexplain
  if not os.path.isfile(X_explain_f):
    raise Exception("Invalid path to xexplain-file: {}".format(X_explain_f))

  y_explain_f = args.yexplain
  if not os.path.isfile(y_explain_f):
    raise Exception("Invalid path to yexplain-file: {}".format(y_explain_f))

  inifile = args.ini
  if inifile and not os.path.isfile(inifile):
    raise Exception("Invalid path to ini-file: {}".format(inifile))


if __name__ == "__main__":
  argparser = argparse.ArgumentParser(description='The pipeline for the explainability experiments.')

  argparser.add_argument('parser', type=str, help='The argparser as a string')
  argparser.add_argument('classifier', type=str, help='The classifier as a string')
  argparser.add_argument('explainer', type=str, help='The explainer as a string')

  argparser.add_argument('--xtrain', type=str, help='Path to X-train. Mandatory.')
  argparser.add_argument('--ytrain', type=str, help='Path to y-train. Mandatory.')

  argparser.add_argument('--xtest', default=None, type=str, help='Path to X-test. Not mandatory, but some classifiers will need this for training.')
  argparser.add_argument('--ytest', default=None, type=str, help='Path to y-test. Not mandatory, but some classifiers will need this for training.')
  
  argparser.add_argument('--xexplain', type=str, help='Path to X-explain. Mandatory.')
  argparser.add_argument('--yexplain', type=str, help='Path to y-explain. Mandatory.')

  argparser.add_argument('--ini', type=str, default=None, help='The explainer as a string')
  argparser.add_argument('--load-classifier', type=str, default=None, help='Path to a pickled classifier file. If provided, this classifier will be loaded rather than a new one trained.')
  argparser.add_argument('--output-path', type=str, default=workdir, help='Output dir for this experiment. Default is the script\'s directory.')

  args = argparser.parse_args()

  test_filepaths(args)

  X_train_f = args.xtrain
  y_train_f = args.ytrain
  X_test_f = args.xtest
  y_test_f = args.ytest
  X_explain_f = args.xexplain
  y_explain_f = args.yexplain

  inifile = args.ini

  pfactory = ParserFactory()
  fileparser = pfactory.create_parser(args.parser)

  print("Parsing the input files.")
  X_train = fileparser.parse(X_train_f)
  y_train = fileparser.parse(y_train_f)

  if X_test_f and y_test_f:
    X_test = fileparser.parse(X_test_f)
    y_test = fileparser.parse(y_test_f)

  X_explain = fileparser.parse(X_explain_f)
  y_explain = fileparser.parse(y_explain_f)

  output_path = args.output_path
  if not os.path.isdir(output_path):
    os.mkdir(output_path)

  load_classifier = args.load_classifier
  if load_classifier:
    if not os.path.isfile(load_classifier):
      raise Exception("Invalid path to classifier provided: {}".format(load_classifier))
    classifier = pickle.load(open(load_classifier, "rb"))
  else:
    cfactory = ClassifierFactory()
    classifier = cfactory.create_classifier(args.classifier)
    print("Starting the training of the classifier.")
    classifier.fit(X_train, y_train)
    classifier.print_wrong_predictions(X_explain, y_explain, output_path)
    pickle.dump(classifier, open(os.path.join(output_path, "classifier.pk"), "wb"))

  print("Starting the explanation step.")
  efactory = ExplainerFactory()
  explainer = efactory.create_explainer(args.explainer)
  explainer.explain(classifier, X_explain, y_explain)

  print("Finished explanations. Saving results to {}".format(output_path))
  explainer.save_results(output_path)