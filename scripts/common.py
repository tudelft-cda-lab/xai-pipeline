"""
Common functions for the scripts in this directory.
"""


def preprocess_df(df):
    df = df.iloc[:, 1:] # drop index
    df.y_pred = df.y_pred.apply(lambda x: int(x.split()[-1]))
    df.y_true = df.y_true.apply(lambda x: int(x.split()[-1].lstrip("[").rstrip("]")))
    df.correct_predicted = df.correct_predicted.apply(lambda x: x.strip())
    df.correct_predicted = df.correct_predicted.map({"correct": True, "wrong": False})
    return df