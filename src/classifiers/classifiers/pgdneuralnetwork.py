from classifierbase import ClassifierBase

import configparser

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

# Disable tenorflow eager execution since it is incompatible with ART
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

from art.estimators.classification import KerasClassifier
from art.attacks.evasion import ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer, AdversarialTrainerMadryPGD

import numpy as np


class PgdNeuralNetwork(ClassifierBase):
    def __init__(self):
        super().__init__()

        self.params = dict()

        config = configparser.ConfigParser()
        config.read("src/classifiers/parameters/pgdneuralnetwork.ini")

        for section in config.sections():
            for param in config[section]:
                value = config[section][param]
                self.params[param] = eval(value)

        self.classifier = PgdNeuralNetworkImpl()
        self.classifier.set_params(**self.params)

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        y_pred = self.classifier.predict(X)
        return y_pred


class PgdNeuralNetworkImpl(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        min_value=-1e10,
        max_value=1e10,
        eps=0.1,
        eps_step=0.01,
        max_iter=20,
        nb_epochs=10,
        batch_size=32,
        verbose=False,
        undersample=True,
        random_state=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.min_value = min_value
        self.max_value = max_value
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.undersample = undersample
        self.random_state = random_state

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        input_shape = X.shape[1:]

        # Define keras sequential model from hidden_layer_sizes
        self.model = Sequential(
            [
                layers.Dense(
                    self.hidden_layer_sizes[0],
                    activation="relu",
                    input_shape=input_shape,
                ),
                *[
                    layers.Dense(self.hidden_layer_sizes[i], activation="relu")
                    for i in range(1, len(self.hidden_layer_sizes))
                ],
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

        if self.undersample:
            X_0 = X[y == 0]
            X_1 = X[y == 1]
            if len(X_0) > len(X_1):
                X_0 = X_0[:len(X_1)]
            else:
                X_1 = X_1[:len(X_0)]
            X = np.concatenate((X_0, X_1), axis=0)
            y = np.concatenate((np.zeros(len(X_0), dtype=np.int8), np.ones(len(X_1), dtype=np.int8)), axis=0)

        if self.verbose:
            print(self.model.summary())
            print("Training on {} samples".format(len(X)))
            if self.undersample:
                print(f"After undersampling (class counts: {np.bincount(y)}):")
            else:
                print(f"Without undersampling (class counts: {np.bincount(y)}):")

        self.classifier = KerasClassifier(
            clip_values=(self.min_value, self.max_value), model=self.model, use_logits=False
        )
        attacker = ProjectedGradientDescent(
            estimator=self.classifier,
            eps=self.eps,
            eps_step=self.eps_step,
            norm=np.inf,
            max_iter=self.max_iter,
            verbose=False,
        )

        X_adv = X.copy()
        for epoch in range(self.nb_epochs):
            self.model.fit(X_adv, y, epochs=1, batch_size=self.batch_size)

            if epoch < self.nb_epochs - 1:
                X_adv = attacker.generate(X)

        predictions = self.model.predict(X).ravel()
        print("Train predictions:", np.bincount(np.round(predictions).astype(np.int8)))

    def predict(self, X):
        X = self.scaler.transform(X)
        return np.round(self.model.predict(X)).ravel().astype(np.int8)
