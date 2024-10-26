

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import ConfigSpace
from config_encoder import ConfigEncoder

class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.encoder = ConfigEncoder(self.config_space)
        self.model = Pipeline([
                               ('model', RandomForestRegressor())])  

    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """
        y = df['score']
        self.features = df.columns[:-1]
        self.label = df.columns[-1]
        df_encoded = self.encoder.transform(df)
        self.df = df_encoded
        self.model.fit(df_encoded[self.features],y)

    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        if isinstance(theta_new, list): # len(theta_new)>1:
            X = pd.DataFrame(theta_new,index=[x for x in range(len(theta_new))])
        elif isinstance(theta_new, ConfigSpace.Configuration):
            X = pd.DataFrame([dict(theta_new)])
        else:
            X = pd.DataFrame(theta_new)
        for col in self.features:
            if col not in X.columns:
                X[col] = None
        T = self.encoder.transform(X)
        return self.model.predict(T[self.features])
