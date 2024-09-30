import ConfigSpace

import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd


class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = None

    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model

        LJS: added first start. I would think we need to use the train_test_split to do a small parameter search to optimize the model. 
        But i'm not sure.
        """
        self.df = df 
        features = df.columns[:-1]
        label = df.columns[-1]
        X_train, X_test, y_train, y_test = train_test_split(df[features],df[label],test_size=0.33)
        self.model.fit(X_train,y_train)
        y_pred = self.model.predict(X_test)
        print('mse:', mean_squared_error(y_pred,y_test),'r2:', r2_score(y_test,y_pred))

    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        raise NotImplementedError()
