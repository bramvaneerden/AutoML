import ConfigSpace

import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd


class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = RandomForestRegressor()

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
        encoded = pd.get_dummies(df)

        features = encoded.columns[:-1]
        label = encoded.columns[-1]

        X_train, X_test, y_train, y_test = train_test_split(encoded[features],encoded[label],test_size=0.33)

        param_dist = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }

        rf_random = RandomizedSearchCV(estimator=self.model, param_distributions=param_dist,
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

        rf_random.fit(X_train,y_train)
        print("best params: " , rf_random.best_params_ , "\n best score: " , rf_random.best_score_)
        # print("cv results: ", rf_random.cv_results_)

        self.model = rf_random.best_estimator_
        y_pred = self.model.predict(X_test)
        print('mse:', mean_squared_error(y_pred,y_test),'r2:', r2_score(y_test,y_pred))




        # features = df.columns[:-1]
        # label = df.columns[-1]
        # X_train, X_test, y_train, y_test = train_test_split(df[features],df[label],test_size=0.33)
        # self.model.fit(X_train,y_train)
        # y_pred = self.model.predict(X_test)
        # print('mse:', mean_squared_error(y_pred,y_test),'r2:', r2_score(y_test,y_pred))

    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        raise NotImplementedError()
