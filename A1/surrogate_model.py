

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

class SurrogateModel:

    def __init__(self, config_space):
        self.config_space = config_space
        self.df = None
        self.model = Pipeline([('encoder',OneHotEncoder()),('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), ('model', RandomForestRegressor())])  
    
    def fit(self, df):
        """
        Receives a data frame, in which each column (except for the last two) represents a hyperparameter, the
        penultimate column represents the anchor size, and the final column represents the performance.

        :param df: the dataframe with performances
        :return: Does not return anything, but stores the trained model in self.model
        """
        self.df = df 
        features = df.columns[:-1]
        label = df.columns[-1]
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(df[features],df[label],test_size=0.2)

        # Define random forest search space
        search_space  = {
                'model__criterion': Categorical(['squared_error', 'absolute_error', 'friedman_mse', 'poisson']),
                'model__max_depth': Integer(4,20),
                'model__min_samples_split': Integer(2,20),
                'model__min_samples_leaf': Integer(2,20),
                'model__max_features':Real(0.1,1),
                'model__bootstrap':Categorical([True,False]),
            }

        # Optimize hyper params
        opt = BayesSearchCV(
        estimator=self.model,
        search_spaces=search_space,
        n_iter=32,              # Number of iterations
        scoring='r2',     # You can change this to other metrics if needed
        cv=5,               # 5-fold cross-validation
        verbose=1
        )

        opt.fit(X_train,y_train)
        y_pred = opt.predict(X_test)
        print("Best Score:", opt.best_score_)
        print(f'R2: {r2_score(y_test,y_pred)},mse: {mean_squared_error(y_test,y_pred)}')
        print("Best params:")
        print(opt.best_params_)

    def predict(self, theta_new):
        """
        Predicts the performance of a given configuration theta_new

        :param theta_new: a dict, where each key represents the hyperparameter (or anchor)
        :return: float, the predicted performance of theta new (which can be considered the ground truth)
        """
        return self.model.predict(theta_new)



if __name__ == '__main__':
    data = pd.read_csv('lcdb_configs.csv')
    sm = SurrogateModel(None)
    sm.fit(data)
