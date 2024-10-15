import ConfigSpace
import numpy as np
import typing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from scipy.stats import norm

class SequentialModelBasedOptimization(object):

    def __init__(self, config_space):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.R = []
        self.theta_inc = {}
        self.theta_inc_performance = 1
        self.cs = config_space
        self.features = [param.name for param in list(config_space.values())]


    def initialize(self, R: typing.List[typing.Tuple[typing.Dict, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """
        self.R = R
        self.gpr = Pipeline([('encoder',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore')),
                        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), 
                        ('model', GaussianProcessRegressor())])  

    def fit_model(self) -> None:
        """
        Fits the internal surrogate model on the complete run list.
        """
        X = pd.DataFrame(columns=self.features)
        y = []
        for i,(x_row,y_row) in enumerate(self.R):
            X =  pd.concat([X,pd.DataFrame([x_row],index=[i])])
            y.append(y_row)
        y = np.array(y)

#        X = pd.DataFrame([row for row,_ in self.R], index = [x for x in range(len(self.R))])
#        y = np.array([y for _,y in self.R])
#        for col in self.features:
#            if col not in X.columns:
#                X[col] = None
        self.gpr.fit(X[self.features],y)


    def select_configuration(self, configurations) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        theta = pd.DataFrame(configurations,index = [x for x in range(len(configurations))])
        EI = self.expected_improvement(self.gpr,self.theta_inc_performance,theta[self.features])
        return EI

    @staticmethod
    def expected_improvement(model_pipeline: Pipeline, f_star: float, theta: np.array) -> np.array:
        """
        Acquisition function that determines which configurations are good and which
        are not good.

        :param model_pipeline: The internal surrogate model (should be fitted already)
        :param f_star: The current incumbent (theta_inc)
        :param theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        y_mu,y_std = model_pipeline.predict(theta,return_std=True)
        EI = (-1*f_star * y_mu) * norm.cdf((-1 * f_star * y_mu)/y_std) + y_std * norm.pdf((-1*f_star*y_mu)/y_std)
        return EI
    
    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        self.R.append(run)

