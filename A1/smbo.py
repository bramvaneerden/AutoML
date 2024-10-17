import ConfigSpace
import numpy as np
import typing

from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from config_encoder import ConfigEncoder

class SequentialModelBasedOptimization(object):

    def __init__(self, config_space):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.config_space = config_space
        self.R = []
        self.theta_inc = {}
        self.theta_inc_performance = 1
        
    def initialize(self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """
        # print(capital_phi)
        self.R.extend(capital_phi)
        encoder = ConfigEncoder(self.config_space)
        self.model = Pipeline([('encoder', encoder),
                               ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), 
                               ('model', GaussianProcessRegressor())]) 
        
        
        for configuration, performance in capital_phi:
            if performance < self.theta_inc_performance:
                self.theta_inc_performance = performance
                self.theta_inc = configuration.copy()
        
        
    def fit_model(self) -> None:
        """
        Fits the internal surrogate model on the complete run list.
        """
        configs, results = zip(*self.R)
        X = pd.DataFrame(configs)
        y = np.array(results)
        self.model.fit(X,y)
        
        
        

    def select_configuration(self, idx) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        rs_probability = 0.3
        if np.random.randint(0,1) < rs_probability:
            return self.config_space.sample_configuration(1)
        
        self.fit_model()
        sample_configs = self.config_space.sample_configuration(3000)
        df = pd.DataFrame(sample_configs)
        
        EI = self.expected_improvement(self.model, self.theta_inc_performance, df)
        best = np.argmax(EI)
        return sample_configs[best]
        
        

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
        
        mu, sigma = model_pipeline.predict(theta, return_std = True)
        
        sigma = np.maximum(sigma, 1e-9)
        
        z = (f_star - mu) / sigma
        
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)

        EI = (f_star - mu) * cdf + sigma*(pdf)
        
        return EI
        

    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        
        self.R.append(run)
        configuration, performance = run
        if performance < self.theta_inc_performance:
            self.theta_inc_performance = performance
            self.theta_inc = configuration
