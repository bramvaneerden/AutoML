import ConfigSpace
from sklearn.base import BaseEstimator, TransformerMixin


'''
The class implements the instance methods fit() and transform(). 
-> need to have both X and y parameters, and transform() should return a pandas DataFrame

map category values to numbers from configspace
self can be self
transform should map categories to numbers

'''
class ConfigEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, config_space):
        self.config_space = config_space
        self.categories = self._extract_mappings()
        
    def _extract_mappings(self):
        mappings = {}
        
        for param in self.config_space.get_hyperparameters():
            if isinstance(param, ConfigSpace.hyperparameters.CategoricalHyperparameter):
                mappings[param.name] = {choice: num for num, choice in enumerate(param.choices)}
                
        return mappings
    
    def fit(self, X, y=None):
        return self            

    def transform(self, df):
        copy_df = df.copy()
        for column in copy_df.columns:
            if column in self.categories:
                copy_df[column] = copy_df[column].map(self.categories[column])
        return copy_df

