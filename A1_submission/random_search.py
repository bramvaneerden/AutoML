import ConfigSpace
import typing


class RandomSearch(object):
    """Class to perform random search. 
    Parameter space has to be defined at initiation of the class.
    Initialize does nothing, select_configuration returns a random
    configuration.

    """    

    def __init__(self, config_space: ConfigSpace.ConfigurationSpace):
        self.config_space = config_space

    def initialize(self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]) -> None:
        pass

    def select_configuration(self) -> ConfigSpace.Configuration:
        return self.config_space.sample_configuration(1)

    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        pass
