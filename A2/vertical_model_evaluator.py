from abc import ABC
import ConfigSpace
import numpy as np
from sklearn.pipeline import Pipeline
import typing


class VerticalModelEvaluator(ABC):

    def __init__(self, surrogate_model: Pipeline, anchors: list) -> None:
        """
        Initialises the vertical model evaluator. Take note of what the arguments are
        
        :param surrogate_model: A sklearn pipeline object, which has already been fitted on LCDB data. 
                You can use the predict model to predict for a numpy array (consisting of configuration 
                information and an anchor size) what the performance of that configuration is. 
        :param minimal_anchor: Smallest anchor to be used
        :param final_anchor: Largest anchor to be used
        """
        self.surrogate_model = surrogate_model
        self.anchors = anchors
        self.minimal_anchor = anchors[0]
        self.final_anchor = anchors[-1]

    def evaluate_model(self, best_so_far: float, configuration: typing.Dict) -> typing.List[float]:
        anchor = self.minimal_anchor
        learning_curve = []
        for anchor in self.anchors:
            configuration['anchor_size'] = anchor 
            expected_performance = self.pipeline(configuration)
            learning_curve.append((anchor,expected_performance))

        return learning_curve

