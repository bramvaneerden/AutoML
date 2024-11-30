import logging
import numpy as np
import typing
import pandas as pd
from scipy.optimize import curve_fit
from vertical_model_evaluator import VerticalModelEvaluator
from config_encoder import ConfigEncoder


class IPL(VerticalModelEvaluator):
    
    @staticmethod
    def extrapolation(results, target_anchor: int
    ) -> float:
        """
        Does the optimistic performance. Since we are working with a simplified
        surrogate model, we can not measure the infimum and supremum of the
        distribution. Just calculate the slope between the points, and
        extrapolate this.

        :param previous_anchor: See name
        :param previous_performance: Performance at previous anchor
        :param current_anchor: See name
        :param current_performance: Performance at current anchor
        :param target_anchor: the anchor at which we want to have the
        optimistic extrapolation
        :return: The optimistic extrapolation of the performance
        """
        def func_powerlaw(x, m, c, c0):
            return c0 + x**m * c
        
        X_values = np.array([x for x,_ in results])
        y_values = np.array([y for _,y in results])
        target_func = func_powerlaw
        popt, pcov = curve_fit(target_func, X_values, y_values,maxfev=5000)
        extrapolated = target_func(target_anchor, *popt)
        return extrapolated
    
    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        """
        Does a staged evaluation of the model, on increasing anchor sizes.
        Determines after the evaluation at every anchor an optimistic
        extrapolation. In case the extrapolation can not improve
        over the best so far, it stops the evaluation.
        In case the best so far is not determined (None), it evaluates
        immediately on the final anchor (determined by self.final_anchor)

        :param best_so_far: indicates which performance has been obtained so far
        :param configuration: A dictionary indicating the configuration

        :return: A tuple of the evaluations that have been done. Each element of
        the tuple consists of two elements: the anchor size and the estimated
        performance.
        """
        encoder = ConfigEncoder(self.surrogate_model.config_space)
        if best_so_far == None:
            # anchor = self.final_anchor
            configuration["anchor_size"] = self.final_anchor
            config = pd.DataFrame([dict(configuration)])
            result = self.surrogate_model.predict(config)[0]
            return [(self.final_anchor, result)]
        
        anchor = self.minimal_anchor
        # anchors = []
        # performances = []
        results = []
        while anchor <= self.final_anchor:
            
            configuration["anchor_size"] = anchor
            config = pd.DataFrame([dict(configuration)])
            performance = self.surrogate_model.predict(config)[0]
            
            # anchors.append(anchor)
            results.append((anchor, performance))
            anchor  *= 2
            
            
            if len(results) >= 3: 
                extrapolated = self.extrapolation(results, self.final_anchor)
                # print (f"extrapolated {extrapolated}, best_so_far { best_so_far}")
                if extrapolated > best_so_far:
                    # print ( "breaking")
                    break
                
        return results

        
            
