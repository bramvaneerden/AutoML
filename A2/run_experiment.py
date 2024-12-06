import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
from lccv import LCCV
from ipl import IPL
from surrogate_model import SurrogateModel
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-6.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--num_iterations', type=int, default=10)

    return parser.parse_args()

def count_evalutions(eval_dict):
    evaluations = 0
    for anchor,evals in eval_dict.items():
        evaluations+=anchor*evals
    return evaluations

def experiment(vertical_eval,iterations,config_space):
    evaluations_dict = {anchor:0 for anchor in vertical_eval.anchors}
    best_so_far = None
    
    for _ in range(iterations):
        theta_new = dict(config_space.sample_configuration())
        result,evaluations_dict = vertical_eval.evaluate_model(best_so_far, theta_new,evaluations_dict)
        final_result = result[-1][1]
        if best_so_far is None or final_result < best_so_far:
            best_so_far = final_result
        x_values = [i[0] for i in result]
        y_values = [i[1] for i in result]
        plt.plot(x_values, y_values, "-o")
    plt.savefig(f'{vertical_eval.method}_{iterations}.png')
    evalutations = count_evalutions(evaluations_dict)
    return evalutations,best_so_far


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    anchors = sorted(df['anchor_size'].unique())
    
    # LCCV
    vertical_eval = LCCV(surrogate_model, anchors)
    lccv_eval = []
    lccv_best = []
    for _ in range(10):
        evalutations,best_score = experiment(vertical_eval,args.num_iterations,config_space)
        lccv_eval.append(evalutations)
        lccv_best.append(best_score)

    # IPL
    ipl_eval = []
    ipl_best = []
    vertical_eval = IPL(surrogate_model, anchors)
    for _ in range(10):
        evalutations,best_score = experiment(vertical_eval,args.num_iterations,config_space)
        ipl_eval.append(evalutations)
        ipl_best.append(best_score)
    lccv_eval = np.array(lccv_eval)/anchors[-1]
    ipl_eval = np.array(ipl_eval)/anchors[-1]
    print(args.configurations_performance_file)
    print(f'LCCV score: {np.mean(lccv_best):.2f} std {np.std(lccv_best):.3f}')
    print(f'LCCV evals: {np.mean(lccv_eval):.2f} std {np.std(lccv_eval):.3f}')
    print(f'IPL score: {np.mean(ipl_best):.2f} std {np.std(ipl_best):.3f}')
    print(f'IPL evals: {np.mean(ipl_eval):.2f} std {np.std(ipl_eval):.3f}')
    
    


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
