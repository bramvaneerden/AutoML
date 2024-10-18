import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=40)

    return parser.parse_args()


def plot_optimization_comparison(results_smbo, results_random):

    plt.figure(figsize=(10, 6))
    
    best_smbo = np.minimum.accumulate(results_smbo)
    best_random = np.minimum.accumulate(results_random)
    
    iterations = range(len(best_smbo))
    plt.plot(iterations, best_smbo, 'b-', label='SMBO', linewidth=2)
    plt.plot(iterations, best_random, 'r--', label='random search', linewidth=2)
    
    plt.xlabel('iteration')
    plt.ylabel('best performance')
    plt.title('SMBO vs random search')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    random_search = RandomSearch(config_space)
    smbo = SequentialModelBasedOptimization(config_space)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    results = {
        'random_search': [1.0],
        'smbo':[1.0]
    }
    results_smbo = [1.0]
    results_random = [1.0]
    thetas = config_space.sample_configuration(100)

    performances = surrogate_model.predict(thetas)
        
    
    capital_phi = zip(thetas, performances)
    smbo.initialize(capital_phi)
    for idx in range(args.num_iterations):
        # smbo
        theta_smbo = smbo.select_configuration(idx)
        perf_smbo = surrogate_model.predict(theta_smbo)[0]
        smbo.update_runs((theta_smbo, perf_smbo))
        results_smbo.append(smbo.theta_inc_performance)
        
        # random Search
        theta_random = random_search.select_configuration()
        perf_random = surrogate_model.predict(theta_random)[0]
        results_random.append(perf_random)
    
    plot_optimization_comparison(results_smbo, results_random)

if __name__ == '__main__':
    run(parse_args())