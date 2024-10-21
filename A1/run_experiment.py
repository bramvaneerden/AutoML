import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization
from config_encoder import ConfigEncoder

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='total_performances_dataset.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=50)

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
    plt.savefig('comparison.png')
    plt.show()
    

def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    random_search = RandomSearch(config_space)
    smbo = SequentialModelBasedOptimization(config_space, max_anchor_size=args.max_anchor_size)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    
    results_smbo = []
    results_random = []
    
    thetas = config_space.sample_configuration(100)
    thetas_df = pd.DataFrame([dict(theta) for theta in thetas])
    thetas_df["anchor_size"] = args.max_anchor_size
    
    performances = surrogate_model.predict(thetas_df)
    capital_phi = list(zip(thetas_df.to_dict('records'), performances))
    # print("performances on thetas: \n", performances)
    smbo.initialize(capital_phi)

    for idx in range(args.num_iterations):
        # smbo
        theta_smbo = smbo.select_configuration(idx)
        theta_df = pd.DataFrame([dict(theta_smbo)])
        theta_df["anchor_size"] = args.max_anchor_size
        perf_smbo = surrogate_model.predict(theta_df)
            
        # maybe plot perf_smbo against smbo.theta_inc_performance
        smbo.update_runs((theta_df.to_dict('records')[0], perf_smbo))
        results_smbo.append(perf_smbo)
        
        # random Search
        theta_random = random_search.select_configuration()
        theta_random_df = pd.DataFrame([dict(theta_random)])
        theta_random_df["anchor_size"] = args.max_anchor_size
        perf_random = surrogate_model.predict(theta_random_df)
        results_random.append(perf_random)
        # if perf_random < np.min(results_random):
        #     results_random.append(perf_random)
        # else:
        #     results_random.append(results_random[-1])
            
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.num_iterations + 1), results_smbo, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Performance ')
    plt.title('Performance Over Iterations SMBO')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.num_iterations + 1), results_random, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Performance ')
    plt.title('Performance Over Iterations RS')
    plt.grid(True)
    plt.show()
    
    plot_optimization_comparison(results_smbo, results_random)

if __name__ == '__main__':
    run(parse_args())