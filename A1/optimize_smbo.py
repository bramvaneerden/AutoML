import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='total_performances_dataset.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=100)

    return parser.parse_args()


def plot_smbo(results):

    plt.figure(figsize=(10, 6))
    
    for budget,curve in results.items():
        iterations = range(len(curve))
        plt.plot(iterations, curve, label=budget)
    plt.xlabel('iteration')
    plt.ylabel('performance')
    plt.title('SMBO trainset optimization')
    plt.legend()
    plt.grid(True)
    plt.savefig('smbo.png')
    plt.show()
    

def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    smbo = SequentialModelBasedOptimization(config_space)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    results = {}
    for budget in [50,70,100,150,200,300,400]:
        results[budget] = []
        thetas = config_space.sample_configuration(budget)
        performances = surrogate_model.predict(thetas)       
        capital_phi = zip(thetas, performances)
        smbo.initialize(capital_phi)

        for idx in range(args.num_iterations):
            # smbo
            theta_smbo = smbo.select_configuration(idx)
            perf_smbo = surrogate_model.predict(theta_smbo)[0]
            smbo.update_runs((theta_smbo, perf_smbo))
            results[budget].append(smbo.theta_inc_performance)

    
    plot_smbo(results)

if __name__ == '__main__':
    run(parse_args()) 