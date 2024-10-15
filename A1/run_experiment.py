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
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--mode', type=str, default='all')

    return parser.parse_args()


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
    thetas = config_space.sample_configuration(10)
    
    performances = surrogate_model.predict(thetas)
    R = [(theta,performance) for theta,performance in zip(thetas,performances)]
    smbo.initialize(R)
    for idx in range(args.num_iterations):
        smbo.fit_model()
        theta_new = smbo.select_configuration(config_space.sample_configuration(100))
        performance = surrogate_model.predict(theta_new)
        smbo.update_runs((theta_new,performance))
        results['smbo'].append(smbo.theta_inc_performance)
    
    plt.plot(range(len(results['smbo'])), results['smbo'])
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    run(parse_args())
