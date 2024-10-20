import pandas as pd 
import numpy as np
import ConfigSpace
from surrogate_model import SurrogateModel
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='total_performances_dataset.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=100)

    return parser.parse_args()

def plot_halving_curves(curves,anchors):
    for i,curve in curves.items():
        plt.plot(anchors[:len(curve)],curve)
    plt.title('SuccessiveHalving')
    plt.xlabel('data size')
    plt.ylabel('error')
    plt.savefig('halving.png')
    plt.show()

class SuccessiveHalving():
    def __init__(self, config_space,df):
        """
        Initializes successive Halving
        """
        self.config_space = config_space
        self.configs = []
        self.curves = {}
        self.df = df
        self.best_error = 1
        self.anchors = []
        self.surrogate_model = SurrogateModel(config_space)

    def initialize(self,iterations = 7, halving_rate = 3):
        self.iterations = iterations
        self.halving_rate = halving_rate 
        budget = halving_rate**iterations 
        self.configs = self.config_space.sample_configuration(budget)
        self.in_race = [i for i in range(len(self.configs))]
        self.curves = {i:[1] for i in self.in_race}

    def run(self):
        start = 16
        for step in range(self.iterations):
            anchor = 16*(2**(step+1))
            self.anchors.append(anchor)
            print(f'Anchor = {anchor}')
            self.surrogate_model.fit(self.df.loc[self.df['anchor_size']==anchor])
            errors = []
            for i,config in enumerate(self.configs):
                if i not in self.in_race:
                    continue
                error = self.surrogate_model.predict(config)[0]
                self.curves[i].append(error)
                errors.append(error)
            self.errors = np.array(errors)
            self.in_race = np.argpartition(errors, len(errors)//self.halving_rate)[:len(errors)//self.halving_rate]
            print(f'Best: {np.max(errors)}, in race: {len(self.in_race)}')

def main(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    halving = SuccessiveHalving(config_space,df)
    halving.initialize(iterations = 7, halving_rate = 3)
    halving.run()
    plot_halving_curves(halving.curves,halving.anchors)

if __name__ == '__main__':
    main(parse_args())

        
    