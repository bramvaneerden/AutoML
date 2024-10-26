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
    plt.title('Successive Halving')
    plt.xlabel('Data Size')
    plt.ylabel('Performance')
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
        self.anchors = [0]
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
            anchor = start*(2**(step+1))
            self.anchors.append(anchor)
            if anchor > self.df.anchor_size.max():
                continue
            print(f'Anchor = {anchor}')
            self.surrogate_model.fit(self.df.loc[(self.df['anchor_size']>anchor/2)&(self.df['anchor_size']<(2*anchor))])
            errors = {}
            for i,config in enumerate(self.configs):
                # convert config to df with anchor size, for new logic of encoder
                config = pd.DataFrame([dict(config)])
                config["anchor_size"] = anchor
                if i not in self.in_race:
                    continue
                error = self.surrogate_model.predict(config)[0]
                self.curves[i].append(error)
                errors[i] = error
            
            error_to_keep = np.max(np.sort([error for error in errors.values()])[:len(errors)//self.halving_rate])
            ind_to_keep = []
            for i in range(len(self.configs)):
                if i in self.in_race: 
                    if errors[i]<= error_to_keep: 
                        ind_to_keep.append(i)


            #self.in_race = np.argpartition(errors, len(errors)//self.halving_rate)[:len(errors)//self.halving_rate]
            self.in_race = ind_to_keep[:len(errors)//self.halving_rate]
            print(f'Best: {np.min([err for err in errors.values()])}, in race: {len(self.in_race)}')
            

def main(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    # Demonstrate successful halving
    halving = SuccessiveHalving(config_space,df)
    halving.initialize(iterations = 7, halving_rate = 3)
    halving.run()
    plot_halving_curves(halving.curves,halving.anchors)

    # Compare halfing results per dataset
    dfs = []
    data_files = ['config_performances_dataset-6.csv','config_performances_dataset-11.csv',
                  'config_performances_dataset-1457.csv','lcdb_configs.csv'] 
    for file in data_files:
        dfs.append(pd.read_csv(f'{file}'))

    colors = ['tab:blue','tab:orange','tab:green','tab:red']
    labels = ['dataset-6','dataset-11','dataset-1457','lcdb-configs']

    curves = {}
    anchors = {}
    for name,df in zip(labels,dfs):
        halving = SuccessiveHalving(config_space,df)
        print(name, 'available anchors:', df.anchor_size.min(),df.anchor_size.max())
        halving.initialize(iterations = 7, halving_rate = 3)
        halving.run()
        curves[name] = halving.curves
        anchors[name] = halving.anchors 
    for name,color,df in zip(labels,colors,dfs):
        for i,curve in curves[name].items():
            if i==0:
                plt.plot(anchors[name][:len(curve)],curve,color=color, label = f'halving for {name}')
            else:
                plt.plot(anchors[name][:len(curve)],curve,color=color)
        if name == 'dataset-6':
            plt.axhline(y = df['score'].min(), color = color,linestyle=':',label='best score in dataset') 
        else:
            plt.axhline(y = df['score'].min(), color = color, linestyle = ':') 
    plt.title('Successive Halving comparison per dataset')
    plt.xlabel('Data Size')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('halving_compared.png')
    plt.show()

if __name__ == '__main__':
    main(parse_args())

        
    