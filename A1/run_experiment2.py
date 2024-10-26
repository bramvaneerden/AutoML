import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization
from config_encoder import ConfigEncoder
 
import numpy as np

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
    

def run():
    config_space = ConfigSpace.ConfigurationSpace.from_json('lcdb_config_space_knn.json')
    random_search = RandomSearch(config_space)
    smbo = SequentialModelBasedOptimization(config_space)
    surrogate_model = SurrogateModel(config_space)
    dfs = []
    data_files = ['config_performances_dataset-6.csv','config_performances_dataset-11.csv',
                  'config_performances_dataset-1457.csv','lcdb_configs.csv'] 
    for file in data_files:
        dfs.append(pd.read_csv(f'{file}'))

    colors = ['tab:blue','tab:orange','tab:green','tab:red']
    labels = ['dataset-6','dataset-11','dataset-1457','lcdb-configs']

    results_smbo = {name:[] for name in labels}
    results_random = {name:[] for name in labels}
    for name,df in zip(labels,dfs):
        print(name)
        surrogate_model.fit(df)
        thetas = config_space.sample_configuration(10)
        thetas_df = pd.DataFrame([dict(theta) for theta in thetas])
        thetas_df["anchor_size"] = 1600
        
        performances = surrogate_model.predict(thetas_df)
        capital_phi = list(zip(thetas_df.to_dict('records'), performances))
        # print("performances on thetas: \n", performances)
        smbo.initialize(capital_phi) 

        for idx in range(100):
            # smbo
            theta_smbo = smbo.select_configuration(idx)
            theta_df = pd.DataFrame([dict(theta_smbo)])
            theta_df["anchor_size"] = 1600
            perf_smbo = surrogate_model.predict(theta_df)
                
            # maybe plot perf_smbo against smbo.theta_inc_performance
            smbo.update_runs((theta_df.to_dict('records')[0], perf_smbo))
            results_smbo[name].append(perf_smbo)
            
            # random Search
            theta_random = random_search.select_configuration()
            theta_random_df = pd.DataFrame([dict(theta_random)])
            theta_random_df["anchor_size"] = 1600
            perf_random = surrogate_model.predict(theta_random_df)
            results_random[name].append(perf_random)
            
    plt.figure(figsize=(8, 6))
    for name,color,df in zip(labels,colors,dfs):
        plot_smbo = [np.min(results_smbo[name][:i]) for i in range(1,len(results_smbo[name]))]
        plot_random = [np.min(results_random[name][:i]) for i in range(1,len(results_random[name]))]
        
        iterations = range(len(plot_smbo))
        plt.plot(range(1, 100), plot_smbo, color = color, label = f'{name} - smbo')
        plt.plot(range(1, 100), plot_random, color = color, linestyle = '--', label = f'{name} - rs')
        if name == 'dataset-6':
            plt.axhline(y = df['score'].min(), color = color, linestyle = ':',label='best score in dataset') 
        else:
            plt.axhline(y = df['score'].min(), color = color, linestyle = ':') 
    plt.ylim((0,1))
    plt.title('SMBO & RS comparison per dataset')
    plt.xlabel('Iteration')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.legend()
    plt.savefig('compare_smbo_rs.png')
    plt.show()
    
    #plot_optimization_comparison(results_smbo, results_random)

if __name__ == '__main__':
    run()