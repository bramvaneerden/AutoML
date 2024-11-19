import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization
from config_encoder import ConfigEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-1457.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=100)

    return parser.parse_args()

def run_and_visualize_smbo(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    smbo = SequentialModelBasedOptimization(config_space, max_anchor_size=args.max_anchor_size)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    
    exploration = .3
    # smbo
    thetas = config_space.sample_configuration(10)
    thetas_df = pd.DataFrame([dict(theta) for theta in thetas])
    thetas_df["anchor_size"] = args.max_anchor_size
    performances = surrogate_model.predict(thetas_df)
    print("performances: ", performances)
    capital_phi = list(zip(thetas_df.to_dict('records'), performances))
    smbo.initialize(capital_phi)
    
    predicted_improvements = []  
    actual_improvements = []     
    predicted_means = []         
    actual_values = []          
    iteration_incumbents = []    
    
    for idx in range(args.num_iterations):
        exploration = exploration * 0.95 ** idx
        current_incumbent = smbo.theta_inc_performance
        # print("current_incumbent: ", current_incumbent)
        iteration_incumbents.append(current_incumbent)
        
        smbo.fit_model()
        if np.random.rand() < exploration:
            selected_config = config_space.sample_configuration(1)
            candidate_df = pd.DataFrame(selected_config)
            candidate_df['anchor_size'] = args.max_anchor_size
            candidate_df_encoded = smbo.encoder.transform(candidate_df)
            gp_means, gp_stds = smbo.model.predict(candidate_df_encoded, return_std=True)
            # print(gp_means)
            predicted_improvements.append(0.01)
            predicted_means.append(gp_means[0])
        else:
            candidate_configs = smbo.config_space.sample_configuration(5000)
            candidate_df = pd.DataFrame([dict(config) for config in candidate_configs])
            candidate_df['anchor_size'] = args.max_anchor_size
            candidate_df_encoded = smbo.encoder.transform(candidate_df)
            
            ei_values = smbo.expected_improvement(smbo.model, current_incumbent, candidate_df_encoded)
            gp_means, gp_stds = smbo.model.predict(candidate_df_encoded, return_std=True)
            
            best_idx = np.argmax(ei_values)
            selected_config = candidate_configs[best_idx]
            
            predicted_improvements.append(ei_values[best_idx])
            predicted_means.append(gp_means[best_idx])
        
        config_df = pd.DataFrame([dict(selected_config)])
        config_df["anchor_size"] = args.max_anchor_size
        actual_value = float(surrogate_model.predict(config_df))
        # print(actual_value, "actual_value")
        
        actual_values.append(actual_value)
        actual_improvement = max(0, current_incumbent - actual_value)
        actual_improvements.append(actual_improvement)
        
        smbo.update_runs((config_df.to_dict('records')[0], actual_value))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # plot 1, predicted vs actual values
    iterations = range(args.num_iterations)
    ax1.plot(iterations, np.maximum(predicted_means, 0), 'b-', label='GP predicted value', alpha=0.7)
    ax1.plot(iterations, actual_values, 'r--', label='actual value', alpha=0.7)
    ax1.plot(iterations, iteration_incumbents, 'g:', label='current best', alpha=0.7)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('value')
    ax1.set_title('GPR predictions vs actual values')
    ax1.legend()
    ax1.grid(True)
    
    # 2, predicted vs actual improvements
    ax2.plot(iterations, predicted_improvements, 'b-', label='expected improvement', alpha=0.7)
    ax2.plot(iterations, actual_improvements, 'r--', label='actual improvement', alpha=0.7)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('improvement')
    ax2.set_title('expected vs actual improvement')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('smbo_analysis.png')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    run_and_visualize_smbo(args)