import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from surrogate_model import SurrogateModel
import ConfigSpace
import matplotlib.pyplot as plt

def evaluate_surrogate_model(config_space_file, performance_data_file):
    config_space = ConfigSpace.ConfigurationSpace.from_json(config_space_file)
    #df = pd.read_csv(performance_data_file)
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax2 = ax1.twinx()  # Create a secondary y-axis
    colors = ['tab:blue','tab:orange','tab:green','tab:red']
    labels = ['dataset-6','dataset-11','dataset-1457','lcdb-configs']
    for df,color,label in zip(dfs,colors,labels):
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        surrogate_model = SurrogateModel(config_space)
        surrogate_model.fit(train_df)
    
        test_predicted = surrogate_model.predict(test_df)
        test_actual = test_df['score']
    
        correlation, p_value = spearmanr(test_actual, test_predicted)
    
        ax1.scatter(test_actual, test_predicted, alpha=0.2,color=color,s=3)
        ax1.plot([min(test_actual), max(test_actual)], [min(test_actual), max(test_actual)], 
                 linestyle='--',color=color)
        ax2.hist(df['score'], bins=30, color=color, alpha=0.3, label=f'{label} - correlation: {correlation:.3f}')
    
    ax1.set_xlabel('actual performance')
    ax1.set_ylabel('predicted performance')
    ax2.set_ylabel('counts')
    plt.legend()
    plt.title(f'spearman correlation')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('surrogate_quality.png')
    
    
if __name__ == "__main__":
    config_space = "lcdb_config_space_knn.json"
    performance_data = "config_performances_dataset-6.csv"
    dfs = []
    data_files = ['config_performances_dataset-6.csv','config_performances_dataset-11.csv',
                  'config_performances_dataset-1457.csv','lcdb_configs.csv'] 
    for file in data_files:
        dfs.append(pd.read_csv(file))
    
    evaluate_surrogate_model(
        config_space,
        dfs
    )
    
    plt.show()