import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from surrogate_model import SurrogateModel
import ConfigSpace
import matplotlib.pyplot as plt

def evaluate_surrogate_model(config_space_file, performance_data_file):
    config_space = ConfigSpace.ConfigurationSpace.from_json(config_space_file)
    df = pd.read_csv(performance_data_file)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(train_df)
    
    test_predicted = surrogate_model.predict(test_df)
    test_actual = test_df['score']
    
    correlation, p_value = spearmanr(test_actual, test_predicted)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(test_actual, test_predicted, alpha=0.5)
    plt.plot([min(test_actual), max(test_actual)], [min(test_actual), max(test_actual)], 'r--')
    plt.xlabel('actual performance')
    plt.ylabel('predicted performance')
    plt.title(f'spearman correlation: {correlation:.3f}')
    plt.grid(True)
    
    
if __name__ == "__main__":
    config_space = "lcdb_config_space_knn.json"
    performance_data = "config_performances_dataset-6.csv"
    
    evaluate_surrogate_model(
        config_space,
        performance_data
    )
    
    plt.show()