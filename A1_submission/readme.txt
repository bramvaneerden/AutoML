AML 2024 Assignment 1 Hyperparameter Optimization
Date: 2024-10-26
Authors: Bram van Eerden & Laurens Schinkelshoek

This file describes the content of all files used to create the visualizations in our report on Hyperparameter Optimization

Experiment files:
- spearman.py creates Figure 1.
- run_experiment2.py creates Figure 2.
- run_experiment.py creates Figures 3, 4 & 5 by running it for each data set.
- halving.py creates Figures 6 and 7.

Support files:
- config_encoder.py contains encoder to encode string columns with ordinal encoding, based on the configspace definition.
- random_seach.py provided and completed file to run random search. 
- smbo.py provided and completed file to run smbo. 
- surrogate_model.py provided and completed file to fit and use the surrogate model.