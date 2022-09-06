## Generate Fig11:
* input file: preceiding generated features_generated_experiences, synthetic_users_scores_for_generated_experiences
* Run Generate_results_for_time_and_space_overhead.py in order to generate data that are saved in folder mq and time_over. The first contains the saved iQoE models at different training steps for all the synthetic users, the latter contains the time needed to select and train a new SA instance at different training steps for each synthetic users.
* Run Plots.py to generate time and space overhead plots