## Generate Fig6:
* input file: preceiding generated features_generated_experiences, synthetic_users_scores_for_generated_experiences
* Run select_nr_of_cluster.py for selecting the right number of cluster for cluster uncertainty sampling strategy using the elbow method
* Run Generate_XGboost_SS_results.py in order to generate data for different sampling strategies and saved in a new generated folder
* Run Plot_continuous_metrics_across_SA.py and plots will be saved in new folder Plot_continuous_metrics_XGboost
* Run Plot_ECDF_50SA.py and plots will be saved in new folder Plot_ECDF
