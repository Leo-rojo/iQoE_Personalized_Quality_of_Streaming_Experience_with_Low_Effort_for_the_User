import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import copy
from sklearn.metrics import silhouette_score

#all features
synthetic_experiences = np.load('./features_generated_experiences/feat_iQoE_for_synth_exp.npy')
scores_synthetic_users = np.load('./synthetic_users_scores_for_generated_experiences/scaled/nrchunks_7.npy')

all_features=copy.copy(synthetic_experiences)
users_scores=copy.copy(scores_synthetic_users)

#elbow
distortions = []
silhouette_avg = []
K = range(2,13)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(all_features)
    distortions.append(kmeanModel.inertia_)
    cluster_labels = kmeanModel.labels_

    silhouette_avg.append(silhouette_score(all_features, cluster_labels))

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#silouette
# silhouette score
plt.figure(figsize=(16,8))
plt.plot(K,silhouette_avg,'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')
plt.show()