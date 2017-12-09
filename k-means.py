import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.datasets.samples_generator import make_blobs
sns.set()
plt.ion()

'''
#####################
Options and Variables
#####################
'''
#Number of Clusters
clusters = 3
#Number of Centroids
K = 3
#Colors for our centroids
centroid_colors = np.zeros(K)
for i in range(K):
    centroid_colors[i] = i
cmap = 'Dark2'

#Number of Epochs
epochs = 10
#Training set
X, y_true = make_blobs(n_samples=clusters*50, centers=clusters,
                       cluster_std=1)
#Indexes of our corresponding examples to centroids
idx = np.zeros(len(X[:, 0]))
#Centroids
centroids = np.zeros((K, 2))
#Previous centroids for plotting
p_centroids = np.zeros((K, 2))
'''
#############################
Randomly Initialize Centroids
#############################
'''
for k in range(K):
    centroids[k] = (random.choice(X))
'''
#########
Algorithm
#########
'''
for epoch in range(epochs):
    p_centroid = centroids

    'Find closest centroids to the ith training example'
    for i in range(len(X[:, 0])):
        #Find centroid closest to X[i, :]
        distances = np.zeros((K, 1))
        for k in range(K):
            distances[k] = np.linalg.norm(X[i, :] - centroids[k, :])**2

        #Assign idx[i] to centroid index closest to X[i, :]s
        v = min(distances)
        index = np.where(distances == v)
        try:
            idx[i] = index[0]
        except ValueError:
            print idx[i], index[0]
            pass


    'Compute and move centroids'
    for k in range(K):
        #Find all idx elements that contain the current centroid k
        indices = np.where(idx == k)
        #Get the index of all those elements and apply to X
        x_indices = X[indices, :]
        #Compute average of all of those elements
        average = sum(x_indices[0]) / len(x_indices[0])
        #Assign that average to the position of the k'th centroid
        centroids[k, :] = average

    'Plot our training set and centroids'
    plt.clf()

    plt.title("Epoch: %d" % (epoch+1))
    plt.scatter(X[:, 0], X[:, 1], s=40, c=idx, cmap=cmap, alpha=0.5, marker='.')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100,  c=centroid_colors, cmap=cmap)

    plt.draw()
    plt.pause(0.05)

while True:
    plt.pause(0.05)
