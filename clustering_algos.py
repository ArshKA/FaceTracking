import hdbscan
import sklearn.metrics.pairwise
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.metrics.pairwise import cosine_similarity
import math
from numpy import dot
from numpy.linalg import norm
import numpy as np
from sklearn import mixture


# def similarity(x, y):
#     dist = sklearn.metrics.pairwise.cosine_distances(x, y)
#     # if x[-1] == y[-1]:
#     #     dist *= 100
#     # print(x[-1], y[-1], x[-1] == y[-1], dist)
#     return dist

def cosine_distance(X, Y):
  cos_sim = dot(X, Y)/(norm(X)*norm(Y))
  return cos_sim

def cluster_and_plot(data, min_clusters=1, min_samples=1):
    reducer = umap.UMAP(n_components=2)
    # reducer = PCA(n_components=2)
    reduced_data = reducer.fit_transform(data)

    # Cluster the data using HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_clusters, min_samples=min_samples, allow_single_cluster=True)
    labels = clusterer.fit_predict(data)

    # Plot the clustered data
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = reduced_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('HDBSCAN Clustering with UMAP')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.show()

    return labels


def cluster_and_plot_optics(data, min_samples, xi=0.05, min_cluster_size=0.05):
    # reducer = PCA(n_components=2)
    reducer = umap.UMAP(n_components=2)
    reduced_data = reducer.fit_transform(data)

    # Cluster the data using OPTICS
    optics_model = OPTICS(min_samples=min_samples)
    optics_model.fit(data)
    labels = optics_model.labels_

    # Plot the clustered data
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = reduced_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('OPTICS Clustering with PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

    return labels

def cluster_and_plot_dpgmm(data, n_components):
    # reducer = PCA(n_components=2)
    reducer = umap.UMAP(n_components=2)
    reduced_data = reducer.fit_transform(data)

    # Cluster the data using DPGMM
    dpgmm_model = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000)
    dpgmm_model.fit(data)
    labels = dpgmm_model.predict(data)

    # Plot the clustered data
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = reduced_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('DPGMM Clustering with PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

    return labels


def cluster_and_plot_gmm(data, n_components):
    # reducer = PCA(n_components=2)
    reducer = umap.UMAP(n_components=2)

    reduced_data = reducer.fit_transform(data)

    # Cluster the data using GMM
    gmm_model = mixture.GaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000)
    gmm_model.fit(data)
    labels = gmm_model.predict(data)

    # Plot the clustered data
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = reduced_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('GMM Clustering with PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

    return labels
