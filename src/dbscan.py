from cProfile import label
from sklearn.datasets import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def checkCorePoints(data, picked_data, epsilon, minimumPoints):
    # Picked data will be the point that will be tested
    # The picked data will be tested against the rest of the points in the data
    # Epsilon is the data distance that will be used to determine if a point is neighboring another point
    # MinimumPoints is the minimum number of points that must be in a neighborhood to be considered a core point
    neighbors_idx = []

    for instance_idx in range(len(data)):
        instance = data.iloc[instance_idx]
        if instance.name != picked_data.name:
            distance = 0
            # Since there are n features per instance of data
            for feature_idx in range(len(instance)):
                # Implementing Euclidean distance to measure distance between points
                distance += (instance[feature_idx] -
                             picked_data[feature_idx])**2
            distance = np.sqrt(distance)
            if distance <= epsilon:
                neighbors_idx.append(instance_idx)

    if len(neighbors_idx) == 0:
        return False, neighbors_idx, 0  # 0 means it's a noise point
    elif len(neighbors_idx) < minimumPoints and len(neighbors_idx) > 0:
        return True, neighbors_idx, 1  # 1 means it's a border point
    elif len(neighbors_idx) >= minimumPoints:
        return True, neighbors_idx, 2  # 2 means it's a core point


def isTwoListTheSame(list1, list2):
    # Check if two lists are the same
    if len(list1) != len(list2):
        return False
    list1_new = list1.sort()
    list2_new = list2.sort()
    if list1_new == list2_new:
        return True
    else:
        return False


def generateClusters(data, epsilon, minimumPoints):
    # Initialize cluster and noise points' lists
    # Keep track of which points have not been visited
    unvisited_points = data.index.tolist()
    # List of list containing clusters and their members
    clusters = [[] for i in range(len(data))]
    # To keep track of noise points since they don't belong in any cluster (has index of -1)
    noises = []
    single_borders = []  # To keep track of single border points since they don't belong in any cluster, may be noise or belong to a cluster

    while len(unvisited_points) != 0 and isTwoListTheSame(unvisited_points, single_borders) == False:
        # Signifies whether the loop is in its first iteration or not (only one point in the cluster)
        first_point = True
        new_clusters_content = []  # List of points that will be added to the current cluster
        cluster_count = 0  # Keep track of the number of clusters
        # Add a random point to the new cluster that has not been visited
        new_clusters_content.append(random.choice(unvisited_points))

        while len(new_clusters_content) != 0:
            # Get the first element in the new_clusters_content list
            current_idx = new_clusters_content.pop(0)
            isNotNoise, neighbors_idx, cluster_type = checkCorePoints(
                data, data.iloc[current_idx], epsilon, minimumPoints)

            # EDGE CASE: If the current point is a border point but it's also the first point in the cluster
            # If the current point is a border point but also the only point in the cluster (pointed out by the `first_point` variable),
            # The programmer decides to store it in a separate list because there is a chance that the border point is a neighbor of a core point
            # In the case of there's no core point in the neighborhood, the border point will be added to the noises list
            # This will be done later by proving that the contents of the `unvisited_points` list is exactly the same as the `single_borders` list
            if isNotNoise and cluster_type == 1 and first_point:
                single_borders.append(current_idx)
                continue
            # Remove the current point from the unvisited points list
            # print("Removing {} from unvisited points".format(current_idx))
            # print("Point {} is a {} point".format(current_idx, cluster_type))
            unvisited_points.remove(current_idx)
            # Normal cases
            # If the current point is a core point, add it to the current cluster. Also add all its neighbors to the new_clusters_content list as long as the neighbors are not already in the current cluster
            if isNotNoise and cluster_type == 2:
                first_point = False
                for neighbor_idx in neighbors_idx:
                    if neighbor_idx not in new_clusters_content and neighbor_idx in unvisited_points:
                        new_clusters_content.append(neighbor_idx)
                clusters[cluster_count].append(current_idx)
            # If the current point is a border point, add it to the current cluster. Do not add its neighbors to the new_clusters_content list
            elif isNotNoise and cluster_type == 1 and not first_point:
                clusters[cluster_count].append(current_idx)
            # If the current point is a noise point, add it to the noises list. Delete it from the unvisited_points list
            elif not isNotNoise:
                noises.append(current_idx)

        if not first_point:  # If the previously checked point wasn't the first point in the cluster, we can add the new cluster to the clusters list
            cluster_count += 1

    if isTwoListTheSame(unvisited_points, single_borders) and len(unvisited_points) != 0:
        # If the unvisited_points list is exactly the same as the single_borders list, we can stop the loop and return the remaining points as noise
        for idx in unvisited_points:
            noises.append(idx)

    if len(noises) != 0:
        # If there are noise points, add them to the clusters list but as the last element
        clusters.append(noises)

    # To filter empty lists in the list of lists
    clusters = [x for x in clusters if x]
    return clusters


def convertIndexToData(data, clusters):
    new_clusters = []
    for cluster in clusters:
        new_cluster_subset = []
        for element in cluster:
            new_cluster_subset.append(data[element])
        new_clusters.append(new_cluster_subset)
    return new_clusters


def mainDBSCAN():
    # Ask for user input for epsilon and minimumPoints
    data = load_iris()
    epsilon = float(input("Enter epsilon: "))
    minimumPoints = int(input("Enter minimumPoints: "))
    clusters = generateClusters(pd.DataFrame(
        data.data), epsilon, minimumPoints)
    column_names = data.feature_names
    print("There are {} features in the data".format(len(column_names)))
    for column_idx in range(len(column_names)):
        print("{}. {}".format(column_idx, column_names[column_idx]))
    x_axis = int(input(
        "\nPlease pick a number for the feature you'd like to become the X axis: "))
    y_axis = int(input(
        "\nPlease pick a number for the feature you'd like to become the Y axis: "))
    # x_axis = column_names[x_axis]
    # y_axis = column_names[y_axis]
    clusters = convertIndexToData(data.data, clusters)
    cluster_idx = 0
    for cluster in clusters:
        # print(len(cluster[x_axis]))
        # print(len(cluster[y_axis]))
        plt.scatter(cluster[x_axis], cluster[y_axis],
                    label="Cluster {}".format(cluster_idx))
        cluster_idx += 1
    plt.title("DBSCAN results for epsilon = {}, minimumPoints = {}".format(
        epsilon, minimumPoints))
    plt.legend()
    plt.show()


mainDBSCAN()
