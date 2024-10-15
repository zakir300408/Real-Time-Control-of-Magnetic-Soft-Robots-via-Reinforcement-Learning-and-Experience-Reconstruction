import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from community import community_louvain
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import circmean, circstd
from collections import defaultdict

# Load the data
file_path = 'high_purity_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Define thresholds for categorization with adjusted values
angle_threshold = 0.01  # Adjusted small angle change considered negligible (5/180)
moderate_rotation_threshold = 0.056  # Adjusted threshold for moderate rotation (10/180)
strong_rotation_threshold = 0.111  # Adjusted threshold for strong rotation (20/180)
movement_threshold = 0.002  # Tighter threshold for movement to define 'Pure Stationary'

# Calculate movement magnitude and assign signs
data['movement_magnitude'] = np.sqrt(
    data['delta_prev_current_center_x'] ** 2 + data['delta_prev_current_center_y'] ** 2) * \
                             np.sign(data['delta_prev_current_center_x'])

# Preserve the sign in direction change magnitude
data['direction_change_magnitude'] = data['delta_angle']  # Preserve sign to indicate direction of change

# Normalize delta_angle by dividing by 180
data['normalized_delta_angle'] = data['delta_angle'] / 180.0

# Filter out small movements
data['filtered_movement_magnitude'] = data['movement_magnitude'].apply(
    lambda x: 0 if abs(x) < movement_threshold else x
)

# Prepare data for clustering with normalized angle
features = data[['filtered_movement_magnitude', 'normalized_delta_angle']]

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Construct a similarity graph using k-nearest neighbors
n_neighbors = 200  # Number of neighbors to consider for the graph
knn_graph = kneighbors_graph(scaled_features, n_neighbors, include_self=False)

# Convert the k-NN graph to a NetworkX graph
G = nx.Graph(knn_graph)

# Apply the Louvain method for community detection
partition = community_louvain.best_partition(G)

# Assign cluster labels to the original data
data['Cluster'] = pd.Series(partition)

# Rule-based descriptive labels for each cluster
descriptive_labels = []
unique_clusters = sorted(data['Cluster'].unique())

# To track the frequency of descriptive labels
label_count = defaultdict(int)

for i in unique_clusters:
    cluster_data = data[data['Cluster'] == i]

    # Calculate means, medians, and centroids
    centroid_x_mean = cluster_data['delta_prev_current_center_x'].mean()
    centroid_y_mean = cluster_data['delta_prev_current_center_y'].mean()
    centroid_direction_mean = circmean(cluster_data['direction_change_magnitude'], high=180, low=-180)
    centroid_movement = cluster_data['filtered_movement_magnitude'].mean()

    # Determine the movement direction
    if abs(centroid_x_mean) > abs(centroid_y_mean):
        movement_direction = "Right" if centroid_x_mean > 0 else "Left"
    else:
        movement_direction = "Forward" if centroid_y_mean > 0 else "Backward"

    # Determine the type of stationary state or movement
    if abs(centroid_movement) < movement_threshold:
        if centroid_direction_mean > angle_threshold:
            descriptive_label = 'Counterclockwise Rotation'
        elif centroid_direction_mean < -angle_threshold:
            descriptive_label = 'Clockwise Rotation'
        else:
            descriptive_label = 'Pure Stationary'
    else:
        descriptive_label = f'{movement_direction}'
        if centroid_direction_mean > moderate_rotation_threshold:
            descriptive_label += ' with Counterclockwise Rotation'
        elif centroid_direction_mean < -moderate_rotation_threshold:
            descriptive_label += ' with Clockwise Rotation'

    # Check if the label has been seen before and update if necessary
    label_count[descriptive_label] += 1
    if label_count[descriptive_label] > 1:
        descriptive_label = f"{descriptive_label} {label_count[descriptive_label]}"

    descriptive_labels.append(descriptive_label)

import matplotlib.patches as mpatches

# Prepare figure
plt.figure(figsize=(16, 12))  # Increase the figure size (width, height)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

# Plot clusters and their convex hulls
for idx, cluster_num in enumerate(unique_clusters):
    cluster_data = data[data['Cluster'] == cluster_num]
    plt.scatter(cluster_data['filtered_movement_magnitude'], cluster_data['direction_change_magnitude'],
                s=50, alpha=0.6, c=[colors[idx]])

    # Compute and plot the convex hull for each cluster
    points = cluster_data[['filtered_movement_magnitude', 'direction_change_magnitude']].values
    if len(points) > 2:  # Need at least three points to define a convex hull
        try:
            hull = ConvexHull(points)
            hull_points = np.append(hull.vertices, hull.vertices[0])
            plt.plot(points[hull_points, 0], points[hull_points, 1], linestyle='-', linewidth=2, color=colors[idx])
        except QhullError:
            print(f"Cluster {cluster_num} skipped for Convex Hull due to dimensionality issues.")


plt.xlabel('Movement Magnitude (Euclidean distance)', fontsize=22, weight='bold', labelpad=10)
plt.ylabel('Direction Change Magnitude (Degrees)', fontsize=22, weight='bold', labelpad=10)

# Adjust layout to accommodate the legend inside the plot
plt.tight_layout()

plt.grid(False)

# Creating custom patches for the legend
legend_patches = [mpatches.Patch(color=colors[idx], label=descriptive_labels[idx]) for idx in range(len(unique_clusters))]

# Plot the legend inside the plot, aligned with the y-axis value of 60
#add a background color to the legend
plt.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=4, fontsize=13, title='Cluster Names', title_fontsize=18,  shadow=True, fancybox=True)

# Adjust y-axis limits to ensure the legend fits inside
plt.ylim(-60, 70)
#increase the font size of the ticks
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

#remove top and right borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

#thicken the plot borders for left and bottom
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
#y axis should stop showing 60 and the line
plt.gca().spines['left'].set_bounds(-60, 50)
#it should stop showing the numbers after 50
plt.gca().set_yticks(np.arange(-50, 51, 10))



# Save the plot
plt.savefig('movement_clusters_louvain_with_legend_inside.png', bbox_inches='tight')
plt.show()








# Print statistical summaries for each cluster
print("\nStatistical Summaries for Each Cluster:")
for idx, cluster_num in enumerate(unique_clusters):
    cluster_data = data[data['Cluster'] == cluster_num]
    print(f"\nCluster {cluster_num} - {descriptive_labels[idx]}:")
    for feature in ['delta_prev_current_center_x', 'delta_prev_current_center_y', 'delta_angle']:
        min_val = cluster_data[feature].min()
        max_val = cluster_data[feature].max()
        if feature == 'delta_angle':
            mean_val = circmean(cluster_data[feature], high=180, low=-180)
        else:
            mean_val = cluster_data[feature].mean()
        median_val = cluster_data[feature].median()
        print(f"{feature}: min = {min_val}, max = {max_val}, mean = {mean_val}, median = {median_val}")
