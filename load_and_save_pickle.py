import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Constants
x = 'prev_current_center_x'
y = 'prev_current_center_y'
angle = 'prev_angle'
spatial_columns = [x, y]
components = [
    'prev_phase_value_x', 'prev_phase_value_y', 'prev_phase_value_z',
    'prev_amplitude_value_x', 'prev_amplitude_value_y', 'prev_amplitude_value_z'
]
movement_threshold = 0.01
angle_threshold = 5
angle_movement_threshold = 5

# Cluster-specific movement ranges
clusters_ranges = {
    'Cluster 0': {
        'delta_prev_current_center_x': (0.002, 0.025),
        'delta_prev_current_center_y': (-0.0199999999999999, 0.023),
        'delta_angle': (-0.9093804491991476, 0.9093804491991476)
    },
    'Cluster 1': {
        'delta_prev_current_center_x': (0.001, 0.0799999999999999),
        'delta_prev_current_center_y': (-0.021, 0.068),
        'delta_angle': (-1.9945693012418813, 32.61924307119281)
    },
    'Cluster 2': {
        'delta_prev_current_center_x': (-0.003, 0.003),
        'delta_prev_current_center_y': (-0.015, 0.0339999999999999),
        'delta_angle': (-32.41789390667503, -5.061196620444832)
    },
    'Cluster 3': {
        'delta_prev_current_center_x': (-0.022, 0.016),
        'delta_prev_current_center_y': (-0.0189999999999999, 0.0409999999999999),
        'delta_angle': (0.0, 49.25383643611923)
    },
    'Cluster 4': {
        'delta_prev_current_center_x': (-0.03, -0.001),
        'delta_prev_current_center_y': (-0.021, 0.0509999999999999),
        'delta_angle': (-40.02197654270799, -5.042451069170909)
    },
    'Cluster 5': {
        'delta_prev_current_center_x': (0.001, 0.0659999999999999),
        'delta_prev_current_center_y': (-0.021, 0.0519999999999999),
        'delta_angle': (-40.8150838748816, -5.054776276103894)
    },
    'Cluster 6': {
        'delta_prev_current_center_x': (-0.03, -0.002),
        'delta_prev_current_center_y': (-0.02, 0.024),
        'delta_angle': (-1.3940911400366076, 0.9821171632241884)
    },
    'Cluster 7': {
        'delta_prev_current_center_x': (-0.03, -0.001),
        'delta_prev_current_center_y': (-0.021, 0.045),
        'delta_angle': (0.0, 29.67308211207768)
    }
}


# Functions
def load_and_preprocess_data(filepath):
    dataset = pd.read_csv(filepath)
    dataset = dataset.groupby('episode').apply(lambda x: x.iloc[1:]).reset_index(drop=True)
    return dataset


def calculate_deltas(df):
    for col in spatial_columns + components:
        df[f'delta_{col}'] = df.groupby('episode')[col].diff()
    df['euclidean_distance'] = np.sqrt(df[f'delta_{x}'] ** 2 + df[f'delta_{y}'] ** 2)
    df[angle] = df[angle] % 360
    df['delta_angle'] = df.groupby('episode')[angle].diff().apply(lambda a: (a + 180) % 360 - 180)
    return df


def filter_valid_episodes(df):
    valid_episodes = df.groupby('episode').filter(lambda x: x['step'].is_monotonic_increasing)
    return valid_episodes


def remove_outliers(df):
    df = df[
        (df[f'delta_{x}'].between(-0.03, 0.1)) &
        (df[f'delta_{y}'].between(-0.021, 0.2)) &
        (df['delta_angle'].between(-40.899, 50))
        ]
    return df


def assign_cluster(row):
    for cluster, ranges in clusters_ranges.items():
        if (
                ranges['delta_prev_current_center_x'][0] <= row[f'delta_{x}'] <= ranges['delta_prev_current_center_x'][
            1] and
                ranges['delta_prev_current_center_y'][0] <= row[f'delta_{y}'] <= ranges['delta_prev_current_center_y'][
            1] and
                ranges['delta_angle'][0] <= row['delta_angle'] <= ranges['delta_angle'][1]
        ):
            return cluster
    return 'Uncategorized'


def plot_spider_plot(df):
    angle_bins = np.linspace(0, 360, 240)  # 12 bins of 30 degrees each
    df['angle_bin'] = pd.cut(df[angle], bins=angle_bins, include_lowest=True)

    # Count the number of unique clusters per angle bin
    unique_movements = df.groupby('angle_bin')['cluster'].nunique()

    categories = [f'{int(angle_bins[i])}-{int(angle_bins[i + 1])}' for i in range(len(angle_bins) - 1)]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    values = unique_movements.tolist()
    values += values[:1]

    # Plotting the spider plot
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values, linewidth=3, linestyle='solid', color='gray')
    ax.fill(angles, values, alpha=0.1, color='cyan')

    plt.xticks(angles[:-1], categories)
    plt.yticks(range(1, max(values) + 1))  # Each ring represents a unique movement
    ax.grid(True)
    #let's not show x axis only every 10 degrees
    ax.set_xticks(angles[::10])
    #increase the font size of the ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, weight='bold')
    #pad the x axis labels
    ax.tick_params(axis='x', pad=15)

    #thicken the polar plot border
    ax.spines['polar'].set_linewidth(2)

    plt.savefig('spider_plot.png', bbox_inches='tight', dpi=300)
    plt.show()


# Main execution
if __name__ == '__main__':
    # Load and preprocess data
    dataset = load_and_preprocess_data('aug_3_24.csv')
    dataset = calculate_deltas(dataset)
    dataset = filter_valid_episodes(dataset)
    dataset = remove_outliers(dataset)

    # Assign clusters based on delta ranges
    dataset['cluster'] = dataset.apply(assign_cluster, axis=1)

    # Plot spider plot
    plot_spider_plot(dataset)
