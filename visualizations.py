import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Example modality combinations
modality_combinations = [
    'Tabular', 
    'Image',
    'Audio', 
    'Tabular+Image', 
    'Tabular+Audio', 
    'Audio+Image', 
    'All Modalities'
]

accuracy = [.625, 0.9524, 0.7647, 0.8966, 0.7922, .8947, .8043]
f1_score = [0.6045, 0.9522, 0.7469, 0.8958, 0.7553, .8651, .7997]
precision = [0.6042, 0.9563, 0.8371, 0.9148, 0.7922, .8947, .8354]
recall = [0.625, 0.9524, 0.7647, 0.8966, 0.76, .8684, .8043]

def plot_bars():
    metrics = [accuracy, f1_score, precision, recall]
    metric_labels = ['Accuracy', 'F1-score', 'Precision', 'Recall']
    colors = ['#01161E', '#124559', '#598392', '#AEC3B0']

    x = np.arange(len(modality_combinations))  # group locations
    width = 0.2  # width of each bar

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6), facecolor = '#fffcf1ff')

    # Plot each metric
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        ax.bar(x + i * width - 1.5 * width, metric, width, label=label, color=color)

    # Aesthetics
    ax.set_xlabel('Modality Combination')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Across Modalities')
    ax.set_xticks(x)
    ax.set_xticklabels(modality_combinations)
    # ax.legend()
    ax.set_ylim(.6, 1.0)
    ax.set_facecolor('#EFF6E0') 

    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()




def plot_heatmap():
# Create DataFrame

    modality_combinations = [
        'Tabular', 
        'Image',
        'Audio', 
        'Tabular+Image', 
        'Tabular+Audio', 
        'Audio+Image', 
        'All Modalities'
    ]
    data = {
        'accuracy' : [.9488, 0.9784, 0.9140, 0.9704, 0.9491, .9638, .9550],
        # 'f1_score' : [0.9490, 0.9783, 0.9133, 0.9704, 0.9491, .9638, .9550],
        # 'precision' : [0.9522, 0.9792, 0.9217, 0.9706, 0.9530, .9642, .9557],
        # 'recall' : [0.9488, 0.9784, 0.9140, 0.9704, 0.9491, .9638, .9550],
    }

    df = pd.DataFrame(data, index=modality_combinations)



    # Plot heatmap
    plt.figure(figsize=(8, 6),  facecolor = '#fffcf1ff')

    sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True, vmin=.6, vmax=1, linewidths=0.5, linecolor='gray')
    plt.title("Model Performance Across Modality Combinations")
    plt.ylabel("Modality")
    plt.xlabel("Metric")
    plt.tight_layout()
    plt.show()


plot_bars()
# plot_heatmap()