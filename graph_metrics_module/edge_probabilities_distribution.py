import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_edge_probabilities(edges_probabilities: Dict[str, List[float]]) -> None:
    """
    This function plots the distribution of edge probabilities.

    Parameters:
    edges_probabilities (Dict[str, List[float]]): A dictionary where keys are model names and values are lists of edge probabilities.

    Returns:
    None
    """

    # Create subplots
    fig, ax = plt.subplots(1,2, figsize=(12,5))

    # Set the title for the plot
    fig.suptitle("Distribution of edge probabilities")
    
    # Loop over the items in the dictionary
    for index, (model, probs) in enumerate(edges_probabilities.items()) :
        # Plot a histogram for each model
        sns.histplot(probs, bins=10, kde=True, label=model, ax=ax[index], color=sns.color_palette()[index])
        
        # Set the x-label
        ax[index].set_xlabel("probability bins")
        
        # Set the y-label
        ax[index].set_ylabel("frequency")
        
        # Set the legend
        ax[index].legend(loc="best")

    # Adjust the layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()