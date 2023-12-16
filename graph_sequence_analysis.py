import json
# Import all functions from the graph_metrics_module
from graph_metrics_module import *

# Open and read the JSON file containing the edge probabilities for the BERT model
with open('edges_probabilities_BERT.json', 'r') as f:
    edges_probabilities_BERT = json.load(f)

# Convert the keys in the dictionary back to their original data types
edges_probabilities_BERT = {eval(key): value for key, value in edges_probabilities_BERT.items()}

# Open and read the JSON file containing the edge probabilities for the Ada model
with open('edges_probabilities_Ada.json', 'r') as f:
    edges_probabilities_Ada = json.load(f)

# Convert the keys in the dictionary back to their original data types
edges_probabilities_Ada = {eval(key): value for key, value in edges_probabilities_Ada.items()}

# Define the thresholds, interactions, and timestamps for the graph metrics
thresholds = [i/10 for i in range(0,11,2)]
interactions = [i for i in range(1,9,3)]
timestamps = [i for i in range(2,25,5)]

# Use the plot_graph_metrics function from the graph_metrics_module to plot the diversity metric
# for the given thresholds, interactions, and timestamps using the edge probabilities from the BERT and Ada models
plot_graph_metrics(["diversity"], thresholds, interactions, timestamps, {"BERT": edges_probabilities_BERT, "Ada": edges_probabilities_Ada})