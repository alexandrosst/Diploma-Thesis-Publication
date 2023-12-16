from .graph_metrics_functions import *
from .edge_probabilities_distribution import *
import yaml
import seaborn as sns

# Open the parameters.yaml file in read mode
with open('parameters.yaml', 'r') as f:
    # Load the parameters from the yaml file into a dictionary
    parameters = yaml.safe_load(f)

# Get the num_nodes parameter from the dictionary
num_nodes = parameters["num_nodes"]

sns.set_theme(style="darkgrid")