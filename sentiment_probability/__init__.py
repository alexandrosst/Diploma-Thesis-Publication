from .sentiment_models import *
from .dialogue_sentiment_analysis import *
import yaml
import pickle
import numpy as np


# Open the parameters.yaml file in read mode
with open('parameters.yaml', 'r') as f:
    # Load the parameters from the yaml file into a dictionary
    parameters = yaml.safe_load(f)

num_nodes = parameters["num_nodes"]
number_of_dialogs = parameters["number_of_dialogs"]


with open('./sentiment_probability/dialogs.pkl', 'rb') as f:
    dialogs = pickle.load(f)

people = [sorted(np.random.choice(range(num_nodes), size=2, replace=False)) for _ in range(number_of_dialogs)]
dialogs = dialogs[:number_of_dialogs]