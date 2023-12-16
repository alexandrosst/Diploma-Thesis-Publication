import json
# Import all functions from the sentiment_probability module
from sentiment_probability import *

# The Ada & BERT models are initialized when the sentiment_probability module is imported

# Use the findEdgesProbabilities function from the sentiment_probability module
# to calculate the edge probabilities for the BERT model
edges_probabilities_BERT = findEdgesProbabilities("BERT")

# Save the edge probabilities for the BERT model to a JSON file
with open('edges_probabilities_BERT.json', 'w') as f:
    json.dump({str(key): value for key, value in edges_probabilities_BERT.items()}, f)

# Calculate the edge probabilities for the Ada model
edges_probabilities_Ada = findEdgesProbabilities("Ada")

# Save the edge probabilities for the Ada model to a JSON file
with open('edges_probabilities_Ada.json', 'w') as f:
    json.dump({str(key): value for key, value in edges_probabilities_Ada.items()}, f)
