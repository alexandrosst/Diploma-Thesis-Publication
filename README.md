# Diploma Thesis: Extended Version
## Project Overview
The project, primarily based on my Diploma Thesis, underwent significant reformations in terms of code structure and result presentation. Despite the internal framework remaining unaffected, new concepts were introduced. These include:
- the application of two distinct neural networks, BERT (a generative model) and Ada (a decision-boundary model), for sentiment analysis of chat messages. These models were chosen to compare the impact of a generative model and a model that creates a decision boundary in understanding and interpreting natural language.
- the implementation of specific metrics - diversity, number of isolated nodes, and graph edit similarity - to quantify the sequence of graph snapshots, also known as the Interaction Graph. These metrics provide a comprehensive measure of the graph’s evolution over time.

These concepts were utilized to construct comparative diagrams for different combinations of internal parameters of the framework. This allows for a comprehensive evaluation of the quality of Interaction Graphs under various conditions.

## Python Package Requirements
The project was developed using `Python v.3.11.6`. The following packages are required, each serving a unique purpose in the project:

- `numpy==1.22.4`: Used for efficient numerical operations.
- `openai==0.28.0`: Utilized for interacting with the OpenAI API.
- `python-dotenv==1.0.0`: Enables the user to specify environment variables in a .env file.
- `PyYAML==6.0.1`: Used for handling YAML files.
- `seaborn==0.13.0`: Facilitates data visualization.
- `torch==2.1.2`: Provides tools for deep learning.
- `transformers==4.36.1`: Used for state-of-the-art Natural Language Processing.

Please ensure that you have these packages installed before proceeding.