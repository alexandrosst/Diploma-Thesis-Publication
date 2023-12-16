# Diploma Thesis: Extended Version
## Project Overview
The project, primarily based on my [Diploma Thesis](https://github.com/alexandrosst/Diploma-Thesis), underwent significant reformations in terms of code structure and result presentation. Despite the internal framework remaining unaffected, new concepts were introduced. These include:
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

## Project Structure & Execution
The project parameters are set in the [parameters.yaml](https://github.com/alexandrosst/Diploma-Thesis-Publication/blob/main/parameters.yaml) file.

The project is composed of two modules:
- [sentiment_probability](https://github.com/alexandrosst/Diploma-Thesis-Publication/tree/main/sentiment_probability): this module’s primary purpose is to calculate the probabilities of edges in the Interaction Graph. It includes class definitions for creating instances of BERT and Ada models. These instances are used to calculate probabilities based on a dialog file that is also included in the module. 
**Note**: In order to use the Ada Model, an OpenAI key is required. This key is used to authenticate your application’s requests to the OpenAI API. If you don’t have one, you can obtain it from the OpenAI website. After you’ve obtained your key, create a `.env` file in this module and add the following line:
	```yaml
	OPENAI_KEY="<your-key>"
	```
	Replace `<your-key>` with your actual OpenAI key. Please ensure that you keep this key confidential to prevent unauthorized access to your OpenAI account.
- [graph_metrics_module](https://github.com/alexandrosst/Diploma-Thesis-Publication/tree/main/graph_metrics_module): This module’s main function is to create Interaction Graphs and comparative diagrams on a parameter space. It uses the probabilities calculated by the `sentiment_probability` module to create these diagrams and graphs.


First, we need to calculate the probabilities of the edges. This can be done by running the [conversation_analysis.py](https://github.com/alexandrosst/Diploma-Thesis-Publication/blob/main/conversation_analysis.py) script, which performs sentiment analysis on the dialogues and calculates the edge probabilities for each pair of nodes using the BERT and Ada models:
```bash
$ python3 conversation_analysis.py 
```
This script will generate two files: `edges_probabilities_Ada` and `edges_probabilities_BERT`. These files contain dictionaries where each key-value pair represents an edge and its corresponding probability in the format (node1, node2): probability.

For convenience, two such files ([edges_probabilities_Ada.json](https://github.com/alexandrosst/Diploma-Thesis-Publication/blob/main/edges_probabilities_Ada.json), [edges_probabilities_BERT.json](https://github.com/alexandrosst/Diploma-Thesis-Publication/blob/main/edges_probabilities_BERT.json)) have already been created in the repository for the given dialog file that exists in the [sentiment_probability](https://github.com/alexandrosst/Diploma-Thesis-Publication/tree/main/sentiment_probability) module.


Finally, we are able to construct the Interaction Graph and present the evolution of our metrics in the parameter space of our framework by executing the  [graph_sequence_analysis.py](https://github.com/alexandrosst/Diploma-Thesis-Publication/blob/main/graph_sequence_analysis.py) script:
```bash
$ python3 graph_sequence_analysis.py
```
This script reads the previously calculated edge probabilities from the `edges_probabilities_Ada` and `edges_probabilities_BERT` files. It then constructs the Interaction Graph and generates comparative diagrams that present the evolution of our metrics in the parameter space of our framework. These visualizations provide a comprehensive evaluation of the quality of Interaction Graphs under various conditions.