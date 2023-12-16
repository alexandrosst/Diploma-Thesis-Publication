from typing import List, Tuple, Dict
from collections import Counter
import sentiment_probability

Ada_model = sentiment_probability.AdaModel()
BERT_model = sentiment_probability.BERTModel()

def create_sentimental_profile(model_name: str, sentences_personA: List[str], sentences_personB: List[str]) -> float:
    """
    This function creates a sentimental profile based on the given model name and sentences from two persons.
    It calculates the intensity of "non negative" sentiments for each person and returns the average intensity.

    Args:
        model_name (str): The name of the model to be used. It can be "BERT" or "Ada".
        sentences_personA (List[str]): A list of sentences from person A.
        sentences_personB (List[str]): A list of sentences from person B.

    Returns:
        float: The average intensity of "non negative" sentiments for both persons.
    """

    # Depending on the model name, the function uses a different model to categorize the sentiments
    if model_name == "BERT":
        model = sentiment_probability.BERT_model
    else:
        model = sentiment_probability.Ada_model

    # The function then applies the chosen model to each sentence from both persons
    categories_personA = [model.get_category(item) for item in sentences_personA]
    categories_personB = [model.get_category(item) for item in sentences_personB]

    # The frequencies of each sentiment category are counted for both persons
    frequencies_personA = Counter(categories_personA)
    frequencies_personB = Counter(categories_personB)

    # The intensity of "non negative" sentiments is calculated for both persons by dividing the frequency of "non negative" sentiments by the total number of sentiments
    intensity_personA = frequencies_personA["non negative"] / len(categories_personA)
    intensity_personB = frequencies_personB["non negative"] / len(categories_personB)

    # The function returns the average intensity of "non negative" sentiments for both persons
    return (intensity_personA + intensity_personB) / 2


def findEdgesProbabilities(model_name: str) -> Dict[Tuple[str, str], float]:
    """
    This function calculates the sentimental profile for each pair of people in a dialog.
    It uses a specified model to create the sentimental profile.

    Args:
        model_name (str): The name of the model to be used for creating the sentimental profile.

    Returns:
        edgesProbabilities (Dict[Tuple[str, str], float]): A dictionary where the keys are tuples representing pairs of people,
        and the values are the average sentimental profile for the dialogs between each pair.

    """
    # Initialize an empty dictionary to store the sentimental profiles
    edgesProbabilities = {}

    # Iterate over each pair of people and their corresponding dialog
    for pair, dialog in zip(people, dialogs):
        try:
            # If the pair already exists in the dictionary, update their sentimental profile
            current = edgesProbabilities[(pair[0], pair[1])]
            edgesProbabilities[(pair[0], pair[1])] = (current[0] + create_sentimental_profile(model_name, dialog[0], dialog[1]), current[1] + 1)
        except:
            # If the pair does not exist in the dictionary, add them with their sentimental profile
            edgesProbabilities[(pair[0], pair[1])] = (create_sentimental_profile(model_name, dialog[0], dialog[1]), 1)

    # Calculate the average sentimental profile for each pair
    for key in edgesProbabilities.keys():
        edgesProbabilities[key] = edgesProbabilities[key][0] / edgesProbabilities[key][1]

    return edgesProbabilities