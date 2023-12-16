import os
from dotenv import load_dotenv
import time
import openai
import torch
from transformers import AutoModel, AutoTokenizer

class AdaModel:
    def __init__(self):
        # Set the API type to "azure"
        openai.api_type = "azure"

        # Load environment variables from a .env file
        load_dotenv()

        # Get the OpenAI key from the environment variables and set it as the API key
        openai.api_key = os.getenv("OPENAI_KEY")

        # Set the base URL for the OpenAI API
        openai.api_base = "https://irene.openai.azure.com/"

        # Set the version of the OpenAI API
        openai.api_version = "2022-12-01"

    def get_category(self, text: str) -> str:
        """
        This function uses the OpenAI API to generate a completion with the "Ada_fine_tuned" engine.
        It then checks the response and returns the corresponding category.

        Args:
            text (str): The text to be completed by the OpenAI API.

        Returns:
            str: The category of the response. It can be "non negative", "negative", or None.
        """

        # Create a completion with the OpenAI API using the "Ada_fine_tuned" engine
        try:
            response = openai.Completion.create(
                        engine="Ada_fine_tuned",  # The engine used for the completion
                        prompt=text,  # The text to complete
                        temperature=0,  # The randomness of the output (0 means deterministic)
                        max_tokens=3  # The maximum number of tokens in the output
                    ).choices[0].text.strip()  # Get the text of the first choice and remove leading/trailing spaces
        except:
            time.sleep(10)
            return "negative"

        time.sleep(10)  # Wait for 10 seconds

        # Check the response and return the corresponding category
        if "non negative" in response:
            return "non negative"
        elif "negative" in response:
            return "negative"
        else:
            return None  # Return None if the response doesn't match any category


class BERTClassifier(torch.nn.Module):
    """
    This class defines a BERT classifier model. It inherits from the PyTorch nn.Module class.
    """
    def __init__(self, model_name):
        """
        The constructor for BERTClassifier class. It initializes the BERT model, dropout, and linear layer.
        """
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.4)
        self.linear = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        The forward method takes input tensors and passes them through the BERT model, dropout, and linear layer.
        """
        bert_output = self.bert_model(input_ids, attention_mask, token_type_ids).pooler_output
        bert_output = self.dropout(bert_output)
        final_output = self.linear(bert_output)
        return final_output

class BERTModel:
    """
    This class defines a BERT model. It uses the BERTClassifier for sentiment analysis.
    """
    def __init__(self, model_name='bert-base-uncased', output_file='bert_model'):
        """
        The constructor for BERTModel class. It initializes the tokenizer, device, and BERTClassifier model.
        """
        torch.manual_seed(0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = 128
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = BERTClassifier(model_name)
        self.model.load_state_dict(torch.load(f'./sentiment_probability/{output_file}.pth', map_location=self.device))
        self.model.eval()

    def get_category(self, text: str) -> str:
        """
        This method takes a text input, tokenizes it, and passes it through the BERTClassifier model.
        It then returns the category with the highest probability.
        """
        t = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True
        )

        input_ids = t["input_ids"].to(self.device, dtype=torch.long)
        attention_mask = t["attention_mask"].to(self.device, dtype=torch.long)
        token_type_ids = t["token_type_ids"].to(self.device, dtype=torch.long)

        calculated_labels = self.model(input_ids, attention_mask, token_type_ids)

        results = dict(zip(["negative", "non negative"], torch.sigmoid(calculated_labels).cpu().detach().numpy().tolist()[0]))

        return max(results, key=results.get)