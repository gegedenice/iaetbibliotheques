#  LLMs

Pieces of code for free LLMs

## HuggingFace LLMs

### Download 

```
!pip install huggingface-hub

from huggingface_hub import snapshot_download, login

HUGGINGFACEHUB_API_TOKEN = "<your_hf_token"
login(HUGGINGFACEHUB_API_TOKEN)

snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B", cache_dir="<local_path>/Meta-Llama-3-8B", local_dir="<local_path>/Meta-Llama-3-8B")
```

### Use with transformers

```
!pip install transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "<local_path>/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
```

### API d'inférence

```
import requests
import json
import os
from typing import Optional, List, Dict, Any

os.environ["HUGGINGFACEHUB_API_TOKEN"] = <hf_token>>
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

model_id = "tiiuae/falcon-7b-instruct"
llm_api_url = f"hhttps://api-inference.huggingface.co/models/{model_id}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def llm_query(prompt: str) -> List[Dict[str, Any]]:
    payload = {
	"inputs": prompt,
	"options":{"wait_for_model":True}
}
	response = requests.post(llm_api_url, headers=headers, json=payload)
	return response.json()

llm_query("what is AI ?")
```

### Client d'inférence

```
from huggingface_hub import InferenceClient

HF_TOKEN = <hf_token>
model = "facebook/deit-small-distilled-patch16-224"
client = InferenceClient(
    model=model, # model is optional, if not provided the client choose the best model
	token=HF_TOKEN
	) 

# Example 1
messages = [{"role": "user", "content": "What is AI?"}]
client.chat_completion(messages, max_tokens=100)

# Example 2
client.feature_extraction("Hi, who are you?")
```

### Langchain

```
from langchain_community.llms import HuggingFaceEndpoint

os.environ["HUGGINGFACEHUB_API_TOKEN"] = <hf_token>>
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

model_id = "tiiuae/falcon-7b-instruct"

llm = HuggingFaceEndpoint(
    repo_id=model_id, max_length=128, temperature=0.5, token=HF_TOKEN
)
```

## Self-made generic client

Based on the OpenAI client and Inference API 

```
!pip install openai requests pandas
```

```
import requests
import json
from openai import OpenAI
from abc import ABC, abstractmethod
from typing import Dict

class LLMLoader(ABC):
    """Abstract Base Class for loading LLM models."""
    @abstractmethod
    def get_base_url(self) -> str:
        """Get the base URL for the LLM provider."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider."""
        pass


class GroqLLMLoader(LLMLoader):
    """Loader for Groq LLM models."""
    def __init__(self):
        self.base_url = "https://api.groq.com/openai/v1"

    def get_base_url(self) -> str:
        return self.base_url

    def get_provider_name(self) -> str:
        return "Groq"


class NvidiaLLMLoader(LLMLoader):
    """Loader for Nvidia LLM models."""
    def __init__(self):
        self.base_url = "https://api.nvidia.com/openai/v1"

    def get_base_url(self) -> str:
        return self.base_url

    def get_provider_name(self) -> str:
        return "Nvidia"


class OllamaLLMLoader(LLMLoader):
    """Loader for Ollama LLM models."""
    def __init__(self):
        self.base_url = "http://localhost:11434/v1/"

    def get_base_url(self) -> str:
        return self.base_url

    def get_provider_name(self) -> str:
        return "Ollama"


class BaseOpenAILLMClient:
    """Base class for LLM OpenAI-compatible providers."""
    def __init__(self, api_key: str = None, model: str = None, llm_loader: LLMLoader = None):
        """Initialize the BaseOpenAILLMClient with API key and model.

        Args:
        - api_key (str): Your API key.
        - model (str): The model to use.
        - llm_loader (LLMLoader): The loader for the LLM provider.
        """
        self.base_url = llm_loader.get_base_url()
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.model = model
        self.llm_loader = llm_loader
        self.openai_client = self._create_openai_client()

    def _create_openai_client(self) -> OpenAI:
        """Create an OpenAI client instance."""
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def list_models(self) -> list:
        """Fetch a list of available models."""
        url = f"{self.base_url}/models"
        response = requests.request("GET", url, headers=self.headers)
        return response.json()

    def _default_completion_options(self) -> Dict:
        """Get default completion options."""
        return {
            "max_tokens": 1024,
            "top_p": 0.7,
            "temperature": 0.7,
        }

    def create_completion_stream(self, prompt: str, **options) -> str:
        """Create a completion stream using the LLM client.

        Args:
        - prompt (str): The prompt to use for the completion.
        - **options: Keyword arguments to customize the completion.

        Returns:
        - str: The completion stream response.
        """
        options = {**self._default_completion_options(), **options}
        response = self.openai_client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=options["temperature"],
            top_p=options["top_p"],
            max_tokens=options["max_tokens"],
            stream=False,
        )
        return response.choices[0].text

    def create_chat_completion(self, user_prompt: str, system_prompt: str = "you are a helpful assistant.", **options) -> str:
        """Create a chat completion using the LLM client.

        Args:
        - user_prompt (str): The prompt to use for the chat completion.
        - system_prompt (str): The system prompt to use for the chat completion. Defaults to "you are a helpful assistant."
        - **options: Keyword arguments to customize the chat completion.

        Returns:
        - str: The chat completion response.
        """
        options = {**self._default_completion_options(), **options}
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=options["temperature"],
            top_p=options["top_p"],
            max_tokens=options["max_tokens"],
            stream=False,
        )
        return completion.choices[0].message.content

    def api_create_chat_completion(self, user_prompt: str, system_prompt: str = "you are a helpful assistant.", **options) -> str:
        """Create a chat completion using the LLM API.

        Args:
        - user_prompt (str): The prompt to use for the chat completion.
        - system_prompt (str): The system prompt to use for the chat completion. Defaults to "you are a helpful assistant."
        - **options: Keyword arguments to customize the chat completion.

        Returns:
        - str: The chat completion response.
        """
        options = {**self._default_completion_options(), **options}
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": options["temperature"],
                "top_p": options["top_p"],
                "max_tokens": options["max_tokens"],
                "stream": False,
            }
        )
        response = requests.request("POST", url, headers=self.headers, data=payload)
        return response.json()["choices"][0]["message"]["content"]

    def create_embedding(self, text: str, model: str = "text-embedding-ada-002") -> Dict:
        """Create an embedding using the LLM client.

        Args:
        - text (str): The text to create an embedding for.
        - model (str): The model to use for creating the embedding. Defaults to "text-embedding-ada-002".

        Returns:
        - Dict: A dictionary containing the embedding.
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model=model,
        )
        return response.data[0]

    def api_create_embedding(self, text: str, model: str = "text-embedding-ada-002") -> Dict:
        """Create an embedding using the LLM API.

        Args:
        - text (str): The text to create an embedding for.
        - model (str): The model to use for creating the embedding. Defaults to "text-embedding-ada-002".

        Returns:
        - Dict: A dictionary containing the embedding.
        """
        url = f"{self.base_url}/embeddings"
        payload = json.dumps(
            {
                "input": text,
                "model": model,
            }
        )
        response = requests.request("POST", url, headers=self.headers, data=payload)
        return response.json()["data"][0]

    def get_embedding_vectors(self, text: str, model: str = "text-embedding-ada-002"):
        """Get the embedding vectors for the given text.

        Args:
        - text (str): The text to get the embedding vectors for.
        - model (str): The model to use for creating the embedding. Defaults to "text-embedding-ada-002".

        Returns:
        - List[float]: A list of float values representing the embedding vectors.
        """
        embedding_response = self.create_embedding(text, model)
        return embedding_response["vector"]

    def compare_embeddings(self, text1: str, text2: str, model: str = "text-embedding-ada-002"):
        """Compare the similarities between two embeddings.

        Args:
        - text1 (str): The first text to compare.
        - text2 (str): The second text to compare.
        - model (str): The model to use for creating the embedding. Defaults to "text-embedding-ada-002".

        Returns:
        - float: A float value representing the similarity between the two embeddings.
        """
        import numpy as np
        vector1 = np.array(self.get_embedding_vectors(text1, model))
        vector2 = np.array(self.get_embedding_vectors(text2, model))
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return similarity
```

Usage examples : list models

```
groq_llm_loader = GroqLLMLoader()
groq_client = BaseOpenAILLMClient(api_key="GROQ_API_KEY", llm_loader=groq_llm_loader)
groq_client.list_models()
```

Usage examples : chat completion
```
groq_llm_loader = GroqLLMLoader()
groq_client = BaseOpenAILLMClient(api_key="GROQ_API_KEY", model="llama3-8b-8192", llm_loader=groq_llm_loader)
system_prompt = "..."
user_prompt = "..."
print(groq_client.api_create_chat_completion(user_prompt, system_prompt))

nvidia_llm_loader = NvidiaLLMLoader()
nvidia_client = BaseOpenAILLMClient(api_key="NVIDIA_API_KEY", model="gpt-4", llm_loader=nvidia_llm_loader)

ollama_llm_loader = OllamaLLMLoader()
ollama_client = BaseOpenAILLMClient(api_key="whatever", model="nomic-text-embed", llm_loader=ollama_llm_loader)
text = "..."
print(ollama_client.create_embedding(text))
```




