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

## Ollama

```
# pip install openai

from openai import OpenAI

class OllamaClient:
    def __init__(
        self,
        model=None,
		api_key=None,
    ):
        self.base_url = "http://localhost:11434"
        self.headers = {"Content-Type": "application/json"}
        self.model = model
        self.api_key = api_key

    def list_models(self):
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors (status codes 4xx and 5xx)
            return response.json()  # returns the response is in JSON format
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"Other error occurred: {err}")

    def embeddings(self):
        return OllamaEmbedding(
            model_name=self.model,
            base_url=self.base_url,
            ollama_additional_kwargs={"mirostat": 0},
        )

    def client_llm(self):
        return Ollama(model=self.model, base_url=self.base_url, request_timeout=120.0)
		
    def openai_api_chat_completion(self,prompt,**options):
        url = f"{self.base_url}/api/chat"
        options = options if options is not None else {"max_tokens":1024,"top_p":0.7,"temperature":0.7}
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}],
                "option": {
                    "num_ctx": self.options["max_tokens"],
                    "top_p": self.options["top_p"],
                    "temperature": self.options["temperature"],
                    # stop_sequences=["<|prompter|>","<|assistant|>","</s>"]
                },
                "stream": False,
            }
        )
        response = requests.request("POST", url, headers=self.headers, data=payload)
        return response.json()["message"]["content"]
		
    def openai_client_chat_completion(self,prompt,**options):
        options = options if options is not None else {"max_tokens":1024,"top_p":0.7,"temperature":0.7}
        client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
        )
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}],
            temperature=options["temperature"],
            top_p=options["top_p"],
            max_tokens=options["max_tokens"],
            stream=False,
        )
        return completion.choices[0].message.content
```

## Groq

```
# pip install openai groq

from openai import OpenAI
from groq import Groq

class GroqClient:
    def __init__(self, api_key=None, model=None):
        self.base_url = "https://api.groq.com/openai/v1"
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.model = model

    def list_models(self):
        url = f"{self.base_url}/models"
        response = requests.request("GET", url, headers=self.headers)
        return response.json()

    def client_llm(self):
        return Groq(model=self.model, api_key=self.api_key)
		
	def openai_api_chat_completion(self,prompt,**options):
        url = f"{self.base_url}/chat/completions"
        options = options if options is not None else {"max_tokens":1024,"top_p":0.7,"temperature":0.7}
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}],
                "temperature": options["temperature"],
                "top_p": options["top_p"],
                "max_tokens": options["max_tokens"],
                "stream": False,
            }
        )
        response = requests.request("POST", url, headers=self.headers, data=payload)
        return response.json()["choices"][0]["message"]["content"]

    def openai_client_chat_completion(self,prompt,**options):
        options = options if options is not None else {"max_tokens":1024,"top_p":0.7,"temperature":0.7}
        client = Groq(
            api_key=self.api_key,
        )
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "you are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.options["temperature"],
            top_p=self.options["top_p"],
            max_tokens=self.options["max_tokens"],
            stream=False,
        )
        return completion.choices[0].message.content
```




