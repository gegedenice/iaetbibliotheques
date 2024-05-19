import requests
import json


class OllamaClient:
    def __init__(
        self,
        api_key=None,
        model=None,
    ):
        self.base_url = "http://localhost:11434"
        self.headers = {"Content-Type": "application/json"}
        self.api_key = api_key
        self.model = model

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

    def list_models_names(self):
        models = self.list_models()
        return [x["name"] for x in models["models"]]

    def api_chat_completion(self, system, prompt, **options):
        url = f"{self.base_url}/api/chat"
        options = (
            options
            if options is not None
            else {"max_tokens": 1024, "top_p": 0.7, "temperature": 0.7}
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
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


class NvidiaClient:
    def __init__(self, api_key=None, model=None):
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.model = model

    def list_models(self):
        url = f"{self.base_url}/models"
        response = requests.request(
            "GET", url
        )  # api_key is not needed to list the available models
        return response.json()

    def list_models_names(self):
        models = self.list_models()
        return [x["id"] for x in models["data"]]

    def api_chat_completion(self, system, prompt, **options):
        url = f"{self.base_url}/chat/completions"
        options = (
            options
            if options is not None
            else {"max_tokens": 1024, "top_p": 0.7, "temperature": 0.7}
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": options["temperature"],
                "top_p": options["top_p"],
                "max_tokens": options["max_tokens"],
                "stream": False,
            }
        )
        response = requests.request("POST", url, headers=self.headers, data=payload)
        return response.json()["choices"][0]["message"]["content"]


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

    def list_models_names(self):
        models = self.list_models()
        return [x["id"] for x in models["data"]]

    def api_chat_completion(self, system, prompt, **options):
        url = f"{self.base_url}/chat/completions"
        options = (
            options
            if options is not None
            else {"max_tokens": 1024, "top_p": 0.7, "temperature": 0.7}
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": options["temperature"],
                "top_p": options["top_p"],
                "max_tokens": options["max_tokens"],
                "stream": False,
            }
        )
        response = requests.request("POST", url, headers=self.headers, data=payload)
        return response.json()["choices"][0]["message"]["content"]
