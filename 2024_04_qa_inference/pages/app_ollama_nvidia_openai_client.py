import requests
import json
import os
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="QA Inference Streamlit App using Ollama and Nvidia with OpenAI Client")


# Cache the header of the app to prevent re-rendering on each load
@st.cache_resource
def display_app_header():
    """Display the header of the Streamlit app."""
    st.title("QA Inference with Ollama & Nvidia : OpenAI Client")
    st.subheader("ChatBot")


# Display the header of the app
display_app_header()

# params
ollama_base_url = "http://localhost:11434"
nvidia_base_url = "https://integrate.api.nvidia.com/v1"


# functions #############################
def list_ollama_models():
    url = f"{ollama_base_url}/api/tags"
    response = requests.request("GET", url).text
    return response


# UI sidebar ##########################################
st.sidebar.subheader("Models")
# LLM
llm_providers = {"Local Ollama": "ollama", "Remote Nvidia": "nvidia"}
llms_from_nvidia = [
    "meta/llama3-70b-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "google/gemma-7b",
    "databricks/dbrx-instruct",
]
llm_provider = st.sidebar.radio(
    "Choose your LLM Provider", llm_providers.keys(), key="llm_provider"
)
if llm_provider == "Remote Nvidia":
    if nvidia_api_token := st.sidebar.text_input("Enter your Nvidia API Key"):
        st.sidebar.info("Vvidia authentification ok")
    else:
        st.sidebar.warning("You must enter your Nvidia API key")
    nvidia_llm = st.sidebar.radio(
        "Choose your Nvidia LLM", llms_from_nvidia, key="nvidia_llm"
    )
elif llm_provider == "Local Ollama":
    ollama_models = [x["name"] for x in json.loads(list_ollama_models())["models"]]
    ollama_llm = st.sidebar.radio(
        "Select model you would like to use", ollama_models, key="ollama_llm"
    )  # retrive with st.session_state["ollama_llm"]

# LLM parameters
st.sidebar.subheader("Parameters")
max_tokens = st.sidebar.number_input("Token numbers", value=1024, key="max_tokens")
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="temperature"
)
top_p = st.sidebar.slider(
    "Top P", min_value=0.0, max_value=1.0, value=0.7, step=0.1, key="top_p"
)


# LLM clients examples #########################################
# The `LlmProvider1` class is a Python class that provides methods for making requests to different
# language model providers based on the specified provider.
class LlmProvider:
    def __init__(self, provider, prompt=None):
        if provider == "ollama":
            self.base_url = f"{ollama_base_url}/v1"
            self.api_key="ollama"
            self.llm = st.session_state["ollama_llm"]
        elif provider == "nvidia":
            self.base_url = nvidia_base_url
            self.api_key=nvidia_api_token
            self.llm = st.session_state["nvidia_llm"]
        self.response = self.request(prompt)
        
    def request(self,prompt):
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        completion = client.chat.completions.create(
            model=self.llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=st.session_state["temperature"],
            top_p=st.session_state["top_p"],
            max_tokens=st.session_state["max_tokens"],
            stream=False,
        )
        return completion.choices[0].message.content


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = LlmProvider(llm_providers[llm_provider],prompt=prompt).response
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
