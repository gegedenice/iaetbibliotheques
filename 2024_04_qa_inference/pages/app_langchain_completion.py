import requests
import json
import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from clients import OllamaClient, GroqClient

st.set_page_config(
    page_title="QA Inference Streamlit App using Ollama, Nvidia and Groq with Langchain framework"
)


# Cache the header of the app to prevent re-rendering on each load
@st.cache_resource
def display_app_header():
    """Display the header of the Streamlit app."""
    st.title("QA Inference with Ollama & Nvidia & Groq as LLMs providers")
    st.subheader("ChatBot based on Langchain framework")


# Display the header of the app
display_app_header()

# UI sidebar ##########################################
st.sidebar.subheader("Models")
# LLM
llm_providers = {
    "Local Ollama": "ollama",
    "Cloud Nvidia": "nvidia",
    "Cloud Groq": "groq",
}
# hard coded because models returned by NvidiaClient().list_models() are not well formed for Langchain ChatNVIDIA class
llms_from_nvidia = [
    "ai-llama3-70b",
    "ai-mistral-large",
    "ai-gemma-7b",
    "ai-codellama-70b",
]
llm_provider = st.sidebar.radio(
    "Choose your LLM Provider", llm_providers.keys(), key="llm_provider"
)
if llm_provider == "Local Ollama":
    ollama_list_models = OllamaClient().list_models()
    ollama_models = [x["name"] for x in ollama_list_models["models"]]
    ollama_llm = st.sidebar.radio(
        "Select your Ollama model", ollama_models, key="ollama_llm"
    )  # retrive with st.session_state["ollama_llm"]
elif llm_provider == "Cloud Nvidia":
    if nvidia_api_token := st.sidebar.text_input("Enter your Nvidia API Key", type="password"):
        os.environ["NVIDIA_API_KEY"] = nvidia_api_token
        st.sidebar.info("nvidia authentification ok")
        # nvidia_models = [model.model_name for model in list_nvidia_models() if (model.model_type == "chat") & (model.model_name is not None)] # list is false
        nvidia_models = llms_from_nvidia
        nvidia_llm = st.sidebar.radio(
            "Select your Nvidia LLM", nvidia_models, key="nvidia_llm"
        )
    else:
        st.sidebar.warning("You must enter your Nvidia API key")
elif llm_provider == "Cloud Groq":
    if groq_api_token := st.sidebar.text_input("Enter your Groq API Key", type="password"):
        st.sidebar.info("Groq authentification ok")
        groq_list_models = GroqClient(api_key=groq_api_token).list_models()
        groq_models = [x["id"] for x in groq_list_models["data"]]
        groq_llm = st.sidebar.radio("Choose your Groq LLM", groq_models, key="groq_llm")
    else:
        st.sidebar.warning("You must enter your Groq API key")

# LLM parameters
st.sidebar.subheader("Parameters")
max_tokens = st.sidebar.number_input("Token numbers", value=1024, key="max_tokens")
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="temperature"
)
top_p = st.sidebar.slider(
    "Top P", min_value=0.0, max_value=1.0, value=0.7, step=0.1, key="top_p"
)


# LLM client #########################################
class LlmProvider:
    def __init__(self, provider):
        if provider == "ollama":
            self.llm = Ollama(
                model=st.session_state["ollama_llm"],
                temperature=st.session_state["temperature"],
                max_tokens=st.session_state["max_tokens"],
                top_p=st.session_state["top_p"],
            )
        elif provider == "nvidia":
            self.llm = ChatNVIDIA(
                model=st.session_state["nvidia_llm"],
                temperature=st.session_state["temperature"],
                max_tokens=st.session_state["max_tokens"],
                top_p=st.session_state["top_p"],
            )
        elif provider == "groq":
            self.llm = ChatGroq(
                groq_api_key = groq_api_token,
                model_name=st.session_state["groq_llm"],
                temperature=st.session_state["temperature"],
                max_tokens=st.session_state["max_tokens"],
                top_p=st.session_state["top_p"],
            )


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
    conversation = ConversationChain(
        llm=LlmProvider(llm_providers[st.session_state["llm_provider"]]).llm,
        memory=ConversationBufferMemory(),
    )
    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # response = LlmProvider1(llm_providers[llm_provider], prompt=prompt).response
        response = conversation.invoke(prompt)["response"]
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
