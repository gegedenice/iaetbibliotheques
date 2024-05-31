import streamlit as st
from huggingface_hub import InferenceClient, AsyncInferenceClient
from PIL import Image
from pathlib import Path
import os, subprocess
import requests

st.set_page_config(page_title='HF Inference Client Demo',layout="wide")
# Cache the header of the app to prevent re-rendering on each load
@st.cache_resource
def display_app_header():
    """Display the header of the Streamlit app."""
    st.title("HF Inference Client Demo")
    st.subheader("Just a little demonstrator")
# Display the header of the app
display_app_header()

# UI sidebar parameters ####################################
st.sidebar.header("Loging")
if hg_token :=st.sidebar.text_input('Enter your HF token', type="password"):
    st.sidebar.info('Logged', icon="ℹ️") 
else:
    st.sidebar.warning("enter your token")

st.sidebar.header("Model")
selected_model = st.sidebar.radio(
        "Choose a model or let the client do it",
        ["Not choose", "Choose"]
        )
if selected_model == "Choose":
    model = st.sidebar.text_input('Enter a model name. ex : facebook/fastspeech2-en-ljspeech')
else:
    model = None

st.sidebar.header("Task")
dict_hg_tasks = {
"Automatic Speech Recognition":"automatic_speech_recognition",
"Text-to-Speech (choose model)":"text_to_speech",
"Image Classification":"image_classification",
"Image Segmentation":"image_segmentation",
"Object Detection":"object_detection",
"Text-to-Image":"text_to_image",
"Visual Question Answering":"visual_question_answering",
"Conversational":"conversational",
"Feature Extraction":"feature_extraction",
"Question Answering":"question_answering",
"Summarization":"summarization",
"Text Classification":"text_classification",
"Text Generation":"text_generation",
"Token Classification":"token_classification",
"Translation (choose model)":"translation",
}

dict_hg_tasks_params = {
    "automatic_speech_recognition": {
        "input": "upload,url",
        "output": "text",
        "prompt": False,
        "context": False
    },
    "text_to_speech": {
        "input": "text",
        "output": "audio",
        "prompt": False,
        "context": False
    },
    "image_classification": {
        "input": "upload,url",
        "output": "image,text",
        "prompt": False,
        "context": False
    },
    "image_segmentation": {
        "input": "upload,url",
        "output": "image,text",
        "prompt": False,
        "context": False
    },
    "object_detection": {
        "input": "upload,url",
        "output": "image,text",
        "prompt": False,
        "context": False
    },
    "text_to_image": {
        "input": "text",
        "output": "image",
        "prompt": False,
        "context": False
    },
    "visual_question_answering": {
        "input": "upload,url",
        "output": "image,text",
        "prompt": True,
        "context": False
    },
    "image_to_image": {
        "input": "upload,url",
        "output": "image,text",
        "prompt": True,
        "context": False
    },
    "feature_extraction": {
        "input": "text",
        "output": "text",
        "prompt": False,
        "context": False
    },
    "conversational": {
        "input": "text",
        "output": "text",
        "prompt": False,
        "context": False
    },
    "question_answering": {
        "input": None,
        "output": "text",
        "prompt": True,
        "context": True
    },
    "text_classification": {
        "input": "text",
        "output": "text",
        "prompt": False,
        "context": False
    },
    "token_classification": {
        "input": "text",
        "output": "text",
        "prompt": False,
        "context": False
    },
    "text_generation": {
        "input": "text",
        "output": "text",
        "prompt": False,
        "context": False
    },
    "text_classification": {
        "input": "text",
        "output": "text",
        "prompt": False,
        "context": False
    },
    "translation": {
        "input": "text",
        "output": "text",
        "prompt": False,
        "context": False
    },
    "summarization": {
        "input": "text",
        "output": "text",
        "prompt": False,
        "context": False
    },
}
selected_task = st.sidebar.radio(
        "Choose the task you want to do", # see https://huggingface.co/docs/huggingface_hub/guides/inference"
        dict_hg_tasks.keys()
        )
st.write(f"The current selected task is : {dict_hg_tasks[selected_task]}")
with st.sidebar.expander("tasks documentation"):
    st.write("https://huggingface.co/docs/huggingface_hub/package_reference/inference_client")

# functions ########################################
cwd = os.getcwd()
def get_input(upload,url,text):
    if upload is not None:
        return upload
    else:
        if url:
            print(url)
            return url
        elif text:
            return text
    return None  # Default return if neither upload nor url is provided

def display_inputs(task):
    if dict_hg_tasks_params[task]["input"] == "upload,url":
        return st.file_uploader("Choose a file"),st.text_input("or enter a file url"),""
    elif dict_hg_tasks_params[task]["input"] == "text":
        return None,"",st.text_input("Enter a text")
    else:
        return None,"",""
    
def display_prompt(task):
    if dict_hg_tasks_params[task]["prompt"] is True:
        return st.text_input("Enter a question")
    return None

def display_context(task):
    if dict_hg_tasks_params[task]["context"] is True:
        return st.text_area("Enter a context")
    return None

# UI main client ####################################

if selected_task :
    response = None
    task = dict_hg_tasks[selected_task]
    if model:
        client = InferenceClient(model=model)
    else:
        client = InferenceClient()    
    uploaded_input,url_input,text_input = display_inputs(task)
    prompt_input = display_prompt(task)
    context_input = display_context(task)
    if get_input(uploaded_input,url_input,text_input):
        input = get_input(uploaded_input,url_input,text_input) 
        response = getattr(client, task)(input)
    elif prompt_input:
        if context_input is not None:
            response = getattr(client, task)(question=prompt_input,context=context_input)
        else:
            response = getattr(client, task)(input,prompt=prompt_input)
    if response is not None:
        col1,col2 = st.columns(2)
        with col1:
            if "text" in dict_hg_tasks_params[task]["output"]:  
                st.write(response)
            elif "audio" in dict_hg_tasks_params[task]["output"]: 
                Path(os.path.join(cwd,"audio.flac")).write_bytes(response)
                st.audio(os.path.join(cwd,"audio.flac"))
        with col2:
            if dict_hg_tasks_params[task]["output"] == "image,text":
                if not(isinstance(input, str)):
                    image = Image.open(input)
                else:
                    image = Image.open(requests.get(input, stream=True).raw)
                st.image(image)
            elif dict_hg_tasks_params[task]["output"] == "image":
                response.save(os.path.join(cwd,"generated_image.png"))
                image = Image.open(os.path.join(cwd,"generated_image.png"))
                st.image(image)