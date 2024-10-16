import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import tool
from gradio_client import Client, file, handle_file
from PIL import Image

# Adjust the width of the Streamlit page
st.set_page_config(
    page_title='Image to Unimarc/XML Converter',
    layout="wide"
)

# Initialize Streamlit app
st.title("Image to Unimarc/XML Converter")

def create_file_path(uploaded_file):
    file_path = os.path.join("files", uploaded_file.name)
    with open(file_path, "wb") as user_file:
        user_file.write(uploaded_file.getbuffer())
    return file_path

# Initialize crew.ai API
llm = LLM(
    model="llama3-70b-8192",
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

@st.cache_resource
def get_gradio_client(space_name):
    return Client(space_name, hf_token=st.secrets["HF_TOKEN"])

@tool("Image to Text Converter")
def image_to_text_tool(image_path: str):
    """Extracts text content from the image passed as argument. Returns the extracted text."""
    text = "Extracts the text from this image taking into account the style (bold words, lists, etc.) and retranscript this content in markdown format"
    
    # Get the client using the cached function
    client = get_gradio_client(st.session_state.current_space_key)
    st.write(image_path)
    try:
        result = client.predict(
            images=handle_file(image_path),
            text=text,
            assistant_prefix="",
            decoding_strategy="Greedy",
            temperature=0.4,
            max_new_tokens=512,
            repetition_penalty=1.2,
            top_p=0.8,
            api_name="/model_inference"
        )
        return result
    except Exception as e:
        st.error(f"Error in image_to_text_tool: {str(e)}")
        return f"Failed to process image: {str(e)}"

agent_image_extractor = Agent(
    llm=llm,
    role="Image extractor",
    goal="Extract text content from this image {image_path}",
    backstory="An AI assistant specialized in extracting text content from images.",
    tools=[image_to_text_tool],
    verbose=True,
)

task_image_extraction = Task(
    expected_output="The text in the image {image_path}",
    description="The extracted textual content from {image_path}",
    agent=agent_image_extractor,
)

agent_marc_converter = Agent(
    llm=llm,
    role="Expert in bibliographic metadata formatted in Unimarc/XML",
    backstory="An AI assistant expert in bibliographic metadata and cataloguing in Unimarc/XML.",
    goal="Generate Unimarc/XML formatted record from unstructured textual metadata",
    context=[task_image_extraction],
    verbose=True,
)

task_marc_conversion = Task(
    description="A Unimarc/xml record",
    expected_output="A Unimarc/XML record",
    agent=agent_marc_converter,
    context=[task_image_extraction],
)

col1,col2 = st.columns(2)
with col1:
    spaces = ["HuggingFaceM4/idefics3","arad1367/Marketing_Vision_HuggingFaceM4_idefics3"]
    space = st.selectbox(":green[Choose the space]",spaces,key="current_space_key")
with col2:
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if st.button("Run"):
    if uploaded_file is not None:
        col1,col2 = st.columns(2)
        # display image
        with col1:
            image = Image.open(uploaded_file)
            st.image(image)
        
        file_path = create_file_path(uploaded_file)
        # Run the crew.ai pipeline        
        my_crew = Crew(agents=[agent_image_extractor, agent_marc_converter], tasks=[task_image_extraction, task_marc_conversion])
        with col2:
            # Display the final answer
            st.write("Final Answer:")
            st.write(my_crew.kickoff(inputs={"image_path": file_path}))
            