import streamlit as st
import numpy as np
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModel, AutoTokenizer, pipeline
import joblib
from typing import List

# Define styles for different sections
custom_styles = """
<style>
    [id^=tabs-] {
        padding: 20px;
        border: 1px solid #ccc;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .tab h2 {
        font-size: 24px;
        margin-top: 0;
        color: black;
    }

    .response-container {
        padding: 20px;
        border: 1px solid #ccc;
        background-color: #f9f9f9;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .response-container h3 {
        font-size: 20px;
        margin-top: 0;
        color: black;
    }
</style>
"""

def setup_page():
    st.set_page_config(page_title="EAD Generation", layout="wide")
    display_app_header()
    st.markdown(custom_styles, unsafe_allow_html=True)
    # Create two columns - one for query history and one for main content
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("### Query History")
        # Create a container for the query history
        query_history_container = st.container()
    
    return col1, col2

# Cache the header of the app to prevent re-rendering on each load
@st.cache_resource
def display_app_header():
    """Display the header of the Streamlit app."""
    st.title("EAD Generation")
    #st.subheader("Tries")
    st.markdown("---")
    # Add a description of the app
    st.markdown("""This app allows you to generate EAD/XML archival descriptions. See this serie of blog posts for explanations :""")                
    st.markdown("""
    - [https://iaetbibliotheques.fr/2024/11/comment-apprendre-lead-a-un-llm](https://iaetbibliotheques.fr/2024/11/comment-apprendre-lead-a-un-llm)
    - [https://iaetbibliotheques.fr/2024/11/comment-apprendre-lead-a-un-llm-rag-23](https://iaetbibliotheques.fr/2024/11/comment-apprendre-lead-a-un-llm-rag-23)
    - [https://iaetbibliotheques.fr/2024/12/comment-apprendre-lead-a-un-llm-fine-tuning-33](https://iaetbibliotheques.fr/2024/12/comment-apprendre-lead-a-un-llm-fine-tuning-33)
    """, unsafe_allow_html=True)
    st.markdown("---")

# Display the header of the app and get columns
history_col, main_col = setup_page()

def setup_sidebar():    
    groq_models = ["llama3-70b-8192", "llama-3.1-70b-versatile","llama3-8b-8192", "llama-3.1-8b-instant", "mixtral-8x7b-32768","gemma2-9b-it", "gemma-7b-it"]
    selected_groq_models = st.sidebar.radio("Choose a model (used in tabs 1, 2 and 3)", groq_models)
    return selected_groq_models

def create_groq_llm(model):
    return ChatGroq(
        model=model,
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )
    
def create_ollama_llm():
    return ChatOllama(
        model="ollama-3.1-8b-instant",
        base_url="https://api.ollama.com",
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
def setup_zero_shot_tab(llm):
    st.header("Zero-shot Prompting")
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in the domain of archival description in the standardized Encoded Archival Description EAD format** and intelligent EAD/XML generator.",
        ),
        ("human", "{question}"),
    ]
    )
    zero_shot_chain = prompt | llm | StrOutputParser()
    return zero_shot_chain

def setup_ead_xsd_tab(llm):
    st.header("One-shot Prompting with EAD 2002 XSD schema")
    ead_xsd_2002 = open("assets/ead_xsd_2002.xml", "r").read()
    
    class CustomRetriever(BaseRetriever):
        def _get_relevant_documents(
          self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            return [Document(page_content=ead_xsd_2002)]
    
    retriever = CustomRetriever()
    
    template = """
    ### [INST]
    **You are an expert in the domain of archival description in the standardized Encoded Archival Description EAD format** and intelligent EAD/XML generator.
    Use the following xsd schema context to answer the question by generating a compliant xml content.

    {context}

    Please follow the EAD schema guidelines to ensure your output is valid and well-formed. Do not include any markup or comments other than what is specified in the schema.

    ### QUESTION:
    {question}

    [/INST]
    """

    prompt = ChatPromptTemplate.from_template(template)
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return retrieval_chain

def setup_rag_tab(llm):
    st.header("RAG")

    model = AutoModel.from_pretrained("Geraldine/msmarco-distilbert-base-v4-ead", token=st.secrets["HF_TOKEN"])
    tokenizer = AutoTokenizer.from_pretrained("Geraldine/msmarco-distilbert-base-v4-ead", token=st.secrets["HF_TOKEN"])
    #pca = hf_hub_download("Geraldine/msmarco-distilbert-base-v4-ead", "pca_model.joblib",local_dir="assets")
    feature_extraction_pipeline = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
    
    class HuggingFaceEmbeddingFunction:
        def __init__(self, pipeline, pca_model_path):
            self.pipeline = pipeline
            self.pca = joblib.load(pca_model_path)
                
        # Function for embedding documents (lists of text)
        def embed_documents(self, texts):
            # Get embeddings as numpy arrays
            embeddings = self.pipeline(texts)
            embeddings = [embedding[0][0] for embedding in embeddings]
            embeddings = np.array(embeddings)

            # Transform embeddings using PCA
            reduced_embeddings = self.pca.transform(embeddings)
            return reduced_embeddings.tolist()

        # Function for embedding individual queries
        def embed_query(self, text):
            embedding = self.pipeline(text)
            embedding = np.array(embedding[0][0]).reshape(1, -1)

            # Transform embedding using PCA
            reduced_embedding = self.pca.transform(embedding)
            return reduced_embedding.flatten().tolist()

    embeddings = HuggingFaceEmbeddingFunction(feature_extraction_pipeline, pca_model_path="assets/pca_model.joblib")
    persist_directory = "assets/chroma_xml_db"
    vector_store = Chroma(
        collection_name="ead-xml",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    retriever = vector_store.as_retriever()
    
    template = """
    # Generate EAD/XML File for Archival Collection

    ## Description
    You are an assistant for the generation of archives encoded in EAD/XML format.
    You are an expert in archival description rules and standards, knowing very well the EAD format for encoding archival metadata.

    ## Instruction
    Answer the question based only on the following context:
    {context}.

    The EAD/XML sections you generate must follow the Library of Congress EAD schema and be in the style of a traditional archival finding aid, as if written by a professional archivist, including the required XML tags and structure.

    ## Question
    {question}

    ## Answer
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def setup_fine_tuned_tab():
    st.header("Fine-tuned Zephir model")
    
    llm = create_ollama_llm()
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in the domain of archival description in the standardized Encoded Archival Description EAD format** and intelligent EAD/XML generator.",
        ),
        ("human", "{question}"),
    ]
    )
    fine_tuned_chain = prompt | llm | StrOutputParser()
    return fine_tuned_chain

def clear_outputs():
    # Clear all stored responses from session state
    for key in list(st.session_state.keys()):
        if key.startswith('response_'):
            del st.session_state[key]

# Initialize query history in session state if it doesn't exist
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Move all main content to the main column
with main_col:
    selected_groq_models = setup_sidebar()

    if 'previous_query' not in st.session_state:
        st.session_state.previous_query = ''
        
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ''

    query = st.chat_input("Enter your query", key="chat_input")
    st.markdown("*Example queries : Generate an EAD description for the personal papers of Marie Curie, Create an EAD inventory for a collection of World War II photographs, Create a EAD compliant `<eadheader>` sections with all necessary attributes and child elements.*")

    tab1, tab2, tab3, tab4 = st.tabs(["Zero-shot prompting","One-shot prompting with EAD schema", "RAG", "Fine-tuned Zephir model"])

    # Display info messages for each tab on app launch
    with tab1:
        st.info("Simple inference with zero-shot prompting : the prompt used to interact with the model does not contain examples or demonstrations. The LLM used id the one selected in the sidebar list.",icon="ℹ️")

    with tab2:
        st.info("One-shot inference with EAD 2002 XSD schema : the prompt used to interact with the model contains the plaintext of the EAD schema as a guideline for the desired output format. The LLM used id the one selected in the sidebar list.",icon="ℹ️")

    with tab3:
        st.info("Retrieval-augmented generation : the prompt used to interact with the model contains the relevant context from an archival collection of EAD files. The LLM used id the one selected in the sidebar list.",icon="ℹ️")

    with tab4:
        st.info("FineLlama-3.2-3b-Instruct-ead model : this is a custom fine-tuned adaptation of llama-3.2-3b-instruct post-trained on a dataset of archival descriptions in the EAD format",icon="ℹ️")

    # Process query for all tabs when submitted
    if query:
        # Add new query to history if it's different from the last one
        if not st.session_state.query_history or query != st.session_state.query_history[-1]:
            st.session_state.query_history.append(query)
        
        # Store the current query
        st.session_state.current_query = query
        # Clear outputs if query has changed
        if query != st.session_state.previous_query:
            clear_outputs()
            st.session_state.previous_query = query
            
        with st.spinner('Processing query across all models...'):
            # Process for Tab 1 - zero-shot inference
            with tab1:
                llm = create_groq_llm(selected_groq_models)
                zero_shot_chain = setup_zero_shot_tab(llm)
                st.session_state.response_zero_shot = zero_shot_chain.invoke({"question": query})
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.response_zero_shot)
            
            # Process for Tab 2 - EAD XSD
            with tab2:
                llm = create_groq_llm("llama-3.1-8b-instant")
                ead_chain = setup_ead_xsd_tab(llm)
                st.session_state.response_ead = ead_chain.invoke(query)
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.response_ead)
            
            # Process for Tab 3 - RAG
            with tab3:
                llm = create_groq_llm(selected_groq_models)
                rag_chain = setup_rag_tab(llm)
                st.session_state.response_rag = rag_chain.invoke(query)
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.response_rag)
                    
            # Process for Tab 4 - Fine-tuned model
            with tab4:
                fine_tuned_chain = setup_fine_tuned_tab()
                st.session_state.response_fine_tuned = fine_tuned_chain.invoke(query)
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.response_fine_tuned)  

# Display query history in the sidebar column
with history_col:
    if st.session_state.query_history:
        for i, past_query in enumerate(reversed(st.session_state.query_history)):
            st.text_area(f"Query {len(st.session_state.query_history) - i}", 
                        past_query, 
                        height=100,
                        key=f"history_{i}",
                        disabled=True)
    else:
        st.write("No queries yet")