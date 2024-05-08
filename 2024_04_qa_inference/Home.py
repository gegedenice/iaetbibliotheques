import streamlit as st
import os,getpass

st.set_page_config(page_title="QA Inference Streamlit App")
# Cache the header of the app to prevent re-rendering on each load
@st.cache_resource
def display_app_header():
    """Display the header of the Streamlit app."""
    st.title("QA Inference Streamlit App")
# Display the header of the app
display_app_header()

