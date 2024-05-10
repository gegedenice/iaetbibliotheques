import streamlit as st

st.set_page_config(page_title="QA Inference Streamlit App using Ollama, Nvidia and Groq", layout="wide")

st.write("# QA Inference with Ollama & Nvidia & Groq as LLMs providers")
st.markdown(
        """
        This app is a demo for showing how to interact with LLMs in the case of three providers : Ollama, the Nvidia Cloud and Groq.

        You can use one, two or the three LLMs hosting solutions according to your environment :
        
        - **[Ollama](https://ollama.com/)** : a local Ollama instance must be running on http://localhost:11434 (change the base_url in clients.py if needed)
        - **[Nvidia Cloud](https://build.nvidia.com/explore/discover)** : if you want to test the LLMs hosted on Nvidia Cloud and mostly the no-latency QA process on Nvidia GPU, you need to create an (free) account and generate an API key
        - **[Groq Cloud](https://console.groq.com/playground)** : if you want to test the LLMs hosted on Groq and especially the speed of execution of the inference process on Groq LPU, you need to create an (free) account and generate an API key


        The app contains two pages implementing the same kind of chatbot, the only difference is how to achieve the LLM answer

        - 👉 **App API completion** page : this page illustrates how to query a LLM by using OpenAI-like APIs or the OpenAI client
        - 👉 **App Langchain completion** page : this page illustrates how to query a LLM using appropriate Langchain components

    """
    )
    
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Contact 🤙 <a style='display: block; text-align: center;' href="mailto:geraldine.geoffroy@epdl.ch" target="_blank">Géraldine Geoffroy</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

