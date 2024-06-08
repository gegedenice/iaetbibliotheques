import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pickle

# Set Streamlit page configuration
st.set_page_config(page_title="App")

st.title("Semantic Search on HAL UNIV-COTEDAZUR SHS articles from 2013 to 2023")
st.subheader("The pre-processed data are accessible and documented from this HF dataset [Geraldine/hal_univcotedazur_shs_articles_2013-2023](https://huggingface.co/datasets/Geraldine/hal_univcotedazur_shs_articles_2013-2023)")

with st.spinner('Loading datasets...'):
    dataset = load_dataset(
      "Geraldine/hal_univcotedazur_shs_articles_2013-2023",
       revision="main"
       )
    # data
    hal_data = load_dataset("Geraldine/hal_univcotedazur_shs_articles_2013-2023", data_files="hal_data.csv")
    df = pd.DataFrame(hal_data["train"])
    df = df.replace(np.nan, '')
    df = df.astype(str)
    # embeddings
    hf_hub_download(repo_id="Geraldine/hal_univcotedazur_shs_articles_2013-2023", 
                filename="hal_embeddings.pkl", 
                repo_type="dataset",
                cache_dir="./", local_dir="./")
    file = open("./hal_embeddings.pkl",'rb')
    corpus_embeddings = pickle.load(file)
    
model_id = "sentence-transformers/all-MiniLM-L6-v2"
def llm_response(query):
    embedder = SentenceTransformer(model_id)
    question_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=5)
    article_data_list = []
    data_list = []
    for hit in hits[0]:
        hit_id = hit['corpus_id']
        article_data = df.iloc[hit_id]
        #article_data_list.append(article_data["combined"])
        article_data_list.append({"title": article_data["title_s"] + ". " + article_data["subTitle_s"],
                                  "date": article_data["producedDate_s"],
                                  "journal" : article_data["journalTitle_s"],
                                  "pub": article_data["journalPublisher_s"],
                                  "abstract": article_data["abstract_s"]
                                  })
    return article_data_list

with st.container():
    if query := st.text_input(
        "Enter your search:"):
        with st.expander(":blue[click here to see the HAL search engine results]"):
            components.iframe(f"https://hal.univ-cotedazur.fr/search/index/?q={query}&rows=30&publicationDateY_i=2023+OR+2022+OR+2021+OR+2020+OR+2019+OR+2018+OR+2017+OR+2016+OR+2015+OR+2014+OR+2013&docType_s=ART", height=800, scrolling=True)
        with st.spinner('Calculating...'):
            response = llm_response(query)
            for x in response:
                st.success("**Title** : " + x["title"] + "  \n  " + "**Date** : " + x["date"] + "  \n  " + "**Journal** : " + x["journal"] + "(" + x["pub"] + ")" + "  \n  " + "**Abstract** : " + x["abstract"]) 

