from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from PIL import Image
import requests
import os
import json
import math
import re
import pandas as pd
import numpy as np
from omeka_s_api_client import OmekaSClient,OmekaSClientError
from typing import List, Dict, Any, Union
import io
from dotenv import load_dotenv

# env var
load_dotenv(os.path.join(os.getcwd(), ".env"))
HF_TOKEN = os.environ.get("HF_TOKEN")

# Nomic vison model
processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

# Nomic text model
text_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, token=HF_TOKEN)

def image_url_to_pil(url: str, max_size=(512, 512)) -> Image:
    """
    Ex usage : image_blobs = df["image_url"].apply(image_url_to_pil).tolist()
    """
    response = requests.get(url, stream=True, timeout=5)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def generate_img_embed(images_urls, batch_size=20):
    """Generate image embeddings in batches to manage memory usage.
    
    Args:
        images_urls (list): List of image URLs
        batch_size (int): Number of images to process at once
    """
    all_embeddings = []
    
    for i in range(0, len(images_urls), batch_size):
        batch_urls = images_urls[i:i + batch_size]
        images = [image_url_to_pil(image_url) for image_url in batch_urls]
        inputs = processor(images, return_tensors="pt")
        img_emb = vision_model(**inputs).last_hidden_state
        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
        all_embeddings.append(img_embeddings.detach().numpy())
    
    return np.vstack(all_embeddings)

def generate_text_embed(sentences: List, batch_size=64):
    """Generate text embeddings in batches to manage memory usage.
    
    Args:
        sentences (List): List of text strings to encode
        batch_size (int): Number of sentences to process at once
    """
    all_embeddings = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        embeddings = text_model.encode(batch_sentences)
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)
            
def add_concatenated_text_field_exclude_keys(item_dict, keys_to_exclude=None, text_field_key="text", pair_separator=" - "):
    if not isinstance(item_dict, dict):
        raise TypeError("Input must be a dictionary.")
    if keys_to_exclude is None:
        keys_to_exclude = set() # Default to empty set
    else:
        keys_to_exclude = set(keys_to_exclude) # Ensure it's a set for efficient lookup

    # Add the target text key to the exclusion set automatically
    keys_to_exclude.add(text_field_key)

    formatted_pairs = []
    for key, value in item_dict.items():
        # 1. Skip any key in the exclusion set
        if key in keys_to_exclude:
            continue

        # 2. Check for empty/invalid values (same logic as before)
        is_empty_or_invalid = False
        if value is None: is_empty_or_invalid = True
        elif isinstance(value, float) and math.isnan(value): is_empty_or_invalid = True
        elif isinstance(value, (str, list, tuple, dict)) and len(value) == 0: is_empty_or_invalid = True

        # 3. Format and add if valid
        if not is_empty_or_invalid:
            formatted_pairs.append(f"{str(key)}: {str(value)}")
            
    concatenated_text = f"search_document: {pair_separator.join(formatted_pairs)}"
    item_dict[text_field_key] = concatenated_text
    return item_dict

def prepare_df_atlas(df: pd.DataFrame, id_col='id', images_col='images_urls'):

    # Drop completely empty columns
    #df = df.dropna(axis=1, how='all')

    # Fill remaining nulls with empty strings
    #df = df.fillna('')

    # Ensure ID column exists
    if id_col not in df.columns:
        df[id_col] = [f'{i}' for i in range(len(df))]

    # Ensure indexed field exists and is not empty
    #if indexed_col not in df.columns:
    #    df[indexed_col] = ''

    #df[images_col] = df[images_col].apply(lambda x: [x[0]] if isinstance(x, list) and len(x) > 1 else x if isinstance(x, list) else [x])
    df[images_col] = df[images_col].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Optional: force all to string (can help with weird dtypes)
    for col in df.columns:
        df[col] = df[col].astype(str)

    return df

def remove_key_value_from_dict(list_of_dict, key_to_remove):
    new_list = []
    for dictionary in list_of_dict:
        new_dict = dictionary.copy()  # Create a copy to avoid modifying the original list
        if key_to_remove in new_dict:
            del new_dict[key_to_remove]
        new_list.append(new_dict)
    return new_list

def remove_key_value_from_dict(input_dict, key_to_remove='text'):
    if not isinstance(input_dict, dict):
        raise TypeError("Input must be a dictionary.")

    if key_to_remove in input_dict:
        del input_dict[key_to_remove]

    return input_dict