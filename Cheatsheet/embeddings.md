#  Embeddings

Pieces of code for free embeddings

## HuggingFace embeddings

### API

```
import pandas as pd
import requests
import json
from typing import Optional, List, Dict, Any

HF_TOKEN = <hf_token>
model_id = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def embeddings_query(text:List) -> List:
    response = requests.post(embeddings_api_url, headers=headers, json={"inputs": text, "options":{"wait_for_model":True}})
    return response.json()

# Exemple 1
embeddings_query("Refonte de theses.fr : éclairage sur les choix informatiques")

# Exemple 2
embeddings_query(["Refonte de theses.fr : éclairage sur les choix informatiques"])

# Exemple 3 (ajout d'une colonne d'embeddings)
df = pd.read_csv("data/hal....csv", sep=",", encoding="utf-8")

df['embeddings'] = df.combined.apply(lambda x:embeddings_query(x.strip()))
```

### Avec sentence-transformers (local CPU)

```
import pandas as pd
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import pickle

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Exemple 1 : création d'un corpus d'embeddings sur les données d'une colonne
df = pd.read_csv("data/hal....csv", sep=",", encoding="utf-8")
corpus_embeddings = embedder.encode(df.combined.to_list(), show_progress_bar=True)

corpus_embeddings[0].shape
# returns (384,) 

with open('data/....pkl', "wb") as fOut:
  pickle.dump(corpus_embeddings, fOut)

# Exemple 2 : ajout d'une colonne d'embeddings

def embeddings_query(text):
    return embedder.encode(text,convert_to_tensor=True)

df['embeddings'] = df.combined.apply(lambda x:embeddings_query(x.strip()))


```

#### Bonus : Calcul de similarité

```
from sentence_transformers import util

df = pd.read_csv("data/hal....csv", sep=",", encoding="utf-8")
file = open("data/hal_articles-052024-embeddings.pkl",'rb')
corpus_embeddings = pickle.load(file)

prompt = "Ecologie"
prompt_embedding = embedder.encode(prompt, convert_to_tensor=True)
hits = util.semantic_search(prompt_embedding, corpus_embeddings, top_k=10)
hits = pd.DataFrame(hits[0], columns=['corpus_id', 'score'])

def get_similar_records(hits):
    article_data_list = []
    data_list = []
    for hit in hits[0]:
        hit_id = hit['corpus_id']
        article_data = df.iloc[hit_id]
        #article_data_list.append(article_data["combined"])
        article_data_list.append({"title": article_data["title_s"],
                                  "subtitle": article_data["subTitle_s"],
                                  "date": article_data["producedDate_s"],
                                  "journal" : article_data["journalTitle_s"],
                                  "pub": article_data["journalPublisher_s"],
                                  "abstract": article_data["abstract_s"]
                                  })
    return article_data_list
	
```

### Langchain et l'API d'HF

```
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

HF_TOKEN = <hf_token>
model_id = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name=model_id
)

# Exemple
embeddings.embed_query("Refonte de theses.fr : éclairage sur les choix informatiques")
```

### LlamaIndex ONNX format (local CPU)

```
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding

model_name = "BAAI/bge-small-en-v1.5"

def convert_hf_embddings_model_to_onnx(model_name):
    if not(os.path.exists(f"./hf_models/{model_name.partition('/')[2]}_onnx")):
        os.makedirs(f"./hf_models/{model_name.partition('/')[2]}_onnx")
        OptimumEmbedding.create_and_save_optimum_model(
              model_name, f"./hf_models/{model_name.partition('/')[2]}_onnx")
    return OptimumEmbedding(folder_name=f"./hf_models/{model_name.partition('/')[2]}_onnx")

embeddings = convert_hf_embddings_model_to_onnx(model_name)
```

### LlamaIndex HF embeddings (local CPU)

```
!pip install llama-index-embeddings-huggingface llama-index-embeddings-instructor


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

### LlamaIndex Nomic Embeddings (remote)

```
!pip install -U llama-index llama-index-embeddings-nomic

nomic_api_key = "<NOMIC_API_KEY>"

import nest_asyncio

nest_asyncio.apply()

from llama_index.embeddings.nomic import NomicEmbedding

embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=128,
    model_name="nomic-embed-text-v1.5",
)

```

## Nomic

https://docs.nomic.ai/reference/endpoints/nomic-embed-text
https://blog.nomic.ai/posts/nomic-embed-matryoshka
https://docs.nomic.ai/reference/endpoints/nomic-embed-vision

### Images embeddings

#### API

```
import requests

NOMIC_API_KEY = "..."

headers = {
    "Authorization": f"Bearer {NOMIC_API_KEY}",
    "Content-Type": "application/x-www-form-urlencoded"
}

data = {
    "model": "nomic-embed-vision-v1.5",
    "urls": ["https://static.nomic.ai/secret-model.png", "https://static.nomic.ai/secret-model-2.png"]
}

response = requests.post("https://api-atlas.nomic.ai/v1/embedding/image", headers=headers, data=data)

print(response.text)
```

##### Client Python

```
! pip install nomic

from nomic import embed
import numpy as np

output = embed.image(
    images=[
        "image_path_1.jpeg",
        "image_path_2.png",
    ],
    model='nomic-embed-vision-v1.5',
)

print(output['usage'])
embeddings = np.array(output['embeddings'])
print(embeddings.shape)
```




