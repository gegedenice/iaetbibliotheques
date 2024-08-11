# Divers

Miscellaneous pieces of code

## Download d'un dataset hebergé sur HF

### Avec le package datasets

```
#!pip install datasets

from datasets import load_dataset
import pandas as pd
```

```
hal_dataset = load_dataset(
  "Geraldine/hal_univcotedazur_shs_articles_2013-2023",
  revision="main"  # tag name, or branch name, or commit hash
)

print(hal_dataset)

# returns
DatasetDict({
    train: Dataset({
        features: ['doiId_s', 'uri_s', 'title_s', 'subTitle_s', 'authFullName_s', 'producedDate_s', 'journalTitle_s', 'journalPublisher_s', 'abstract_s', 'domain_s', 'openAccess_bool', 'combined'],
        num_rows: 2417
    })
})
# dataset to pandas dataframe
df = pd.DataFrame(hal_dataset["train"])
```

Ou directement

```
# dataset to pandas dataframe
hal_dataset = load_dataset("Geraldine/hal_univcotedazur_shs_articles_2013-2023", data_files="hal_data.csv")
df = pd.DataFrame(hal_dataset["train"])
```

Ou sur la branche refs/convert/parquert créée automatiquement par Huggingface après le dépôt

```
hal_dataset = load_dataset(
  "Geraldine/hal_univcotedazur_shs_articles_2013-2023",
  revision="refs/convert/parquet"  # tag name, or branch name, or commit hash
)
df = pd.DataFrame(hal_dataset["train"])
```



### Avec le package huggingface_hub

```
from huggingface_hub import hf_hub_download

folder_to_store_the_data = "./data"

hf_hub_download(repo_id="Geraldine/hal_univcotedazur_shs_articles_2013-2023",
                filename="hal_embeddings.pkl",
                repo_type="dataset",
                cache_dir=folder_to_store_the_data, local_dir=folder_to_store_the_data)
```

