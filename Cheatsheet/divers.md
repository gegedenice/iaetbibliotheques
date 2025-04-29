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

```
dataset = load_dataset("Geraldine/Ead-Instruct-full-20k", split="train")
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

## datasets

```
#!pip install datasets

from datasets import Dataset,load_dataset
import pandas as pd
```

### Save & read jsonl file locally

```
# Save from json array or python dict
with open(f"{GDRIVE_PATH}/training-datasets-generation/Ead-Instruct-5k.jsonl", "w") as f:
    for entry in sample_dataset_5k:
        f.write(json.dumps(entry) + "\n"
```

```
# Reload in dataframe
df = pd.read_json(f"{GDRIVE_PATH}/training-datasets-generation/Ead-Instruct-20k.jsonl", lines=True,orient="records")
```

```
dataset = df.to_dict(orient="records")
df = pd.DataFrame(dataset)
```

### datasets & dataframe

```
df = pd.read_json(f"{GDRIVE_PATH}/training-datasets-generation/Ead-Instruct-20k.jsonl", lines=True,orient="records")
data = Dataset.from_pandas(df)
```

### Split Dataset

```
dataset = load_dataset("Geraldine/Ead-Instruct-50k", split="train")
ds = dataset.train_test_split(test_size=0.3)
train_dataset = ds["train"]
eval_dataset = ds["test"]
```

### Map dataset

```
def preprocess_data(example):
  example["text"] = example["text"] + EOS_TOKEN
  return example

dataset = dataset.map(preprocess_data, batched=False)
```

```
def clean_xml_tags(text):
  """
  Cleans XML tags by removing extra spaces between them.

  Args:
      text: The text containing XML tags.

  Returns:
      The cleaned text.
  """
  import re
  text = re.sub(r'>\s+<', '><', text)
  return text
  
cleaned_dataset = dataset.map(lambda example: {
      column_name: clean_xml_tags(example[column_name].replace('\n', ' ').replace('\t', ' '))
  })
```

```
def clean_values(dataset, column_name):
  """
  Cleans values in a specified column of a dataset by removing newline, tab, and other unwanted characters.

  Args:
      dataset: The dataset to clean.
      column_name: The name of the column to clean.

  Returns:
      The cleaned dataset.
  """

  cleaned_dataset = dataset.map(lambda example: {
      column_name: clean_xml_tags(example[column_name].replace('\n', ' ').replace('\t', ' '))
  })
  return cleaned_dataset
  
dataset = clean_values(dataset, "prompt")
```

### Push to HuggingFace hub

```
!pip install huggingface_hub
```

```
from huggingface_hub import login, logout, whoami
from google.colab import userdata

login(token = userdata.get('WRITE_HF_TOKEN'))
whoami()
```

```
dataset.push_to_hub("Geraldine/Ead-Instruct-full-20k", private=False)
```

### Miscellaneous

```
# Remove columns
dataset = dataset.remove_columns(["prompt", "completion"])
```

```
import torch
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

```
# see size
print(dataset['input_ids'].shape)
# ex result : torch.Size([10000, 128])
```

```
# see data type
print(type(dataset[0]["input_ids"]))
```

## Calculate the minimum required GPU memory per LLM

Source : https://gist.github.com/philschmid/d188034c759811a7183e7949e1fa0aa4




