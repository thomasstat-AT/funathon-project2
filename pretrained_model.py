# %%

import s3fs
import torchTextClassifiers
import mlflow
from dotenv import load_dotenv
load_dotenv(override=True)
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers




fs = s3fs.S3FileSystem(
    anon=True,  # public bucket
    endpoint_url="https://minio.lab.sspcloud.fr",
)

local_dir = "./mlflow-artifacts/"
fs.get(
    "projet-funathon/diffusion/mlflow-artifacts/",
    local_dir,
    recursive=True,
)
# Rebuild the torchTextClassifiers object from the downloaded files
ttc = torchTextClassifiers.load(local_dir)

ttc.pytorch_model.eval()
# %%
from split_the_data.py import X_test
example_texts = X_test[1:10]
# %%
preds = ttc.predict(np.array(example_texts), top_k=5,explain_with_captum=True)
# %%
