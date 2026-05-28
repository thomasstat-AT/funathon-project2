# %%

import mlflow
from dotenv import load_dotenv

load_dotenv(override=True)

import polars as pl


df = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet")
print(df)

df.head(10)
df.height

n_classes = df["code"].n_unique()
print(n_classes)

## 3.1
import numpy as np
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, random_state=1, test_size=0.3)
test, validation = train_test_split(test, random_state=1, test_size=0.5)


X_train, y_train = train["label"].to_numpy(), train["code"].to_numpy()
X_val, y_val = validation["label"].to_numpy(), validation["code"].to_numpy()
X_test, y_test = test["label"].to_numpy(), test["code"].to_numpy()

print(f"Train: {len(train)} | Val: {len(validation)} | Test: {len(test)}")

