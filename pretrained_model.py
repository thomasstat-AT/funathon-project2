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

from split_the_data import *
example_texts = X_test[:6]
example_true = y_test[:6]
print(y_test)

preds = ttc.predict(np.array(example_texts), top_k=1,explain_with_captum=True)

print(preds)
print(example_texts)
print(example_true)

predictions = preds["prediction"]
print(predictions)
captum = preds["captum_attributions"]
print(captum)

from torchTextClassifiers.utilities.plot_explainability import *


text_idx = 0
top_k_idx = 0
text_sample         = example_texts[text_idx]
offsets             = preds["offset_mapping"][text_idx]
word_ids            = preds["word_ids"][text_idx]
predicted_code = preds["prediction"][text_idx][top_k_idx]
attributions  = preds["captum_attributions"][text_idx][top_k_idx]
print(attributions)

words, word_attributions = map_attributions_to_word(
    attributions.unsqueeze(0), text_sample, word_ids, offsets
)
char_attributions = map_attributions_to_char(attributions.unsqueeze(0), offsets, text_sample)

titles = [f"Attributions for NACE code {predicted_code}"]

figshow(plot_attributions_at_char(
    text=text_sample, attributions_per_char=char_attributions, titles=titles,
)[0])

figshow(plot_attributions_at_word(
    text=text_sample, words=words.values(), attributions_per_word=word_attributions, titles=titles,
)[0])

print(X_test[0])
# %%
gefinkelt = np.array(["industrial bakery"])


pred_gefinkelt = ttc.predict(np.array(gefinkelt), top_k=2,explain_with_captum=True)

print(pred_gefinkelt)

results_test = ttc.predict(X_test, top_k=1)
preds    = results_test["prediction"].squeeze(1)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
# %%
test = f"Test"


# %%
