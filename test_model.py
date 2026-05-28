from split_the_data import *

## 3.2 Encode the Labels
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
encoder.fit(train['code'].to_numpy())

# Make sure, all codes are contained in test data
np.unique(y_train).size

## 3.3
# Import the ValueEncoder class from the 
# torchTextClassifiers.value_encoder subpackage, 
# then create a value_encoder object by passing your fitted 
# label_encoder as an argument.
from torchTextClassifiers.value_encoder import ValueEncoder

value_encoder = ValueEncoder(label_encoder=encoder)

# You are going to use a WordPieceTokenizer from the 
# torchTextClassifiers package (see the documentation). 
# Train a WordPieceTokenizer from scratch on the training labels 
# with vocab_size=5000 and output_dim=10. Use the fitted tokenizer 
# to tokenize one observations (with the tokenize method), 
# and inspect the result to understand how the text is split.
import torchTextClassifiers
print(dir(torchTextClassifiers))
## 4.3 Tokenizer
from torchTextClassifiers.tokenizers import WordPieceTokenizer

tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
# WordPieceTokenizer(vocab_size=5000, output_dim=10)

tokenizer.train(X_train)

print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

print(tokenizer.tokenize(X_train[1]))
print(X_train[1])


# Look at an example of tokenization
print("Raw text", X_train[0])
print(
    "Tokens id:",
    tokenizer.tokenize(X_train[1]).input_ids.squeeze(0)
)
print(
    "Tokens:",
    tokenizer.tokenizer.convert_ids_to_tokens(
        tokenizer.tokenize(X_train[1]).input_ids.squeeze(0)
    )
)

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers


model_config = ModelConfig(embedding_dim=96, num_classes=n_classes,)

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)


training_config = TrainingConfig(lr=5e-4, batch_size=128, num_epochs=1, patience_early_stopping=5,)

# %%
mlflow.set_experiment("funathon-2026-project2")
mlflow.pytorch.autolog()

# %%
with mlflow.start_run() as run:
    # This should take approximately 1-2mn
    ttc.train(
        X_train,
        y_train,
        training_config=training_config,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )

    mlflow.log_artifacts(
        training_config.save_path,   # local folder produced by ttc.train()
        artifact_path="model_artifacts",
    )

# %%
#| label: load-from-run
#| code-overflow: scroll
#| output: true
local_dir = mlflow.artifacts.download_artifacts(
    f"runs:/{run.info.run_id}/model_artifacts"
)

# Rebuild the torchTextClassifiers object from the downloaded files
ttc_loaded = torchTextClassifiers.load(local_dir)

# %%
example_texts = X_test[1:10]
preds = ttc_loaded.predict(np.array(example_texts), top_k=5,explain_with_captum=True)