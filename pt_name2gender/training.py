from pt_name2gender.config import SEED, MODEL_DIR
from pt_name2gender.preprocessing import load_name_dataset, explode_names
from pt_name2gender.model import create_model

from sklearn import model_selection
from sklearn.metrics import classification_report
from tensorflow import keras
import numpy as np
import pandas as pd


def train_test_split(complete_names):
    train, test = model_selection.train_test_split(complete_names, test_size=0.1, random_state=SEED)
    train, val = model_selection.train_test_split(train, test_size=0.1, random_state=SEED)

    X_train = train["all_names_norm"]
    X_val = val["all_names_norm"]
    X_test = test["all_names_norm"]

    y_train = train["classification"].replace({"M": 1, "F": 0}).astype(int)
    y_val = val["classification"].replace({"M": 1, "F": 0}).astype(int)
    y_test = test["classification"].replace({"M": 1, "F": 0}).astype(int)

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )


def tokenize_names(encoder, X_train, X_val):
    tokenized_X_train = encoder.texts_to_sequences(X_train)
    tokenized_X_val = encoder.texts_to_sequences(X_val)

    max_length = max(map(len, tokenized_X_train))

    padded_X_train = keras.preprocessing.sequence.pad_sequences(
        tokenized_X_train, maxlen=max_length
    )
    padded_X_val = keras.preprocessing.sequence.pad_sequences(
        tokenized_X_val, maxlen=max_length
    )

    return padded_X_train, padded_X_val


def test_model(model, X_test, y_test):
    tokenized_X_test = encoder.texts_to_sequences(X_test)
    max_length = max(map(len, tokenized_X_test))

    padded_X_test = keras.preprocessing.sequence.pad_sequences(
        tokenized_X_test, maxlen=max_length
    )

    y_pred = model.predict(padded_X_test, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)

    report = classification_report(y_test, y_pred, target_names=["F", "M"])
    return report


if __name__ == '__main__':
    # Suppress warning in replace method
    pd.set_option('future.no_silent_downcasting', True)

    names = load_name_dataset()
    complete_names = explode_names(names)

    data_split = train_test_split(complete_names)

    X_train, y_train, X_val, y_val, X_test, y_test = data_split

    # Tokenize names at character level
    encoder = keras.preprocessing.text.Tokenizer(
        char_level=True, lower=False, filters=None
    )

    encoder.fit_on_texts(X_train)

    tokenized_X_train, tokenized_X_val = tokenize_names(
        encoder,
        X_train,
        X_val
    )

    # Hyperparameters
    embedding_dim = 8
    embedding_lenght = len(encoder.index_word) + 1

    model = create_model(
        embedding_dim,
        embedding_lenght,
        tokenized_X_train.shape
    )

    history = model.fit(
        x=tokenized_X_train, y=y_train,
        epochs=10, shuffle=True,
        batch_size=128, 
        validation_data=(tokenized_X_val, y_val)
    )

    print("Training finished. Evaluating model...")
    report = test_model(model, X_test, y_test)

    print(report)

    # Save model and encoder
    model.save(MODEL_DIR / "PT-Name2Gender.keras")

    with open(MODEL_DIR / "Name-Encoder.json", "w") as f:
        f.write(encoder.to_json())
