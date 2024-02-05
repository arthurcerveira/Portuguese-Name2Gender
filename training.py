from sklearn import model_selection
from tensorflow import keras

from preprocessing import load_name_dataset, explode_names
from model import create_model


SEED = 1907
LABELS = {0: 'F', 1: 'M'}


def train_test_split(complete_names):
    train, test = model_selection.train_test_split(complete_names, test_size=0.1, random_state=SEED)
    train, val = model_selection.train_test_split(train, test_size=0.1, random_state=SEED)

    X_train = train["all_names_norm"]
    X_val = val["all_names_norm"]
    X_test = test["all_names_norm"]

    y_train = train["classification"].replace({"M": 1, "F": 0})
    y_val = val["classification"].replace({"M": 1, "F": 0})
    y_test = test["classification"].replace({"M": 1, "F": 0})

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


if __name__ == '__main__':
    names = load_name_dataset()
    complete_names = explode_names(names)

    data_split = train_test_split(complete_names)

    X_train, y_train, X_val, y_val, *_ = data_split

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

    # Save model and encoder
    model.save("model/PT-Name2Gender.keras")

    with open("model/Name-Encoder.json", "w") as f:
        f.write(encoder.to_json())
