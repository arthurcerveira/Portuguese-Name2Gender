import json

from tensorflow import keras
import numpy as np

from preprocessing import remove_accents
from config import MODEL_DIR, DATA_DIR


def name_to_gender_pipeline(name, name2gender=None, encoder=None, names=None):
    model, tokenizer, names_dict = load_resources()

    # If not provided, use the loaded resources
    name2gender = model if name2gender is None else name2gender
    encoder = tokenizer if encoder is None else encoder
    names = names_dict if names is None else names

    name = remove_accents(name)

    if name in names:
        return names[name]

    tokenized_name = encoder.texts_to_sequences([name])

    pred = name2gender.predict(tokenized_name, verbose=0)
    gender = np.argmax(pred, axis=1)[0]

    return 'M' if gender == 1 else 'F'


def load_resources():
    model_path = MODEL_DIR / "PT-Name2Gender.keras"
    encoder_path = MODEL_DIR / "Name-Encoder.json"
    names_path = DATA_DIR / "names.json"

    name2gender = keras.models.load_model(model_path)
    encoder = keras.preprocessing.text.tokenizer_from_json(open(encoder_path).read())
    names_dict = json.load(open(names_path))

    return name2gender, encoder, names_dict


if __name__ == "__main__":
    # "Devandra" is not in the dataset and should be classified as F
    names = ["João", "Maria", "Pedro", "Devandra"]

    for name in names:
        gender = name_to_gender_pipeline(name)
        print(name, gender)
    