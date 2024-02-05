import json
from tensorflow import keras
import numpy as np

from preprocessing import remove_accents


def name_to_gender_pipeline(name, name2gender, encoder, names):
    name = remove_accents(name)

    if name in names:
        return names[name]

    tokenized_name = encoder.texts_to_sequences([name])

    pred = name2gender.predict(tokenized_name, verbose=0)
    gender = np.argmax(pred, axis=1)[0]

    return 'M' if gender == 1 else 'F'


if __name__ == "__main__":
    name2gender = keras.models.load_model("model/PT-Name2Gender.h5")
    encoder = keras.preprocessing.text.tokenizer_from_json(open("model/Name-Encoder.json").read())
    names_dict = json.load(open("data/names.json"))

    names = ["João", "Maria", "Pedro", "Ana"]

    for name in names:
        gender = name_to_gender_pipeline(name, name2gender, encoder, names_dict)
        print(name, gender)
    