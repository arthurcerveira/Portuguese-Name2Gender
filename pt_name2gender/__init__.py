from pt_name2gender.config import MODEL_DIR, DATA_DIR
from pt_name2gender.preprocessing import remove_accents

import json
from dataclasses import dataclass

from tensorflow import keras
import numpy as np


@dataclass
class Name2Gender:
    """
    Class to predict the gender of a name using a pre-trained LSTM model

    args:
        model_path: str = Path to the pre-trained Keras model
        encoder_path: str = Path to the tokenizer in JSON format
        names_path: str = Path to the names dataset in JSON format
    """
    model_path: str = MODEL_DIR / "PT-Name2Gender.keras"
    encoder_path: str = MODEL_DIR / "Name-Encoder.json"
    names_path: str = DATA_DIR / "names.json"

    def __post_init__(self):
        self.model, self.encoder, self.names = self._load_resources()

    def pipeline(self, name: str):
        """
        Pipeline to predict the gender of a PT-BR name
        - Preprocess the name string
        - If the name is in the dataset, return the gender
        - If not, predict the gender using the pre-trained model

        args:
            name: str = Name to be predicted
        """
        name = remove_accents(name)

        if name in self.names:
            return self.names[name]

        gender = self._predict(name)

        return 'M' if gender == 1 else 'F'

    def _predict(self, name):
        tokenized_name = self.encoder.texts_to_sequences([name])

        pred = self.model.predict(tokenized_name, verbose=0)
        gender = np.argmax(pred, axis=1)[0]

        return gender

    def _load_resources(self):
        name2gender = keras.models.load_model(self.model_path)
        encoder = keras.preprocessing.text.tokenizer_from_json(open(self.encoder_path).read())
        names_dict = json.load(open(self.names_path))

        return name2gender, encoder, names_dict
