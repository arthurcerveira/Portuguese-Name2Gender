from tensorflow import keras
from tensorflow.keras import layers


def create_model(embedding_dim, embeddings_input_lenght, input_shape):
    model = keras.Sequential()

    model.add(layers.Embedding(embeddings_input_lenght, embedding_dim))

    model.add(layers.LSTM(16, activation="tanh",
                        return_sequences=True, dropout=.2))

    model.add(layers.LSTM(16, activation="tanh",
                        return_sequences=False, dropout=.2))

    model.add(layers.Dense(2, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(
        optimizer=optimizer, 
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.build(input_shape=input_shape)

    return model


if __name__ == '__main__':
    model = create_model(
        embedding_dim=8,
        embeddings_input_lenght=27,
        input_shape=(420131, 14) 
    )

    model.summary()
