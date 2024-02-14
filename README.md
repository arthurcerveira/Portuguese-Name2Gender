# Portuguese Name2Gender

Treinamento e disponibilização de um modelo de rede neural LSTM para identificar o gênero a partir de um nome em português. O modelo foi treinado com um dataset de nomes em português e possui uma acurácia de >91%.

## Pipeline de classificação

O pipeline de classificação é composto por 3 etapas:

1. **Pré-processamento**: O nome é convertido para minúsculas e removidos caracteres especiais.
2. **Verificação de nome comum**: O nome é verificado em um dataset de nomes comuns em português. Caso o nome seja encontrado, o gênero é retornado.
3. **Classificação**: Caso o nome não seja encontrado no dataset de nomes comuns, o modelo de classificação é utilizado para identificar o gênero a partir do nome.

## Instruções

Para utilizar o modelo, basta clonar o repositório e instalar as dependências:

```bash
$ git clone https://github.com/arthurcerveira/Portuguese-Name2Gender.git
$ cd Portuguese-Name2Gender/
$ pip install -r requirements.txt
```

Após instalar as dependências, é possível utilizar o modelo de acordo com o exemplo abaixo (disponível no arquivo `example.py`):

```python
from pipeline import name_to_gender_pipeline

# "Adrevaldo" and "Devandra" are not in the dataset and should be classified as M and F, respectively
names = ["João", "Maria", "Adrevaldo", "Devandra"]

for name in names:
    gender = name_to_gender_pipeline(name)
    print(name, gender)
```

O código acima irá imprimir o seguinte resultado:

```
João M
Maria F
Adrevaldo M
Devandra F
```

## Report de classificação no conjunto de teste

O modelo foi treinado com um dataset de nomes em português e foi avaliado em um conjunto de teste. O relatório de classificação no conjunto de teste é apresentado abaixo:

```
              precision    recall  f1-score   support

           F       0.92      0.93      0.92      5741
           M       0.91      0.90      0.91      4590

    accuracy                           0.92     10331
   macro avg       0.91      0.91      0.91     10331
weighted avg       0.92      0.92      0.92     10331
```