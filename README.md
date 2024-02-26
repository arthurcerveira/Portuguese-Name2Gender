# Portuguese Name2Gender

Treinamento e disponibilização de um modelo de rede neural LSTM para identificar o gênero a partir de um nome em português. O modelo foi treinado com um dataset de nomes em português brasileiro e possui uma acurácia de >91%.

## Pipeline de classificação

O pipeline de classificação é composto por 3 etapas:

1. **Pré-processamento**: O nome é convertido para minúsculas e removidos caracteres especiais.
2. **Verificação de nome comum**: O nome é verificado em um dataset de nomes comuns em português. Caso o nome seja encontrado, o gênero é retornado.
3. **Classificação**: Caso o nome não seja encontrado no dataset de nomes comuns, o modelo de classificação é utilizado para identificar o gênero a partir do nome.

## Instruções

Para utilizar o modelo, basta instalar o pacote via pip através da URL do repositório:

```bash
$ pip install git+https://github.com/arthurcerveira/Portuguese-Name2Gender.git
```

Após instalar as dependências, é possível utilizar o modelo de acordo com o exemplo abaixo (disponível no arquivo `example.py`):

```python
from pt_name2gender import Name2Gender

# "Adrevaldo" and "Devandra" are not in the dataset and should be classified as M and F, respectively
names = ["João", "Maria", "Adrevaldo", "Devandra"]

name2gender = Name2Gender()

for name in names:
    gender = name2gender.pipeline(name)
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

## Fonte dos dados

O dataset de classificação de gênero em nomes brasileiros foi obtido através da página [Brasil.IO](https://brasil.io/dataset/genero-nomes/nomes/).
