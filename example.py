from pipeline import name_to_gender_pipeline

# "Adrevaldo" and "Devandra" are not in the dataset and should be classified as M and F, respectively
names = ["João", "Maria", "Adrevaldo", "Devandra"]

for name in names:
    gender = name_to_gender_pipeline(name)
    print(name, gender)
