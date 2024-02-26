from pt_name2gender import Name2Gender

# "Adrevaldo" and "Devandra" are not in the dataset and should be classified as M and F, respectively
names = ["João", "Maria", "Adrevaldo", "Devandra"]

name2gender = Name2Gender()

for name in names:
    gender = name2gender.pipeline(name)
    print(name, gender)
