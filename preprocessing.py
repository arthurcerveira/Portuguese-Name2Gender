from unicodedata import normalize, category
import pandas as pd


SEED = 1907


def remove_accents(string):
    if string is None:
        return ""
    
    no_accent = ''.join(c for c in normalize('NFD', string) if category(c) != 'Mn')
    lowercase = no_accent.lower()

    return lowercase.strip()


def explode_names(complete_names):
    # "all_names" column is a list of names
    unique_complete_names = (
        complete_names
            .explode('all_names', ignore_index=True)
            .replace('', pd.NA)
            .dropna(subset=['all_names'])[["all_names", "classification"]]
    )

    unique_complete_names["all_names_norm"] = unique_complete_names["all_names"].apply(remove_accents)
    unique_complete_names = unique_complete_names.drop_duplicates(subset=['all_names_norm'])

    return unique_complete_names


def load_name_dataset(path="data/names.csv"):
    complete_names = pd.read_csv(path).sample(frac=1, random_state=SEED)

    # fillna("") fix concatenation issue of returning None
    complete_names['all_names'] = (
        complete_names['first_name'].fillna("") + '|' + 
        complete_names['alternative_names'].fillna("")
    )

    complete_names['all_names'] = complete_names['all_names'].str.split('|')

    return complete_names


if __name__ == "__main__":
    names = load_name_dataset()
    complete_names = explode_names(names)

    json_path = "data/names.json" 
    print(f"Saving names to {json_path}")

    complete_names.set_index('all_names_norm')["classification"].to_json(
        json_path, indent=4
    )
