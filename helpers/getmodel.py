import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline

from helpers.preprocess import (AddImportantTokens, CleanAndListifyIngredients,
                                DropColumns, FillNA, LogTransformation,
                                MergeWithFoodNutrients, NaiveBayesScores)

DATA_PATH = 'data/food_train.csv'

def get_data():
    food_train = pd.read_csv("data/food_train.csv")

    features_df = food_train.drop("category", axis=1)
    labels_df = food_train["category"]

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        features_df, labels_df, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.25, random_state=42
    )

    steps = [
        FillNA(),
        MergeWithFoodNutrients(),
        CleanAndListifyIngredients(),
        AddImportantTokens(min_token_frequeny=100, ntokens=50, colname="ingredients"),
        AddImportantTokens(min_token_frequeny=100, ntokens=50, colname="description"),
        NaiveBayesScores(colname="brand", preprocess_func=lambda x: x.replace(" ", "")),
        NaiveBayesScores(
            colname="description",
            vectorizer_kwgs=dict(
                stop_words="english", ngram_range=(1, 6), strip_accents="unicode"
            ),
        ),
        NaiveBayesScores(
            colname="ingredients",
            vectorizer_kwgs=dict(
                stop_words="english", ngram_range=(1, 6), strip_accents="unicode"
            ),
        ),
        NaiveBayesScores(
            colname="household_serving_fulltext",
            vectorizer_kwgs=dict(stop_words="english", ngram_range=(1, 6)),
        ),
        LogTransformation(columns=["serving_size"]),
        DropColumns(columns=["serving_size_unit"]),
    ]

    pipe = Pipeline([(f"{i}", step) for i, step in enumerate(steps)])

    X_train = pipe.fit_transform(X_train, y_train)
    X_val = pipe.transform(X_val)

    return X_train, X_val, y_train, y_val


def get_train_val_indices():
    food_train = pd.read_csv("data/food_train.csv")

    features_df = food_train.drop("category", axis=1)
    labels_df = food_train["category"]

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        features_df, labels_df, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.25, random_state=42
    )
    
    train_indices = pd.concat([X_train, X_val]).index.tolist()
    test_indices = X_test.index.tolist()
    
    return train_indices, test_indices


def get_data_for_bert():
    food_train = pd.read_csv(DATA_PATH).fillna('na')
    food_train = CleanAndListifyIngredients().fit_transform(food_train)
    food_train['ingredients'] = food_train['ingredients'].apply(lambda x: x.replace(" ", ", ").replace("_", " "))
    

    text = food_train.apply(lambda x: f"brand: {x['brand']};\ndescription: {x['description']};\nserving: {x['household_serving_fulltext']};\ningredients: {x['ingredients']};", axis=1).tolist()
    categories = food_train['category'].tolist()
    
    return text, categories


def get_train_val_indices():
    food_train = pd.read_csv(DATA_PATH)

    features_df = food_train.drop("category", axis=1)
    labels_df = food_train["category"]

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        features_df, labels_df, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.25, random_state=42
    )
    
    train_indices = pd.concat([X_train, X_val]).index.tolist()
    val_indices = X_test.index.tolist()
    
    return train_indices, val_indices