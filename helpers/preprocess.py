import re

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

from helpers.utils import highest_accuracy_category, remove_parentheses


class FillNA(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df.fillna("na", inplace=True)
        return df


class MergeWithFoodNutrients(BaseEstimator, TransformerMixin):
    def __init__(self, nutrient_min_freq=None):
        self.nutrient_min_freq = nutrient_min_freq

    def fit(self, df, y=None):
        self.nutrients = pd.read_csv("data/food_nutrients_merged.csv", index_col=0)
        if self.nutrient_min_freq is not None:
            filtered = (
                ((self.nutrients > 0).sum() > self.nutrient_min_freq)
                .replace(False, None)
                .dropna()
                .index.tolist()
            )
            self.nutrients = self.nutrients[filtered]
        return self

    def transform(self, df, y=None):
        df = df.merge(self.nutrients, left_on="idx", right_index=True)
        return df


class NaiveBayesScores(BaseEstimator, TransformerMixin):
    def __init__(
        self, colname, preprocess_func=None, vectorizer_kwgs={}, mode="scores", use_tfidf=False
    ):
        self.colname = colname
        self.preprocess_func = preprocess_func
        self.vectorizer_kwgs = vectorizer_kwgs
        self.mode = mode
        self.use_tfidf = use_tfidf

    def fit(self, df, y=None):
        sentences = df[self.colname]
        if self.preprocess_func is not None:
            sentences = sentences.apply(self.preprocess_func)

        if not self.use_tfidf:
            vectorizer = CountVectorizer(**self.vectorizer_kwgs)
        else:
            vectorizer = TfidfVectorizer(**self.vectorizer_kwgs)
        X = vectorizer.fit_transform(sentences)

        nb_classifier = MultinomialNB()
        nb_classifier.fit(X, y)

        self.vectorizer = vectorizer
        self.nb_classifier = nb_classifier
        return self

    def transform(self, df, y=None):
        sentences = df[self.colname]
        if self.preprocess_func is not None:
            sentences = sentences.apply(self.preprocess_func)

        X = self.vectorizer.transform(sentences)

        if self.mode == "scores":
            scores = self.nb_classifier.predict_log_proba(X)
            for i, cname in enumerate(self.nb_classifier.classes_):
                df[f"{self.colname}_nb_score_{cname}"] = scores[:, i]
        else:
            feature_names = self.vectorizer.get_feature_names_out()
            new_cols = pd.DataFrame(X.todense(), columns = [f"{self.colname}_count_{feature}" for feature in feature_names], index=df.index)
            df = pd.concat([df, new_cols], axis=1)

        df.drop(columns=self.colname, inplace=True)
        return df


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df.drop(columns=self.columns, inplace=True)
        return df


class LogTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, columns=["serving_size"]):
        self.columns = columns

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df.loc[:, self.columns] = np.log(df[self.columns])
        return df


class CleanAndListifyIngredients(BaseEstimator, TransformerMixin):
    def __init__(self, keep_top_n=None):
        self.keep_top_n = keep_top_n

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        ingredients = df["ingredients"].apply(lambda x: remove_parentheses(x))
        ingredients = (
            ingredients.str.lower()
            .apply(lambda x: re.sub(r"[^a-z ,]", "", x))
            .apply(lambda x: re.sub(r"\s+", " ", x))
        )
        ingredients = (
            ingredients.str.split(",")
            .apply(lambda x: self.strip_ingredients(x))
            .apply(lambda x: self._rm_less(x))
        )

        if self.keep_top_n is not None:
            ingredients = ingredients.apply(lambda x: x[: self.keep_top_n])

        ingredients = ingredients.apply(lambda x: " ".join(x))

        df.loc[:, "ingredients"] = ingredients
        return df

    def _rm_less(
        self, x
    ):  # all ingridients after 'less' are not relevant (less that 2%)
        i = 0
        for idx, ingredient in enumerate(x):
            if "less" in ingredient:
                i = idx
        if i > 0:
            return x[:i]
        return x

    def strip_ingredients(self, x):
        return [ingredient.strip().replace(" ", "_") for ingredient in x]


class AddImportantTokens(BaseEstimator, TransformerMixin):
    def __init__(self, min_token_frequeny=35, ntokens=100, colname="ingredients"):
        self.min_token_frequeny = min_token_frequeny
        self.ntokens = ntokens
        self.colname = colname

    def fit(self, df, y):
        tokens = df[self.colname].str.split(" ").explode().str.strip()
        tokens_frequencies = tokens.value_counts()

        important_tokens = highest_accuracy_category(
            df=df,
            frequent_tokens=tokens_frequencies,
            colname=self.colname,
            min_token_frequeny=self.min_token_frequeny,
            y=y,
        ).head(self.ntokens)

        self.important_tokens = important_tokens[self.colname]
        return self

    def transform(self, df, y=None):
        tokens_list = df[self.colname].str.split(" ")
        new_df = pd.DataFrame(
            {
                f"{self.colname}_{token}": tokens_list.apply(lambda x: x.count(token))
                for token in self.important_tokens
            },
            index=df.index,
        )
        q = df.index
        df = pd.concat([df, new_df], axis=1)
        p = df.index
        if any(q.sort_values() != p.sort_values()):
            raise ValueError("Index mismatch")
        return df


class StandardScale(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, df, y=None):
        idx = df["idx"].to_numpy()
        df.drop(columns=["idx"], inplace=True)
        self.scaler.fit(df)
        df.insert(0, "idx", idx)
        return self

    def transform(self, df, y=None):
        idx = df["idx"].to_numpy()
        df.drop(columns=["idx"], inplace=True)
        df = pd.DataFrame(self.scaler.transform(df), columns=df.columns, index=df.index)
        df.insert(0, "idx", idx)
        return df


class AddCategoryTokensApperance(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.textual_cols = [
            "brand",
            "description",
            "ingredients",
            "household_serving_fulltext",
        ]

    def fit(self, df, y):
        self.categories_tokens = [
            token.rstrip("s")
            for category in y.unique()
            for token in category.split("_")
        ]
        return self

    def transform(self, df, y=None):
        for token in self.categories_tokens:
            for col in self.textual_cols:
                df[f"token_{col}_{token}"] = df[col].str.count(token)
        return df


class StemDescription(BaseEstimator, TransformerMixin):
    def fit(self, df, y):
        return self

    def transform(self, df, y=None):
        stemmer = SnowballStemmer(language="english")
        df["description"] = df["description"].apply(
            lambda x: " ".join([stemmer.stem(word) for word in x.split(" ")])
        )
        return df
