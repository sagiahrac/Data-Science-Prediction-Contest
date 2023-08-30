import os
from time import sleep
import re

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding
from sklearn.metrics import PrecisionRecallDisplay

if "OPENAI_API_KEY" not in os.environ:
    raise Exception("OPENAI_API_KEY environment variable not set.")
openai.api_key = os.environ["OPENAI_API_KEY"]

EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"

labels = [
    "cakes_cupcakes_snack_cakes",
    "candy",
    "chips_pretzels_snacks",
    "chocolate",
    "cookies_biscuits",
    "popcorn_peanuts_seeds_related_snacks",
]

labels = [label.replace('_', ' ') for label in labels]


close_ended_answers = [f"{i+1}. {label}" for i, label in enumerate(labels)]
close_ended_answers = "\n".join(close_ended_answers)


def get_categories_embeddings():
    return {label: get_embedding(label, engine=EMBEDDING_MODEL) for label in labels}


def evaluate_gpt_embeddings(series: pd.Series, save_path: str):
    embeddings = series.apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    embeddings.to_csv(save_path, index=False)
    return embeddings


def load_gpt_embeddings(path: str):
    embed = pd.read_csv(path)
    embed = embed.iloc[:, 0].apply(eval).apply(np.array)
    return embed.to_frame()


def extract_answer_from_gpt_text_output(text):
    for l in labels:
        if l in text:
            return l

    digit = re.sub(r'[^0-9 ]', '', text).strip()[-1]
    
    if digit.isdigit() and int(digit) in range(1, 7):
        return labels[int(digit)-1]
    
    return None


def gpt_zero_shot_classify(snacks, debug_mode=False):
    prompt = (
        "Classify the following snacks as one of the categories bellow:\n"
        "Categories:\n"
        f"{close_ended_answers}\n\n"
        "snacks:\n"
        f"{snacks}"
    )

    completion = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a smart student, answer close ended questions only with a signle digit of the right answer.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-03-02",
            },
            {"role": "user", "content": prompt},
        ],
    )

    if debug_mode:
        print(f"[DEBUG] OUTPUT: {completion.choices[0].message.content}")
    
    gpt_output = completion.choices[0].message.content
    gpt_output = gpt_output.split("\n")
    gpt_output = [extract_answer_from_gpt_text_output(s) for s in gpt_output]
    return gpt_output
