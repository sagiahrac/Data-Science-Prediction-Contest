import pandas as pd


def remove_parentheses(string):
    for left, right in ("()", "[]", "{}"):
        if left in string:
            result = ""
            paren = 0
            for ch in string:
                if ch == left:
                    paren = paren + 1
                elif (ch == right) and paren:
                    paren = paren - 1
                elif not paren:
                    result += ch
            string = result
    return string


def highest_accuracy_category(
    df, frequent_tokens, colname, min_token_frequeny, y=None, verbose=False
):
    if y is not None:
        df = df.copy()
        df["y"] = y
    frequent_tokens = frequent_tokens[frequent_tokens > min_token_frequeny]
    vals = []
    for x in frequent_tokens.index:
        freqs = df[df[colname].apply(lambda feature: x in str(feature))][
            "y"
        ].value_counts(normalize=True)
        try:
            vals += [(x, frequent_tokens.loc[x], freqs.max(), freqs.idxmax())]
        except ValueError:
            if verbose:
                print(x)

    return pd.DataFrame(
        vals, columns=[colname, "count", "rate", "category"]
    ).sort_values(by=["rate", "count"], ascending=False)
