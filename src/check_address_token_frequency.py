import pandas as pd
import re
from collections import Counter

DATA_DIR = "/home/ec2-user/entity-resolution/data"

token_counts = Counter()


def get_tokens(text):
    if pd.isna(text):
        return []

    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = text.split()

    # Keep useful tokens
    return [
        token
        for token in tokens
        if len(token) >= 4
    ]


for source_file in [
    "train_source2.tsv",
    "train_source3.tsv"
]:

    print(f"Processing {source_file}...")

    path = f"{DATA_DIR}/{source_file}"

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["business_address"],
        chunksize=100000
    ):

        for address in chunk["business_address"]:
            tokens = get_tokens(address)

            # Count each token once per address
            token_counts.update(set(tokens))

        del chunk


print("\n===== TOKEN FREQUENCY =====")

for token in [
    "high",
    "point",
    "westchester",
    "drive",
    "carolina",
    "north"
]:

    print(
        f"{token}: {token_counts.get(token, 0):,}"
    )
