import pandas as pd
import re
from collections import Counter
from itertools import combinations

DATA_DIR = "/home/ec2-user/entity-resolution/data"

pair_counts = Counter()


def get_tokens(text):
    if pd.isna(text):
        return set()

    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    return {
        token
        for token in text.split()
        if len(token) >= 4
    }


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

            # Count each pair only once per address
            for pair in combinations(sorted(tokens), 2):
                pair_counts[pair] += 1

        del chunk


print("\n===== ADDRESS PAIR FREQUENCY =====")

test_pairs = [
    ("high", "point"),
    ("high", "westchester"),
    ("point", "westchester")
]

for pair in test_pairs:
    print(
        f"{pair}: {pair_counts.get(pair, 0):,}"
    )
