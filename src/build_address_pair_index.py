import pandas as pd
import pickle
import re
from collections import Counter, defaultdict
from itertools import combinations

DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"

MAX_PAIR_FREQUENCY = 100


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


print("Step 1: Counting address token pairs...")

pair_counts = Counter()

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

            for pair in combinations(
                sorted(tokens),
                2
            ):
                pair_counts[pair] += 1

        del chunk


print("Pair counting complete.")

rare_pairs = {
    pair
    for pair, count in pair_counts.items()
    if count <= MAX_PAIR_FREQUENCY
}

print(
    f"Rare pairs (frequency <= {MAX_PAIR_FREQUENCY}):",
    len(rare_pairs)
)


print("\nStep 2: Building pair index...")

pair_index = defaultdict(list)

for source_file in [
    "train_source2.tsv",
    "train_source3.tsv"
]:

    print(f"Indexing {source_file}...")

    path = f"{DATA_DIR}/{source_file}"

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_address"
        ],
        chunksize=100000
    ):

        for entity_id, address in zip(
            chunk["entity_id"],
            chunk["business_address"]
        ):

            tokens = get_tokens(address)

            for pair in combinations(
                sorted(tokens),
                2
            ):

                if pair in rare_pairs:
                    pair_index[pair].append(entity_id)

        del chunk


output_path = (
    f"{OUTPUT_DIR}/rare_address_pair_index.pkl"
)

with open(output_path, "wb") as file:
    pickle.dump(
        dict(pair_index),
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )


print("\n===== RESULT =====")
print("Rare address pairs:", len(pair_index))
print(
    "Entity references:",
    sum(len(values) for values in pair_index.values())
)
print("Output:", output_path)
