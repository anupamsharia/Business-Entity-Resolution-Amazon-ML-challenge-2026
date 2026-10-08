import pandas as pd
import pickle
import re
from collections import Counter, defaultdict

DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"

MAX_FREQUENCY = 5000


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


def count_tokens():

    counts = Counter()

    for source_file in [
        "train_source2.tsv",
        "train_source3.tsv"
    ]:

        print(f"Counting tokens in {source_file}...")

        path = f"{DATA_DIR}/{source_file}"

        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=["business_address"],
            chunksize=100000
        ):

            for address in chunk["business_address"]:
                counts.update(get_tokens(address))

            del chunk

    return counts


def build_index(token_counts):

    index = defaultdict(list)

    for source_file in [
        "train_source2.tsv",
        "train_source3.tsv"
    ]:

        print(f"Building index from {source_file}...")

        path = f"{DATA_DIR}/{source_file}"

        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=["entity_id", "business_address"],
            chunksize=100000
        ):

            for entity_id, address in zip(
                chunk["entity_id"],
                chunk["business_address"]
            ):

                tokens = get_tokens(address)

                for token in tokens:

                    if token_counts[token] <= MAX_FREQUENCY:
                        index[token].append(entity_id)

            del chunk

    return dict(index)


print("Counting address tokens...")

token_counts = count_tokens()

print("\nBuilding rare-token index...")

rare_index = build_index(token_counts)

output_path = f"{OUTPUT_DIR}/rare_address_token_index.pkl"

with open(output_path, "wb") as file:
    pickle.dump(
        rare_index,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("\n===== RESULT =====")
print("Rare tokens:", len(rare_index))

print(
    "Indexed entity references:",
    sum(len(values) for values in rare_index.values())
)

print("Output:", output_path)
