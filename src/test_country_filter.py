import pandas as pd
import pickle
import re

DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"


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


print("Loading rare address index...")

with open(
    f"{OUTPUT_DIR}/rare_address_token_index.pkl",
    "rb"
) as f:
    rare_index = pickle.load(f)


print("Loading S2/S3 country information...")

country_map = {}

for source_file in [
    "train_source2.tsv",
    "train_source3.tsv"
]:

    path = f"{DATA_DIR}/{source_file}"

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id", "country"],
        chunksize=100000
    ):

        for entity_id, country in zip(
            chunk["entity_id"],
            chunk["country"]
        ):

            country_map[entity_id] = str(country).strip().lower()

        del chunk


print("Country map built.")

s1 = pd.read_csv(
    f"{DATA_DIR}/train_source1.tsv",
    sep="\t",
    nrows=10000,
    usecols=[
        "entity_id",
        "business_address",
        "country"
    ]
)


candidate_pairs = 0
s1_with_candidates = 0
max_candidates = 0

for row in s1.itertuples(index=False):

    tokens = get_tokens(row.business_address)

    source_country = str(row.country).strip().lower()

    candidates = set()

    for token in tokens:

        for entity_id in rare_index.get(token, []):

            if country_map.get(entity_id) == source_country:
                candidates.add(entity_id)

    candidate_count = len(candidates)

    if candidate_count:
        s1_with_candidates += 1

    candidate_pairs += candidate_count

    max_candidates = max(
        max_candidates,
        candidate_count
    )


print("\n===== COUNTRY-FILTERED ADDRESS BLOCKING =====")
print("S1 records tested:", len(s1))
print("S1 with candidates:", s1_with_candidates)
print("S1 without candidates:", len(s1) - s1_with_candidates)
print("Candidate pairs:", candidate_pairs)
print("Average candidates per S1:", round(
    candidate_pairs / len(s1),
    2
))
print("Maximum candidates:", max_candidates)
