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


with open(
    f"{OUTPUT_DIR}/rare_address_token_index.pkl",
    "rb"
) as f:
    rare_index = pickle.load(f)


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

    candidates = set()

    for token in tokens:

        for entity_id in rare_index.get(token, []):

            candidates.add(entity_id)

    candidate_count = len(candidates)

    if candidate_count:
        s1_with_candidates += 1

    candidate_pairs += candidate_count

    max_candidates = max(
        max_candidates,
        candidate_count
    )


print("\n===== RARE ADDRESS BASELINE =====")
print("S1 records tested:", len(s1))
print("S1 with candidates:", s1_with_candidates)
print("Candidate pairs:", candidate_pairs)
print("Average candidates per S1:", round(
    candidate_pairs / len(s1),
    2
))
print("Maximum candidates:", max_candidates)
