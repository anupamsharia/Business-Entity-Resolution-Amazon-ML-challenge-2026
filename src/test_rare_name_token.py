import pandas as pd
import pickle
import re
from collections import Counter

DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"

MAX_TOKEN_FREQUENCY = 1000


def get_tokens(text):
    if not text:
        return set()

    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    return {
        token
        for token in text.split()
        if len(token) >= 4
    }


print("Loading name indexes...")

with open(
    f"{OUTPUT_DIR}/s2_name_index.pkl",
    "rb"
) as f:
    s2_name_index = pickle.load(f)

with open(
    f"{OUTPUT_DIR}/s3_name_index.pkl",
    "rb"
) as f:
    s3_name_index = pickle.load(f)


print("Counting name-token frequencies...")

token_counts = Counter()

for name in s2_name_index:
    token_counts.update(get_tokens(name))

for name in s3_name_index:
    token_counts.update(get_tokens(name))


print("Unique name tokens:", len(token_counts))


test_tokens = [
    "orelee",
    "barbershop",
    "services",
    "retail",
    "consulting"
]

print("\n===== TOKEN FREQUENCIES =====")

for token in test_tokens:
    print(
        f"{token}: {token_counts.get(token, 0):,}"
    )


s1 = pd.read_csv(
    f"{DATA_DIR}/train_source1.tsv",
    sep="\t",
    nrows=10000,
    usecols=[
        "entity_id",
        "business_name"
    ]
)


candidate_pairs = 0
s1_with_candidates = 0
max_candidates = 0


for row in s1.itertuples(index=False):

    tokens = get_tokens(row.business_name)

    rare_tokens = {
        token
        for token in tokens
        if token_counts.get(token, 0)
        <= MAX_TOKEN_FREQUENCY
    }

    candidates = set()

    for token in rare_tokens:

        for name, entity_ids in s2_name_index.items():

            if token in get_tokens(name):
                candidates.update(entity_ids)

        for name, entity_ids in s3_name_index.items():

            if token in get_tokens(name):
                candidates.update(entity_ids)

    candidate_count = len(candidates)

    if candidate_count:
        s1_with_candidates += 1

    candidate_pairs += candidate_count

    max_candidates = max(
        max_candidates,
        candidate_count
    )


print("\n===== RARE NAME TOKEN BLOCKING =====")
print("S1 records tested:", len(s1))
print("S1 with candidates:", s1_with_candidates)
print(
    "S1 without candidates:",
    len(s1) - s1_with_candidates
)
print("Candidate pairs:", candidate_pairs)
print(
    "Average candidates per S1:",
    round(candidate_pairs / len(s1), 2)
)
print(
    "Maximum candidates:",
    max_candidates
)

