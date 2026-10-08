import pandas as pd
import pickle
import re
from collections import defaultdict, Counter

DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"

MAX_TOKEN_FREQUENCY = 1000


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


print("Building token frequency...")

token_counts = Counter()

for name in s2_name_index:
    token_counts.update(get_tokens(name))

for name in s3_name_index:
    token_counts.update(get_tokens(name))


print("Building inverted index...")

token_index = defaultdict(set)

for name, entity_ids in s2_name_index.items():

    for token in get_tokens(name):

        if token_counts[token] <= MAX_TOKEN_FREQUENCY:

            token_index[token].update(entity_ids)


for name, entity_ids in s3_name_index.items():

    for token in get_tokens(name):

        if token_counts[token] <= MAX_TOKEN_FREQUENCY:

            token_index[token].update(entity_ids)


print("Loading S1 and ground truth...")

s1 = pd.read_csv(
    f"{DATA_DIR}/train_source1.tsv",
    sep="\t",
    nrows=10000,
    usecols=[
        "entity_id",
        "business_name"
    ]
)

gt = pd.read_csv(
    f"{DATA_DIR}/train_ground_truth.tsv",
    sep="\t",
    nrows=10000
)


gt_dict = {}

for row in gt.itertuples(index=False):

    value = row.matched_entity_ids

    if pd.isna(value) or str(value).strip() == "":
        gt_dict[row.source1_entity_id] = set()
    else:
        gt_dict[row.source1_entity_id] = {
            entity_id.strip()
            for entity_id in str(value).split(",")
            if entity_id.strip()
        }


total_true_matches = 0
captured_true_matches = 0

s1_with_true_match = 0
s1_with_captured_match = 0


for row in s1.itertuples(index=False):

    true_matches = gt_dict.get(
        row.entity_id,
        set()
    )

    total_true_matches += len(true_matches)

    candidates = set()

    for token in get_tokens(row.business_name):

        if token_counts.get(token, 0) <= MAX_TOKEN_FREQUENCY:

            candidates.update(
                token_index.get(token, set())
            )

    captured = true_matches & candidates

    captured_true_matches += len(captured)

    if true_matches:
        s1_with_true_match += 1

    if captured:
        s1_with_captured_match += 1


print("\n===== RARE NAME RECALL =====")
print("S1 records tested:", len(s1))
print("Total true matches:", total_true_matches)
print("Captured true matches:", captured_true_matches)

print(
    "Pair-level recall:",
    f"{captured_true_matches / total_true_matches:.4%}"
)

print(
    "S1-level recall:",
    f"{s1_with_captured_match / s1_with_true_match:.4%}"
)

