import pandas as pd
import re
from collections import defaultdict, Counter

DATA_DIR = "/home/ec2-user/entity-resolution/data"

N = 10000
MAX_PAIR_FREQUENCY = 500


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def get_tokens(text):
    return {
        token
        for token in normalize(text).split()
        if len(token) >= 4
    }


def get_pairs(text):
    token_list = sorted(get_tokens(text))

    pairs = set()

    for i in range(len(token_list)):
        for j in range(i + 1, len(token_list)):
            pairs.add(
                (token_list[i], token_list[j])
            )

    return pairs


def parse_gt(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


print("Loading data...")

s1 = pd.read_csv(
    f"{DATA_DIR}/train_source1.tsv",
    sep="\t",
    nrows=N,
    dtype=str
)

s2 = pd.read_csv(
    f"{DATA_DIR}/train_source2.tsv",
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    f"{DATA_DIR}/train_source3.tsv",
    sep="\t",
    dtype=str
)

gt = pd.read_csv(
    f"{DATA_DIR}/train_ground_truth.tsv",
    sep="\t",
    nrows=N,
    dtype=str
)


print("Building address pair frequencies...")

pair_frequency = Counter()

for address in s2["business_address"]:
    for pair in get_pairs(address):
        pair_frequency[pair] += 1

for address in s3["business_address"]:
    for pair in get_pairs(address):
        pair_frequency[pair] += 1


print("Building selective pair index...")

pair_index = defaultdict(set)

for _, row in s2.iterrows():

    for pair in get_pairs(row["business_address"]):

        if pair_frequency[pair] <= MAX_PAIR_FREQUENCY:
            pair_index[pair].add(row["entity_id"])


for _, row in s3.iterrows():

    for pair in get_pairs(row["business_address"]):

        if pair_frequency[pair] <= MAX_PAIR_FREQUENCY:
            pair_index[pair].add(row["entity_id"])


print("Testing...")

total_true = 0
captured_true = 0
candidate_pairs = 0
s1_with_candidates = 0

for i, row in s1.iterrows():

    candidates = set()

    for pair in get_pairs(row["business_address"]):

        if pair in pair_index:
            candidates.update(
                pair_index[pair]
            )

    candidate_pairs += len(candidates)

    if candidates:
        s1_with_candidates += 1

    true_matches = parse_gt(
        gt.iloc[i]["matched_entity_ids"]
    )

    total_true += len(true_matches)

    captured_true += len(
        true_matches.intersection(candidates)
    )


print()
print("===== ADDRESS PAIR BLOCKING =====")
print("S1 records tested:", N)
print("Total true matches:", total_true)
print("Captured true matches:", captured_true)
print(
    "Pair-level recall:",
    f"{captured_true / total_true:.4%}"
)
print(
    "S1-level candidate coverage:",
    f"{s1_with_candidates / N:.4%}"
)
print("Candidate pairs:", candidate_pairs)
print(
    "Average candidates per S1:",
    f"{candidate_pairs / N:.2f}"
)
print("Maximum pair frequency:", MAX_PAIR_FREQUENCY)
