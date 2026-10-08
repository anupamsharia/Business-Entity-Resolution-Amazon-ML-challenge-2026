import pandas as pd
import pickle
import re
from collections import defaultdict

DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"

MAX_TOKEN_FREQUENCY = 1000
N = 10000


def get_tokens(text):
    if pd.isna(text):
        return set()

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)

    return {
        token
        for token in text.split()
        if len(token) >= 4
    }


def parse_matches(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


print("Loading indexes...")

with open(f"{OUTPUT_DIR}/s2_name_index.pkl", "rb") as f:
    s2_name_index = pickle.load(f)

with open(f"{OUTPUT_DIR}/s3_name_index.pkl", "rb") as f:
    s3_name_index = pickle.load(f)

with open(f"{OUTPUT_DIR}/s2_address_index.pkl", "rb") as f:
    s2_address_index = pickle.load(f)

with open(f"{OUTPUT_DIR}/s3_address_index.pkl", "rb") as f:
    s3_address_index = pickle.load(f)


print("Building rare name token index...")

token_counts = defaultdict(int)

for name in s2_name_index:
    for token in get_tokens(name):
        token_counts[token] += 1

for name in s3_name_index:
    for token in get_tokens(name):
        token_counts[token] += 1


token_index = defaultdict(list)

for name, entity_ids in s2_name_index.items():
    tokens = get_tokens(name)

    for token in tokens:
        if token_counts[token] <= MAX_TOKEN_FREQUENCY:
            token_index[token].extend(entity_ids)

for name, entity_ids in s3_name_index.items():
    tokens = get_tokens(name)

    for token in tokens:
        if token_counts[token] <= MAX_TOKEN_FREQUENCY:
            token_index[token].extend(entity_ids)


print("Loading data...")

s1 = pd.read_csv(
    f"{DATA_DIR}/train_source1.tsv",
    sep="\t",
    nrows=N
)

gt = pd.read_csv(
    f"{DATA_DIR}/train_ground_truth.tsv",
    sep="\t",
    nrows=N
)

print("Testing combined blocking...")

total_true = 0
captured_true = 0

s1_with_candidate = 0
candidate_pairs = 0

for i in range(N):

    row = s1.iloc[i]

    name = str(row["business_name"]).lower()
    name = re.sub(r"[^\w\s]", " ", name)
    name = " ".join(name.split())

    address = row["business_address"]

    if pd.isna(address):
        address_norm = ""
    else:
        address_norm = str(address).lower()
        address_norm = re.sub(r"[^\w\s]", " ", address_norm)
        address_norm = " ".join(address_norm.split())

    candidates = set()

    # Exact name blocking
    if name in s2_name_index:
        candidates.update(s2_name_index[name])

    if name in s3_name_index:
        candidates.update(s3_name_index[name])

    # Exact address blocking
    if address_norm:
        if address_norm in s2_address_index:
            candidates.update(s2_address_index[address_norm])

        if address_norm in s3_address_index:
            candidates.update(s3_address_index[address_norm])

    # Rare name token blocking
    for token in get_tokens(row["business_name"]):
        if token in token_index:
            candidates.update(token_index[token])

    candidate_pairs += len(candidates)

    if candidates:
        s1_with_candidate += 1

    true_matches = parse_matches(gt.iloc[i]["matched_entity_ids"])

    total_true += len(true_matches)
    captured_true += len(true_matches.intersection(candidates))


pair_recall = (
    captured_true / total_true
    if total_true
    else 0
)

print()
print("===== COMBINED BLOCKING RECALL =====")
print(f"S1 records tested: {N}")
print(f"Total true matches: {total_true}")
print(f"Captured true matches: {captured_true}")
print(f"Pair-level recall: {pair_recall:.4%}")
print(f"S1-level candidate coverage: {s1_with_candidate / N:.4%}")
print(f"Candidate pairs: {candidate_pairs}")
print(f"Average candidates per S1: {candidate_pairs / N:.2f}")
