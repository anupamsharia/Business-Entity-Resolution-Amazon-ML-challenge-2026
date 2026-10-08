import pandas as pd
import re
from collections import defaultdict

DATA_DIR = "/home/ec2-user/entity-resolution/data"

N = 10000


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def shape(text):
    text = normalize(text)

    if not text:
        return ""

    compact = text.replace(" ", "")

    if len(compact) < 3:
        return compact

    return (
        compact[0]
        + compact[-1]
        + str(len(compact) // 5)
    )


def parse_gt(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


print("Loading source data...")

s1 = pd.read_csv(
    f"{DATA_DIR}/train_source1.tsv",
    sep="\t",
    nrows=N
)

s2 = pd.read_csv(
    f"{DATA_DIR}/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    f"{DATA_DIR}/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    f"{DATA_DIR}/train_ground_truth.tsv",
    sep="\t",
    nrows=N
)

print("Building shape index...")

shape_index = defaultdict(set)

for _, row in s2.iterrows():
    key = shape(row["business_name"])

    if key:
        shape_index[key].add(row["entity_id"])

for _, row in s3.iterrows():
    key = shape(row["business_name"])

    if key:
        shape_index[key].add(row["entity_id"])


print("Testing blocking...")

total_true = 0
captured_true = 0
candidate_pairs = 0
s1_with_candidates = 0

for i in range(N):

    row = s1.iloc[i]

    key = shape(row["business_name"])

    candidates = shape_index.get(key, set())

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
print("===== NAME SHAPE BLOCKING =====")
print(f"S1 records tested: {N}")
print(f"Total true matches: {total_true}")
print(f"Captured true matches: {captured_true}")
print(f"Pair-level recall: {captured_true / total_true:.4%}")
print(f"S1-level candidate coverage: {s1_with_candidates / N:.4%}")
print(f"Candidate pairs: {candidate_pairs}")
print(f"Average candidates per S1: {candidate_pairs / N:.2f}")

