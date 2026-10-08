import pandas as pd
import pickle
import re

DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"


def normalize_name(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def get_composite_key(name):
    normalized = normalize_name(name)

    tokens = normalized.split()

    if len(tokens) < 2:
        return normalized

    return f"{tokens[0]}|{tokens[-1]}"


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


s2_composite = {}
s3_composite = {}

for name, entity_ids in s2_name_index.items():

    key = get_composite_key(name)

    if key:
        s2_composite.setdefault(key, []).extend(entity_ids)


for name, entity_ids in s3_name_index.items():

    key = get_composite_key(name)

    if key:
        s3_composite.setdefault(key, []).extend(entity_ids)


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

    key = get_composite_key(row.business_name)

    candidates = set()

    candidates.update(
        s2_composite.get(key, [])
    )

    candidates.update(
        s3_composite.get(key, [])
    )

    candidate_count = len(candidates)

    if candidate_count:
        s1_with_candidates += 1

    candidate_pairs += candidate_count

    max_candidates = max(
        max_candidates,
        candidate_count
    )


print("\n===== COMPOSITE NAME BLOCKING =====")
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
