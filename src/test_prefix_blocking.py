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


def get_prefix(name, length=5):
    normalized = normalize_name(name)

    # Remove spaces for compact prefix matching
    compact = normalized.replace(" ", "")

    return compact[:length]


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


s1 = pd.read_csv(
    f"{DATA_DIR}/train_source1.tsv",
    sep="\t",
    nrows=10000,
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ]
)


print("Testing prefix blocking...")

# Build prefix maps from existing normalized-name indexes
s2_prefix_index = {}
s3_prefix_index = {}

for name, entity_ids in s2_name_index.items():

    prefix = name.replace(" ", "")[:5]

    if prefix:
        s2_prefix_index.setdefault(prefix, []).extend(
            entity_ids
        )


for name, entity_ids in s3_name_index.items():

    prefix = name.replace(" ", "")[:5]

    if prefix:
        s3_prefix_index.setdefault(prefix, []).extend(
            entity_ids
        )


candidate_pairs = 0
s1_with_candidates = 0
max_candidates = 0


for row in s1.itertuples(index=False):

    prefix = get_prefix(row.business_name)

    candidates = set()

    candidates.update(
        s2_prefix_index.get(prefix, [])
    )

    candidates.update(
        s3_prefix_index.get(prefix, [])
    )

    candidate_count = len(candidates)

    if candidate_count:
        s1_with_candidates += 1

    candidate_pairs += candidate_count

    max_candidates = max(
        max_candidates,
        candidate_count
    )


print("\n===== PREFIX BLOCKING =====")
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
