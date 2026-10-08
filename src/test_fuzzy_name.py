import pandas as pd
import re
from rapidfuzz import process, fuzz

DATA_DIR = "/home/ec2-user/entity-resolution/data"

N = 10000
LIMIT = 20
SAMPLE_SOURCE_SIZE = 500000


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


print("Loading data...")

s1 = pd.read_csv(
    f"{DATA_DIR}/train_source1.tsv",
    sep="\t",
    nrows=N
)

s2 = pd.read_csv(
    f"{DATA_DIR}/train_source2.tsv",
    sep="\t",
    nrows=SAMPLE_SOURCE_SIZE
)

s3 = pd.read_csv(
    f"{DATA_DIR}/train_source3.tsv",
    sep="\t",
    nrows=SAMPLE_SOURCE_SIZE
)

print("Normalizing names...")

s2_names = s2["business_name"].fillna("").map(normalize).tolist()
s3_names = s3["business_name"].fillna("").map(normalize).tolist()

print("Testing fuzzy name candidate generation...")

total_candidates = 0
s1_with_candidates = 0

for i in range(N):

    query = normalize(s1.iloc[i]["business_name"])

    results2 = process.extract(
        query,
        s2_names,
        scorer=fuzz.WRatio,
        limit=LIMIT,
        score_cutoff=70
    )

    results3 = process.extract(
        query,
        s3_names,
        scorer=fuzz.WRatio,
        limit=LIMIT,
        score_cutoff=70
    )

    candidates = len(results2) + len(results3)

    total_candidates += candidates

    if candidates:
        s1_with_candidates += 1

print()
print("===== FUZZY NAME CANDIDATE TEST =====")
print(f"S1 records tested: {N}")
print(f"Source 2 names searched: {len(s2_names)}")
print(f"Source 3 names searched: {len(s3_names)}")
print(f"S1 with candidates: {s1_with_candidates}")
print(f"Candidate pairs: {total_candidates}")
print(f"Average candidates per S1: {total_candidates / N:.2f}")
print(f"Top candidates per source: {LIMIT}")
print("Score cutoff: 70")
