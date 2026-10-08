import pandas as pd
from rapidfuzz import process, fuzz

DATA_DIR = "/home/ec2-user/entity-resolution/data"


def normalize_name(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Keep letters, numbers, and spaces
    text = "".join(
        char if char.isalnum() or char.isspace() else " "
        for char in text
    )

    return " ".join(text.split())


print("Loading sample data...")

s1 = pd.read_csv(
    f"{DATA_DIR}/train_source1.tsv",
    sep="\t",
    nrows=100
)

s2 = pd.read_csv(
    f"{DATA_DIR}/train_source2.tsv",
    sep="\t",
    nrows=10000
)

s1["name_norm"] = s1["business_name"].map(normalize_name)
s2["name_norm"] = s2["business_name"].map(normalize_name)

name_choices = s2["name_norm"].drop_duplicates().tolist()

print("S1 records:", len(s1))
print("S2 unique names:", len(name_choices))

print("\nFuzzy candidate examples:")

for row in s1.head(10).itertuples():

    query = row.name_norm

    if not query:
        continue

    results = process.extract(
        query,
        name_choices,
        scorer=fuzz.WRatio,
        limit=5,
        score_cutoff=70
    )

    print("\nS1:", row.entity_id)
    print("Name:", row.business_name)

    for match_name, score, _ in results:
        print(f"  {score:.2f} -> {match_name}")
