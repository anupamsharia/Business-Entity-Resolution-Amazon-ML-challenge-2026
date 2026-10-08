import pandas as pd

DATA_DIR = "/home/ec2-user/entity-resolution/data"


def normalize_name(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = "".join(
        char if char.isalnum() or char.isspace() else " "
        for char in text
    )
    return " ".join(text.split())


def build_name_index(df):
    index = {}

    for row in df.itertuples(index=False):
        name = normalize_name(row.business_name)

        if not name:
            continue

        index.setdefault(name, []).append(row.entity_id)

    return index


def generate_candidates(source1, source2):
    index = build_name_index(source2)

    candidates = []

    for row in source1.itertuples(index=False):
        name = normalize_name(row.business_name)

        matched_ids = index.get(name, [])

        for entity_id in matched_ids:
            candidates.append(
                {
                    "source1_entity_id": row.entity_id,
                    "candidate_entity_id": entity_id,
                }
            )

    return pd.DataFrame(candidates)


if __name__ == "__main__":

    print("Loading sample data...")

    s1 = pd.read_csv(
        f"{DATA_DIR}/train_source1.tsv",
        sep="\t",
        nrows=10000
    )

    s2 = pd.read_csv(
        f"{DATA_DIR}/train_source2.tsv",
        sep="\t",
        nrows=100000
    )

    print("Source 1 rows:", len(s1))
    print("Source 2 rows:", len(s2))

    candidates = generate_candidates(s1, s2)

    print("\nCandidate pairs:", len(candidates))

    print("\nSample candidates:")
    print(candidates.head(20).to_string(index=False))
