import pandas as pd
import pickle
import re
import csv
import gc
from rapidfuzz import fuzz

DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"

TEST_S1 = f"{DATA_DIR}/test_source1.tsv"

S2_NAME_INDEX = f"{OUTPUT_DIR}/s2_name_index.pkl"
S3_NAME_INDEX = f"{OUTPUT_DIR}/s3_name_index.pkl"
S2_ADDRESS_INDEX = f"{OUTPUT_DIR}/s2_address_index.pkl"
S3_ADDRESS_INDEX = f"{OUTPUT_DIR}/s3_address_index.pkl"

MATCH_OUTPUT = f"{OUTPUT_DIR}/matching_results.tsv"
CANDIDATE_OUTPUT = f"{OUTPUT_DIR}/candidate_pairs.tsv"


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def score_name(name1, name2):
    a = normalize_text(name1)
    b = normalize_text(name2)
    if not a or not b:
        return 0.0
    return fuzz.WRatio(a, b)


print("Loading test Source 1...")

s1 = pd.read_csv(
    TEST_S1,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("Test Source 1 rows:", len(s1))


print("Creating output files...")

with open(
    MATCH_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as match_file, open(
    CANDIDATE_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as candidate_file:

    match_writer = csv.writer(
        match_file,
        delimiter="\t"
    )

    candidate_writer = csv.writer(
        candidate_file,
        delimiter="\t"
    )

    match_writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    candidate_writer.writerow([
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "block_type"
    ])


    print("Loading S2 name index...")

    with open(S2_NAME_INDEX, "rb") as f:
        s2_name_index = pickle.load(f)

    print("Processing S2 exact-name candidates...")

    for row in s1.itertuples(index=False):

        name = normalize_text(row.business_name)

        for entity_id in s2_name_index.get(name, []):

            candidate_writer.writerow([
                row.entity_id,
                entity_id,
                "S2",
                "exact_name"
            ])

    del s2_name_index
    gc.collect()


    print("Loading S3 name index...")

    with open(S3_NAME_INDEX, "rb") as f:
        s3_name_index = pickle.load(f)

    print("Processing S3 exact-name candidates...")

    for row in s1.itertuples(index=False):

        name = normalize_text(row.business_name)

        for entity_id in s3_name_index.get(name, []):

            candidate_writer.writerow([
                row.entity_id,
                entity_id,
                "S3",
                "exact_name"
            ])

    del s3_name_index
    gc.collect()


    print("Loading S2 address index...")

    with open(S2_ADDRESS_INDEX, "rb") as f:
        s2_address_index = pickle.load(f)

    print("Processing S2 exact-address candidates...")

    for row in s1.itertuples(index=False):

        address = normalize_text(row.business_address)

        if not address:
            continue

        for entity_id in s2_address_index.get(address, []):

            candidate_writer.writerow([
                row.entity_id,
                entity_id,
                "S2",
                "exact_address"
            ])

    del s2_address_index
    gc.collect()


    print("Loading S3 address index...")

    with open(S3_ADDRESS_INDEX, "rb") as f:
        s3_address_index = pickle.load(f)

    print("Processing S3 exact-address candidates...")

    for row in s1.itertuples(index=False):

        address = normalize_text(row.business_address)

        if not address:
            continue

        for entity_id in s3_address_index.get(address, []):

            candidate_writer.writerow([
                row.entity_id,
                entity_id,
                "S3",
                "exact_address"
            ])

    del s3_address_index
    gc.collect()


print("Candidate generation completed.")
print("Candidate file:", CANDIDATE_OUTPUT)
print("Matching file:", MATCH_OUTPUT)
