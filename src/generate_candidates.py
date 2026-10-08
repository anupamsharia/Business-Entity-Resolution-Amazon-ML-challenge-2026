import pandas as pd
import pickle
import re
import csv
from collections import defaultdict


DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"

S1_PATH = f"{DATA_DIR}/train_source1.tsv"

S2_NAME_INDEX = f"{OUTPUT_DIR}/s2_name_index.pkl"
S3_NAME_INDEX = f"{OUTPUT_DIR}/s3_name_index.pkl"

S2_ADDRESS_INDEX = f"{OUTPUT_DIR}/s2_address_index.pkl"
S3_ADDRESS_INDEX = f"{OUTPUT_DIR}/s3_address_index.pkl"

OUTPUT_PATH = f"{OUTPUT_DIR}/candidate_pairs_sample.tsv"

MAX_TOKEN_FREQUENCY = 1000


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def get_tokens(text):
    normalized = normalize_text(text)

    return {
        token
        for token in normalized.split()
        if len(token) >= 4
    }


print("Loading indexes...")

with open(S2_NAME_INDEX, "rb") as f:
    s2_name_index = pickle.load(f)

with open(S3_NAME_INDEX, "rb") as f:
    s3_name_index = pickle.load(f)

with open(S2_ADDRESS_INDEX, "rb") as f:
    s2_address_index = pickle.load(f)

with open(S3_ADDRESS_INDEX, "rb") as f:
    s3_address_index = pickle.load(f)

print("All indexes loaded.")


print("Building rare name token index...")

token_frequency = defaultdict(int)

for name in s2_name_index:
    for token in get_tokens(name):
        token_frequency[token] += 1

for name in s3_name_index:
    for token in get_tokens(name):
        token_frequency[token] += 1


rare_token_index = defaultdict(set)

for name, entity_ids in s2_name_index.items():

    for token in get_tokens(name):

        if token_frequency[token] <= MAX_TOKEN_FREQUENCY:
            rare_token_index[token].update(entity_ids)


for name, entity_ids in s3_name_index.items():

    for token in get_tokens(name):

        if token_frequency[token] <= MAX_TOKEN_FREQUENCY:
            rare_token_index[token].update(entity_ids)


print("Rare name tokens:", len(rare_token_index))


s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=10000,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)


candidate_count = 0
s1_with_candidates = 0


with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file, delimiter="\t")

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "block_type"
    ])


    for row in s1.itertuples(index=False):

        name = normalize_text(row.business_name)
        address = normalize_text(row.business_address)

        candidates = set()


        # Exact normalized name blocking

        for entity_id in s2_name_index.get(name, []):
            candidates.add(("S2", entity_id, "exact_name"))

        for entity_id in s3_name_index.get(name, []):
            candidates.add(("S3", entity_id, "exact_name"))


        # Exact normalized address blocking

        if address:

            for entity_id in s2_address_index.get(address, []):
                candidates.add(("S2", entity_id, "exact_address"))

            for entity_id in s3_address_index.get(address, []):
                candidates.add(("S3", entity_id, "exact_address"))


        # Rare name token blocking

        for token in get_tokens(row.business_name):

            if token in rare_token_index:

                for entity_id in rare_token_index[token]:

                    if entity_id.startswith("S2-"):
                        candidates.add(
                            ("S2", entity_id, "rare_name_token")
                        )

                    elif entity_id.startswith("S3-"):
                        candidates.add(
                            ("S3", entity_id, "rare_name_token")
                        )


        for source, entity_id, block_type in candidates:

            writer.writerow([
                row.entity_id,
                entity_id,
                source,
                block_type
            ])

            candidate_count += 1


        if candidates:
            s1_with_candidates += 1


print()
print("===== RESULT =====")
print("S1 records processed:", len(s1))
print("S1 with candidates:", s1_with_candidates)
print("S1 without candidates:", len(s1) - s1_with_candidates)
print("Candidate pairs:", candidate_count)
print("Average candidates per S1:", candidate_count / len(s1))
print("Output:", OUTPUT_PATH)
