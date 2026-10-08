import pandas as pd
import csv
import re
import gc
from collections import defaultdict
from rapidfuzz import fuzz

DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"

TEST_S1 = f"{DATA_DIR}/test_source1.tsv"
TEST_S2 = f"{DATA_DIR}/test_source2.tsv"
TEST_S3 = f"{DATA_DIR}/test_source3.tsv"

CANDIDATE_FILE = f"{OUTPUT_DIR}/candidate_pairs.tsv"
MATCH_OUTPUT = f"{OUTPUT_DIR}/matching_results.tsv"

THRESHOLD = 85.0


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def name_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return fuzz.WRatio(a, b)


def address_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    return fuzz.WRatio(a, b)


def token_similarity(a, b):
    a = set(normalize_text(a).split())
    b = set(normalize_text(b).split())

    if not a or not b:
        return 0.0

    return 100.0 * len(a & b) / len(a | b)


def combined_score(s1, s2):
    name_score = name_similarity(
        s1["business_name"],
        s2["business_name"]
    )

    address_score = address_similarity(
        s1["business_address"],
        s2["business_address"]
    )

    token_score = token_similarity(
        s1["business_address"],
        s2["business_address"]
    )

    c1 = normalize_text(s1["country"])
    c2 = normalize_text(s2["country"])

    country_score = 100.0 if c1 and c2 and c1 == c2 else 0.0

    if s1["business_address"] and s2["business_address"]:
        return (
            0.65 * name_score +
            0.20 * address_score +
            0.10 * token_score +
            0.05 * country_score
        )

    return (
        0.80 * name_score +
        0.20 * country_score
    )


print("Loading Source 1...")

s1 = pd.read_csv(
    TEST_S1,
    sep="\t",
    dtype=str
)

s1_map = s1.set_index("entity_id").to_dict("index")

print("Source 1 loaded:", len(s1_map))


print("Loading candidate pairs...")

candidate_map = defaultdict(list)

with open(
    CANDIDATE_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:

        candidate_map[row["candidate_entity_id"]].append(
            (
                row["source1_entity_id"],
                row["candidate_source"]
            )
        )

print("Candidate IDs:", len(candidate_map))


def process_source(path, source_name):

    print("Processing", source_name)

    matched = defaultdict(list)

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=usecols,
        chunksize=100000
    ):

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            if entity_id not in candidate_map:
                continue

            target = {
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country
            }

            for s1_id, candidate_source in candidate_map[entity_id]:

                if candidate_source != source_name:
                    continue

                source1 = s1_map[s1_id]

                score = combined_score(
                    source1,
                    target
                )

                if score >= THRESHOLD:
                    matched[s1_id].append(
                        (
                            score,
                            entity_id
                        )
                    )

        del chunk
        gc.collect()

    return matched


s2_matches = process_source(
    TEST_S2,
    "S2"
)

gc.collect()

s3_matches = process_source(
    TEST_S3,
    "S3"
)

gc.collect()


print("Writing final matching results...")

with open(
    MATCH_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f,
        delimiter="\t"
    )

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    for s1_id in s1["entity_id"]:

        matches = []

        for score, entity_id in s2_matches.get(s1_id, []):
            matches.append((score, entity_id))

        for score, entity_id in s3_matches.get(s1_id, []):
            matches.append((score, entity_id))

        best = {}

        for score, entity_id in matches:

            if (
                entity_id not in best
                or score > best[entity_id]
            ):
                best[entity_id] = score

        selected = [
            entity_id
            for entity_id, score in best.items()
            if score >= THRESHOLD
        ]

        writer.writerow([
            s1_id,
            ",".join(selected)
        ])

print("Matching completed.")
print("Output:", MATCH_OUTPUT)
