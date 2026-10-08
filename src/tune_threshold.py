import pandas as pd
import pickle
import sys

sys.path.insert(
    0,
    "/home/ec2-user/entity-resolution/src"
)

from matching import combined_score


DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"

N = 10000


print("Loading training data...")

s1 = pd.read_csv(
    f"{DATA_DIR}/train_source1.tsv",
    sep="\t",
    nrows=N,
    dtype=str
)

s2 = pd.read_csv(
    f"{DATA_DIR}/train_source2.tsv",
    sep="\t",
    dtype=str
)

s3 = pd.read_csv(
    f"{DATA_DIR}/train_source3.tsv",
    sep="\t",
    dtype=str
)

gt = pd.read_csv(
    f"{DATA_DIR}/train_ground_truth.tsv",
    sep="\t",
    nrows=N,
    dtype=str
)


s2_map = s2.set_index("entity_id").to_dict("index")
s3_map = s3.set_index("entity_id").to_dict("index")


print("Loading candidate pairs...")

candidates = pd.read_csv(
    f"{OUTPUT_DIR}/candidate_pairs_sample.tsv",
    sep="\t",
    dtype=str
)


true_scores = []
false_scores = []


print("Scoring candidates...")

for _, candidate in candidates.iterrows():

    s1_id = candidate["source1_entity_id"]
    candidate_id = candidate["candidate_entity_id"]

    s1_row = s1[
        s1["entity_id"] == s1_id
    ]

    if s1_row.empty:
        continue

    s1_row = s1_row.iloc[0]

    if candidate_id.startswith("S2-"):
        target = s2_map.get(candidate_id)
    else:
        target = s3_map.get(candidate_id)

    if target is None:
        continue

    score = combined_score(
        s1_row["business_name"],
        target["business_name"],
        s1_row["business_address"],
        target["business_address"],
        s1_row["country"],
        target["country"]
    )

    gt_row = gt[
        gt["source1_entity_id"] == s1_id
    ]

    if gt_row.empty:
        continue

    true_ids = {
        x.strip()
        for x in str(
            gt_row.iloc[0]["matched_entity_ids"]
        ).split(",")
        if x.strip()
    }

    if candidate_id in true_ids:
        true_scores.append(score)
    else:
        false_scores.append(score)


print()
print("===== SCORE DISTRIBUTION =====")

print("True match scores:", len(true_scores))
print("Non-match scores:", len(false_scores))

if true_scores:
    print()
    print("TRUE MATCH")
    print("Min:", min(true_scores))
    print("Average:", sum(true_scores) / len(true_scores))
    print("Max:", max(true_scores))

    for threshold in [50, 60, 70, 75, 80, 85, 90, 95]:
        captured = sum(
            x >= threshold
            for x in true_scores
        )

        print(
            f"Recall at {threshold}:",
            f"{captured / len(true_scores):.4%}"
        )

if false_scores:
    print()
    print("NON-MATCH")
    print("Min:", min(false_scores))
    print("Average:", sum(false_scores) / len(false_scores))
    print("Max:", max(false_scores))

    for threshold in [50, 60, 70, 75, 80, 85, 90, 95]:
        accepted = sum(
            x >= threshold
            for x in false_scores
        )

        print(
            f"False acceptance at {threshold}:",
            f"{accepted / len(false_scores):.4%}"
        )
