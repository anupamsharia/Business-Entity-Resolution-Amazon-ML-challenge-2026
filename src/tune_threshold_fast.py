import pandas as pd
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


print("Creating fast lookups...")

s1_map = s1.set_index("entity_id").to_dict("index")
s2_map = s2.set_index("entity_id").to_dict("index")
s3_map = s3.set_index("entity_id").to_dict("index")
gt_map = gt.set_index(
    "source1_entity_id"
)["matched_entity_ids"].to_dict()


print("Loading candidates...")

candidates = pd.read_csv(
    f"{OUTPUT_DIR}/candidate_pairs_sample.tsv",
    sep="\t",
    dtype=str
)


true_scores = []
false_scores = []


print("Scoring candidates...")

for row in candidates.itertuples(index=False):

    s1_row = s1_map.get(
        row.source1_entity_id
    )

    if s1_row is None:
        continue

    if row.candidate_entity_id.startswith("S2-"):
        target = s2_map.get(
            row.candidate_entity_id
        )
    else:
        target = s3_map.get(
            row.candidate_entity_id
        )

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

    true_ids = {
        x.strip()
        for x in str(
            gt_map.get(
                row.source1_entity_id,
                ""
            )
        ).split(",")
        if x.strip()
    }

    if row.candidate_entity_id in true_ids:
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
    print(
        "Average:",
        round(
            sum(true_scores) / len(true_scores),
            2
        )
    )
    print("Max:", max(true_scores))

    for threshold in [
        50, 60, 70, 75,
        80, 85, 90, 95
    ]:

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
    print(
        "Average:",
        round(
            sum(false_scores) / len(false_scores),
            2
        )
    )
    print("Max:", max(false_scores))

    for threshold in [
        50, 60, 70, 75,
        80, 85, 90, 95
    ]:

        accepted = sum(
            x >= threshold
            for x in false_scores
        )

        print(
            f"False acceptance at {threshold}:",
            f"{accepted / len(false_scores):.4%}"
        )
