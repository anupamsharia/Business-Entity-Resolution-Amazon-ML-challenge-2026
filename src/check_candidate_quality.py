import pandas as pd

DATA_DIR = "/home/ec2-user/entity-resolution/data"
OUTPUT_DIR = "/home/ec2-user/entity-resolution/output"

GT_PATH = f"{DATA_DIR}/train_ground_truth.tsv"
CANDIDATE_PATH = f"{OUTPUT_DIR}/candidate_pairs_sample.tsv"

print("Loading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    nrows=10000
)

print("Loading candidates...")

candidates = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t"
)

gt_dict = {}

for row in gt.itertuples(index=False):

    value = row.matched_entity_ids

    if pd.isna(value) or str(value).strip() == "":
        gt_dict[row.source1_entity_id] = set()
    else:
        gt_dict[row.source1_entity_id] = {
            entity_id.strip()
            for entity_id in str(value).split(",")
            if entity_id.strip()
        }

candidate_pairs = 0
true_candidate_pairs = 0

s1_with_candidate = set()
s1_with_true_candidate = set()

for row in candidates.itertuples(index=False):

    s1_id = row.source1_entity_id
    candidate_id = row.candidate_entity_id

    s1_with_candidate.add(s1_id)

    candidate_pairs += 1

    true_matches = gt_dict.get(s1_id, set())

    if candidate_id in true_matches:
        true_candidate_pairs += 1
        s1_with_true_candidate.add(s1_id)

print("\n===== CORRECTED CANDIDATE QUALITY =====")
print("Candidate pairs:", candidate_pairs)
print("True candidate pairs:", true_candidate_pairs)

if candidate_pairs:
    precision = true_candidate_pairs / candidate_pairs
    print(f"Candidate precision: {precision:.4%}")

print("S1 with at least one candidate:", len(s1_with_candidate))
print(
    "S1 with at least one true candidate:",
    len(s1_with_true_candidate)
)

print(
    "S1 candidate recall:",
    f"{len(s1_with_true_candidate) / len(gt):.4%}"
)
