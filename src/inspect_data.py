import pandas as pd

DATA_DIR = "/home/ec2-user/entity-resolution/data"

S1_PATH = f"{DATA_DIR}/train_source1.tsv"
S2_PATH = f"{DATA_DIR}/train_source2.tsv"
S3_PATH = f"{DATA_DIR}/train_source3.tsv"
GT_PATH = f"{DATA_DIR}/train_ground_truth.tsv"


def inspect_file(path, name):
    print(f"\n===== {name} =====")

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=10000
    )

    print("Columns:")
    print(df.columns.tolist())

    print("\nSample rows:")
    print(df.head(3).to_string(index=False))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nRows inspected:", len(df))


print("BUSINESS ENTITY RESOLUTION - DATA INSPECTION")

inspect_file(S1_PATH, "TRAIN SOURCE 1")
inspect_file(S2_PATH, "TRAIN SOURCE 2")
inspect_file(S3_PATH, "TRAIN SOURCE 3")
inspect_file(GT_PATH, "GROUND TRUTH")
