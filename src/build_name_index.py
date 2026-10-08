import pandas as pd
import pickle
import re
import sys


def normalize_name(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    return " ".join(text.split())


def build_index(source_path, output_path):

    index = {}

    print(f"Building index from: {source_path}")

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        usecols=["entity_id", "business_name"],
        dtype=str,
        chunksize=100000
    ):

        chunk["name_norm"] = chunk["business_name"].map(
            normalize_name
        )

        for name, entity_id in zip(
            chunk["name_norm"],
            chunk["entity_id"]
        ):

            if not name:
                continue

            if name not in index:
                index[name] = []

            index[name].append(entity_id)

        print(
            f"Processed {len(chunk):,} rows | "
            f"Unique names: {len(index):,}"
        )

        del chunk

    with open(output_path, "wb") as file:
        pickle.dump(
            index,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(f"Index saved to: {output_path}")
    print(f"Unique normalized names: {len(index):,}")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python3 build_name_index.py "
            "<input_tsv> <output_pkl>"
        )
        sys.exit(1)

    build_index(
        sys.argv[1],
        sys.argv[2]
    )
