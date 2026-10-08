import re
import unicodedata


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(text):
    return normalize_text(text)


def normalize_address(text):
    return normalize_text(text)


if __name__ == "__main__":
    examples = [
        "Orelee's Barbershop",
        "B+ Retail Inc",
        "85 Wayne Avenue, Ticonderoga, NY",
    ]

    print("Normalization test:")

    for value in examples:
        print(f"Original:   {value}")
        print(f"Normalized: {normalize_text(value)}")
        print()
