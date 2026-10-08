import re


s1_address = "1795 Westchester Drive, High Point, NC"
s3_address = "Westchester Dr, High Point, North Carolina"


def normalize_tokens(text):
    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Split into tokens
    return set(text.split())


tokens_s1 = normalize_tokens(s1_address)
tokens_s3 = normalize_tokens(s3_address)

common_tokens = tokens_s1 & tokens_s3

print("S1 address tokens:")
print(sorted(tokens_s1))

print("\nS3 address tokens:")
print(sorted(tokens_s3))

print("\nCommon tokens:")
print(sorted(common_tokens))

print("\nNumber of common tokens:", len(common_tokens))
