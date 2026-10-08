import re


s1_address = "1795 Westchester Drive, High Point, NC"
s3_address = "Westchester Dr, High Point, North Carolina"

token_frequency = {
    "high": 27023,
    "point": 45771,
    "westchester": 1477,
    "drive": 457748,
    "carolina": 216961,
    "north": 399761
}


def get_tokens(text):
    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    return {
        token
        for token in text.split()
        if len(token) >= 4
    }


tokens_s1 = get_tokens(s1_address)
tokens_s3 = get_tokens(s3_address)

common_tokens = tokens_s1 & tokens_s3

rare_common_tokens = {
    token
    for token in common_tokens
    if token_frequency.get(token, 999999999) <= 5000
}

print("Common tokens:")
print(sorted(common_tokens))

print("\nRare common tokens:")
print(sorted(rare_common_tokens))

if rare_common_tokens:
    print("\nBLOCKING RESULT: MATCH CANDIDATE")
else:
    print("\nBLOCKING RESULT: NOT A CANDIDATE")

