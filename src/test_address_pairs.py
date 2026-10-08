import re
from itertools import combinations


s1_address = "1795 Westchester Drive, High Point, NC"
s3_address = "Westchester Dr, High Point, North Carolina"


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

common_tokens = sorted(tokens_s1 & tokens_s3)

print("Common address tokens:")
print(common_tokens)

pairs = list(combinations(common_tokens, 2))

print("\nCommon token pairs:")

for pair in pairs:
    print(pair)

print("\nNumber of common token pairs:", len(pairs))

