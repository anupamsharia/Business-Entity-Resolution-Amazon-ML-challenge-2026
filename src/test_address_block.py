import re


s1_address = "1795 Westchester Drive, High Point, NC"
s3_address = "Westchester Dr, High Point, North Carolina"


def get_tokens(text):
    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = text.split()

    # Ignore very short tokens
    return {
        token
        for token in tokens
        if len(token) >= 4
    }


tokens_s1 = get_tokens(s1_address)
tokens_s3 = get_tokens(s3_address)

common = tokens_s1 & tokens_s3

print("S1 useful tokens:")
print(sorted(tokens_s1))

print("\nS3 useful tokens:")
print(sorted(tokens_s3))

print("\nCommon useful tokens:")
print(sorted(common))

print("\nCommon token count:", len(common))

if len(common) >= 2:
    print("BLOCKING RESULT: MATCH CANDIDATE")
else:
    print("BLOCKING RESULT: NOT A CANDIDATE")
