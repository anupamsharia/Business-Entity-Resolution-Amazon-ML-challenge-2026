from rapidfuzz import fuzz


s1_name = "Orelee's Barbershop"
s3_name = "Orelee'S Services"

s1_address = "1795 Westchester Drive, High Point, NC"
s3_address = "Westchester Dr, High Point, North Carolina"


def normalize(text):
    text = str(text).lower()
    text = text.replace(",", " ")
    text = text.replace("'", " ")
    return " ".join(text.split())


n1 = normalize(s1_name)
n2 = normalize(s3_name)

a1 = normalize(s1_address)
a2 = normalize(s3_address)


print("Normalized names:")
print(n1)
print(n2)

print("\nNormalized addresses:")
print(a1)
print(a2)

print("\n===== SIMILARITY =====")

print("Name ratio:", round(fuzz.ratio(n1, n2), 2))
print("Name WRatio:", round(fuzz.WRatio(n1, n2), 2))
print(
    "Name token_set:",
    round(fuzz.token_set_ratio(n1, n2), 2)
)

print("\nAddress ratio:", round(fuzz.ratio(a1, a2), 2))
print(
    "Address WRatio:",
    round(fuzz.WRatio(a1, a2), 2)
)
print(
    "Address token_set:",
    round(fuzz.token_set_ratio(a1, a2), 2)
)
