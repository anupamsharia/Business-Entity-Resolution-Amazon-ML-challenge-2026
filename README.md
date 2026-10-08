# Business Entity Resolution --- Theory to Implementation

**Challenge:** Amazon ML Challenge 2026\
**Stack:** Python · Pandas · RapidFuzz · AWS EC2\
**Evaluation:** Macro F0.5\
**Last verified official score in the project history:** `0.455032`

> This guide explains the problem first, visualizes each concept, and
> then connects the theory to the implementation. The diagrams use
> Mermaid, which GitHub can render in Markdown.

------------------------------------------------------------------------

## 1. Understand the problem

### The real-world problem

The same business can appear in different datasets with different
spellings, word order, abbreviations, addresses, or languages.

**Source 1**

  ------------------------------------------------------------------------
  entity_id         business_name     business_address   country
  ----------------- ----------------- ------------------ -----------------
  S1-001            Physical Therapy  4303 Elkins        US
                    Associates Group  Avenue, Nashville, 
                                      TN                 

  ------------------------------------------------------------------------

**Source 2**

  ------------------------------------------------------------------------
  entity_id         business_name     business_address   country
  ----------------- ----------------- ------------------ -----------------
  S2-101            Group Physical    004303 Elkins Ave, US
                    Therapy           Nashville, TN      
                    Asseatiaes                           

  ------------------------------------------------------------------------

**Source 3**

  ------------------------------------------------------------------------
  entity_id         business_name     business_address   country
  ----------------- ----------------- ------------------ -----------------
  S3-201            Physical Therapy  4303 Elkins Ave,   US
                                      Nashville,         
                                      Tennessee          

  ------------------------------------------------------------------------

These records may refer to the same business even though the text is not
identical.

### What the system must do

For every Source 1 entity, find all valid matching records in Source 2
and Source 3. A source record may have **zero, one, or multiple
matches**.

``` mermaid
flowchart TD
    A["Source 1 business"] --> B["Find possible records in Source 2"]
    A --> C["Find possible records in Source 3"]
    B --> D["Evaluate candidate records"]
    C --> D
    D --> E{"Valid matches?"}
    E -->|None| F["Return an empty match list"]
    E -->|One| G["Return one entity ID"]
    E -->|Multiple| H["Return all matching entity IDs"]
```

**Theory:** This is an entity-resolution / record-linkage problem, not
simply a duplicate-row deletion task. The goal is to connect records
that describe the same real-world entity.

**Implementation inputs:** - `train_source1.tsv` - `train_source2.tsv` -
`train_source3.tsv` - `train_ground_truth.tsv`

Test inputs use the corresponding `test_*.tsv` files.

The source files contain `entity_id`, `business_name`,
`business_address`, and `country`. The ground truth maps
`source1_entity_id` to `matched_entity_ids`.

------------------------------------------------------------------------

## 2. Understand the scale

The training data contains approximately:

  Dataset                      Approximate rows
  -------------------------- ------------------
  Source 1                          2.2 million
  Source 2                          5.0 million
  Source 3                          5.3 million
  **Total source records**     **12.5 million**

### Why not compare every record with every other record?

A brute-force approach compares each Source 1 record against every
Source 2 and Source 3 record.

``` mermaid
flowchart LR
    A["One S1 record"] --> B["Compare with millions of S2 records"]
    A --> C["Compare with millions of S3 records"]
    B --> D["Huge number of comparisons"]
    C --> D
```

Using the approximate sizes, the cross-source comparison count is:

\[
2.2`\text{M}`{=tex}`\times`{=tex}(5.0`\text{M}`{=tex}+5.3`\text{M}`{=tex})
`\approx 22.66`{=tex}`\text{ trillion comparisons}`{=tex} \]

This is a **potential brute-force search space**, not the number of
comparisons actually executed.

**Theory:** Avoid unnecessary comparisons before using an expensive
similarity function.

**Implementation principle:** Use blocking to produce a much smaller
candidate set, then score only those candidates.

------------------------------------------------------------------------

## 3. Step 1 --- Inspect and validate the data

Before writing the matching logic, inspect: - row counts and column
names; - missing business names and addresses; - country values; -
duplicate or highly frequent names; - the number of true matches per
Source 1 entity.

### Why this matters

If addresses are frequently missing, a matching formula must not require
an address for every record. If entities can have multiple true matches,
the solution must not keep only the top-scoring candidate.

``` mermaid
flowchart TD
    A["Read TSV files"] --> B["Check schema and row counts"]
    B --> C["Check missing values"]
    C --> D["Inspect ground-truth match counts"]
    D --> E["Choose preprocessing and matching rules"]
```

**Implementation example:**

``` python
import pandas as pd

source1 = pd.read_csv(
    "data/train_source1.tsv",
    sep="\t",
    dtype=str
)

print(source1.shape)
print(source1.columns.tolist())
print(source1[["business_name", "business_address", "country"]].isna().sum())
```

`sep="\t"` is important because these datasets are tab-separated files.

------------------------------------------------------------------------

## 4. Step 2 --- Normalize text

### Problem

These strings differ in capitalization and punctuation:

-   `ABC Pvt. Ltd.`
-   `abc pvt ltd`

Exact string comparison treats them as different strings.

### Theory

Normalization standardizes easy-to-fix differences before matching. It
does **not** solve every spelling mistake, word-order change, or
language difference.

``` mermaid
flowchart LR
    A["ABC Pvt. Ltd."] --> B["Lowercase"]
    B --> C["Replace punctuation"]
    C --> D["Collapse extra spaces"]
    D --> E["abc pvt ltd"]
```

**Implementation:**

``` python
import re
import pandas as pd

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())
```

Use the same normalization consistently when building indexes and
looking up candidates.

**Important limitation:** Lowercasing and punctuation removal do not
make transliterated names in different scripts identical. Those cases
need additional evidence, such as address similarity or language-aware
matching.

------------------------------------------------------------------------

## 5. Step 3 --- Blocking

### What is blocking?

Blocking is a way to select plausible candidates before calculating
detailed similarity scores.

Without blocking:

``` text
S1 record → millions of S2/S3 records
```

With blocking:

``` text
S1 record → a smaller candidate set → detailed scoring
```

``` mermaid
flowchart TD
    A["All Source 2 and Source 3 records"] --> B["Build lookup indexes"]
    C["Source 1 entity"] --> D["Generate blocking keys"]
    B --> E["Retrieve records sharing keys"]
    D --> E
    E --> F["Candidate pairs"]
    F --> G["Fuzzy matching"]
```

### Blocking method A --- Exact normalized name

Example key:

`physical therapy associates group`

Records with the same normalized name can be retrieved quickly using a
dictionary index.

**Strength:** Fast and precise for identical names.\
**Weakness:** Misses spelling errors, reordered names, partial names,
and translated names.

### Blocking method B --- Rare address tokens

Example address:

`4303 Elkins Avenue, Nashville, Tennessee`

Possible tokens include `4303`, `elkins`, `avenue`, and `nashville`.

A rare token can identify a useful candidate group. Very common tokens
such as `street` or `hospital` may generate too many candidates.

``` mermaid
flowchart LR
    A["Business address"] --> B["Normalize and tokenize"]
    B --> C["Count token frequency"]
    C --> D["Keep selected rare tokens"]
    D --> E["Look up records sharing a token"]
```

**Strength:** Can recover records whose names differ but addresses
overlap.\
**Weakness:** Missing or heavily altered addresses can cause missed
matches; common tokens can create too many candidates.

### Blocking method C --- Name signatures

A signature uses selected characters from a normalized name, for example
the first and last few characters.

**Strength:** Can retrieve some variations quickly.\
**Weakness:** Similar signatures can produce very large candidate sets,
and typos near the signature characters can still cause misses.

### Combining blockers

Candidate sets can be combined with a union:

\[
C(s)=C\_{`\text{name}`{=tex}}(s)`\cup `{=tex}C\_{`\text{address}`{=tex}}(s)
\]

The union improves the chance of retrieving true matches, but may
increase the number of candidates.

``` mermaid
flowchart TD
    A["Source 1 record"] --> B["Exact-name blocker"]
    A --> C["Address-token blocker"]
    A --> D["Name-signature blocker"]
    B --> E["Union and deduplicate candidate IDs"]
    C --> E
    D --> E
    E --> F["Candidate set"]
```

### How to evaluate a blocker

Track both: - **Pair recall:** what fraction of known true pairs were
retrieved? - **Candidate volume:** how many candidate pairs were
generated?

A blocker with high recall but an enormous candidate set may be too
expensive. A very small candidate set that misses true matches is also
unsuitable.

In one 10,000-Source-1 training sample experiment, exact normalized name
plus rare address-token blocking retrieved **102 of 195 known true pairs
(52.31% pair recall)** and generated about **1.25 million candidates**.
This is a sample experiment, not a final test-set metric.

------------------------------------------------------------------------

## 6. Step 4 --- Generate candidate pairs

A candidate pair links one Source 1 entity to a possible Source 2 or
Source 3 entity.

``` text
source1_entity_id    candidate_entity_id    block_type
S1-001               S2-101                 exact_name
S1-001               S3-201                 address_token
```

``` mermaid
flowchart LR
    A["Source 1"] --> C["Blocking indexes"]
    B["Source 2 + Source 3"] --> C
    C --> D["Candidate pairs"]
    D --> E["Deduplicate candidate IDs"]
    E --> F["Candidate-pairs TSV"]
```

**Theory:** Candidate generation is a recall-sensitive stage. If the
correct record is never generated as a candidate, the scoring model
cannot recover it.

**Implementation notes:** - Store the Source 1 ID, candidate ID, source
and/or blocking reason. - Avoid duplicate candidate IDs for the same
Source 1 entity. - Keep final predictions restricted to IDs allowed by
the challenge output rules. - Measure candidate recall against ground
truth during training.

The project uses `output/candidate_pairs.tsv` for candidate pairs.

------------------------------------------------------------------------

## 7. Step 5 --- Fuzzy matching

Once candidates are generated, compare their names and addresses more
carefully.

### Why fuzzy matching?

Exact comparison cannot handle minor spelling changes.

``` mermaid
flowchart TD
    A["Source 1 name"] --> C["String similarity"]
    B["Candidate name"] --> C
    C --> D["Name similarity score"]
    E["Source 1 address"] --> F["Address similarity"]
    G["Candidate address"] --> F
    F --> H["Address similarity score"]
```

The project uses RapidFuzz, including `fuzz.WRatio`, to measure string
similarity.

**Implementation example:**

``` python
from rapidfuzz import fuzz

name_score = fuzz.WRatio(
    normalize_text(name1),
    normalize_text(name2)
)

address_score = fuzz.WRatio(
    normalize_text(address1),
    normalize_text(address2)
)
```

Scores generally range from 0 to 100; higher scores indicate more
similar strings.

**Caution:** A high similarity score is evidence, not proof, that two
records represent the same business. Common names and duplicate-looking
records can be misleading.

------------------------------------------------------------------------

## 8. Step 6 --- Combine matching signals

A name alone may be insufficient. The project explored combining: -
business-name similarity; - address similarity; - address-token
overlap; - country agreement.

### Conceptual scoring model

\[ S = w_nN+w_aA+w_tT+w_cC \]

Where: - (N) = name similarity; - (A) = address similarity; - (T) =
token overlap; - (C) = country agreement; - (w_n,w_a,w_t,w_c) = weights
chosen and validated on training data.

``` mermaid
flowchart TD
    A["Name similarity"] --> E["Combine signals"]
    B["Address similarity"] --> E
    C["Token overlap"] --> E
    D["Country agreement"] --> E
    E --> F["Combined score"]
    F --> G["Decision rule"]
```

When an address is missing, do not automatically treat it as a real
zero-similarity address. A missing-data rule should be tested
separately.

**Implementation principle:** The project scoring script combines
similarity signals and uses a precision-oriented threshold. The exact
formula and threshold should be kept consistent between validation and
final inference.

------------------------------------------------------------------------

## 9. Step 7 --- Select matches; do not blindly choose top-1

A Source 1 entity can have multiple valid matches.

``` mermaid
flowchart TD
    A["Candidate scores for one S1"] --> B["Apply validated decision rule"]
    B --> C{"How many candidates qualify?"}
    C -->|0| D["Empty match list"]
    C -->|1| E["Return one ID"]
    C -->|2 or more| F["Return all qualifying IDs"]
```

**Why not always choose the highest score?**

The highest-scoring candidate is not necessarily the only correct match.
The ground truth can contain multiple valid IDs. A top-1-only rule can
therefore lose recall.

**Why use a threshold?**

A threshold helps control false positives. But increasing the threshold
can also reject real matches. Thresholds must be validated against the
challenge metric rather than selected arbitrarily.

------------------------------------------------------------------------

## 10. Step 8 --- Evaluate correctly

The challenge uses **macro F0.5**, which places more emphasis on
precision than recall.

-   **Precision:** Of the predicted matches, how many are correct?
-   **Recall:** Of the true matches, how many were found?
-   **F0.5:** A combined metric that weights precision more heavily than
    recall.

``` mermaid
flowchart TD
    A["Predicted matches"] --> B["Compare with ground truth"]
    C["Ground-truth matches"] --> B
    B --> D["Precision"]
    B --> E["Recall"]
    D --> F["Macro F0.5"]
    E --> F
```

For the standard F-beta definition:

\[ F\_`\beta`{=tex}=(1+`\beta`{=tex}\^2) `\frac{PR}{\beta^2P+R}`{=tex}
\]

For F0.5, (`\beta=0.5`{=tex}).

**Practical implication:** Adding many uncertain candidates can hurt
precision. Raising the threshold too far can hurt recall. Both blocking
and matching need validation.

### Current score

The last official evaluator score confirmed in the project history is:

**Macro F0.5 = `0.455032`**

Only replace this with a newer score when it is confirmed by the
official evaluator. Do not label an experimental training result as the
final challenge score.

------------------------------------------------------------------------

## 11. Step 9 --- Large-file and memory management

The source datasets contain millions of rows, while the EC2 environment
has limited memory. Loading all intermediate structures at once can
cause memory pressure.

### Chunk-based processing

``` mermaid
flowchart TD
    A["Large TSV file"] --> B["Read chunk 1"]
    B --> C["Process chunk"]
    C --> D["Write or update result"]
    D --> E{"More chunks?"}
    E -->|Yes| F["Read next chunk"]
    F --> C
    E -->|No| G["Finish"]
```

**Implementation example:**

``` python
for chunk in pd.read_csv(
    "data/train_source2.tsv",
    sep="\t",
    dtype=str,
    chunksize=100000
):
    # Process this chunk before loading the next one.
    print(len(chunk))
```

Dictionary-based indexes can make lookups fast, but they still consume
memory. Build and load only the indexes needed for the current stage,
and measure peak memory and runtime.

------------------------------------------------------------------------

## 12. Step 10 --- Create and validate submission files

The challenge expects:

-   `output/matching_results.tsv`
-   `output/candidate_pairs.tsv`

Before submitting, check: 1. Every test Source 1 entity has exactly one
row in `matching_results.tsv`. 2. Empty match lists are represented
according to the required output format. 3. Every predicted ID is
present in the candidate set, if required by the challenge rules. 4. IDs
are correctly formatted and separated. 5. No duplicate predicted IDs
exist for one Source 1 entity. 6. The output headers and delimiters
match the specification.

``` mermaid
flowchart TD
    A["Candidate pairs"] --> B["Score and select matches"]
    B --> C["Write matching_results.tsv"]
    A --> D["Write candidate_pairs.tsv"]
    C --> E["Validate row count and IDs"]
    D --> E
    E --> F["Submit to official evaluator"]
```

------------------------------------------------------------------------

## 13. Repository implementation map

The current project repository documents these main scripts:

  -----------------------------------------------------------------------
  Script                              Purpose
  ----------------------------------- -----------------------------------
  `src/build_name_index.py`           Build the business-name lookup
                                      index

  `src/build_address_index.py`        Build the address lookup index

  `src/run_matching.py`               Generate candidates and output
                                      files

  `src/score_candidates.py`           Score candidate pairs and write
                                      predictions
  -----------------------------------------------------------------------

The expected high-level execution order is:

``` mermaid
flowchart LR
    A["Build name index"] --> C["Run matching"]
    B["Build address index"] --> C
    C --> D["Score candidates"]
    D --> E["matching_results.tsv"]
    C --> F["candidate_pairs.tsv"]
```

Check each script's current arguments and configuration before running
it. File paths and flags may differ between local development and EC2.

------------------------------------------------------------------------

## 14. Limitations and lessons learned

1.  **Blocking controls recall:** True matches excluded during blocking
    cannot be recovered by later scoring.
2.  **Similarity is not identity:** A score of 100 does not always prove
    that two records should be matched.
3.  **Multilingual records are difficult:** Basic normalization does not
    translate or transliterate names.
4.  **Missing addresses need special handling:** Do not interpret
    missing data as strong negative evidence.
5.  **Multiple matches matter:** Do not reduce a multi-match problem to
    top-1 classification.
6.  **Scale affects design:** Candidate count, memory use and runtime
    must be measured alongside matching quality.
7.  **Validation must be honest:** Sample experiments, baseline scores
    and official test scores are different quantities.

------------------------------------------------------------------------

## 15. How to explain the project in an interview

> "I built a business entity-resolution pipeline for matching records
> across three large data sources. The records contained spelling
> errors, abbreviations, word-order differences, missing addresses and
> multilingual names. Comparing every Source 1 record with every Source
> 2 and Source 3 record would require roughly 22.66 trillion potential
> comparisons, so I used blocking and candidate generation to reduce the
> search space. I then applied RapidFuzz similarity measures to names
> and addresses and combined them with token overlap and country
> agreement. I evaluated blocking recall and matching decisions using
> the challenge's macro F0.5 metric. The main challenge was balancing
> recall, precision and computational cost while allowing zero, one or
> multiple matches per entity."

------------------------------------------------------------------------

## 16. Recommended learning order

Study each topic, understand the example, then implement and test it:

1.  **Problem formulation** --- understand source records and ground
    truth.
2.  **Data inspection** --- load TSVs and validate schemas.
3.  **Normalization** --- write and test a reusable normalization
    function.
4.  **Blocking** --- implement exact-name blocking and measure recall.
5.  **Address-token blocking** --- add rare-token lookup and measure
    candidate volume.
6.  **Candidate generation** --- combine and deduplicate candidate IDs.
7.  **Fuzzy matching** --- calculate name and address scores.
8.  **Combined scoring** --- validate weights and missing-data behavior.
9.  **Decision rules** --- evaluate thresholds and multi-match handling.
10. **Scale and submission** --- process chunks, validate output files,
    and submit to the official evaluator.

**Core lesson:** First ensure that good candidates are retrieved; then
rank and select them carefully. A sophisticated scoring model cannot
recover a true match that blocking never generated.
