# Business Entity Resolution Documentation

## 1. Problem Analysis

The task is to match each Source 1 business entity with all corresponding entities in Source 2 and Source 3.

The data contains noisy business names and addresses. A Source 1 entity may have zero, one, or multiple matching records.

## 2. Solution Strategy

The solution uses a two-stage entity-resolution pipeline:

1. Normalize business names and addresses.
2. Generate candidate pairs using exact normalized-name and exact normalized-address blocking.
3. Score candidates using:
   - Fuzzy business-name similarity
   - Fuzzy address similarity
   - Address token similarity
   - Country agreement
4. Keep candidates above the matching threshold.
5. Generate the final matching results.

## 3. Blocking

Candidate generation uses:

- Exact normalized business name
- Exact normalized business address

This reduces the number of comparisons compared with comparing every Source 1 record against every Source 2 and Source 3 record.

## 4. Matching Model

The matching score combines:

- Business name similarity
- Address similarity
- Address token similarity
- Country agreement

For records with addresses:

Score = 0.65 × Name + 0.20 × Address + 0.10 × Address Token + 0.05 × Country

For records without comparable addresses:

Score = 0.80 × Name + 0.20 × Country

A threshold of 85 is used for final matching.

## 5. Results and Error Analysis

The final test dataset contains 1,732,544 Source 1 entities.

The generated candidate file contains 11,576,879 candidate pairs.

The final matching file contains exactly one row for every Source 1 entity.

1,142,436 Source 1 entities received at least one predicted match.

A final integrity check confirmed that all 3,322,234 predicted entity matches are present in the candidate-pair file.

## 6. Conclusion

The pipeline provides a scalable entity-resolution workflow using normalization, blocking, fuzzy similarity, and precision-oriented thresholding.

No external business lookup, geocoding, or external entity-resolution API was used.
