# Submission Format (quick reference)

Submit a single `predictions.csv`, zipped, to the matching CodaBench task.

## Task A — Binary
Header: `id,label`
Labels: Hate | Non-Hate  (1 | 0 also accepted)

## Task B — Fine-Grained
Header: `id,label`
Labels: Gender | Political | Religion | Geo-political | Violence | Others
(GEN | POL | REL | GEO | VIO | OTH also accepted)

Rules: one row per id from the input file; UTF-8; keep ids unchanged; zip only predictions.csv.
