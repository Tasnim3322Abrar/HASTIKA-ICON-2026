#!/usr/bin/env python3
"""
HASTIKA baseline — writes a majority-class predictions.csv from an input CSV.
Task A -> predicts 'Non-Hate' (majority in training).
Task B -> predicts 'Gender'   (largest category in training).
Replace with your own model; this only guarantees a correctly formatted submission.
"""
import csv, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with an id column and a Comment column")
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()

    # utf-8-sig strips a BOM if present on the id header
    rows = list(csv.DictReader(open(a.input, encoding="utf-8-sig")))
    # find the id column regardless of exact spelling/case
    id_key = None
    if rows:
        for k in rows[0].keys():
            if (k or "").strip().lower() in ("id", "index"):
                id_key = k; break
    if id_key is None:
        raise SystemExit("Could not find an 'id' column in the input file.")

    default = "Non-Hate" if a.task == "a" else "Gender"
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for r in rows:
            w.writerow([r[id_key], default])
    print(f"Wrote {a.out} with {len(rows)} rows (all '{default}').")

if __name__ == "__main__":
    main()
