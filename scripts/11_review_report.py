"""Real-world version of Exercise 5: analyze a CSV file of product reviews.

Reads reviews from a CSV, gets a structured analysis of each one from the
local model, then saves:
  - outputs/review_results.csv   one row per review, with the analysis
  - outputs/review_report.md     sentiment shares and most mentioned features

Run with:
  uv run python scripts/11_review_report.py
  uv run python scripts/11_review_report.py --input my_reviews.csv --limit 5
"""

import argparse
import csv
import time
from pathlib import Path

from promptlab.review_analyzer import analyze_review, report_markdown, review_analysis_chain, summarize

# --- Command-line options ---------------------------------------------------
parser = argparse.ArgumentParser(description="Analyze product reviews with a local LLM.")
parser.add_argument("--input", default="data/sample_reviews.csv", help="CSV file with a 'review' column")
parser.add_argument("--output-dir", default="outputs", help="folder for the results")
parser.add_argument("--limit", type=int, default=None, help="only analyze the first N reviews")
args = parser.parse_args()

# --- Read the reviews --------------------------------------------------------
with open(args.input, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

if not rows or "review" not in rows[0]:
    raise SystemExit(f"{args.input} must have a column named 'review'.")

if args.limit:
    rows = rows[: args.limit]

# --- Analyze each review --------------------------------------------------------
chain = review_analysis_chain()
results = []      # successful analyses, used for the report
output_rows = []  # every review, with its analysis or error, for the CSV
failed = 0

print(f"Analyzing {len(rows)} reviews...\n")

for i, row in enumerate(rows, start=1):
    start = time.perf_counter()
    try:
        analysis = analyze_review(chain, row["review"])
        results.append(analysis)
        output_rows.append({
            **row,
            "sentiment": analysis.sentiment,
            "features": "; ".join(analysis.features),
            "summary": analysis.summary,
            "status": "ok",
        })
        outcome = f"{analysis.sentiment:<8} {', '.join(analysis.features)}"
    except Exception as error:
        failed += 1
        output_rows.append({**row, "sentiment": "", "features": "", "summary": "", "status": f"error: {error}"})
        outcome = f"FAILED ({error})"

    seconds = time.perf_counter() - start
    print(f"[{i}/{len(rows)}] {seconds:4.1f}s  {outcome}")

# --- Save the results -------------------------------------------------------------
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

results_path = output_dir / "review_results.csv"
# utf-8-sig lets Excel on Windows show special characters correctly.
with open(results_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()))
    writer.writeheader()
    writer.writerows(output_rows)

report = report_markdown(summarize(results, failed))
report_path = output_dir / "review_report.md"
report_path.write_text(report, encoding="utf-8")

print("\n" + report)
print(f"Saved: {results_path}")
print(f"Saved: {report_path}")
