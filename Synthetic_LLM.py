import re
import json
import time
import pathlib
import pandas as pd
from openai import OpenAI

# =============================================================================
# CONFIG
# =============================================================================

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
MONTHS          = pd.date_range("2022-01", "2022-11", freq="MS", tz="UTC")
N_PER_MONTH     = 2000
AI_CATEGORIES   = {"cs.CV", "cs.LG", "cs.AI", "cs.IR", "cs.CL"}
RANDOM_SEED     = 42
BATCH_JSONL     = "batch_input.jsonl"
BATCH_ID_FILE   = "batch_id.txt"
OUTPUT_CSV      = "synthetic_llm.csv"

client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# STEP 1 — SAMPLE 2,000 PAPERS PER MONTH (Jan–Nov 2022)
# =============================================================================

def load_sample() -> pd.DataFrame:
    print("Scanning arXiv metadata...")
    buckets = {m: [] for m in MONTHS}

    with open("arxiv-metadata-oai-snapshot.json", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line.strip())
            versions = record.get("versions")
            if not versions:
                continue
            date = pd.to_datetime(
                versions[0].get("created", ""),
                format="%a, %d %b %Y %H:%M:%S %Z"
            )
            month_start = date.normalize().replace(day=1)
            if month_start not in buckets:
                continue
            if set(record.get("categories", "").split()) & AI_CATEGORIES:
                continue
            abstract = (record.get("abstract", "") or "").strip()
            if not abstract:
                continue
            buckets[month_start].append(abstract)

            if (i + 1) % 500_000 == 0:
                print(f"  {i+1:,} records scanned")

    frames = []
    rng = pd.core.common  # just use pandas sample with seed below
    for month, abstracts in buckets.items():
        df_m = pd.DataFrame({"Abstract": abstracts, "month": month.strftime("%Y-%m")})
        if len(df_m) > N_PER_MONTH:
            df_m = df_m.sample(n=N_PER_MONTH, random_state=RANDOM_SEED)
        print(f"  {month.strftime('%Y-%m')}: {len(df_m):,} papers sampled")
        frames.append(df_m)

    df = pd.concat(frames, ignore_index=True)
    df["custom_id"] = [f"abs_{i}" for i in range(len(df))]
    print(f"Total: {len(df):,} abstracts across {len(buckets)} months\n")
    return df


# =============================================================================
# STEP 2 — BUILD BATCH JSONL AND SUBMIT
# =============================================================================

PROMPT_TEMPLATE = (
    "You are simulating the writing process of a scientific author.\n"
    "Internally identify the abstract's key points, then rewrite it in polished academic prose.\n"
    "Preserve the original meaning, scientific claims, scope, and level of specificity.\n"
    "Keep the rewritten abstract similar in length to the original.\n"
    "Return your answer in exactly this format and nothing else:\n"
    "<REWRITTEN_ABSTRACT>\n"
    "[rewritten abstract]\n"
    "</REWRITTEN_ABSTRACT>\n\n"
    "## INPUT_ABSTRACT\n"
    "{abstract}"
)

def build_and_submit(df: pd.DataFrame) -> str:
    print("Building batch JSONL...")
    with open(BATCH_JSONL, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            request = {
                "custom_id": row["custom_id"],
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo-0125",
                    "messages": [
                        {
                            "role":    "user",
                            "content": PROMPT_TEMPLATE.format(abstract=row["Abstract"])
                        }
                    ],
                    "max_tokens": 2048,
                    "temperature": 1,
                }
            }
            f.write(json.dumps(request) + "\n")
    print(f"  Written {len(df):,} requests to {BATCH_JSONL}")

    print("Uploading batch file...")
    with open(BATCH_JSONL, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")

    print("Submitting batch job...")
    batch = client.batches.create(
        input_file_id    = batch_file.id,
        endpoint         = "/v1/chat/completions",
        completion_window= "24h",
    )
    print(f"  Batch ID: {batch.id}  |  Status: {batch.status}")
    pathlib.Path(BATCH_ID_FILE).write_text(batch.id)
    print(f"  Batch ID saved to {BATCH_ID_FILE}")
    return batch.id


# =============================================================================
# STEP 3 — POLL UNTIL COMPLETE AND RETRIEVE RESULTS
# =============================================================================

def poll_and_retrieve(batch_id: str, df: pd.DataFrame):
    print(f"\nPolling batch {batch_id}...")
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  Status: {batch.status}  |  "
              f"completed: {counts.completed}  failed: {counts.failed}  total: {counts.total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(60)

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status: {batch.status}")

    print("Downloading results...")
    result_content = client.files.content(batch.output_file_id).text

    # Parse results into a dict keyed by custom_id
    results = {}
    for line in result_content.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid  = obj["custom_id"]
        body = obj.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if choices:
            raw = choices[0]["message"]["content"].strip()
            m = re.search(r"<REWRITTEN_ABSTRACT>\s*(.*?)\s*</REWRITTEN_ABSTRACT>", raw, re.DOTALL)
            results[cid] = m.group(1).strip() if m else None
        else:
            results[cid] = None

    df["GPT_35_Abstract"] = df["custom_id"].map(results)
    n_missing = df["GPT_35_Abstract"].isna().sum()
    if n_missing:
        print(f"  Warning: {n_missing} abstracts have no result (failed requests)")

    out = df[["Abstract", "GPT_35_Abstract"]].dropna(subset=["GPT_35_Abstract"])
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out):,} rows to {OUTPUT_CSV}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # If a batch was already submitted, skip straight to retrieval
    if pathlib.Path(BATCH_ID_FILE).exists():
        batch_id = pathlib.Path(BATCH_ID_FILE).read_text().strip()
        print(f"Found existing batch ID: {batch_id}")
        df = load_sample()
        poll_and_retrieve(batch_id, df)
    else:
        df = load_sample()
        batch_id = build_and_submit(df)
        poll_and_retrieve(batch_id, df)
