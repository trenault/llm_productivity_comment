import os, hashlib, json, re
import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

import random as _random
import pandas as pd
import numpy as np
from collections import Counter
from scipy.optimize import minimize_scalar

# =============================================================================
# 1. LLM DETECTION MODEL 
# =============================================================================
print("Building LLM detection model...")
_synthetic = pd.read_csv("synthetic_llm.csv")

TOKEN_RE = re.compile(r"\b[a-z]+\b")

def tokenize_words(text: str):
    return TOKEN_RE.findall(text.lower())

def build_occurrence_model(df, human_col="Abstract", ai_col="GPT_35_Abstract", eps=1e-6):
    human_docs = df[human_col].dropna().tolist()
    ai_docs    = df[ai_col].dropna().tolist()
    human_dc   = Counter()
    ai_dc      = Counter()
    for text in human_docs:
        human_dc.update(set(tokenize_words(text)))
    for text in ai_docs:
        ai_dc.update(set(tokenize_words(text)))
    n_h   = len(human_docs)
    n_a   = len(ai_docs)
    vocab = set(human_dc) | set(ai_dc)
    p_h   = {w: max(human_dc[w] / n_h, eps) for w in vocab}
    p_a   = {w: max(ai_dc[w]   / n_a, eps) for w in vocab}
    lr    = {w: np.log(p_a[w]) - np.log(p_h[w]) for w in vocab}
    return {"lr": lr}

def estimate_alpha(abstract: str, lr_dict: dict) -> float:
    if not isinstance(abstract, str) or not abstract.strip():
        return 0.0
    scores = []
    for s in sent_tokenize(abstract.strip()):
        toks  = {t.lower() for t in word_tokenize(s) if t.isalpha()}
        score = sum(lr_dict[w] for w in toks if w in lr_dict)
        if score != 0.0:
            scores.append(score)
    if not scores:
        return 0.0
    x = np.array(scores, dtype=float)
    def neg_ll(a):
        vals = (1.0 - a) + a * np.exp(np.clip(x, -500, 500))
        return -np.sum(np.log(np.maximum(vals, 1e-300)))
    return float(minimize_scalar(neg_ll, bounds=(0.0, 1.0), method="bounded").x)

_model   = build_occurrence_model(_synthetic)
_LR_DICT = _model["lr"]
print(f"  Vocabulary size: {len(_LR_DICT):,} words")
LLM_THRESHOLD    = 0.1
KEYWORDS_DATA = re.compile(r"\bdata\b",    re.IGNORECASE)
KEYWORDS_PAPER   = re.compile(r"\bpaper\b",  re.IGNORECASE)
KEYWORDS_FIND    = re.compile(r"\bfind\b",   re.IGNORECASE)

# =============================================================================
# 2. LOAD ARXIV METADATA
# =============================================================================
AI_CATEGORIES = {"cs.CV", "cs.LG", "cs.AI", "cs.IR", "cs.CL"}
DATE_MIN      = pd.Timestamp("2016-01-01", tz="UTC")   # need 2016-2019 for pre-ChatGPT placebo filter
DATE_MAX      = pd.Timestamp("2024-06-30", tz="UTC")
CHATGPT_DATE  = pd.Timestamp("2022-12-01", tz="UTC")
PLACEBO_DATE  = pd.Timestamp("2020-12-01", tz="UTC")   # pseudo ChatGPT release for pre-period placebo

rng_placebo = _random.Random(42)   # reproducible random assignment

print("Loading arXiv metadata...")
rows = []
with open("arxiv-metadata-oai-snapshot.json", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if (i + 1) % 500_000 == 0:
            print(f"  {i+1:,} records scanned, {len(rows):,} kept...")
        record   = json.loads(line.strip())
        versions = record.get("versions")
        if not versions:
            continue
        date = pd.to_datetime(versions[0].get("created", ""),
                              format="%a, %d %b %Y %H:%M:%S %Z")
        if not (DATE_MIN <= date <= DATE_MAX):
            continue
        if set(record.get("categories", "").split()) & AI_CATEGORIES:
            continue
        abstract   = record.get("abstract", "") or ""
        after_main = date > CHATGPT_DATE
        after_pre  = date > PLACEBO_DATE

        # Compute alpha only when needed (post-cutoff papers)
        alpha = estimate_alpha(abstract, _LR_DICT) if (after_main or after_pre) else 0.0

        is_llm     = int(after_main and alpha > LLM_THRESHOLD)
        is_llm_pre = int(after_pre  and alpha > LLM_THRESHOLD)   # pre-ChatGPT placebo
        _r         = rng_placebo.random()
        is_random10 = int(after_main and _r < 0.1)
        is_random20 = int(after_main and _r < 0.2)
        is_random30 = int(after_main and _r < 0.3)
        has_data = int(after_main and bool(KEYWORDS_DATA.search(abstract)))
        has_paper   = int(after_main and bool(KEYWORDS_PAPER.search(abstract)))
        has_find    = int(after_main and bool(KEYWORDS_FIND.search(abstract)))

        authors = [" ".join(p for p in a if p).strip()
                   for a in record.get("authors_parsed", [])]
        rows.append((record.get("id"), record.get("title"), authors, len(authors),
                     date, is_llm, is_llm_pre, is_random10, is_random20, is_random30,
                     has_data, has_paper, has_find))

print("Building DataFrame...")
df = pd.DataFrame(rows, columns=["id", "title", "author", "n_author", "date",
                                  "is_llm", "is_llm_pre",
                                  "is_random10", "is_random20", "is_random30",
                                  "has_data", "has_paper", "has_find"])
df = df.drop_duplicates("title")
df = df[df["n_author"] < 50]
df = df.explode("author").reset_index(drop=True)
df["count"] = 1
df["date"]  = pd.to_datetime(df["date"]).dt.tz_localize(None)
print(f"  {df['id'].nunique():,} unique articles, {df['author'].nunique():,} unique authors")

# =============================================================================
# 3. ACTIVE-AUTHOR FILTERS
# =============================================================================

# Main: ≥4 papers in 2018-2021, obs window 2022-01 to 2024-06
active_main = (
    df[df["date"].dt.year.between(2018, 2021)]
    .groupby("author")["count"].sum()
    .pipe(lambda s: s[s >= 4].index)
)
df_main = df[
    df["author"].isin(active_main) &
    df["date"].between("2022-01-01", "2024-06-30")
].copy()

# Pre-ChatGPT placebo: ≥4 papers in 2016-2019, obs window 2020-01 to 2022-06
active_pre = (
    df[df["date"].dt.year.between(2016, 2019)]
    .groupby("author")["count"].sum()
    .pipe(lambda s: s[s >= 4].index)
)
df_pre = df[
    df["author"].isin(active_pre) &
    df["date"].between("2020-01-01", "2022-06-30")
].copy()

print(f"  Main active authors: {len(active_main):,} | Articles in window: {df_main['id'].nunique():,}")
print(f"  Pre-ChatGPT authors: {len(active_pre):,} | Articles in window: {df_pre['id'].nunique():,}")

# =============================================================================
# 4. NAME FILTER
# =============================================================================
def _name_ok(name: str) -> bool:
    parts = [w.rstrip(".") for w in name.split()]
    return len(parts) >= 3 or sum(len(w) >= 2 for w in parts) >= 2

# =============================================================================
# 5. BUILD STACKED PANEL
# =============================================================================
def build_stacked_production(df_obs, treatment_col,
                             obs_start, obs_end,
                             pseudo_start, pseudo_end,
                             seed=42):

    # ── Monthly panel ──────────────────────────────────────────────────────
    panel = (
        df_obs.groupby(["author", pd.Grouper(key="date", freq="ME")])
        [[treatment_col, "count"]].sum().reset_index()
    )
    
    panel = panel[panel["author"].map(_name_ok)]
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)

    all_months  = pd.date_range(obs_start, obs_end, freq="ME").tz_localize(None)
    all_authors = panel["author"].unique()

    # Balanced panel
    full_idx  = pd.MultiIndex.from_product([all_authors, all_months],
                                            names=["author", "date"])
    panel_bal = (
        panel.set_index(["author", "date"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )

    # Month index (1 = first calendar month of obs window)
    date_to_month = {d: i + 1 for i, d in enumerate(sorted(panel_bal["date"].unique()))}
    panel_bal["month"] = panel_bal["date"].map(date_to_month)

    # First treatment month per author
    first_treat = (
        panel_bal[panel_bal[treatment_col] > 0]
        .groupby("author")["date"].min()
        .rename("first_treatment")
    )
    panel_bal = panel_bal.join(first_treat, on="author")

    # Never-treated: assign pseudo-treatment month
    rng           = np.random.default_rng(seed=seed)
    never_treated = panel_bal[panel_bal["first_treatment"].isna()]["author"].unique()
    pseudo_months = pd.date_range(pseudo_start, pseudo_end, freq="ME").tz_localize(None)
    pseudo_map    = pd.Series(
        rng.choice(pseudo_months, size=len(never_treated), replace=True),
        index=never_treated, name="first_treatment"
    )
    mask = panel_bal["author"].isin(never_treated)
    panel_bal.loc[mask, "first_treatment"] = panel_bal.loc[mask, "author"].map(pseudo_map)
    panel_bal["never_treated"] = mask.astype(int)

    # Convert first_treatment date → month index
    panel_bal["first_treatment_month"] = panel_bal["first_treatment"].map(date_to_month)

    # rel_month
    panel_bal["rel_month"] = panel_bal["month"] - panel_bal["first_treatment_month"]

    # ── Stack cohort by cohort ─────────────────────────────────────────────
    cohorts = (
        panel_bal.loc[panel_bal["never_treated"] == 0, "first_treatment_month"]
        .dropna().unique()
    )
    frames = []
    for cohort in cohorts:
        treated  = panel_bal[(panel_bal["never_treated"] == 0) &
                              (panel_bal["first_treatment_month"] == cohort)]["author"].unique()
        controls = panel_bal[(panel_bal["never_treated"] == 1) &
                              (panel_bal["first_treatment_month"] == cohort)]["author"].unique()
        cdf = panel_bal[panel_bal["author"].isin(
                            np.concatenate([treated, controls]))].copy()
        cdf["cohort"]  = int(cohort)
        cdf["treated"] = cdf["author"].isin(treated).astype(int)
        frames.append(cdf)

    stacked = pd.concat(frames, ignore_index=True)

    # ── Rename outcome ─────────────────────────────────────────────────────
    stacked = stacked.rename(columns={"count": "monthly_productivity"})

    # ── Generate interaction dummies ───────────────────────────────────────
    for xx in range(2, 30):
        col = f"rel_month_pre_{xx:02d}_treated"
        stacked[col] = ((stacked["treated"] == 1) &
                        (stacked["rel_month"] == -xx)).astype(int)

    for xx in range(0, 18):
        col = f"rel_month_post_{xx:02d}_treated"
        stacked[col] = ((stacked["treated"] == 1) &
                        (stacked["rel_month"] == xx)).astype(int)

    pre_cols  = [f"rel_month_pre_{xx:02d}_treated"  for xx in range(29, 1, -1)]
    post_cols = [f"rel_month_post_{xx:02d}_treated" for xx in range(0, 18)]

    out_cols = (["hashed_author", "treated", "cohort", "month",
                 "monthly_productivity", "rel_month"]
                + pre_cols + post_cols)

    return stacked[out_cols]


# =============================================================================
# 6. BUILD AND SAVE ALL DATASETS
# =============================================================================
DATASETS = [
    dict(
        label         = "Replication (LLM detector)",
        df            = df_main,
        treatment_col = "is_llm",
        obs_start     = "2022-01-31",
        obs_end       = "2024-06-30",
        pseudo_start  = "2023-01-01",
        pseudo_end    = "2024-06-30",
        seed          = 42,
        out           = "stacked_replication.csv",
    ),
    dict(
        label         = "Placebo: random treatment (p = 0.1)",
        df            = df_main,
        treatment_col = "is_random10",
        obs_start     = "2022-01-31",
        obs_end       = "2024-06-30",
        pseudo_start  = "2023-01-01",
        pseudo_end    = "2024-06-30",
        seed          = 43,
        out           = "stacked_placebo_random10.csv",
    ),
    dict(
        label         = "Placebo: random treatment (p = 0.2)",
        df            = df_main,
        treatment_col = "is_random20",
        obs_start     = "2022-01-31",
        obs_end       = "2024-06-30",
        pseudo_start  = "2023-01-01",
        pseudo_end    = "2024-06-30",
        seed          = 43,
        out           = "stacked_placebo_random20.csv",
    ),
    dict(
        label         = "Placebo: random treatment (p = 0.3)",
        df            = df_main,
        treatment_col = "is_random30",
        obs_start     = "2022-01-31",
        obs_end       = "2024-06-30",
        pseudo_start  = "2023-01-01",
        pseudo_end    = "2024-06-30",
        seed          = 43,
        out           = "stacked_placebo_random30.csv",
    ),
    dict(
        label         = "Placebo: neutral keyword (data)",
        df            = df_main,
        treatment_col = "has_data",
        obs_start     = "2022-01-31",
        obs_end       = "2024-06-30",
        pseudo_start  = "2023-01-01",
        pseudo_end    = "2024-06-30",
        seed          = 44,
        out           = "stacked_placebo_keyword_data.csv",
    ),
    dict(
        label         = "Placebo: keyword (paper)",
        df            = df_main,
        treatment_col = "has_paper",
        obs_start     = "2022-01-31",
        obs_end       = "2024-06-30",
        pseudo_start  = "2023-01-01",
        pseudo_end    = "2024-06-30",
        seed          = 44,
        out           = "stacked_placebo_keyword_paper.csv",
    ),
    dict(
        label         = "Placebo: keyword (find)",
        df            = df_main,
        treatment_col = "has_find",
        obs_start     = "2022-01-31",
        obs_end       = "2024-06-30",
        pseudo_start  = "2023-01-01",
        pseudo_end    = "2024-06-30",
        seed          = 44,
        out           = "stacked_placebo_keyword_find.csv",
    ),
    dict(
        label         = "Placebo: pre-ChatGPT period (2020-2022)",
        df            = df_pre,
        treatment_col = "is_llm_pre",
        obs_start     = "2020-01-31",
        obs_end       = "2022-06-30",
        pseudo_start  = "2021-01-01",
        pseudo_end    = "2022-06-30",
        seed          = 45,
        out           = "stacked_placebo_prechatgpt.csv",
    ),
]

for cfg in DATASETS:
    print(f"\n[{cfg['label']}] Building stacked dataset...")
    stacked = build_stacked_production(
        cfg["df"],
        cfg["treatment_col"],
        cfg["obs_start"], cfg["obs_end"],
        cfg["pseudo_start"], cfg["pseudo_end"],
        seed=cfg["seed"],
    )
    stacked.to_csv(cfg["out"], index=False)
    n_t = stacked.loc[stacked["treated"] == 1, "hashed_author"].nunique()
    n_c = stacked.loc[stacked["treated"] == 0, "hashed_author"].nunique()
    print(f"  {len(stacked):,} rows | {n_t:,} treated | {n_c:,} control authors")
    print(f"  Saved to {cfg['out']}")

print("\nDone.")
