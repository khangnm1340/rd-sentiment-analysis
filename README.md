# Analyzing Public Discussions for Product Insights

**[📄 View the Final Report (PDF)](./final_report.pdf)**

This repository contains the code and assets for a data-mining capstone project that unifies Reddit discussion dumps and Tiki product reviews into a reproducible topic–aspect analysis pipeline. The goal is to surface recurring product issues, pros/cons, and topic trends across 16 hardware-centric communities. 

By leveraging Arctic Shift to bypass Reddit API limits, we achieved multi-year coverage and normalized the data into harmonized Parquet files. Our automated CLI (`extra/run_pipeline.py`) transforms these filtered corpora into TF-IDF/SVD features, groups them with KMeans clustering, and applies aspect-level sentiment scoring using VADER.

![Unified posts-and-comments Parquet preview](images/data_for_ML.png)

### Key Deliverables

- **Configurable CLI (`extra/run_pipeline.py`)**: Filters corpora, vectorizes with TF-IDF, reduces dimensionality with SVD, clusters with KMeans, and layers aspect-level sentiment.
- **Reusable Assets**: Exploratory plots and CSVs under `eda/` and modelling artefacts under `extra/artifacts_*`.
- **Comprehensive Reports**: Typst reports (`final_report.typ`, `final_report_vi.typ`) that document the full workflow in English and Vietnamese.

The instructions below walk through environment setup, data collection, preprocessing, analysis, and report generation so you can reproduce the results end-to-end.

---

## 1. Repository Layout

```
.
├── eda/                      # Exploratory plots (PNG) + tabular summaries (CSV)
├── extra/
│   ├── artifacts_ft1/        # Sample pipeline outputs for Fiio FT1
│   ├── artifacts_hd600/      # Sample pipeline outputs for Sennheiser HD600
│   ├── artifacts_m4/         # Sample pipeline outputs for Sony WF-1000XM4
│   ├── bai_thuyet_trinh.typ  # Typst slide deck delivered in class
│   └── run_pipeline.py       # Main CLI for topic + aspect sentiment pipeline
├── images/                   # Figures used in the reports
├── final_report.typ          # English report (Typst source)
├── final_report_vi.typ       # Vietnamese report (Typst source)
├── plan.md                   # Design notes and trade-off log
└── README.md                 # You are here
```

---

## 2. Prerequisites

### Core tooling

| Tool | Why it is needed |
| ---- | ---------------- |
| **Python ≥ 3.10** | Runs the data pipeline, EDA scripts, and plotting helpers. |
| **pip / virtualenv** | Keeps project dependencies isolated. |
| **Typst ≥ 0.11** | Compiles `final_report.typ` and `final_report_vi.typ` to PDF. |
| **GNU Make / Bash** *(optional)* | Simplifies scripted workflows; commands below assume a POSIX shell. |

### Python packages

Install the following into your virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install \
  "polars>=0.20" \
  "numpy>=1.24" \
  "scikit-learn>=1.4" \
  joblib \
  vaderSentiment \
  matplotlib \
  seaborn
```

> These match the imports used by `extra/run_pipeline.py` and the notebooks/scripts that generated the figures in `eda/`.

### Data acquisition utilities

| Tool | Purpose |
| ---- | ------- |
| **Arctic Shift** | Pulls historical Reddit submissions/comments beyond live API limits. |
| **Academic Torrents account** *(optional)* | Source of bulk subreddit dumps consumed by Arctic Shift. |
| **Nushell** *(optional)* | Used during the project to batch-normalise JSONL exports; any shell + Python combination will work. |

Install Arctic Shift following its upstream instructions (e.g., `pip install arctic-shift` or building from source). Ensure you can authenticate with Reddit if you plan to hit the live API for incremental updates.

---

## 3. Collect Raw Data

We mine 16 hardware-centric Reddit communities (plus complementary Tiki reviews). Reddit supplies rich, text-first, community-moderated threads; Tiki adds verified purchase feedback from the Vietnamese market.

<p align="center">
  <img src="images/subreddits.png" width="45%" alt="Representative subreddit slate" />
  <img src="images/tiki.png" width="45%" alt="Sample of Tiki camera reviews" />
</p>

1. **Clone this repository** and move into it:
   ```bash
   git clone <repo-url>
   cd bao_cao_cuoi_ky
   ```
2. **Download Reddit threads** for the 16 hardware-related subreddits (or your own selection). With Arctic Shift this typically looks like:
   ```bash
   arctic-shift pull \
     --subreddits r/headphones r/GamingLaptops r/macbookpro r/iphone \
     --start 2019-01-01 --end 2024-06-30 \
     --out data/raw/reddit
   ```
   The command above writes compressed JSONL files under `data/raw/reddit/<subreddit>/`.
3. **Fetch Tiki product reviews** (if desired) by exporting CSV/JSONL dumps to `data/raw/tiki/`.
4. Verify you have both posts and comments for each subreddit/product before proceeding.

---

## 4. Normalise to Parquet

The pipeline expects Parquet/CSV/JSONL with (at minimum) the columns `name`, `subreddit`, and `body`. The project stores unified corpora in `posts.parquet` and `comments.parquet` (one row per document). To reproduce:

```bash
python <<'PY'
import glob
import polars as pl
from pathlib import Path

raw_dir = Path('data/raw/reddit')
rows = []
for path in raw_dir.rglob('*.jsonl'):
    df = pl.read_ndjson(path)
    kind = 'comments' if 'comment' in path.name else 'posts'
    cols = set(df.columns)
    if kind == 'posts' and 'selftext' in cols:
        df = df.rename({'selftext': 'body'})
    if 'name' not in cols and 'id' in cols:
        df = df.with_columns(pl.col('id').alias('name'))
    rows.append(df.select(['name', 'subreddit', 'body']))

full = pl.concat(rows, how='vertical_relaxed')
full.write_parquet('data/combined/posts_and_comments.parquet')
print(f"Saved {full.height} rows")
PY
```

Adjust the script if you keep posts/comments separate. The important part is producing a Parquet file whose rows contain the free-text body along with unique identifiers and subreddit metadata.

---

## 5. Run Exploratory Data Analysis (EDA)

The `eda/` directory already contains the PNG/CSV outputs referenced in the report. Below are some highlights from our exploratory data analysis, including community sizes, posting cadences, and word clouds.

<p align="center">
  <img src="eda/01_posts_by_subreddit.png" width="45%" alt="Post counts by subreddit" />
  <img src="eda/15_weekday_hour_heatmap.png" width="45%" alt="Weekly/hourly heatmap of activity" />
</p>
<p align="center">
  <img src="eda/13_wordcloud_titles.png" width="45%" alt="Title word cloud" />
  <img src="eda/14_wordcloud_comments.png" width="45%" alt="Comment word cloud" />
</p>

To regenerate them for a new dataset you can reuse the following template:

```bash
python <<'PY'
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8')
Path('eda').mkdir(exist_ok=True)

df = pl.read_parquet('data/combined/posts_and_comments.parquet')

# Example: posts per subreddit
posts = df.filter(pl.col('body').is_not_null())
counts = posts.group_by('subreddit').count().sort('count', descending=True)
counts.write_csv('eda/posts_by_subreddit.csv')
plt.figure(figsize=(10,5))
ax = sns.barplot(data=counts.to_pandas(), x='subreddit', y='count', color='#4C78A8')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda/01_posts_by_subreddit.png')
plt.close()

# Extend with additional charts (score vs length, top authors, word clouds, etc.)
PY
```

Feel free to follow the naming convention already present in `eda/` so downstream documents pick up the refreshed assets without edits.

---

## 6. Run the Topic + Aspect Sentiment Pipeline

`extra/run_pipeline.py` is the core automation entry point. It:

1. Filters the dataset by aliases/product names (case-insensitive substring match).
2. Cleans the text, builds a TF-IDF matrix, optionally reduces it with Truncated SVD.
3. Sweeps KMeans cluster counts (`k_min..k_max`) to pick the highest silhouette score.
4. Seeds aspect dictionaries based on subreddit categories, assigns sentences to aspects, and scores them with VADER sentiment.
5. Writes artefacts (vectorizer, SVD, clusters, aspect summaries, per-document assignments) to the output directory.

### Example command

```bash
python extra/run_pipeline.py \
  --data data/combined/posts_and_comments.parquet \
  --product "Fiio FT1" \
  --aliases "fiio ft1,ft1" \
  --subs "r/headphones" \
  --out extra/artifacts_ft1_refresh \
  --save-plots
```

Key arguments:

- `--data`: Parquet/CSV/JSONL path created in step 4.
- `--product`: Human-readable product label stored in artefacts.
- `--aliases`: Comma-separated list of strings used to filter rows.
- `--subs`: *(optional)* restricts to specific subreddits.
- `--min-df`, `--max-df`, `--max-feat`, `--ngram-*`: Control TF-IDF behaviour.
- `--svd`: Set to `auto`, a positive integer, or `skip` if you want to keep the full TF-IDF matrix.
- `--method`: Currently `kmeans` is fully supported; NMF is scaffolded.
- `--aspect-category`: Use `auto` or pick from categories defined near the top of the script.
- `--expand-seeds`: Incorporate high-TF-IDF terms into the aspect seed dictionaries.

Outputs written to `--out` include:

- `tfidf_vectorizer.joblib`, `svd_model.joblib`, `svd_explained_variance.json`
- `kmeans_clusters.json`, `method.json`
- `assignments.jsonl`, `aspect_summary.json`, `aspect_summary_by_subreddit.json`
- Optional PNGs when `--save-plots` is supplied.

---

## 7. Update the Reports

Once visuals and artefacts are regenerated, compile the Typst reports:

```bash
# English version
typst compile final_report.typ final_report.pdf

# Vietnamese version
typst compile final_report_vi.typ final_report_vi.pdf
```

Typst looks up assets using relative paths, so ensure the regenerated figures retain the same filenames (or adjust the Typst sources accordingly).

---

## 8. Tips & Troubleshooting

- **Missing VADER**: `pip install vaderSentiment` is required before running the pipeline.
- **Matplotlib errors in headless environments**: Set `MPLBACKEND=Agg` before invoking scripts that create plots.
- **Empty TF-IDF vocabulary**: The CLI automatically retries with `min_df=1`, but you may need to broaden aliases if no rows match.
- **Large datasets**: Increase memory limits or run the pipeline per-product to keep the TF-IDF matrix manageable.
- **Aspect coverage**: Use `--expand-seeds` for new product domains so aspects capture domain-specific jargon.

---

## 9. Reproducibility Checklist

1. ✅ Install Python dependencies and Typst.
2. ✅ Download Reddit/Tiki dumps using Arctic Shift (or equivalent).
3. ✅ Normalise to Parquet with the required columns.
4. ✅ Regenerate EDA plots and CSVs, saving them in `eda/`.
5. ✅ Run `extra/run_pipeline.py` for each product of interest, storing outputs in `extra/artifacts_*` directories.
6. ✅ Compile the Typst reports for the final deliverables.

Following the sequence above will reconstruct the datasets, analyses, and documents delivered in the original project.
