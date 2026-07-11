# Counterfactual Causal Inference for Biological-Age Reversal

Research prototype (proof-of-concept) for the GenMI lab (MBZUAI). It applies
counterfactual causal inference (`CausalForestDML`) to estimate **which
modifiable lifestyle factor produces the greatest reversal in biological-age
acceleration**, and tests whether the **oral microbiome** mediates that effect.
Builds on Li et al. 2025 (arXiv:2510.12384), focusing on the pre-inflection
**40–50 year** intervention window.

The pipeline (`prototype.py`) runs in 12 steps:

1–8. **PhenoAge analysis** — Levine et al. 2018 PhenoAge / age-acceleration from
NHANES blood biomarkers, then a causal ranking of three lifestyle interventions
(diet quality, sleep, physical activity) with covariate-balance diagnostics.

9–12. **Microbiome extension** — merges NHANES oral 16S data, builds a
genus-based microbiome age clock, re-runs the causal analysis with microbiome
**Shannon diversity** as the outcome, and runs a mediation analysis
(lifestyle → microbiome → PhenoAge).

## ⚠️ Data is NOT stored in this repo

All raw inputs are **public but large**, so they are git-ignored (see
[`.gitignore`](.gitignore)) and must be downloaded separately. Only code and the
generated `results/` (figures + CSVs) are committed. Download the files below and
place them in the exact folders shown.

### 1. NHANES 2009–2010 (cycle F) — CDC

Eight SAS transport (`.xpt`) files. Direct download links (CDC / NCHS):

| File | Component | Place in | Download |
|------|-----------|----------|----------|
| `DEMO_F.XPT`   | Demographics                | `NHANES/Demographics/` | https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DEMO_F.XPT |
| `BMX_F.XPT`    | Body Measures (exam)        | `NHANES/Exam/`          | https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/BMX_F.XPT |
| `BIOPRO_F.XPT` | Standard Biochemistry (lab) | `NHANES/Lab/`           | https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/BIOPRO_F.XPT |
| `CBC_F.XPT`    | Complete Blood Count (lab)  | `NHANES/Lab/`           | https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/CBC_F.XPT |
| `CRP_F.XPT`    | C-Reactive Protein (lab)    | `NHANES/Lab/`           | https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/CRP_F.XPT |
| `DR1TOT_F.XPT` | Day-1 Total Nutrients (diet)| `NHANES/Dietary/`       | https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/DR1TOT_F.XPT |
| `SLQ_F.XPT`    | Sleep Disorders (quest.)    | `NHANES/Questionnaire/` | https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/SLQ_F.XPT |
| `PAQ_F.XPT`    | Physical Activity (quest.)  | `NHANES/Questionnaire/` | https://wwwn.cdc.gov/Nchs/Nhanes/2009-2010/PAQ_F.XPT |

Component landing pages (documentation / codebooks) live at
`https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2009`.
Other `.xpt` files that may be present locally (e.g. `BPX_F`, `PFQ_F`, `DR1IFF_F`,
`DS2TOT_F`, `DSQTOT_F`) are **not required** by `prototype.py`.

### 2. NHANES oral microbiome 16S — dada2-processed

Three tab-separated text files (oral rinse 16S rRNA, keyed by NHANES `SEQN`),
placed in `Microbiome/dada2rsv/`:

| File | Contents |
|------|----------|
| `dada2rsv-alpha.txt`             | Alpha diversity per SEQN (Shannon, richness, etc. at multiple rarefaction depths) |
| `dada2rsv-genus-relative.txt`    | Genus-level relative abundances (participant × genus) |
| `dada2rsv-taxonomy-annotate.txt` | Genus ID → SILVA taxonomy lineage |

> **Source:** these are the NHANES oral microbiome 16S (dada2 RSV) release files.
> They are **not** distributed from CDC's main NHANES portal. Download them from
> the same NHANES microbiome data release they were originally obtained from and
> **fill in the exact URL here** before sharing the repo. (`dada2rb-*` reference-
> based variants may also be present locally but are not used by the pipeline.)

## Setup & run

```bash
# from the repository root (this MBZUAI/ folder)
python -m venv .venv && . .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
python prototype.py
```

Python 3.11 recommended. First run auto-installs `econml` if missing. All outputs
are written to [`results/`](results/) — 8 figures (300 DPI PNG) and 6 CSVs:

- `figure1`–`figure5`: PhenoAge cohort, ranked lifestyle effects, top-intervention
  ITEs, dose-response, covariate balance.
- `figure6`–`figure8`: microbiome age clock, lifestyle→diversity effects, mediation.
- CSVs: intervention ranking, per-participant ITEs, covariate balance, microbiome
  diversity ranking, mediation results.

## Honesty notes

This is a small-sample POC. At the accelerated-ager 40–50 study population
(n ≈ 193; ≈ 156 with microbiome), most causal and all mediation estimates have
95% CIs that cross zero — figures mark non-significant effects explicitly, and the
microbiome clock (r ≈ 0.31) is deliberately **not** used as a causal outcome.
Treat results as hypothesis-generating pending the powered HPP-cohort study.

## Repository layout

```
prototype.py           # full 12-step pipeline
requirements.txt
results/               # committed: figures + CSVs
NHANES/                # data folders (structure committed, .xpt git-ignored)
  Demographics/ Exam/ Lab/ Dietary/ Questionnaire/
Microbiome/            # data folders (structure committed, .txt git-ignored)
  dada2rsv/
```
