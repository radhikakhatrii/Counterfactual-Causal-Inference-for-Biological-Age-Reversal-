"""
hpp_prototype.py
================================================================================
HPP adaptation of the counterfactual causal-inference pipeline for biological-
age reversal -- full-phenotype version for the MBZUAI GenMI lab.

All field names are confirmed
from the HPP Knowledgebase data dictionaries except the blood-test biomarker
block (see BLOOD_TEST_FIELDS below), which requires confirmation from someone
with access to the dataset:

    import pandas as pd
    bt = pd.read_parquet(HPP / "blood_tests/blood_tests.parquet")
    print(bt.columns.tolist())

Pipeline:
  1.  Load population (demographics + age)
  2.  Load and merge lifestyle, diet, blood tests, blood pressure
  3.  Compute PhenoAge / AgeAccel (Levine et al. 2018)
  4.  Filter to study population (accelerated agers aged 40-50)
  5.  CausalForestDML per lifestyle intervention -> AgeAccel
  6.  Rank interventions + covariate-balance diagnostics
  7.  Figures 1-5 + CSVs
  8.  Load gut microbiome (MetaPhlAn 4 genus-level)
  9.  Gut microbiome age clock (ElasticNet on genus abundances)
  10. CausalForestDML: lifestyle -> Shannon diversity
  11. Mediation: lifestyle -> microbiome -> PhenoAge
  12. Sweep all curated phenotypes as additional causal outcomes

Run inside the HPP environment:
    python hpp_prototype.py

Python 3.11 recommended. Requires: pandas, numpy, scipy, scikit-learn,
econml>=0.15, matplotlib, statsmodels.
================================================================================
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =========================================================================== #
# CONFIGURATION                                                                #
# =========================================================================== #
HPP          = Path("~/studies/hpp_datasets").expanduser()
RESULTS      = Path("~/hpp_results").expanduser()
RANDOM_STATE = 42
STUDY_YEAR   = 2023    # midpoint of HPP data collection; update if needed
MEDIATION_MIN_N = 150  # minimum n to attempt mediation analysis

RESULTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Blood-test field names for PhenoAge (Levine et al. 2018)                    #
# Field names below are best-guess from HPP naming conventions.               #
# Verify each against the 016-blood_tests data dictionary before running.     #
# Needs confirmation from someone with access to the dataset.                 #
# --------------------------------------------------------------------------- #
BLOOD_TEST_FIELDS = {
    # HPP tabular_field_name      : (internal role,  required unit)
    "albumin":                     ("albumin",    "g/L"),
    "creatinine":                  ("creatinine", "umol/L"),
    "glucose":                     ("glucose",    "mmol/L"),
    "crp":                         ("crp",        "mg/dL"),
    "lymphocyte_percent":          ("lymph_pct",  "%"),
    "mean_corpuscular_volume":     ("mcv",        "fL"),
    "red_cell_distribution_width": ("rdw",        "%"),
    "alkaline_phosphatase":        ("alp",        "U/L"),
    "white_blood_cell_count":      ("wbc",        "10^9/L"),
}

# Conversion factors to reach Levine 2018 required units.
# If HPP already stores in the target unit the factor is 1.0 (no-op).
UNIT_CONVERSIONS = {
    "albumin":    {"g/dL": 10.0,      "g/L":    1.0},
    "creatinine": {"mg/dL": 88.42,    "umol/L": 1.0},
    "glucose":    {"mg/dL": 1/18.018, "mmol/L": 1.0},
}

# Set True if HPP stores CRP in mg/L rather than mg/dL (Levine uses mg/dL).
CRP_AS_MGL = False

# --------------------------------------------------------------------------- #
# Dataset paths (confirmed from HPP Knowledgebase data dictionaries)          #
# --------------------------------------------------------------------------- #
DS = {
    "population":         HPP / "population/population.parquet",
    "lifestyle":          HPP / "lifestyle_and_environment/lifestyle_and_environment.parquet",
    "diet_events":        HPP / "diet_logging/diet_logging_events.parquet",
    "blood_tests":        HPP / "blood_tests/blood_tests.parquet",
    "blood_pressure":     HPP / "blood_pressure/blood_pressure.parquet",
    "gut_microbiome":     HPP / "gut_microbiome/gut_microbiome.parquet",
    "gut_genus":          HPP / "gut_microbiome/abundance/"
                               "gut_microbiome__metaphlan_abundance_genus_parquet.parquet",
    "oral_microbiome":    HPP / "oral_microbiome/oral_microbiome.parquet",
    "oral_genus":         HPP / "oral_microbiome/abundance/"
                               "oral_microbiome__metaphlan_abundance_genus_parquet.parquet",
    "oral_pathways":      HPP / "oral_microbiome/abundance/"
                               "oral_microbiome__humann_pathway_abundance_pathway_level_parquet.parquet",
    "curated_phenotypes": HPP / "curated_phenotypes",
}

# Lifestyle interventions: display name -> column in merged frame
INTERVENTIONS = {
    "Diet quality":               "diet_score",
    "Sleep duration (h)":         "sleep_hours",
    "Physical activity (min/wk)": "pa_weekly_min",
    "Smoking (current)":          "smoking_current_tobacco",
    "Alcohol frequency":          "alcohol_current_frequency",
}

# Covariates used in all causal models
COVARIATES = ["age", "sex_numeric", "sbp", "dbp"]

# Curated phenotypes to sweep in Step 12
CURATED_PHENOTYPES = [
    ("abdominal_adiposity",    "abdominal_adiposity__curated_phenotype"),
    ("bmi",                    "bmi__curated_phenotype"),
    ("diabetes",               "diabetes__curated_phenotype"),
    ("hypertension",           "hypertension__curated_phenotype"),
    ("hyperlipidemia",         "hyperlipidemia__curated_phenotype"),
    ("ischemic_heart_disease", "ischemic_heart_disease__curated_phenotype"),
    ("mafld",                  "mafld__curated_phenotype"),
    ("ckd",                    "ckd__curated_phenotype"),
    ("osteoporosis",           "osteoporosis__curated_phenotype"),
    ("depression",             "depression__curated_phenotype"),
    ("anxiety",                "anxiety__curated_phenotype"),
    ("adhd",                   "adhd__curated_phenotype"),
    ("osa",                    "osa__curated_phenotype"),
    ("sleep_quality",          "sleep_quality__curated_phenotype"),
    ("migraine",               "migraine__curated_phenotype"),
    ("endometriosis",          "endometriosis__curated_phenotype"),
    ("menopause",              "menopause__curated_phenotype"),
    ("nmsc",                   "nmsc__curated_phenotype"),
]


# =========================================================================== #
# UTILITIES                                                                    #
# =========================================================================== #
def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _load(key: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = DS[key]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{key}' not found at expected path:\n  {path}")
    df = pd.read_parquet(path, columns=columns)
    print(f"  [{key:<22}] shape={df.shape}")
    return df


def _ensure_econml() -> None:
    try:
        import econml  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "econml>=0.15", "--quiet"])


# Publication-quality colour palette
C = dict(teal="#3a7d7b", rust="#a0522d", grey="#9aa0a6",
         green="#4a8c5f", red="#b0483a", ink="#33373b")


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Georgia", "Times New Roman", "DejaVu Serif"],
        "font.size":          11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.edgecolor":     "#444444",
        "axes.linewidth":     0.8,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "figure.dpi":         110,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })
    import matplotlib.pyplot as plt
    return plt


# =========================================================================== #
# STEP 1  Load population                                                      #
# Confirmed fields: study_id, year_of_birth (int), sex (category)             #
# =========================================================================== #
def step1_population() -> pd.DataFrame:
    banner("STEP 1  Load population")
    pop = _load("population")

    pop["age"] = STUDY_YEAR - pop["year_of_birth"].astype(float)
    pop["sex_numeric"] = (pop["sex"]
                          .map({"Male": 0, "Female": 1, 1: 0, 2: 1})
                          .astype("Int64"))

    print(f"  participants : {pop['study_id'].nunique():,}")
    print(f"  age range    : {pop['age'].min():.0f}–{pop['age'].max():.0f} yr  "
          f"(mean {pop['age'].mean():.1f})")
    print(f"  sex          : {pop['sex'].value_counts().to_dict()}")
    return pop[["study_id", "year_of_birth", "age", "sex", "sex_numeric"]].copy()


# =========================================================================== #
# STEP 2  Load and merge lifestyle / diet / blood tests / blood pressure       #
# =========================================================================== #
def step2_load_merge(pop: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 2  Load and merge datasets")
    d = pop.copy()

    # Lifestyle and environment (confirmed field names from data dictionary)
    lif = _load("lifestyle")
    _warn_missing(lif, [
        "activity_vigorous_days_weekly", "activity_vigorous_minutes_daily",
        "activity_moderate_days_weekly", "activity_moderate_minutes_daily",
        "activity_walking_10min_days_weekly", "activity_walking_minutes_daily",
        "sleep_hours_daily", "sleep_chronotype",
        "smoking_current_tobacco", "alcohol_current_frequency",
    ])
    d = d.merge(lif.rename(columns={"sleep_hours_daily": "sleep_hours"}),
                on="study_id", how="left")
    d["pa_weekly_min"] = _build_pa(d)

    # Diet logging: aggregate event-level to person-level
    d = d.merge(_build_diet_score(), on="study_id", how="left")

    # Blood tests: PhenoAge biomarkers
    bt = _load("blood_tests")
    rename_map = {"study_id": "study_id"}
    for field, (role, unit) in BLOOD_TEST_FIELDS.items():
        if field in bt.columns:
            rename_map[field] = f"bt_{role}"
        else:
            print(f"  WARNING: blood_tests field '{field}' not found -- "
                  f"needs confirmation from someone with the dataset")
    bt_cols = [c for c in rename_map.values() if c in bt.rename(columns=rename_map)]
    d = d.merge(bt.rename(columns=rename_map)[bt_cols], on="study_id", how="left")

    # Blood pressure: covariate only
    bp = _load("blood_pressure",
               columns=["study_id", "sitting_blood_pressure_systolic",
                        "sitting_blood_pressure_diastolic"])
    d = d.merge(bp.rename(columns={
        "sitting_blood_pressure_systolic":  "sbp",
        "sitting_blood_pressure_diastolic": "dbp",
    }), on="study_id", how="left")

    print(f"\n  merged shape = {d.shape}")
    return d


def _warn_missing(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c not in df.columns:
            print(f"  WARNING: expected column '{c}' not found")


def _build_pa(d: pd.DataFrame) -> pd.Series:
    """Weekly MVPA minutes (vigorous x2 + moderate + walking). MET convention.
    HPP stores days/week and minutes/day separately; weekly = days * min_per_day.
    """
    def safe(col):
        return d[col].astype(float) if col in d.columns else pd.Series(
            np.nan, index=d.index)

    vig_d = safe("activity_vigorous_days_weekly").clip(0, 7)
    vig_m = safe("activity_vigorous_minutes_daily").clip(1, 960)
    mod_d = safe("activity_moderate_days_weekly").clip(0, 7)
    mod_m = safe("activity_moderate_minutes_daily").clip(1, 960)
    wal_d = safe("activity_walking_10min_days_weekly").clip(0, 7)
    wal_m = safe("activity_walking_minutes_daily").clip(1, 960)

    weekly = (
        (vig_d * vig_m).fillna(0) * 2 +
        (mod_d * mod_m).fillna(0) +
        (wal_d * wal_m).fillna(0)
    )
    weekly[vig_d.isna() & mod_d.isna() & wal_d.isna()] = np.nan
    print(f"  PA weekly: n={weekly.notna().sum():,}, "
          f"median={weekly.median():.0f} min/wk")
    return weekly


def _build_diet_score() -> pd.DataFrame:
    """Aggregate diet_logging_events to person-level diet-quality score.
    Score = z(fibre/1000 kcal) - z(fat/1000 kcal) - z(sodium/1000 kcal).
    """
    print("  loading diet events...")
    ev = _load("diet_events")

    if "study_id" not in ev.columns:
        print("  WARNING: 'study_id' not found in diet events -- "
              "needs confirmation from someone with the dataset")
        return pd.DataFrame(columns=["study_id"])

    nutrient_cols = [c for c in ["calories_kcal", "dietary_fiber_g", "lipid_g",
                                  "sodium_mg", "alcohol_g", "carbohydrate_g",
                                  "protein_g"] if c in ev.columns]
    date_col   = "collection_date" if "collection_date" in ev.columns else None
    group_cols = ["study_id"] + ([date_col] if date_col else [])

    daily  = ev.groupby(group_cols)[nutrient_cols].sum().reset_index()
    person = daily.groupby("study_id")[nutrient_cols].mean().reset_index()

    if "calories_kcal" in person.columns and person["calories_kcal"].gt(0).any():
        per1k = 1000.0 / person["calories_kcal"].where(person["calories_kcal"] > 0)
        parts, labels = [], []
        for col, sign in [("dietary_fiber_g", +1), ("lipid_g", -1), ("sodium_mg", -1)]:
            if col in person.columns:
                x = person[col] * per1k
                parts.append(sign * (x - x.mean()) / x.std())
                labels.append(f"{'+'if sign>0 else '-'}{col}")
        if parts:
            person["diet_score"] = sum(parts)
            print(f"  diet score: {' '.join(labels)} per 1000 kcal (z-scored)")

    print(f"  diet participants: {len(person):,}")
    return person


# =========================================================================== #
# STEP 3  PhenoAge (Levine et al. 2018)                                        #
# =========================================================================== #
def step3_phenoage(d: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 3  PhenoAge (Levine et al. 2018)")

    def get(role, unit_key=None):
        col = f"bt_{role}"
        if col not in d.columns or d[col].isna().all():
            print(f"  WARNING: '{col}' missing or all-null -- "
                  f"needs confirmation from someone with the dataset")
            return pd.Series(np.nan, index=d.index)
        s = d[col].astype(float).copy()
        if unit_key and unit_key in UNIT_CONVERSIONS:
            detected = _detect_unit(s, unit_key)
            factor   = UNIT_CONVERSIONS[unit_key].get(detected, 1.0)
            if factor != 1.0:
                s = s * factor
            print(f"  {role:<12}: {detected}, n={s.notna().sum():,}")
        else:
            print(f"  {role:<12}: n={s.notna().sum():,}")
        return s

    albumin = get("albumin",    "albumin")
    creat   = get("creatinine", "creatinine")
    glucose = get("glucose",    "glucose")
    crp     = get("crp").clip(lower=1e-4)
    if CRP_AS_MGL:
        crp = crp * 10
    ln_crp  = np.log(crp)
    lymph   = get("lymph_pct")
    mcv     = get("mcv")
    rdw     = get("rdw")
    alp     = get("alp")
    wbc     = get("wbc")
    age     = d["age"]

    xb = (-19.9067
          - 0.0336  * albumin
          + 0.0095  * creat
          + 0.1953  * glucose
          + 0.0954  * ln_crp
          - 0.0120  * lymph
          + 0.0268  * mcv
          + 0.3306  * rdw
          + 0.00188 * alp
          + 0.0554  * wbc
          + 0.0804  * age)

    g    = 0.0076927
    mort = (1.0 - np.exp(-np.exp(xb) * (np.exp(120.0 * g) - 1.0) / g)
            ).clip(1e-8, 1 - 1e-8)
    d["PhenoAge"] = 141.50 + np.log(-0.00553 * np.log(1.0 - mort)) / 0.090165
    d["AgeAccel"] = d["PhenoAge"] - age

    valid = d[["PhenoAge", "age", "AgeAccel"]].dropna()
    print(f"\n  complete PhenoAge : n={len(valid):,}")
    print(f"  mean chron. age   : {valid['age'].mean():.2f} "
          f"(SD {valid['age'].std():.2f})")
    print(f"  mean PhenoAge     : {valid['PhenoAge'].mean():.2f} "
          f"(SD {valid['PhenoAge'].std():.2f})")
    print(f"  mean AgeAccel     : {valid['AgeAccel'].mean():.2f} "
          f"(SD {valid['AgeAccel'].std():.2f})")
    return d


def _detect_unit(s: pd.Series, key: str) -> str:
    m = float(s.dropna().median()) if s.notna().any() else 0.0
    if key == "albumin":    return "g/dL"    if m < 10  else "g/L"
    if key == "creatinine": return "mg/dL"   if m < 10  else "umol/L"
    if key == "glucose":    return "mg/dL"   if m > 10  else "mmol/L"
    return "unknown"


# =========================================================================== #
# STEP 4  Study population (accelerated agers, 40-50 y)                        #
# =========================================================================== #
def step4_population(d: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 4  Study population (accelerated agers, 40-50 y)")
    mask  = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    study = d[mask].copy()
    print(f"  full cohort with AgeAccel : {d['AgeAccel'].notna().sum():,}")
    print(f"  aged 40-50                : {d['age'].between(40, 50).sum():,}")
    print(f"  accelerated (AgeAccel>0)  : {(d['AgeAccel'] > 0).sum():,}")
    print(f"  STUDY POPULATION          : {len(study):,}")
    print(f"  mean AgeAccel             : {study['AgeAccel'].mean():.2f} yr")
    return study


# =========================================================================== #
# STEP 5  CausalForestDML per intervention                                     #
# =========================================================================== #
def step5_causal(pop: pd.DataFrame, outcome: str = "AgeAccel",
                 interventions: dict | None = None) -> dict:
    banner(f"STEP 5  CausalForestDML  (outcome = {outcome})")
    _ensure_econml()
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    if interventions is None:
        interventions = INTERVENTIONS
    covars  = [c for c in COVARIATES if c in pop.columns]
    results = {}

    for name, tcol in interventions.items():
        if tcol not in pop.columns:
            print(f"\n  [{name}] SKIPPED: column '{tcol}' not in dataset")
            continue
        print(f"\n  [{name}]  T = {tcol}")
        sub = pop[[outcome, tcol] + covars].dropna()
        if tcol == "sleep_hours":
            sub = sub[sub[tcol].between(2, 14)]
        print(f"    n = {len(sub):,}")
        if len(sub) < 80:
            print("    SKIPPED (n < 80)")
            continue

        Y   = sub[outcome].values
        T   = sub[tcol].values.astype(float)
        X   = sub[covars].values
        est = CausalForestDML(
            model_y=GradientBoostingRegressor(random_state=RANDOM_STATE),
            model_t=GradientBoostingRegressor(random_state=RANDOM_STATE),
            n_estimators=500, cv=5, random_state=RANDOM_STATE)
        est.fit(Y, T, X=X)

        ate     = float(est.ate(X))
        lb, ub  = [float(v) for v in est.ate_interval(X, alpha=0.05)]
        ite     = est.effect(X)
        sd_t    = float(np.std(T))
        ns      = "  (n.s.)" if lb < 0 < ub else ""

        print(f"    ATE         = {ate:+.4f} per unit  "
              f"[95% CI {lb:+.4f}, {ub:+.4f}]{ns}")
        print(f"    std. effect = {ate*sd_t:+.4f} per +1 SD")
        print(f"    % benefit   = {100*np.mean(ite < 0):.1f}%")

        results[name] = dict(
            tcol=tcol, ate=ate, ci=(lb, ub), sd_t=sd_t,
            std_effect=ate * sd_t, ite=ite,
            sub=sub.reset_index(drop=True),
            balance=_covariate_balance(sub, tcol, covars),
        )
    return results


def _covariate_balance(sub: pd.DataFrame, tcol: str,
                       covars: list[str]) -> pd.DataFrame:
    from scipy.stats import pearsonr
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict

    T     = sub[tcol].values.astype(float)
    X     = sub[covars].values
    t_hat = cross_val_predict(
        GradientBoostingRegressor(random_state=RANDOM_STATE), X, T, cv=5)
    t_res = T - t_hat

    rows = [dict(covariate=c,
                 corr_raw=float(pearsonr(sub[c], T)[0]),
                 corr_adj=float(pearsonr(sub[c], t_res)[0]))
            for c in covars]
    bal = pd.DataFrame(rows)
    print("    balance |corr| raw -> ortho: " +
          ", ".join(f"{r.covariate} {abs(r.corr_raw):.2f}->{abs(r.corr_adj):.2f}"
                    for _, r in bal.iterrows()))
    return bal


# =========================================================================== #
# STEP 6  Rank interventions by standardised effect size                       #
# =========================================================================== #
def step6_rank(results: dict) -> pd.DataFrame:
    banner("STEP 6  Intervention ranking")
    tbl = _rank_results(results)
    print(tbl[["Rank", "Intervention", "ATE", "Std_effect", "Pct_benefit"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return tbl


def _rank_results(results: dict) -> pd.DataFrame:
    rows = [dict(
        Intervention=name, ATE=r["ate"],
        CI_low=r["ci"][0], CI_high=r["ci"][1],
        SD_treatment=r["sd_t"], Std_effect=r["std_effect"],
        Reversal=-r["std_effect"],
        Pct_benefit=100 * np.mean(r["ite"] < 0),
    ) for name, r in results.items()]
    tbl = (pd.DataFrame(rows)
             .sort_values("Reversal", ascending=False)
             .reset_index(drop=True))
    tbl.insert(0, "Rank", range(1, len(tbl) + 1))
    return tbl


# =========================================================================== #
# STEP 7  Figures and CSVs                                                     #
# =========================================================================== #
def _ranked_figure(ranked: pd.DataFrame, path: Path, title: str, xlabel: str):
    plt = _style()
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.6 * len(ranked) + 1.5)))
    r       = ranked.sort_values("Reversal").reset_index(drop=True)
    crosses = (r["CI_low"] < 0) & (r["CI_high"] > 0)
    colors  = [C["green"] if v < 0 else C["red"] for v in r["Std_effect"]]
    err     = np.abs((r["CI_high"] - r["CI_low"]) / 2.0 * r["SD_treatment"])
    maxabs  = float(r["Std_effect"].abs().max()) or 1.0

    for yi in range(len(r)):
        ns = bool(crosses.iloc[yi])
        ax.barh(yi, r["Std_effect"].iloc[yi],
                color=colors[yi], alpha=0.35 if ns else 0.85,
                hatch="///" if ns else "", edgecolor=C["ink"], linewidth=0.8,
                xerr=err.iloc[yi],
                error_kw=dict(ecolor=C["ink"], lw=1, capsize=3))
        if not ns:
            v = r["Std_effect"].iloc[yi]
            ax.text(v + np.sign(v) * 0.02 * maxabs, yi, "*",
                    va="center", fontsize=14, color=C["ink"])

    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r["Intervention"], fontsize=9)
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.text(0.98, 0.02,
            "hatched/faded = n.s. (CI crosses 0)    * = CI excludes 0",
            transform=ax.transAxes, ha="right", fontsize=7, color=C["grey"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.name}")


def step7_figures(d: pd.DataFrame, study: pd.DataFrame,
                  results: dict, ranked: pd.DataFrame):
    banner("STEP 7  Figures (300 DPI) and CSVs")
    plt = _style()
    import matplotlib.pyplot as plt  # noqa

    # Figure 1 -- cohort overview
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    full = d["AgeAccel"].dropna()
    axes[0].hist(full, bins=60, color=C["grey"], alpha=0.55, density=True,
                 label=f"Full cohort (n={len(full):,})")
    axes[0].hist(study["AgeAccel"], bins=30, color=C["teal"], alpha=0.75, density=True,
                 label=f"Study pop 40-50 (n={len(study):,})")
    axes[0].axvline(0, color=C["ink"], lw=0.8, ls="--")
    axes[0].set_xlabel("Age acceleration (yr)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("A  Biological age acceleration")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_xlim(-25, 25)

    sc = axes[1].scatter(d["age"], d["PhenoAge"], c=d["AgeAccel"],
                         cmap="RdBu_r", vmin=-15, vmax=15, s=4, alpha=0.35)
    lims = [d["age"].min(), d["age"].max()]
    axes[1].plot(lims, lims, color=C["ink"], lw=0.8, ls="--")
    axes[1].set_xlabel("Chronological age (yr)")
    axes[1].set_ylabel("PhenoAge (yr)")
    axes[1].set_title("B  PhenoAge vs chronological age")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.02).set_label(
        "AgeAccel (yr)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure1_ageaccel_overview.png")
    plt.close(fig)
    print("  saved figure1_ageaccel_overview.png")

    # Figure 2 -- ranked effects
    _ranked_figure(ranked, RESULTS / "figure2_ranked_effects.png",
                   "Ranked causal impact on biological-age reversal",
                   "Standardised effect on AgeAccel (yr per +1 SD)")

    if not results:
        return

    # Figure 3 -- top-intervention ITEs
    top_name    = ranked.iloc[0]["Intervention"]
    top         = results[top_name]
    sub, ite    = top["sub"], top["ite"]
    fig, axes   = plt.subplots(1, 2, figsize=(11, 4.4))
    pct         = 100 * np.mean(ite < 0)
    axes[0].hist(ite, bins=40, color=C["teal"], alpha=0.8)
    axes[0].axvline(0, color=C["ink"], lw=0.9, ls="--")
    axes[0].set_xlabel("Individual treatment effect (yr)")
    axes[0].set_ylabel("Participants")
    axes[0].set_title(f"A  ITE distribution  —  {top_name}")
    axes[0].text(0.03, 0.93, f"{pct:.0f}% benefit (ITE < 0)",
                 transform=axes[0].transAxes, fontsize=9, color=C["green"])
    axes[1].scatter(sub["age"], ite, s=8, alpha=0.5, color=C["teal"])
    axes[1].axhline(0, color=C["ink"], lw=0.8, ls="--")
    axes[1].set_xlabel("Chronological age (yr)")
    axes[1].set_ylabel("ITE (yr)")
    axes[1].set_title("B  Effect heterogeneity by age")
    fig.tight_layout()
    fig.savefig(RESULTS / "figure3_top_intervention_ite.png")
    plt.close(fig)
    print("  saved figure3_top_intervention_ite.png")

    # Figure 4 -- dose-response
    fig, ax = plt.subplots(figsize=(8, 4.6))
    t     = sub[top["tcol"]].values.astype(float)
    y_obs = sub["AgeAccel"].values
    y_cf  = y_obs - ite * (t - np.median(t))
    grp   = pd.DataFrame({"q": pd.qcut(t, 5, duplicates="drop"),
                           "obs": y_obs, "cf": y_cf, "t": t})
    agg   = grp.groupby("q", observed=True).mean(numeric_only=True)
    xp    = np.arange(len(agg))
    ax.plot(xp, agg["obs"], "-o",  color=C["rust"], label="Observed AgeAccel")
    ax.plot(xp, agg["cf"],  "--s", color=C["teal"], label="Counterfactual (median dose)")
    ax.set_xticks(xp)
    ax.set_xticklabels([f"Q{i+1}\n({v:.1f})" for i, v in enumerate(agg["t"])],
                       fontsize=8)
    ax.set_xlabel(f"Quintile of {top_name}")
    ax.set_ylabel("Mean AgeAccel (yr)")
    ax.set_title(f"Dose-response  —  {top_name}")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure4_dose_response.png")
    plt.close(fig)
    print("  saved figure4_dose_response.png")

    # Figure 5 -- covariate balance (Love plots)
    names  = list(results.keys())
    ncols  = min(len(names), 4)
    nrows  = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.2 * ncols, 3.8 * nrows),
                              sharex=True)
    axes  = np.atleast_1d(np.array(axes)).flatten()
    gmax  = max(results[n]["balance"]["corr_raw"].abs().max() for n in names)
    gmax  = max(gmax, 0.12) * 1.12

    for ax, name in zip(axes, names):
        bal = results[name]["balance"]
        yb  = np.arange(len(bal))
        ax.hlines(yb, bal["corr_raw"].abs(), bal["corr_adj"].abs(),
                  color=C["grey"], lw=1, zorder=1)
        ax.scatter(bal["corr_raw"].abs(), yb, s=50, facecolors="none",
                   edgecolors=C["rust"], label="raw", zorder=2)
        ax.scatter(bal["corr_adj"].abs(), yb, s=50, color=C["teal"],
                   label="orthogonalised", zorder=3)
        ax.axvline(0.1, color=C["ink"], lw=0.8, ls=":")
        ax.set_yticks(yb)
        ax.set_yticklabels(bal["covariate"])
        ax.set_xlabel("|corr| with treatment")
        ax.set_title(name, fontsize=9)
        ax.set_xlim(-0.01, gmax)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Covariate balance: confounding removed by DML orthogonalisation",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.005,
             "dotted = 0.1 threshold   open = raw correlation   "
             "filled = after orthogonalisation",
             ha="center", fontsize=8, color=C["grey"])
    fig.savefig(RESULTS / "figure5_covariate_balance.png")
    plt.close(fig)
    print("  saved figure5_covariate_balance.png")

    # CSVs
    ranked.to_csv(RESULTS / "intervention_ranking.csv", index=False)
    pd.concat([r["balance"].assign(intervention=n)
               for n, r in results.items()]).to_csv(
        RESULTS / "covariate_balance.csv", index=False)
    out = results[top_name]["sub"].copy()
    out["ITE_AgeAccel"] = results[top_name]["ite"]
    out.to_csv(RESULTS / "top_intervention_ITE.csv", index=False)
    print("  wrote intervention_ranking.csv, covariate_balance.csv, "
          "top_intervention_ITE.csv")


# =========================================================================== #
# STEP 8  Load gut microbiome (MetaPhlAn 4 genus-level)                        #
# Column format: k__Bacteria|p__...|c__...|o__...|f__...|g__GENUS (float64, %) #
# =========================================================================== #
def step8_load_microbiome(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    banner("STEP 8  Load gut microbiome (MetaPhlAn 4, genus-level)")
    meta  = _load("gut_microbiome",
                  columns=["study_id", "collection_date", "sample_name"])
    genus = _load("gut_genus")

    # Genus columns: full taxonomic path ending at g__ with no species suffix
    gcols = [c for c in genus.columns if "|g__" in c and "|s__" not in c]
    if not gcols:
        gcols = [c for c in genus.columns if "g__" in c]
    print(f"  genus columns: {len(gcols)}")

    if "study_id" in genus.columns:
        genus_wide = genus[["study_id"] + gcols].copy()
    else:
        genus_wide = (meta.merge(genus, on="sample_name", how="inner")
                          [["study_id"] + gcols])

    # Shannon diversity from percent relative abundances
    X      = genus_wide[gcols].fillna(0.0).values / 100.0
    X_safe = np.where(X > 0, X, 1e-10)
    genus_wide["gut_shannon"]  = -np.sum(X_safe * np.log(X_safe), axis=1)
    genus_wide["gut_richness"] = (X > 0).sum(axis=1)

    # For longitudinal data keep the most recent sample per participant
    if "collection_date" in meta.columns and "sample_name" in genus_wide.columns:
        latest = (meta.sort_values("collection_date", ascending=False)
                      .drop_duplicates("study_id", keep="first"))
        genus_wide = (latest[["study_id", "sample_name"]]
                      .merge(genus_wide.drop(columns=["study_id"], errors="ignore"),
                             on="sample_name", how="inner"))

    genus_per_person = genus_wide.drop_duplicates("study_id")
    print(f"  participants with gut microbiome: {len(genus_per_person):,}")
    print(f"  Shannon: mean={genus_per_person['gut_shannon'].mean():.2f}, "
          f"SD={genus_per_person['gut_shannon'].std():.2f}")

    merge_cols = ["study_id", "gut_shannon", "gut_richness"] + gcols
    d = d.merge(genus_per_person[merge_cols], on="study_id", how="left")
    return d, gcols


# =========================================================================== #
# STEP 9  Gut microbiome age clock                                             #
# =========================================================================== #
def step9_microbiome_clock(d: pd.DataFrame,
                           gcols: list[str]) -> tuple[pd.DataFrame, float, float]:
    banner("STEP 9  Gut microbiome age clock (ElasticNet, genus abundances)")
    from scipy.stats import pearsonr
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    avail    = [c for c in gcols if c in d.columns]
    clock_df = d[["study_id", "age"] + avail].dropna(subset=["age"]).copy()
    clock_df[avail] = clock_df[avail].fillna(0.0)

    prev = (clock_df[avail] > 0).mean()
    keep = prev[prev > 0.10].index.tolist()
    print(f"  {len(keep)}/{len(avail)} genera pass prevalence >10%")
    if len(keep) < 5:
        print("  Insufficient genera for clock -- skipping")
        return d, 0.0, 0.0

    X    = clock_df[keep].values
    y    = clock_df["age"].values
    cv   = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(
        make_pipeline(StandardScaler(),
                      ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=50,
                                   cv=5, random_state=RANDOM_STATE,
                                   max_iter=5000)),
        X, y, cv=cv)
    r   = float(pearsonr(pred, y)[0])
    mae = float(np.mean(np.abs(pred - y)))
    print(f"  cross-validated clock: r={r:.3f}, MAE={mae:.1f} yr")
    print(f"  quality: {'strong' if r >= 0.50 else 'moderate' if r >= 0.30 else 'weak'}")

    clock_df["gut_pred_age"]  = pred
    clock_df["gut_age_accel"] = pred - y
    d = d.merge(clock_df[["study_id", "gut_pred_age", "gut_age_accel"]],
                on="study_id", how="left")

    # Figure 6
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lo, hi  = y.min() - 5, y.max() + 5
    ax.scatter(y, pred, s=8, alpha=0.3, color=C["teal"])
    ax.plot([lo, hi], [lo, hi], color=C["ink"], lw=0.9, ls="--")
    ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological age (yr)")
    ax.set_ylabel("Microbiome-predicted age (yr)")
    ax.set_title("Gut microbiome age clock (MetaPhlAn 4, ElasticNet)")
    ax.text(0.04, 0.90, f"r = {r:.2f}\nMAE = {mae:.1f} yr",
            transform=ax.transAxes, fontsize=10, color=C["ink"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure6_microbiome_clock.png")
    plt.close(fig)
    print("  saved figure6_microbiome_clock.png")
    return d, r, mae


# =========================================================================== #
# STEP 10  Lifestyle -> gut microbiome Shannon diversity                        #
# =========================================================================== #
def step10_causal_microbiome(d: pd.DataFrame) -> tuple[dict, pd.DataFrame | None]:
    banner("STEP 10  Lifestyle -> gut microbiome Shannon diversity")
    _ensure_econml()
    covars  = [c for c in COVARIATES if c in d.columns]
    mask    = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    popmb   = d[mask & d["gut_shannon"].notna()].copy()
    print(f"  n = {len(popmb):,}")

    results = {}
    for name, tcol in INTERVENTIONS.items():
        if tcol not in popmb.columns:
            continue
        sub = popmb[["gut_shannon", tcol] + covars].dropna()
        if tcol == "sleep_hours":
            sub = sub[sub[tcol].between(2, 14)]
        print(f"\n  [{name}]  n={len(sub):,}")
        if len(sub) < 80:
            print("    SKIPPED (n < 80)")
            continue
        res = _fit_cf(sub, tcol, "gut_shannon", covars)
        ns  = " (n.s.)" if res["ci"][0] < 0 < res["ci"][1] else ""
        print(f"    ATE = {res['ate']:+.5f} Shannon/unit  "
              f"[95% CI {res['ci'][0]:+.5f}, {res['ci'][1]:+.5f}]{ns}")
        results[name] = res

    if not results:
        print("  no models fit -- skipping Figure 7")
        return results, None

    ranked = _rank_results(results)
    ranked.to_csv(RESULTS / "microbiome_diversity_ranking.csv", index=False)
    _ranked_figure(ranked, RESULTS / "figure7_microbiome_diversity_effects.png",
                   "Lifestyle causal effect on gut microbiome diversity",
                   "Standardised effect on Shannon diversity (per +1 SD)")
    print("  wrote microbiome_diversity_ranking.csv")
    return results, ranked


def _fit_cf(sub: pd.DataFrame, tcol: str, ycol: str,
            covars: list[str]) -> dict:
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor
    Y   = sub[ycol].values
    T   = sub[tcol].values.astype(float)
    X   = sub[covars].values
    est = CausalForestDML(
        model_y=GradientBoostingRegressor(random_state=RANDOM_STATE),
        model_t=GradientBoostingRegressor(random_state=RANDOM_STATE),
        n_estimators=500, cv=5, random_state=RANDOM_STATE)
    est.fit(Y, T, X=X)
    ate    = float(est.ate(X))
    lb, ub = [float(v) for v in est.ate_interval(X, alpha=0.05)]
    ite    = est.effect(X)
    sd_t   = float(np.std(T))
    return dict(tcol=tcol, ate=ate, ci=(lb, ub), sd_t=sd_t,
                std_effect=ate * sd_t, ite=ite,
                sub=sub.reset_index(drop=True))


# =========================================================================== #
# STEP 11  Mediation: lifestyle -> gut microbiome -> PhenoAge                  #
# =========================================================================== #
def step11_mediation(d: pd.DataFrame) -> pd.DataFrame | None:
    banner("STEP 11  Mediation (lifestyle -> gut diversity -> PhenoAge accel)")
    import statsmodels.api as sm

    covars  = [c for c in COVARIATES if c in d.columns]
    mask    = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    popmb   = d[mask & d["gut_shannon"].notna()].copy()
    M, Y    = "gut_shannon", "AgeAccel"

    def ols(y_vec, x_df):
        return sm.OLS(y_vec, sm.add_constant(x_df)).fit()

    def path_coefficients(s: pd.DataFrame, T: str):
        a      = ols(s[M], s[[T] + covars]).params[T]
        full   = ols(s[Y], s[[M, T] + covars])
        b, cp  = full.params[M], full.params[T]
        c      = ols(s[Y], s[[T] + covars]).params[T]
        return a, b, cp, c

    rng  = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name, T in [(k, v) for k, v in INTERVENTIONS.items()
                    if v in popmb.columns]:
        s = popmb[[M, Y, T] + covars].dropna()
        if T == "sleep_hours":
            s = s[s[T].between(2, 14)]
        n = len(s)
        print(f"\n  [{name}]  n={n:,}")
        if n < MEDIATION_MIN_N:
            print(f"    SKIPPED (n < {MEDIATION_MIN_N})")
            continue

        a, b, cp, c = path_coefficients(s, T)
        indirect = a * b
        prop     = indirect / c if abs(c) > 1e-9 else np.nan

        boot_ind, boot_prop = [], []
        idx = np.arange(n)
        for _ in range(1000):
            bs = s.iloc[rng.choice(idx, n, replace=True)]
            try:
                aa, bb, _, cc = path_coefficients(bs, T)
                boot_ind.append(aa * bb)
                boot_prop.append(aa * bb / cc if abs(cc) > 1e-9 else np.nan)
            except Exception:
                continue

        ci_ind  = np.nanpercentile(boot_ind,  [2.5, 97.5])
        ci_prop = np.nanpercentile(boot_prop, [2.5, 97.5])
        sig     = "" if ci_ind[0] < 0 < ci_ind[1] else "  (CI excludes 0)"

        print(f"    a  (T -> M)           = {a:+.5f}")
        print(f"    b  (M -> Y | T)       = {b:+.4f}")
        print(f"    c  total (T -> Y)     = {c:+.4f}")
        print(f"    c' direct (T -> Y|M)  = {cp:+.4f}")
        print(f"    indirect (a*b)        = {indirect:+.4f} "
              f"[95% CI {ci_ind[0]:+.4f}, {ci_ind[1]:+.4f}]{sig}")
        print(f"    proportion mediated   = {100*prop:5.1f}% "
              f"[{100*ci_prop[0]:.1f}%, {100*ci_prop[1]:.1f}%]")

        rows.append(dict(
            Intervention=name, n=n, a=a, b=b, total_c=c, direct_cprime=cp,
            indirect=indirect, indirect_lo=ci_ind[0], indirect_hi=ci_ind[1],
            prop_mediated=prop, prop_lo=ci_prop[0], prop_hi=ci_prop[1],
        ))

    if not rows:
        print("  No mediation models could be estimated.")
        return None

    med = pd.DataFrame(rows)
    med.to_csv(RESULTS / "mediation_results.csv", index=False)

    plt = _style()
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.7 * len(med) + 1.5)))
    yb   = np.arange(len(med))
    prop = 100 * med["prop_mediated"].values
    lo   = 100 * med["prop_lo"].values
    hi   = 100 * med["prop_hi"].values
    ns   = [(med["indirect_lo"].iloc[i] < 0 < med["indirect_hi"].iloc[i])
            for i in range(len(med))]
    ax.barh(yb, prop,
            xerr=np.vstack([prop - lo, hi - prop]),
            color=[C["grey"] if n else C["teal"] for n in ns],
            alpha=0.85, error_kw=dict(ecolor=C["ink"], lw=1, capsize=3))
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_yticks(yb)
    ax.set_yticklabels(med["Intervention"])
    ax.set_xlabel("% of lifestyle -> PhenoAge effect mediated by gut microbiome")
    ax.set_title("Gut microbiome mediation of lifestyle effects")
    ax.text(0.98, 0.04, "grey = n.s. (indirect CI crosses 0)",
            transform=ax.transAxes, ha="right", fontsize=8, color=C["grey"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure8_mediation.png")
    plt.close(fig)
    print("\n  saved figure8_mediation.png")
    print("  wrote mediation_results.csv")
    return med


# =========================================================================== #
# STEP 12  Curated-phenotype sweep                                             #
# Runs CausalForestDML for every (intervention, phenotype) pair.               #
# Output: phenotype_sweep.csv + figure9_phenotype_heatmap.png                 #
# =========================================================================== #
def step12_phenotype_sweep(d: pd.DataFrame) -> pd.DataFrame | None:
    banner("STEP 12  Curated-phenotype sweep (lifestyle -> each outcome)")
    _ensure_econml()
    curated_dir = DS["curated_phenotypes"]
    if not curated_dir.exists():
        print(f"  Directory not found: {curated_dir}")
        return None

    covars   = [c for c in COVARIATES if c in d.columns]
    all_rows = []

    for stem, outcome_col in CURATED_PHENOTYPES:
        parquet = curated_dir / f"{stem}.parquet"
        if not parquet.exists():
            print(f"  [{stem}] parquet not found -- skipping")
            continue
        pheno = pd.read_parquet(parquet)
        if outcome_col not in pheno.columns:
            print(f"  [{stem}] column '{outcome_col}' not found -- skipping")
            continue

        df = d.merge(pheno[["study_id", outcome_col]], on="study_id", how="inner")
        if df[outcome_col].dtype == object or str(df[outcome_col].dtype) == "category":
            df[outcome_col] = (pd.Categorical(df[outcome_col])
                               .codes.astype(float)
                               .where(lambda x: x >= 0))

        n_total = df[outcome_col].notna().sum()
        if n_total < 100:
            print(f"  [{stem}] n={n_total} -- skipping")
            continue

        print(f"\n  [{stem}]  n={n_total:,}")
        for intv_name, tcol in INTERVENTIONS.items():
            if tcol not in df.columns:
                continue
            sub = df[[outcome_col, tcol] + covars].dropna()
            if len(sub) < 100:
                continue
            try:
                res = _fit_cf(sub, tcol, outcome_col, covars)
                ns  = res["ci"][0] < 0 < res["ci"][1]
                print(f"    {intv_name:<30} ATE={res['ate']:+.4f} "
                      f"[{res['ci'][0]:+.4f}, {res['ci'][1]:+.4f}]"
                      + (" n.s." if ns else " ***"))
                all_rows.append(dict(
                    phenotype=stem, intervention=intv_name, n=len(sub),
                    ate=res["ate"], ci_low=res["ci"][0], ci_high=res["ci"][1],
                    std_effect=res["std_effect"], ns=ns))
            except Exception as exc:
                print(f"    {intv_name}: ERROR -- {exc}")

    if not all_rows:
        print("  No results produced.")
        return None

    sweep = pd.DataFrame(all_rows)
    sweep.to_csv(RESULTS / "phenotype_sweep.csv", index=False)
    print(f"\n  wrote phenotype_sweep.csv  "
          f"({len(sweep)} intervention-phenotype pairs)")

    # Figure 9 -- effect heatmap
    try:
        pivot = sweep.pivot(index="phenotype", columns="intervention",
                            values="std_effect")
        sig   = sweep.pivot(index="phenotype", columns="intervention", values="ns")
        plt   = _style()
        import matplotlib.pyplot as plt  # noqa
        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(pivot.columns)),
                                         max(5, 0.5 * len(pivot))))
        im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                       vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i, pheno in enumerate(pivot.index):
            for j, intv_n in enumerate(pivot.columns):
                if not sig.loc[pheno, intv_n]:
                    ax.text(j, i, "*", ha="center", va="center",
                            fontsize=12, color="black")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(
            "Standardised effect (per +1 SD)", fontsize=9)
        ax.set_title("Lifestyle causal effects across curated phenotypes\n"
                     "* = 95% CI excludes 0", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(RESULTS / "figure9_phenotype_heatmap.png")
        plt.close(fig)
        print("  saved figure9_phenotype_heatmap.png")
    except Exception as exc:
        print(f"  Heatmap could not be generated: {exc}")

    return sweep


# =========================================================================== #
# STEP 13  Load oral microbiome (MetaPhlAn 4 genus-level)                     #
# Same column format as gut: k__...|p__...|c__...|o__...|f__...|g__GENUS     #
# =========================================================================== #
def step13_load_oral_microbiome(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    banner("STEP 13  Load oral microbiome (MetaPhlAn 4, genus-level)")
    meta  = _load("oral_microbiome",
                  columns=["study_id", "collection_date", "sample_name"])
    genus = _load("oral_genus")

    gcols = [c for c in genus.columns if "|g__" in c and "|s__" not in c]
    if not gcols:
        gcols = [c for c in genus.columns if "g__" in c]
    print(f"  genus columns: {len(gcols)}")

    if "study_id" in genus.columns:
        genus_wide = genus[["study_id"] + gcols].copy()
    else:
        genus_wide = (meta.merge(genus, on="sample_name", how="inner")
                          [["study_id"] + gcols])

    X      = genus_wide[gcols].fillna(0.0).values / 100.0
    X_safe = np.where(X > 0, X, 1e-10)
    genus_wide["oral_shannon"]  = -np.sum(X_safe * np.log(X_safe), axis=1)
    genus_wide["oral_richness"] = (X > 0).sum(axis=1)

    if "collection_date" in meta.columns and "sample_name" in genus_wide.columns:
        latest = (meta.sort_values("collection_date", ascending=False)
                      .drop_duplicates("study_id", keep="first"))
        genus_wide = (latest[["study_id", "sample_name"]]
                      .merge(genus_wide.drop(columns=["study_id"], errors="ignore"),
                             on="sample_name", how="inner"))

    genus_per_person = genus_wide.drop_duplicates("study_id")
    print(f"  participants with oral microbiome: {len(genus_per_person):,}")
    print(f"  Shannon: mean={genus_per_person['oral_shannon'].mean():.2f}, "
          f"SD={genus_per_person['oral_shannon'].std():.2f}")

    oral_cols = ["study_id", "oral_shannon", "oral_richness"] + gcols
    d = d.merge(genus_per_person[oral_cols], on="study_id", how="left",
                suffixes=("", "_oral"))
    return d, gcols


# =========================================================================== #
# STEP 14  Oral microbiome age clock                                           #
# =========================================================================== #
def step14_oral_clock(d: pd.DataFrame,
                      oral_gcols: list[str]) -> tuple[pd.DataFrame, float, float]:
    banner("STEP 14  Oral microbiome age clock (ElasticNet, genus abundances)")
    from scipy.stats import pearsonr
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    avail    = [c for c in oral_gcols if c in d.columns]
    clock_df = d[["study_id", "age"] + avail].dropna(subset=["age"]).copy()
    clock_df[avail] = clock_df[avail].fillna(0.0)

    prev = (clock_df[avail] > 0).mean()
    keep = prev[prev > 0.10].index.tolist()
    print(f"  {len(keep)}/{len(avail)} genera pass prevalence >10%")
    if len(keep) < 5:
        print("  Insufficient genera for clock -- skipping")
        return d, 0.0, 0.0

    X    = clock_df[keep].values
    y    = clock_df["age"].values
    cv   = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(
        make_pipeline(StandardScaler(),
                      ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=50,
                                   cv=5, random_state=RANDOM_STATE,
                                   max_iter=5000)),
        X, y, cv=cv)
    r   = float(pearsonr(pred, y)[0])
    mae = float(np.mean(np.abs(pred - y)))
    print(f"  cross-validated clock: r={r:.3f}, MAE={mae:.1f} yr")
    print(f"  quality: {'strong' if r >= 0.50 else 'moderate' if r >= 0.30 else 'weak'}")

    clock_df["oral_pred_age"]  = pred
    clock_df["oral_age_accel"] = pred - y
    d = d.merge(clock_df[["study_id", "oral_pred_age", "oral_age_accel"]],
                on="study_id", how="left")

    plt = _style()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lo, hi  = y.min() - 5, y.max() + 5
    ax.scatter(y, pred, s=8, alpha=0.3, color=C["rust"])
    ax.plot([lo, hi], [lo, hi], color=C["ink"], lw=0.9, ls="--")
    ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
    ax.set_xlabel("Chronological age (yr)")
    ax.set_ylabel("Microbiome-predicted age (yr)")
    ax.set_title("Oral microbiome age clock (MetaPhlAn 4, ElasticNet)")
    ax.text(0.04, 0.90, f"r = {r:.2f}\nMAE = {mae:.1f} yr",
            transform=ax.transAxes, fontsize=10, color=C["ink"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure10_oral_clock.png")
    plt.close(fig)
    print("  saved figure10_oral_clock.png")
    return d, r, mae


# =========================================================================== #
# STEP 15  Lifestyle -> oral microbiome Shannon diversity                       #
# =========================================================================== #
def step15_causal_oral_microbiome(d: pd.DataFrame) -> tuple[dict, pd.DataFrame | None]:
    banner("STEP 15  Lifestyle -> oral microbiome Shannon diversity")
    _ensure_econml()
    covars = [c for c in COVARIATES if c in d.columns]
    mask   = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    popmb  = d[mask & d["oral_shannon"].notna()].copy()
    print(f"  n = {len(popmb):,}")

    results = {}
    for name, tcol in INTERVENTIONS.items():
        if tcol not in popmb.columns:
            continue
        sub = popmb[["oral_shannon", tcol] + covars].dropna()
        if tcol == "sleep_hours":
            sub = sub[sub[tcol].between(2, 14)]
        print(f"\n  [{name}]  n={len(sub):,}")
        if len(sub) < 80:
            print("    SKIPPED (n < 80)")
            continue
        res = _fit_cf(sub, tcol, "oral_shannon", covars)
        ns  = " (n.s.)" if res["ci"][0] < 0 < res["ci"][1] else ""
        print(f"    ATE = {res['ate']:+.5f} Shannon/unit  "
              f"[95% CI {res['ci'][0]:+.5f}, {res['ci'][1]:+.5f}]{ns}")
        results[name] = res

    if not results:
        print("  no models fit -- skipping Figure 11")
        return results, None

    ranked = _rank_results(results)
    ranked.to_csv(RESULTS / "oral_diversity_ranking.csv", index=False)
    _ranked_figure(ranked, RESULTS / "figure11_oral_diversity_effects.png",
                   "Lifestyle causal effect on oral microbiome diversity",
                   "Standardised effect on Shannon diversity (per +1 SD)")
    print("  wrote oral_diversity_ranking.csv")
    return results, ranked


# =========================================================================== #
# STEP 16  Mediation: lifestyle -> oral microbiome -> PhenoAge                 #
# =========================================================================== #
def step16_oral_mediation(d: pd.DataFrame) -> pd.DataFrame | None:
    banner("STEP 16  Mediation (lifestyle -> oral diversity -> PhenoAge accel)")
    import statsmodels.api as sm

    covars = [c for c in COVARIATES if c in d.columns]
    mask   = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    popmb  = d[mask & d["oral_shannon"].notna()].copy()
    M, Y   = "oral_shannon", "AgeAccel"

    def ols(y_vec, x_df):
        return sm.OLS(y_vec, sm.add_constant(x_df)).fit()

    def path_coefficients(s: pd.DataFrame, T: str):
        a      = ols(s[M], s[[T] + covars]).params[T]
        full   = ols(s[Y], s[[M, T] + covars])
        b, cp  = full.params[M], full.params[T]
        c      = ols(s[Y], s[[T] + covars]).params[T]
        return a, b, cp, c

    rng  = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name, T in [(k, v) for k, v in INTERVENTIONS.items()
                    if v in popmb.columns]:
        s = popmb[[M, Y, T] + covars].dropna()
        if T == "sleep_hours":
            s = s[s[T].between(2, 14)]
        n = len(s)
        print(f"\n  [{name}]  n={n:,}")
        if n < MEDIATION_MIN_N:
            print(f"    SKIPPED (n < {MEDIATION_MIN_N})")
            continue

        a, b, cp, c = path_coefficients(s, T)
        indirect = a * b
        prop     = indirect / c if abs(c) > 1e-9 else np.nan

        boot_ind, boot_prop = [], []
        idx = np.arange(n)
        for _ in range(1000):
            bs = s.iloc[rng.choice(idx, n, replace=True)]
            try:
                aa, bb, _, cc = path_coefficients(bs, T)
                boot_ind.append(aa * bb)
                boot_prop.append(aa * bb / cc if abs(cc) > 1e-9 else np.nan)
            except Exception:
                continue

        ci_ind  = np.nanpercentile(boot_ind,  [2.5, 97.5])
        ci_prop = np.nanpercentile(boot_prop, [2.5, 97.5])
        sig     = "" if ci_ind[0] < 0 < ci_ind[1] else "  (CI excludes 0)"

        print(f"    a  (T -> M)           = {a:+.5f}")
        print(f"    b  (M -> Y | T)       = {b:+.4f}")
        print(f"    c  total (T -> Y)     = {c:+.4f}")
        print(f"    c' direct (T -> Y|M)  = {cp:+.4f}")
        print(f"    indirect (a*b)        = {indirect:+.4f} "
              f"[95% CI {ci_ind[0]:+.4f}, {ci_ind[1]:+.4f}]{sig}")
        print(f"    proportion mediated   = {100*prop:5.1f}% "
              f"[{100*ci_prop[0]:.1f}%, {100*ci_prop[1]:.1f}%]")

        rows.append(dict(
            Intervention=name, n=n, a=a, b=b, total_c=c, direct_cprime=cp,
            indirect=indirect, indirect_lo=ci_ind[0], indirect_hi=ci_ind[1],
            prop_mediated=prop, prop_lo=ci_prop[0], prop_hi=ci_prop[1],
        ))

    if not rows:
        print("  No mediation models could be estimated.")
        return None

    med = pd.DataFrame(rows)
    med.to_csv(RESULTS / "oral_mediation_results.csv", index=False)

    plt = _style()
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.7 * len(med) + 1.5)))
    yb   = np.arange(len(med))
    prop = 100 * med["prop_mediated"].values
    lo   = 100 * med["prop_lo"].values
    hi   = 100 * med["prop_hi"].values
    ns   = [(med["indirect_lo"].iloc[i] < 0 < med["indirect_hi"].iloc[i])
            for i in range(len(med))]
    ax.barh(yb, prop,
            xerr=np.vstack([prop - lo, hi - prop]),
            color=[C["grey"] if n else C["rust"] for n in ns],
            alpha=0.85, error_kw=dict(ecolor=C["ink"], lw=1, capsize=3))
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_yticks(yb)
    ax.set_yticklabels(med["Intervention"])
    ax.set_xlabel("% of lifestyle -> PhenoAge effect mediated by oral microbiome")
    ax.set_title("Oral microbiome mediation of lifestyle effects")
    ax.text(0.98, 0.04, "grey = n.s. (indirect CI crosses 0)",
            transform=ax.transAxes, ha="right", fontsize=8, color=C["grey"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure12_oral_mediation.png")
    plt.close(fig)
    print("\n  saved figure12_oral_mediation.png")
    print("  wrote oral_mediation_results.csv")
    return med


# =========================================================================== #
# STEP 17  HumanN pathway analysis + gut vs oral concordance figure            #
# Compares lifestyle -> pathway abundance across the top HumanN metabolic      #
# pathways in the oral microbiome, and plots gut vs oral effect concordance.   #
# =========================================================================== #
def step17_pathway_concordance(d: pd.DataFrame,
                               gut_mb_results: dict,
                               oral_mb_results: dict) -> pd.DataFrame | None:
    banner("STEP 17  HumanN pathway analysis + gut vs oral concordance")
    _ensure_econml()
    plt    = _style()
    covars = [c for c in COVARIATES if c in d.columns]
    mask   = (d["AgeAccel"] > 0) & (d["age"].between(40, 50))
    pop    = d[mask].copy()

    # ------------------------------------------------------------------ #
    # Part A  HumanN pathway abundance: top variable pathways             #
    # ------------------------------------------------------------------ #
    pathway_rows = []
    pw_path = DS["oral_pathways"]
    if pw_path.exists():
        print("  loading HumanN pathway abundance...")
        pw = pd.read_parquet(pw_path)
        print(f"  shape={pw.shape}")

        # pathway columns: string names like "PWY-123: glycolysis"
        pw_cols = [c for c in pw.columns if c != "study_id" and
                   not c.startswith(("UNMAPPED", "UNINTEGRATED"))]
        if pw_cols and "study_id" in pw.columns:
            pw_merged = pop.merge(pw[["study_id"] + pw_cols],
                                  on="study_id", how="inner")
            # select top 20 most variable pathways
            var      = pw_merged[pw_cols].var().nlargest(20)
            top_pw   = var.index.tolist()
            print(f"  {len(top_pw)} most variable pathways selected for causal analysis")

            for pw_name in top_pw:
                for intv_name, tcol in INTERVENTIONS.items():
                    if tcol not in pw_merged.columns:
                        continue
                    sub = pw_merged[[pw_name, tcol] + covars].dropna()
                    if len(sub) < 100:
                        continue
                    try:
                        res = _fit_cf(sub, tcol, pw_name, covars)
                        ns  = res["ci"][0] < 0 < res["ci"][1]
                        pathway_rows.append(dict(
                            pathway=pw_name[:60], intervention=intv_name,
                            n=len(sub), ate=res["ate"],
                            ci_low=res["ci"][0], ci_high=res["ci"][1],
                            std_effect=res["std_effect"], ns=ns))
                    except Exception:
                        pass

        if pathway_rows:
            pw_df = pd.DataFrame(pathway_rows)
            pw_df.to_csv(RESULTS / "oral_pathway_effects.csv", index=False)
            print(f"  wrote oral_pathway_effects.csv "
                  f"({len(pw_df)} intervention-pathway pairs)")

            # Pathway heatmap
            try:
                piv = pw_df.pivot(index="pathway", columns="intervention",
                                  values="std_effect")
                sig = pw_df.pivot(index="pathway", columns="intervention",
                                  values="ns")
                fig, ax = plt.subplots(
                    figsize=(max(8, 1.5 * len(piv.columns)),
                             max(5, 0.4 * len(piv))))
                im = ax.imshow(piv.values, cmap="RdBu_r", aspect="auto",
                               vmin=-0.3, vmax=0.3)
                ax.set_xticks(range(len(piv.columns)))
                ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=9)
                ax.set_yticks(range(len(piv.index)))
                ax.set_yticklabels(piv.index, fontsize=8)
                for i, pw_n in enumerate(piv.index):
                    for j, iv_n in enumerate(piv.columns):
                        if not sig.loc[pw_n, iv_n]:
                            ax.text(j, i, "*", ha="center", va="center",
                                    fontsize=10, color="black")
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(
                    "Standardised effect (per +1 SD)", fontsize=9)
                ax.set_title("Lifestyle causal effects on oral HumanN metabolic pathways\n"
                             "* = 95% CI excludes 0", fontsize=12, fontweight="bold")
                fig.tight_layout()
                fig.savefig(RESULTS / "figure13_oral_pathway_heatmap.png")
                plt.close(fig)
                print("  saved figure13_oral_pathway_heatmap.png")
            except Exception as exc:
                print(f"  Pathway heatmap could not be generated: {exc}")
    else:
        print(f"  HumanN pathway parquet not found at expected path -- skipping")

    # ------------------------------------------------------------------ #
    # Part B  Gut vs oral concordance (Figure 14)                         #
    # Interventions with consistent direction in both compartments are     #
    # the strongest candidates for host-wide lifestyle-microbiome effects. #
    # ------------------------------------------------------------------ #
    if gut_mb_results and oral_mb_results:
        shared = [n for n in gut_mb_results if n in oral_mb_results]
        if shared:
            gut_eff  = [gut_mb_results[n]["std_effect"]  for n in shared]
            oral_eff = [oral_mb_results[n]["std_effect"] for n in shared]
            gut_ns   = [gut_mb_results[n]["ci"][0]  < 0 < gut_mb_results[n]["ci"][1]
                        for n in shared]
            oral_ns  = [oral_mb_results[n]["ci"][0] < 0 < oral_mb_results[n]["ci"][1]
                        for n in shared]
            concordant = [g * o > 0 for g, o in zip(gut_eff, oral_eff)]

            fig, ax = plt.subplots(figsize=(6.5, 6.0))
            for i, name in enumerate(shared):
                color  = C["teal"]  if concordant[i] else C["rust"]
                marker = "o" if not (gut_ns[i] or oral_ns[i]) else "^"
                ax.scatter(gut_eff[i], oral_eff[i], s=90,
                           color=color, marker=marker, zorder=3,
                           alpha=0.55 if (gut_ns[i] or oral_ns[i]) else 0.9)
                ax.annotate(name, (gut_eff[i], oral_eff[i]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=C["ink"])

            lim = max(abs(v) for v in gut_eff + oral_eff) * 1.3 or 0.3
            ax.axhline(0, color=C["ink"], lw=0.7, ls="--")
            ax.axvline(0, color=C["ink"], lw=0.7, ls="--")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("Standardised effect on gut Shannon diversity")
            ax.set_ylabel("Standardised effect on oral Shannon diversity")
            ax.set_title("Gut vs oral microbiome concordance\n"
                         "interventions in the same quadrant act on both compartments",
                         fontsize=11, fontweight="bold")
            n_conc = sum(concordant)
            ax.text(0.03, 0.97,
                    f"{n_conc}/{len(shared)} interventions concordant\n"
                    f"teal = concordant   rust = discordant\n"
                    f"circle = both sig.   triangle = one or both n.s.",
                    transform=ax.transAxes, va="top", fontsize=8, color=C["ink"])
            fig.tight_layout()
            fig.savefig(RESULTS / "figure14_gut_oral_concordance.png")
            plt.close(fig)
            print("  saved figure14_gut_oral_concordance.png")

            conc_df = pd.DataFrame(dict(
                intervention=shared, gut_std_effect=gut_eff,
                oral_std_effect=oral_eff, concordant=concordant,
                gut_ns=gut_ns, oral_ns=oral_ns))
            conc_df.to_csv(RESULTS / "gut_oral_concordance.csv", index=False)
            print("  wrote gut_oral_concordance.csv")
            return conc_df

    print("  Insufficient results for concordance plot.")
    return None


# =========================================================================== #
# MAIN                                                                         #
# =========================================================================== #
def main():
    print("HPP Counterfactual Causal-Inference Pipeline  |  GenMI lab, MBZUAI")
    print(f"Data   : {HPP}")
    print(f"Results: {RESULTS}\n")

    pop     = step1_population()
    d       = step2_load_merge(pop)
    d       = step3_phenoage(d)
    study   = step4_population(d)
    results = step5_causal(study)
    ranked  = step6_rank(results)
    step7_figures(d, study, results, ranked)

    # Gut microbiome track
    d, gut_gcols              = step8_load_microbiome(d)
    d, gut_clock_r, gut_mae   = step9_microbiome_clock(d, gut_gcols)
    gut_mb_res, _             = step10_causal_microbiome(d)
    gut_med                   = step11_mediation(d)

    # Oral microbiome track
    d, oral_gcols             = step13_load_oral_microbiome(d)
    d, oral_clock_r, oral_mae = step14_oral_clock(d, oral_gcols)
    oral_mb_res, _            = step15_causal_oral_microbiome(d)
    oral_med                  = step16_oral_mediation(d)

    # Cross-compartment analysis + HumanN pathways
    concordance = step17_pathway_concordance(d, gut_mb_res, oral_mb_res)

    # Curated phenotype sweep
    sweep = step12_phenotype_sweep(d)

    banner("ANALYSIS COMPLETE")
    print(f"  Study population (accelerated agers 40-50) : n={len(study):,}")
    print(f"  Gut  microbiome clock : r={gut_clock_r:.2f}, MAE={gut_mae:.1f} yr")
    print(f"  Oral microbiome clock : r={oral_clock_r:.2f}, MAE={oral_mae:.1f} yr")
    if ranked is not None and len(ranked):
        top = ranked.iloc[0]
        print(f"  Top intervention      : {top['Intervention']} "
              f"(std. effect {top['Std_effect']:+.3f} yr per +1 SD)")
    if concordance is not None:
        n_conc = concordance["concordant"].sum()
        print(f"  Gut-oral concordance  : {n_conc}/{len(concordance)} "
              f"interventions act consistently on both microbiomes")
    if sweep is not None:
        sig = (~sweep["ns"]).sum()
        print(f"  Phenotype sweep       : {sig}/{len(sweep)} pairs reach significance")
    print(f"\n  All outputs written to: {RESULTS}/")
    print("\nDONE.\n")


if __name__ == "__main__":
    main()
