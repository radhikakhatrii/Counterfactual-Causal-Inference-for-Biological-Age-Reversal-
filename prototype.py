"""
prototype.py
================================================================================
Counterfactual causal inference for microbiome-driven biological age
acceleration -- NHANES 2009-2010 proof-of-concept.

Research prototype for the GenMI lab (Prof. Imran Razzak, MBZUAI). Builds on
Li et al. 2025 (arXiv:2510.12384): estimate which modifiable lifestyle factor
produces the greatest reversal in biological-age acceleration, focusing on the
pre-inflection 40-50y intervention window.

This POC uses NHANES phenotypic aging (Levine PhenoAge) as a stand-in outcome
for the microbiome-driven biological clock; the same CausalForestDML machinery
transfers directly to the HPP microbiome cohort in the full study.

Pipeline:
  1. Load + merge XPT files on SEQN
  2. Resolve and select analysis variables (prints exact columns found)
  3. Compute PhenoAge / AgeAccel (Levine et al. 2018)
  4. Filter to accelerated agers aged 40-50 (study population)
  5. CausalForestDML per intervention (diet, sleep, activity, glycemia)
  6. Rank interventions by standardised effect size
  7. Publication figures (Fig 1-4, 300 DPI)
  8. Summary table + participant-level ITE CSV

Run:  python prototype.py
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

# --------------------------------------------------------------------------- #
# Paths -- everything is anchored to this script's location (MBZUAI/)          #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent
NHANES = BASE / "NHANES"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# XPT files (already sorted into NHANES/<domain>/). Only the files needed for
# this analysis are loaded -- the 98 MB individual-foods file is skipped.
FILES = {
    "DEMO": NHANES / "Demographics" / "DEMO_F.xpt",
    "BMX": NHANES / "Exam" / "BMX_F.xpt",
    "BIOPRO": NHANES / "Lab" / "BIOPRO_F.xpt",
    "CBC": NHANES / "Lab" / "CBC_F.xpt",
    "CRP": NHANES / "Lab" / "CRP_F.xpt",
    "DR1TOT": NHANES / "Dietary" / "DR1TOT_F.xpt",
    "SLQ": NHANES / "Questionnaire" / "SLQ_F.xpt",
    "PAQ": NHANES / "Questionnaire" / "PAQ_F.xpt",
}

RANDOM_STATE = 42

# CRP unit handling ---------------------------------------------------------- #
# Levine et al. 2018 take ln(CRP) with CRP in mg/dL. NHANES LBXCRP is already
# mg/dL (median ~0.13), so NO conversion is applied by default. Setting this to
# True reproduces the "x10 -> mg/L" convention from the brief, which shifts
# every PhenoAge up by 0.0954*ln(10) ~= 0.22 yr. Left False for fidelity to the
# published formula.
CRP_AS_MGL = False


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def pick(df: pd.DataFrame, candidates: list[str], label: str):
    """Return (name, series) for the first candidate present, else (None, None)."""
    for c in candidates:
        if c in df.columns:
            print(f"    [{label:<22}] using '{c}'")
            return c, df[c]
    print(f"    [{label:<22}] NOT FOUND (tried {candidates})")
    return None, None


# =========================================================================== #
# STEP 1 -- LOAD AND MERGE                                                     #
# =========================================================================== #
def step1_load_merge() -> pd.DataFrame:
    banner("STEP 1  Data loading and merging")
    frames = {}
    for key, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        df = pd.read_sas(path)
        df["SEQN"] = df["SEQN"].astype("int64")
        frames[key] = df
        print(f"  loaded {key:<7} {str(path.relative_to(BASE)):<40} "
              f"shape={df.shape}")

    # Start from demographics (one row per participant) and left-merge the rest
    merged = frames["DEMO"]
    print(f"\n  base DEMO shape={merged.shape}")
    for key in ["BMX", "BIOPRO", "CBC", "CRP", "DR1TOT", "SLQ", "PAQ"]:
        merged = merged.merge(frames[key], on="SEQN", how="left")
        print(f"  + {key:<7} -> shape={merged.shape}  "
              f"(cols now {merged.shape[1]})")

    print(f"\n  final merged shape = {merged.shape}")
    print(f"  first 20 columns: {list(merged.columns[:20])}")
    return merged


# =========================================================================== #
# STEP 2 -- VARIABLE SELECTION                                                 #
# =========================================================================== #
def step2_select(m: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 2  Variable selection (exact columns found)")
    d = pd.DataFrame({"SEQN": m["SEQN"]})

    print("  Demographics / covariates:")
    d["age"], _ = m["RIDAGEYR"], pick(m, ["RIDAGEYR"], "age")[1]
    d["age"] = m["RIDAGEYR"]
    d["sex"] = pick(m, ["RIAGENDR"], "sex")[1]
    d["race"] = pick(m, ["RIDRETH1"], "race")[1]
    d["educ"] = pick(m, ["DMDEDUC2"], "education")[1]
    d["bmi"] = pick(m, ["BMXBMI"], "BMI")[1]

    print("\n  PhenoAge biomarkers (SI columns preferred = Levine units):")
    # albumin g/L, creatinine umol/L, glucose mmol/L come from the SI columns
    d["albumin_gL"] = pick(m, ["LBDSALSI"], "albumin g/L")[1]
    d["albumin_gdL"] = pick(m, ["LBXSAL"], "albumin g/dL")[1]
    d["creat_umolL"] = pick(m, ["LBDSCRSI"], "creatinine umol/L")[1]
    d["creat_mgdL"] = pick(m, ["LBXSCR"], "creatinine mg/dL")[1]
    d["glucose_mmolL"] = pick(m, ["LBDSGLSI"], "glucose mmol/L")[1]
    d["glucose_mgdL"] = pick(m, ["LBXSGL"], "glucose mg/dL")[1]
    d["crp_mgdL"] = pick(m, ["LBXCRP", "LBXHSCRP"], "CRP mg/dL")[1]
    d["lymph_pct"] = pick(m, ["LBXLYPCT"], "lymphocyte %")[1]
    d["mcv_fL"] = pick(m, ["LBXMCVSI", "LBXMCV"], "MCV fL")[1]
    d["rdw"] = pick(m, ["LBXRDW"], "RDW %")[1]
    d["alp_UL"] = pick(m, ["LBXSAPSI", "LBXSAP"], "ALP U/L")[1]
    d["wbc"] = pick(m, ["LBXWBCSI", "LBXWBC"], "WBC 1000/uL")[1]

    print("\n  Intervention inputs:")
    d["kcal"] = pick(m, ["DR1TKCAL", "DRXTKCAL"], "diet kcal")[1]
    d["fibre"] = pick(m, ["DR1TFIBE", "DRXTFIBE"], "diet fibre")[1]
    d["sugar"] = pick(m, ["DR1TSUGR"], "diet sugar")[1]
    d["satfat"] = pick(m, ["DR1TSFAT"], "diet sat-fat")[1]
    d["sodium"] = pick(m, ["DR1TSODI"], "diet sodium")[1]
    d["sleep_hours"] = pick(m, ["SLD010H"], "sleep hours")[1]

    # Physical activity: build weekly MVPA minutes from whatever PAQ recreational
    # columns exist. Vigorous minutes are weighted x2 (MET convention).
    d["pa_weekly_min"] = _build_pa(m)

    # HbA1c not distributed with these files; glycemia proxy = serum glucose.
    d["hba1c"] = pick(m, ["LBXGH"], "HbA1c")[1]

    print(f"\n  analysis frame shape = {d.shape}")
    return d


def _build_pa(m: pd.DataFrame) -> pd.Series:
    """Weekly moderate-to-vigorous physical activity (MVPA) minutes from the
    full GPAQ block, vigorous domains weighted x2 (MET convention).

    NHANES GPAQ uses a gate/days/min-per-day structure per domain. The key to
    retaining sample is the skip pattern: a respondent who answers 'No' to a
    domain gate (code 2) has that domain's days/minutes left blank, which is a
    true ZERO, not missing. Treating those blanks as NaN (as the first version
    did) needlessly discards most of the cohort. Only respondents who did not
    take the module at all (gate NaN) or refused/don't-know (gate 7/9) are NaN.
    """
    print("    [physical activity     ] constructing weekly MVPA minutes (GPAQ)")

    def col(name):
        return m[name] if name in m.columns else pd.Series(np.nan, index=m.index)

    # domain -> (gate, days/week, minutes/day, vigorous?)
    domains = {
        "vig_work":  ("PAQ605", "PAQ610", "PAD615", True),
        "mod_work":  ("PAQ620", "PAQ625", "PAD630", False),
        "walk_bike": ("PAQ635", "PAQ640", "PAD645", False),
        "vig_rec":   ("PAQ650", "PAQ655", "PAD660", True),
        "mod_rec":   ("PAQ665", "PAQ670", "PAD675", False),
    }

    total = pd.Series(0.0, index=m.index)
    valid = pd.Series(False, index=m.index)  # at least one interpretable domain
    for name, (g, d, mi, vig) in domains.items():
        gate, days, mins = col(g), col(d), col(mi)
        # strip refused / don't-know sentinels (days 77/99, minutes 7777/9999)
        # and implausible values; keep days 0-7 and minutes 1-960 (<=16h/day)
        days = days.where(days.between(0, 7))
        mins = mins.where(mins.between(1, 960))
        dom = pd.Series(np.nan, index=m.index)
        no = gate == 2                       # answered "No" -> 0 minutes
        yes = gate == 1                      # answered "Yes" -> days * min/day
        dom[no] = 0.0
        dom[yes] = (days[yes] * mins[yes])
        weight = 2.0 if vig else 1.0
        # accumulate: NaN domains don't corrupt the sum, but a respondent needs
        # at least one interpretable domain to get a non-NaN weekly total
        total = total.add(dom.fillna(0.0) * weight, fill_value=0.0)
        valid = valid | dom.notna()

    weekly = total.where(valid, np.nan)
    print(f"      domains combined = {list(domains)} (vigorous x2)")
    print(f"      non-null n = {weekly.notna().sum()}  "
          f"(median {weekly.median():.0f} min/wk, "
          f"active>0 = {(weekly > 0).sum()})")
    return weekly


# =========================================================================== #
# STEP 3 -- PHENOAGE (Levine et al. 2018)                                      #
# =========================================================================== #
def step3_phenoage(d: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 3  PhenoAge calculation (Levine et al. 2018)")

    # Resolve each biomarker into Levine's required units, auto-converting if
    # only the conventional-unit column is available.
    albumin = d["albumin_gL"].copy()                       # g/L
    if albumin.isna().all():
        albumin = d["albumin_gdL"] * 10.0                  # g/dL -> g/L
        print("  albumin: converted g/dL x10 -> g/L")
    else:
        print("  albumin: using SI g/L directly")

    creat = d["creat_umolL"].copy()                        # umol/L
    if creat.isna().all():
        creat = d["creat_mgdL"] * 88.42                    # mg/dL -> umol/L
        print("  creatinine: converted mg/dL x88.42 -> umol/L")
    else:
        print("  creatinine: using SI umol/L directly")

    glucose = d["glucose_mmolL"].copy()                    # mmol/L
    if glucose.isna().all():
        glucose = d["glucose_mgdL"] / 18.018               # mg/dL -> mmol/L
        print("  glucose: converted mg/dL /18.018 -> mmol/L")
    else:
        print("  glucose: using SI mmol/L directly")

    crp = d["crp_mgdL"].astype(float).copy()               # mg/dL
    if CRP_AS_MGL:
        crp = crp * 10.0
        print("  CRP: x10 -> mg/L before ln (per brief)")
    else:
        print("  CRP: ln(mg/dL) directly (canonical Levine)")
    # guard against ln(0): clip to a small positive floor
    crp = crp.clip(lower=1e-4)
    ln_crp = np.log(crp)

    lymph = d["lymph_pct"]
    mcv = d["mcv_fL"]
    rdw = d["rdw"]
    alp = d["alp_UL"]
    wbc = d["wbc"]
    age = d["age"]

    xb = (-19.9067
          - 0.0336 * albumin
          + 0.0095 * creat
          + 0.1953 * glucose
          + 0.0954 * ln_crp
          - 0.0120 * lymph
          + 0.0268 * mcv
          + 0.3306 * rdw
          + 0.00188 * alp
          + 0.0554 * wbc
          + 0.0804 * age)

    g = 0.0076927  # gamma (Gompertz)
    mort = 1.0 - np.exp(-np.exp(xb) * (np.exp(120.0 * g) - 1.0) / g)
    mort = mort.clip(1e-8, 1 - 1e-8)  # keep ln() finite
    phenoage = 141.50 + np.log(-0.00553 * np.log(1.0 - mort)) / 0.090165

    d["PhenoAge"] = phenoage
    d["AgeAccel"] = phenoage - age

    valid = d[["PhenoAge", "age", "AgeAccel"]].dropna()
    print(f"\n  n with complete PhenoAge = {len(valid)}")
    print(f"  mean chronological age = {valid['age'].mean():6.2f} "
          f"(SD {valid['age'].std():.2f})")
    print(f"  mean PhenoAge          = {valid['PhenoAge'].mean():6.2f} "
          f"(SD {valid['PhenoAge'].std():.2f})")
    print(f"  mean AgeAccel          = {valid['AgeAccel'].mean():6.2f} "
          f"(SD {valid['AgeAccel'].std():.2f})")
    return d


# =========================================================================== #
# STEP 4 -- STUDY POPULATION                                                   #
# =========================================================================== #
def step4_population(d: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 4  Study population (accelerated agers, 40-50y)")
    pop = d[(d["AgeAccel"] > 0) & (d["age"].between(40, 50))].copy()
    print(f"  full cohort with AgeAccel      = {d['AgeAccel'].notna().sum()}")
    print(f"  aged 40-50                     = "
          f"{d['age'].between(40, 50).sum()}")
    print(f"  accelerated agers (AgeAccel>0) = {(d['AgeAccel'] > 0).sum()}")
    print(f"  STUDY POPULATION (both)        = {len(pop)}")
    print(f"    mean AgeAccel in study pop   = {pop['AgeAccel'].mean():.2f} yr")
    return pop


# =========================================================================== #
# STEP 5 -- CAUSAL FOREST DML                                                  #
# =========================================================================== #
def _ensure_econml():
    try:
        import econml  # noqa: F401
    except ImportError:
        print("  econml not found -- installing (one-time)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "econml>=0.15", "--quiet"])
    from econml.dml import CausalForestDML  # noqa: F401


def _diet_score(pop: pd.DataFrame) -> pd.Series:
    """Simple diet-quality proxy: nutrient densities per 1000 kcal combined as
    z(fibre) - z(sugar) - z(sat-fat) - z(sodium). Higher = healthier."""
    k = pop["kcal"].where(pop["kcal"] > 0)
    per1000 = 1000.0 / k

    def z(col):
        x = (pop[col] * per1000)
        return (x - x.mean()) / x.std()

    parts, used = [], []
    for col, sign in [("fibre", +1), ("sugar", -1), ("satfat", -1), ("sodium", -1)]:
        if col in pop and pop[col].notna().any():
            parts.append(sign * z(col))
            used.append(f"{'+' if sign > 0 else '-'}{col}")
    score = sum(parts)
    print(f"    diet score = {' '.join(used)} (density per 1000 kcal, z-scored)")
    return score


def step5_causal(pop: pd.DataFrame):
    banner("STEP 5  CausalForestDML per intervention")
    _ensure_econml()
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    pop = pop.copy()
    pop["diet_score"] = _diet_score(pop)

    # Three modifiable lifestyle interventions. Serum glucose was dropped from
    # the ranking: non-fasting glucose is endogenous to the aging outcome (it is
    # itself a PhenoAge input) and is not a directly chosen lifestyle lever.
    interventions = {
        "Diet quality": "diet_score",
        "Sleep duration (h)": "sleep_hours",
        "Physical activity (min/wk)": "pa_weekly_min",
    }
    covars = ["age", "sex", "bmi", "race", "educ"]

    results = {}
    for name, tcol in interventions.items():
        print(f"\n  --- {name}  (T = {tcol}) ---")
        cols = ["AgeAccel", tcol] + covars
        sub = pop[cols].dropna()
        # clean sleep coding: NHANES tops sleep at codes; keep plausible 2-14h
        if tcol == "sleep_hours":
            sub = sub[sub[tcol].between(2, 14)]
        print(f"    n after dropna = {len(sub)}")
        if len(sub) < 40:
            print("    SKIPPED (insufficient n)")
            continue

        Y = sub["AgeAccel"].values
        T = sub[tcol].values.astype(float)
        X = sub[covars].values

        est = CausalForestDML(
            model_y=GradientBoostingRegressor(random_state=RANDOM_STATE),
            model_t=GradientBoostingRegressor(random_state=RANDOM_STATE),
            n_estimators=500, cv=5, random_state=RANDOM_STATE,
        )
        est.fit(Y, T, X=X)

        ate = float(est.ate(X))
        lb, ub = est.ate_interval(X, alpha=0.05)
        ite = est.effect(X)
        sd_t = float(np.std(T))
        std_effect = ate * sd_t  # expected dAgeAccel per +1 SD of treatment

        print(f"    ATE            = {ate:+.4f} yr per unit  "
              f"[95% CI {float(lb):+.4f}, {float(ub):+.4f}]")
        print(f"    SD(treatment)  = {sd_t:.3f}")
        print(f"    std. effect    = {std_effect:+.4f} yr per +1 SD")
        print(f"    %% benefiting   = {100 * np.mean(ite < 0):.1f}% "
              f"(ITE < 0 = reduces AgeAccel)")

        balance = _covariate_balance(sub, tcol, covars)

        results[name] = dict(
            tcol=tcol, ate=ate, ci=(float(lb), float(ub)), sd_t=sd_t,
            std_effect=std_effect, ite=ite, sub=sub.reset_index(drop=True),
            balance=balance,
        )
    return results


def _covariate_balance(sub: pd.DataFrame, tcol: str,
                       covars: list[str]) -> pd.DataFrame:
    """Continuous-treatment balance check for DML.

    With a continuous treatment there is no treated/control split, so classic
    standardised mean differences don't apply. Instead we test the assumption
    DML actually leverages: after residualising the treatment on covariates
    (T_res = T - E[T|X], the cross-fitted treatment model), the covariates
    should be uncorrelated with the residualised treatment. Large raw
    covariate-treatment correlations shrinking toward zero after
    orthogonalisation is direct evidence the confounding channel is being
    removed. |corr| < 0.1 is the conventional "balanced" threshold.
    """
    from scipy.stats import pearsonr
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict

    X = sub[covars].values
    T = sub[tcol].values.astype(float)
    # out-of-fold E[T|X] so residuals aren't overfit (mirrors DML cross-fitting)
    t_hat = cross_val_predict(
        GradientBoostingRegressor(random_state=RANDOM_STATE), X, T, cv=5)
    t_res = T - t_hat

    rows = []
    for c in covars:
        raw = pearsonr(sub[c], T)[0]
        adj = pearsonr(sub[c], t_res)[0]
        rows.append(dict(covariate=c, corr_raw=raw, corr_adj=adj))
    bal = pd.DataFrame(rows)
    print(f"    covariate balance (|corr| with T, raw -> orthogonalised):")
    for _, r in bal.iterrows():
        flag = "" if abs(r["corr_adj"]) < 0.1 else "  <-- check"
        print(f"      {r['covariate']:<6} {r['corr_raw']:+.3f} -> "
              f"{r['corr_adj']:+.3f}{flag}")
    print(f"      mean |corr| {bal['corr_raw'].abs().mean():.3f} -> "
          f"{bal['corr_adj'].abs().mean():.3f}")
    return bal


# =========================================================================== #
# STEP 6 -- RANKING                                                            #
# =========================================================================== #
def step6_rank(results: dict) -> pd.DataFrame:
    banner("STEP 6  Ranking by standardised effect size")
    rows = []
    for name, r in results.items():
        rows.append(dict(
            Intervention=name,
            ATE=r["ate"],
            CI_low=r["ci"][0], CI_high=r["ci"][1],
            SD_treatment=r["sd_t"],
            Std_effect=r["std_effect"],
            Reversal=-r["std_effect"],  # positive = lowers AgeAccel = good
            Pct_benefit=100 * np.mean(r["ite"] < 0),
        ))
    tbl = pd.DataFrame(rows).sort_values("Reversal", ascending=False)
    tbl.insert(0, "Rank", range(1, len(tbl) + 1))
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(tbl.to_string(index=False))
    print("\n  Reversal > 0 => increasing the factor lowers biological-age "
          "acceleration.")
    return tbl


# =========================================================================== #
# STEP 7 -- FIGURES                                                            #
# =========================================================================== #
def _style():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt


# Muted, Nature-appropriate palette
C = dict(navy="#2c3e50", teal="#3a7d7b", sand="#c9a66b", rust="#a0522d",
         grey="#9aa0a6", green="#4a8c5f", red="#b0483a", ink="#33373b")


def step7_figures(d, pop, results, ranked):
    banner("STEP 7  Figures (300 DPI)")
    plt = _style()
    import matplotlib.pyplot as plt  # noqa

    # ---- Figure 1 : cohort AgeAccel + PhenoAge vs chronological age --------- #
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    full = d["AgeAccel"].dropna()
    ax[0].hist(full, bins=60, color=C["grey"], alpha=0.55,
               label=f"Full cohort (n={len(full)})", density=True)
    ax[0].hist(pop["AgeAccel"], bins=30, color=C["teal"], alpha=0.75,
               label=f"Study pop 40-50 (n={len(pop)})", density=True)
    ax[0].axvline(0, color=C["ink"], lw=0.8, ls="--")
    ax[0].set_xlabel("Age acceleration (yr)")
    ax[0].set_ylabel("Density")
    ax[0].set_title("A  Biological age acceleration")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_xlim(-25, 25)

    sc = ax[1].scatter(d["age"], d["PhenoAge"], c=d["AgeAccel"],
                       cmap="RdBu_r", vmin=-15, vmax=15, s=6, alpha=0.5)
    lims = [d["age"].min(), d["age"].max()]
    ax[1].plot(lims, lims, color=C["ink"], lw=0.8, ls="--")
    ax[1].set_xlabel("Chronological age (yr)")
    ax[1].set_ylabel("PhenoAge (yr)")
    ax[1].set_title("B  PhenoAge vs chronological age")
    cb = fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.02)
    cb.set_label("AgeAccel (yr)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure1_ageaccel_overview.png")
    plt.close(fig)
    print("  saved figure1_ageaccel_overview.png")

    # ---- Figure 2 : ranked standardised effects --------------------------- #
    fig, ax = plt.subplots(figsize=(8, 4.2))
    r = ranked.iloc[::-1]  # smallest reversal at bottom -> best on top
    # 95% CI (on the ATE, equivalently on the standardised effect) crosses zero?
    crosses = (r["CI_low"] < 0) & (r["CI_high"] > 0)
    colors = [C["green"] if a < 0 else C["red"] for a in r["Std_effect"]]
    # non-significant bars: hatched + de-saturated so direction is still visible
    hatches = ["///" if c else "" for c in crosses]
    alphas = [0.35 if c else 0.85 for c in crosses]
    # error bars in the same (standardised) scale as the plotted effect
    err = np.abs((r["CI_high"] - r["CI_low"]) / 2.0 * r["SD_treatment"])
    y = np.arange(len(r))
    for yi, (val, col, hz, al, e) in enumerate(
            zip(r["Std_effect"], colors, hatches, alphas, err)):
        ax.barh(yi, val, color=col, alpha=al, hatch=hz, edgecolor=C["ink"],
                linewidth=0.8,
                xerr=e, error_kw=dict(ecolor=C["ink"], lw=1, capsize=3))
    # asterisk the significant (CI excludes zero) effects
    for yi, (val, cross) in enumerate(zip(r["Std_effect"], crosses)):
        if not cross:
            ax.text(val + np.sign(val) * 0.05, yi, "*", va="center",
                    ha="left" if val > 0 else "right", fontsize=14,
                    color=C["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(r["Intervention"])
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_xlabel("Standardised effect on AgeAccel (yr per +1 SD)")
    ax.set_title("Ranked causal impact on biological-age reversal")
    ax.text(0.98, 0.10, "green = beneficial   red = harmful",
            transform=ax.transAxes, ha="right", fontsize=8, color=C["grey"])
    ax.text(0.98, 0.03,
            "hatched / faded = 95% CI crosses 0 (n.s.)    * = CI excludes 0",
            transform=ax.transAxes, ha="right", fontsize=8, color=C["grey"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure2_ranked_effects.png")
    plt.close(fig)
    print("  saved figure2_ranked_effects.png")

    if not results:
        print("  (no fitted models -> skipping Fig 3-4)")
        return

    # top-ranked intervention
    top_name = ranked.iloc[0]["Intervention"]
    top = results[top_name]
    sub, ite = top["sub"], top["ite"]

    # ---- Figure 3 : ITE distribution + ITE vs age ------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    pct = 100 * np.mean(ite < 0)
    ax[0].hist(ite, bins=30, color=C["teal"], alpha=0.8)
    ax[0].axvline(0, color=C["ink"], lw=0.9, ls="--")
    ax[0].set_xlabel("Individual treatment effect (yr)")
    ax[0].set_ylabel("Participants")
    ax[0].set_title(f"A  ITE distribution -- {top_name}")
    ax[0].text(0.03, 0.92, f"{pct:.0f}% benefit (ITE<0)",
               transform=ax[0].transAxes, fontsize=9, color=C["green"])

    sc = ax[1].scatter(sub["age"], ite, c=sub["bmi"], cmap="viridis",
                       s=18, alpha=0.75)
    ax[1].axhline(0, color=C["ink"], lw=0.8, ls="--")
    ax[1].set_xlabel("Chronological age (yr)")
    ax[1].set_ylabel("ITE (yr)")
    ax[1].set_title("B  Effect heterogeneity by age / BMI")
    cb = fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.02)
    cb.set_label("BMI", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure3_top_intervention_ite.png")
    plt.close(fig)
    print("  saved figure3_top_intervention_ite.png")

    # ---- Figure 4 : dose-response observed vs counterfactual -------------- #
    fig, ax = plt.subplots(figsize=(8, 4.6))
    t = sub[top["tcol"]].values.astype(float)
    y_obs = sub["AgeAccel"].values
    t_ref = np.median(t)
    # counterfactual: remove the estimated causal contribution relative to the
    # median dose  ->  y_cf = y_obs - ITE * (T - T_ref)
    y_cf = y_obs - ite * (t - t_ref)

    qs = pd.qcut(t, 5, duplicates="drop")
    grp = pd.DataFrame({"q": qs, "obs": y_obs, "cf": y_cf, "t": t})
    agg = grp.groupby("q", observed=True).mean(numeric_only=True)
    xpos = np.arange(len(agg))
    ax.plot(xpos, agg["obs"], "-o", color=C["rust"], label="Observed AgeAccel")
    ax.plot(xpos, agg["cf"], "--s", color=C["teal"],
            label="Counterfactual (dose set to median)")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"Q{i+1}\n({v:.1f})" for i, v in enumerate(agg["t"])],
                       fontsize=8)
    ax.set_xlabel(f"Quantile of {top_name}")
    ax.set_ylabel("Mean AgeAccel (yr)")
    ax.set_title(f"Dose-response -- {top_name}")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "figure4_dose_response.png")
    plt.close(fig)
    print("  saved figure4_dose_response.png")

    # ---- Figure 5 : covariate balance (Love plot per intervention) -------- #
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.4),
                             sharex=True)
    if len(names) == 1:
        axes = [axes]
    # common x-limit that includes every raw correlation across panels
    gmax = max(results[n]["balance"]["corr_raw"].abs().max() for n in names)
    gmax = max(gmax, 0.12) * 1.12
    for ax, name in zip(axes, names):
        bal = results[name]["balance"]
        yb = np.arange(len(bal))
        ax.hlines(yb, bal["corr_raw"].abs(), bal["corr_adj"].abs(),
                  color=C["grey"], lw=1, zorder=1)
        ax.scatter(bal["corr_raw"].abs(), yb, s=55, facecolors="none",
                   edgecolors=C["rust"], label="raw", zorder=2)
        ax.scatter(bal["corr_adj"].abs(), yb, s=55, color=C["teal"],
                   label="orthogonalised", zorder=3)
        ax.axvline(0.1, color=C["ink"], lw=0.8, ls=":")
        ax.set_yticks(yb)
        ax.set_yticklabels(bal["covariate"])
        ax.set_xlabel("|corr| with treatment")
        ax.set_title(name, fontsize=10)
        ax.set_xlim(-0.01, gmax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Covariate balance: confounding removed by orthogonalisation",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.005, "dotted line = 0.1 balance threshold   |   "
             "open = raw correlation, filled = after orthogonalisation",
             ha="center", fontsize=8, color=C["grey"])
    fig.savefig(RESULTS / "figure5_covariate_balance.png")
    plt.close(fig)
    print("  saved figure5_covariate_balance.png")


# =========================================================================== #
# STEP 8 -- SUMMARY + CSV                                                      #
# =========================================================================== #
def step8_summary(pop, results, ranked):
    banner("STEP 8  Summary and outputs")

    ranked_out = RESULTS / "intervention_ranking.csv"
    ranked.to_csv(ranked_out, index=False)
    print(f"  wrote {ranked_out.relative_to(BASE)}")

    if results:
        bal_all = pd.concat(
            [r["balance"].assign(intervention=n) for n, r in results.items()],
            ignore_index=True)
        bal_out = RESULTS / "covariate_balance.csv"
        bal_all.to_csv(bal_out, index=False)
        print(f"  wrote {bal_out.relative_to(BASE)}")

    print("\n  FINAL RANKED SUMMARY")
    print("  " + "-" * 74)
    print(f"  {'Rank':<5}{'Intervention':<28}{'ATE':>10}{'Std.eff':>10}"
          f"{'%benefit':>10}")
    print("  " + "-" * 74)
    for _, row in ranked.iterrows():
        print(f"  {int(row['Rank']):<5}{row['Intervention']:<28}"
              f"{row['ATE']:>+10.4f}{row['Std_effect']:>+10.4f}"
              f"{row['Pct_benefit']:>9.1f}%")
    print("  " + "-" * 74)

    if results:
        top_name = ranked.iloc[0]["Intervention"]
        top = results[top_name]
        out = top["sub"].copy()
        out["ITE_AgeAccel"] = top["ite"]
        out["intervention"] = top_name
        csv = RESULTS / "top_intervention_ITE.csv"
        out.to_csv(csv, index=False)
        print(f"\n  TOP INTERVENTION: {top_name}")
        print(f"  participant-level ITE -> {csv.relative_to(BASE)} "
              f"(n={len(out)})")

    print(f"\n  All outputs in: {RESULTS.relative_to(BASE)}/")


# =========================================================================== #
# MICROBIOME EXTENSION  (Steps 9-12)                                           #
# Oral 16S microbiome from Microbiome/dada2rsv/. Added on top of the PhenoAge  #
# analysis -- nothing above is modified.                                       #
# =========================================================================== #
MIC = BASE / "Microbiome" / "dada2rsv"
POP_MASK = lambda df: (df["AgeAccel"] > 0) & (df["age"].between(40, 50))
MEDIATION_MIN_N = 80  # min sample for mediation to be attempted


def _load_alpha(depth: int = 10000) -> pd.DataFrame:
    """Shannon diversity + richness per SEQN, averaged over the 10 resamplings
    at a chosen rarefaction depth."""
    a = pd.read_csv(MIC / "dada2rsv-alpha.txt", sep="\t")
    a["SEQN"] = a["SEQN"].astype("int64")
    sh = [c for c in a.columns if c.startswith(f"RSV_ShanWienDiv_{depth}_")]
    ri = [c for c in a.columns if c.startswith(f"RSV_ObservedOTUs_{depth}_")]
    # some cells use '.' for missing -> columns read as object; coerce to float
    a[sh + ri] = a[sh + ri].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({
        "SEQN": a["SEQN"],
        "mb_shannon": a[sh].mean(axis=1),
        "mb_richness": a[ri].mean(axis=1),
    })
    print(f"    alpha: {len(a)} participants, depth {depth}, averaged "
          f"{len(sh)} Shannon + {len(ri)} richness resamplings")
    return out


# ----- STEP 9  merge microbiome + report survival -------------------------- #
def step9_merge_microbiome(d: pd.DataFrame):
    banner("STEP 9  Merge oral microbiome alpha diversity")
    alpha = _load_alpha()
    pop_mask_before = POP_MASK(d)
    d = d.merge(alpha, on="SEQN", how="left")
    pop_mask = POP_MASK(d)

    full_pheno = int(d["AgeAccel"].notna().sum())
    full_mb = int((d["AgeAccel"].notna() & d["mb_shannon"].notna()).sum())
    pop_n = int(pop_mask.sum())
    pop_mb = int((pop_mask & d["mb_shannon"].notna()).sum())
    covs = ["age", "sex", "bmi", "race", "educ"]
    pop_mb_cov = int((pop_mask & d["mb_shannon"].notna()
                      & d[covs].notna().all(axis=1)).sum())

    print("\n  MERGE SURVIVAL")
    print(f"    full cohort with PhenoAge          : {full_pheno}")
    print(f"    full cohort with PhenoAge + microbiome : {full_mb} "
          f"({100*full_mb/full_pheno:.0f}% retained)")
    print(f"    study pop (accel agers 40-50)      : {pop_n}")
    print(f"    study pop WITH microbiome          : {pop_mb} "
          f"({100*pop_mb/max(pop_n,1):.0f}% retained)")
    print(f"    study pop WITH microbiome + covars : {pop_mb_cov}")

    if pop_mb_cov < MEDIATION_MIN_N:
        print(f"\n  *** WARNING: study-population microbiome overlap is "
              f"{pop_mb_cov} (< {MEDIATION_MIN_N}). Causal estimates on this "
              f"subset will be underpowered; mediation will be SKIPPED. ***")
    else:
        print(f"\n  OK: {pop_mb_cov} participants (>= {MEDIATION_MIN_N}) "
              f"support microbiome causal + mediation analysis.")
    return d, pop_mb_cov


# ----- STEP 10  microbiome age clock --------------------------------------- #
def step10_microbiome_clock(d: pd.DataFrame):
    banner("STEP 10  Microbiome age clock (genus ElasticNet)")
    from scipy.stats import pearsonr
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    genus = pd.read_csv(MIC / "dada2rsv-genus-relative.txt", sep="\t")
    genus["SEQN"] = pd.to_numeric(genus["SEQN"], errors="coerce")
    genus = genus.dropna(subset=["SEQN"])
    genus["SEQN"] = genus["SEQN"].astype("int64")
    gcols = [c for c in genus.columns if c.startswith("RSV_genus")]
    # coerce abundance columns to numeric ('.' missing markers -> NaN -> 0)
    genus[gcols] = genus[gcols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df = genus.merge(d[["SEQN", "age"]], on="SEQN", how="inner").dropna(
        subset=["age"])
    # prevalence filter: keep genera detected in >10% of samples
    prev = (df[gcols] > 0).mean()
    keep = prev[prev > 0.10].index.tolist()
    X = df[keep].values
    y = df["age"].values
    print(f"  training clock on {len(df)} participants, "
          f"{len(keep)}/{len(gcols)} genera (prevalence >10%)")

    model = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=50, cv=5,
                     random_state=RANDOM_STATE, max_iter=5000))
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(model, X, y, cv=cv)
    r = float(pearsonr(pred, y)[0])
    mae = float(np.mean(np.abs(pred - y)))
    print(f"  cross-validated clock: Pearson r = {r:.3f}, MAE = {mae:.1f} yr")
    verdict = ("usable" if r >= 0.30 else
               "WEAK -- oral 16S is a modest age predictor; interpret the "
               "microbiome-age residual cautiously")
    print(f"  -> clock quality: {verdict}")

    df["mb_pred_age"] = pred
    df["mb_age_accel"] = pred - y
    d = d.merge(df[["SEQN", "mb_pred_age", "mb_age_accel"]], on="SEQN",
                how="left")

    # Figure 6 -- predicted vs actual age
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(y, pred, s=10, alpha=0.35, color=C["teal"])
    # clip the VIEW to the real age range (a few ElasticNet predictions overshoot
    # to +/-300 yr and would otherwise squash the cloud); r/MAE use all data
    lo, hi = y.min() - 5, y.max() + 5
    ax.plot([lo, hi], [lo, hi], color=C["ink"], lw=0.9, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    n_out = int(((pred < lo) | (pred > hi)).sum())
    ax.set_xlabel("Chronological age (yr)")
    ax.set_ylabel("Microbiome-predicted age (yr)")
    ax.set_title("Oral microbiome age clock")
    ax.text(0.04, 0.90, f"r = {r:.2f}\nMAE = {mae:.1f} yr"
            + (f"\n({n_out} outlier preds off-view)" if n_out else ""),
            transform=ax.transAxes, fontsize=10, color=C["ink"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure6_microbiome_clock.png")
    plt.close(fig)
    print("  saved figure6_microbiome_clock.png")
    return d, r, mae


# ----- shared: fit one CausalForestDML ------------------------------------- #
def _fit_cf(sub, tcol, ycol, covars):
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor
    Y = sub[ycol].values
    T = sub[tcol].values.astype(float)
    X = sub[covars].values
    est = CausalForestDML(
        model_y=GradientBoostingRegressor(random_state=RANDOM_STATE),
        model_t=GradientBoostingRegressor(random_state=RANDOM_STATE),
        n_estimators=500, cv=5, random_state=RANDOM_STATE)
    est.fit(Y, T, X=X)
    ate = float(est.ate(X))
    lb, ub = est.ate_interval(X, alpha=0.05)
    ite = est.effect(X)
    sd_t = float(np.std(T))
    return dict(tcol=tcol, ate=ate, ci=(float(lb), float(ub)), sd_t=sd_t,
                std_effect=ate * sd_t, ite=ite, sub=sub.reset_index(drop=True))


def _rank(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append(dict(
            Intervention=name, ATE=r["ate"], CI_low=r["ci"][0],
            CI_high=r["ci"][1], SD_treatment=r["sd_t"],
            Std_effect=r["std_effect"], Reversal=-r["std_effect"],
            Pct_benefit=100 * np.mean(r["ite"] < 0)))
    tbl = pd.DataFrame(rows).sort_values("Reversal", ascending=False)
    tbl.insert(0, "Rank", range(1, len(tbl) + 1))
    return tbl


def _ranked_figure(ranked, path, title, xlabel, rank_col="Reversal"):
    """Figure-2-style ranked horizontal bars with n.s. marking (CI crosses 0)."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    r = ranked.sort_values(rank_col).reset_index(drop=True)  # best on top
    crosses = (r["CI_low"] < 0) & (r["CI_high"] > 0)
    colors = [C["green"] if a < 0 else C["red"] for a in r["Std_effect"]]
    err = np.abs((r["CI_high"] - r["CI_low"]) / 2.0 * r["SD_treatment"])
    maxabs = float(np.max(np.abs(r["Std_effect"]))) or 1.0
    for yi in range(len(r)):
        c = bool(crosses.iloc[yi])
        ax.barh(yi, r["Std_effect"].iloc[yi],
                color=colors[yi], alpha=0.35 if c else 0.85,
                hatch="///" if c else "", edgecolor=C["ink"], linewidth=0.8,
                xerr=err.iloc[yi],
                error_kw=dict(ecolor=C["ink"], lw=1, capsize=3))
        if not c:
            v = r["Std_effect"].iloc[yi]
            ax.text(v + np.sign(v) * 0.02 * maxabs, yi, "*",
                    va="center", fontsize=14, color=C["ink"])
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r["Intervention"])
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.text(0.98, 0.03,
            "hatched / faded = 95% CI crosses 0 (n.s.)    * = CI excludes 0",
            transform=ax.transAxes, ha="right", fontsize=8, color=C["grey"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----- STEP 11  causal effect of lifestyle ON the microbiome --------------- #
def step11_causal_microbiome(d: pd.DataFrame):
    banner("STEP 11  Does lifestyle causally affect the microbiome? "
           "(outcome = Shannon diversity)")
    _ensure_econml()
    covars = ["age", "sex", "bmi", "race", "educ"]
    popmb = d[POP_MASK(d) & d["mb_shannon"].notna()].copy()
    popmb["diet_score"] = _diet_score(popmb)
    print(f"  microbiome study population n = {len(popmb)}")

    interventions = {
        "Diet quality": "diet_score",
        "Sleep duration (h)": "sleep_hours",
        "Physical activity (min/wk)": "pa_weekly_min",
    }
    results = {}
    for name, tcol in interventions.items():
        sub = popmb[["mb_shannon", tcol] + covars].dropna()
        if tcol == "sleep_hours":
            sub = sub[sub[tcol].between(2, 14)]
        print(f"\n  --- {name} -> Shannon  (n={len(sub)}) ---")
        if len(sub) < 40:
            print("    SKIPPED (insufficient n)")
            continue
        res = _fit_cf(sub, tcol, "mb_shannon", covars)
        ns = "  (n.s., CI crosses 0)" if res["ci"][0] < 0 < res["ci"][1] else ""
        print(f"    ATE = {res['ate']:+.5f} Shannon/unit "
              f"[95% CI {res['ci'][0]:+.5f}, {res['ci'][1]:+.5f}]{ns}")
        print(f"    std. effect = {res['std_effect']:+.4f} Shannon per +1 SD")
        results[name] = res

    if not results:
        print("  no models fit -> skipping Figure 7")
        return results, None

    ranked = _rank(results)
    ranked.to_csv(RESULTS / "microbiome_diversity_ranking.csv", index=False)
    _ranked_figure(ranked, RESULTS / "figure7_microbiome_diversity_effects.png",
                   "Lifestyle effect on oral microbiome diversity",
                   "Standardised effect on Shannon diversity (per +1 SD)")
    print("\n  saved figure7_microbiome_diversity_effects.png")
    print("  wrote microbiome_diversity_ranking.csv")
    return results, ranked


# ----- STEP 12  mediation: lifestyle -> microbiome -> PhenoAge ------------- #
def step12_mediation(d: pd.DataFrame, overlap_n: int):
    banner("STEP 12  Mediation (lifestyle -> microbiome -> PhenoAge accel)")
    if overlap_n < MEDIATION_MIN_N:
        print(f"  SKIPPED: only {overlap_n} participants have microbiome + all "
              f"covariates (< {MEDIATION_MIN_N}). Mediation would be unreliable "
              f"at this n, so it is not run. Revisit at the HPP stage.")
        return None

    import statsmodels.api as sm
    covars = ["age", "sex", "bmi", "race", "educ"]
    popmb = d[POP_MASK(d) & d["mb_shannon"].notna()].copy()
    popmb["diet_score"] = _diet_score(popmb)
    M, Y = "mb_shannon", "AgeAccel"

    def ols(yv, Xdf):
        return sm.OLS(yv, sm.add_constant(Xdf)).fit()

    def paths(s, T):
        a = ols(s[M], s[[T] + covars]).params[T]              # T -> M
        full = ols(s[Y], s[[M, T] + covars])                  # M,T -> Y
        b, cprime = full.params[M], full.params[T]
        c = ols(s[Y], s[[T] + covars]).params[T]              # total T -> Y
        return a, b, cprime, c

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name, T in [("Diet quality", "diet_score"),
                    ("Sleep duration (h)", "sleep_hours"),
                    ("Physical activity (min/wk)", "pa_weekly_min")]:
        s = popmb[[M, Y, T] + covars].dropna()
        if T == "sleep_hours":
            s = s[s[T].between(2, 14)]
        n = len(s)
        print(f"\n  --- {name} (n={n}) ---")
        if n < MEDIATION_MIN_N:
            print(f"    skipped (n={n} < {MEDIATION_MIN_N})")
            continue
        a, b, cprime, c = paths(s, T)
        indirect = a * b
        prop = indirect / c if abs(c) > 1e-9 else np.nan

        # nonparametric bootstrap of the indirect effect + proportion
        bi, bp = [], []
        idx = np.arange(n)
        for _ in range(1000):
            bs = s.iloc[rng.choice(idx, n, replace=True)]
            try:
                aa, bb, _, cc = paths(bs, T)
                bi.append(aa * bb)
                bp.append(aa * bb / cc if abs(cc) > 1e-9 else np.nan)
            except Exception:
                continue
        ci_ind = np.nanpercentile(bi, [2.5, 97.5])
        ci_prop = np.nanpercentile(bp, [2.5, 97.5])
        sig = "" if ci_ind[0] < 0 < ci_ind[1] else "  (indirect CI excludes 0)"
        print(f"    a (T->M)          = {a:+.5f}")
        print(f"    b (M->Y|T)        = {b:+.4f}")
        print(f"    total  c (T->Y)   = {c:+.4f}")
        print(f"    direct c'(T->Y|M) = {cprime:+.4f}")
        print(f"    indirect (a*b)    = {indirect:+.4f} "
              f"[95% CI {ci_ind[0]:+.4f}, {ci_ind[1]:+.4f}]{sig}")
        print(f"    proportion mediated = {100*prop:5.1f}% "
              f"[95% CI {100*ci_prop[0]:.1f}%, {100*ci_prop[1]:.1f}%]")
        rows.append(dict(Intervention=name, n=n, a=a, b=b, total_c=c,
                         direct_cprime=cprime, indirect=indirect,
                         indirect_lo=ci_ind[0], indirect_hi=ci_ind[1],
                         prop_mediated=prop, prop_lo=ci_prop[0],
                         prop_hi=ci_prop[1]))

    if not rows:
        print("  no mediation models estimable.")
        return None
    med = pd.DataFrame(rows)
    med.to_csv(RESULTS / "mediation_results.csv", index=False)

    # Figure 8 -- proportion mediated with bootstrap CI
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    yb = np.arange(len(med))
    prop = 100 * med["prop_mediated"].values
    lo = 100 * med["prop_lo"].values
    hi = 100 * med["prop_hi"].values
    xerr = np.vstack([prop - lo, hi - prop])
    ns = [(med["indirect_lo"].iloc[i] < 0 < med["indirect_hi"].iloc[i])
          for i in range(len(med))]
    cols = [C["grey"] if n else C["teal"] for n in ns]
    ax.barh(yb, prop, xerr=xerr, color=cols, alpha=0.85,
            error_kw=dict(ecolor=C["ink"], lw=1, capsize=3))
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_yticks(yb)
    ax.set_yticklabels(med["Intervention"])
    ax.set_xlabel("% of lifestyle -> PhenoAge effect mediated by microbiome")
    ax.set_title("Microbiome mediation of lifestyle effects")
    ax.text(0.98, 0.04, "grey = indirect-effect CI crosses 0 (n.s.)",
            transform=ax.transAxes, ha="right", fontsize=8, color=C["grey"])
    fig.tight_layout()
    fig.savefig(RESULTS / "figure8_mediation.png")
    plt.close(fig)
    print("\n  saved figure8_mediation.png")
    print("  wrote mediation_results.csv")
    return med


def microbiome_summary(clock_r, clock_mae, mb_ranked, med):
    banner("MICROBIOME ANALYSIS -- plain-English summary")
    print("  1. Age clock: oral microbiome genus composition predicts "
          f"chronological age at r={clock_r:.2f} (MAE {clock_mae:.1f} yr). "
          + ("A usable clock." if clock_r >= 0.30 else
             "This is weak -- oral 16S carries only modest age signal, so the "
             "microbiome-age residual is a noisy proxy here."))
    if mb_ranked is not None and len(mb_ranked):
        top = mb_ranked.iloc[0]
        anysig = ((mb_ranked["CI_low"] > 0) | (mb_ranked["CI_high"] < 0)).any()
        print("  2. Lifestyle -> microbiome diversity: strongest apparent "
              f"effect is {top['Intervention']}. "
              + ("At least one effect's 95% CI excludes zero." if anysig else
                 "NONE of the three effects reach significance (all CIs cross "
                 "zero) -- no clear causal signal on diversity at this n."))
    if med is None:
        print("  3. Mediation: not run (sample too small) -- see Step 12.")
    else:
        best = med.iloc[med["prop_mediated"].abs().idxmax()]
        anysig = (~((med["indirect_lo"] < 0) & (med["indirect_hi"] > 0))).any()
        print("  3. Mediation: point estimates attribute up to "
              f"{100*best['prop_mediated']:.0f}% of the {best['Intervention']} "
              "effect to the microbiome, but "
              + ("at least one indirect effect is significant."
                 if anysig else
                 "every indirect-effect CI crosses zero -- there is NO "
                 "statistically reliable mediation at this sample size. Treat "
                 "as hypothesis-generating only."))


# =========================================================================== #
def main():
    print("PhenoAge causal-intervention prototype  |  NHANES 2009-2010")
    merged = step1_load_merge()
    d = step2_select(merged)
    d = step3_phenoage(d)
    pop = step4_population(d)
    results = step5_causal(pop)
    ranked = step6_rank(results)
    step7_figures(d, pop, results, ranked)
    step8_summary(pop, results, ranked)

    # --- microbiome extension (Steps 9-12) --- #
    d, overlap_n = step9_merge_microbiome(d)
    d, clock_r, clock_mae = step10_microbiome_clock(d)
    _, mb_ranked = step11_causal_microbiome(d)
    med = step12_mediation(d, overlap_n)
    microbiome_summary(clock_r, clock_mae, mb_ranked, med)

    print("\nDONE.\n")


if __name__ == "__main__":
    main()
