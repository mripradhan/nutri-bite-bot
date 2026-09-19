"""
Build the NutriBiteBot clinical cohort from the raw MIMIC-IV hosp/icu tables.

One row per adult hospital admission (hadm_id):
  - 14 baseline features drawn from the first 24 h of the admission
    (plus the most recent outpatient BMI/BP from OMR)
  - four observed-outcome targets measured after the first 24 h
    (potassium, renal, carbohydrate) or in outpatient follow-up (sodium/BP)

Only aggregate statistics are printed or written to the cohort card; the
row-level parquet stays in data/derived/ (git-ignored, DUA-restricted).

Usage:
    python mimic_extract.py --mimic-dir ../physionet.org/files/mimiciv/3.1 --tag v3.1
    python mimic_extract.py --mimic-dir ../mimic-iv-clinical-database-demo-2.2 --tag demo
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LAB_ITEMS = {
    "sodium": 50983,
    "potassium": 50971,
    "creatinine": 50912,
    "glucose": 50931,
    "hba1c": 50852,
}

# Physiologically plausible ranges; values outside are treated as entry errors.
PLAUSIBLE = {
    "sodium": (100, 180),
    "potassium": (1.5, 10),
    "creatinine": (0.1, 25),
    "glucose": (20, 2000),
    "hba1c": (3, 20),
    "sbp": (50, 300),
    "dbp": (20, 200),
    "bmi": (10, 80),
}

SBP_ITEMS = (220179, 224167, 227243, 220050)  # NIBP, manual L/R, arterial
DBP_ITEMS = (220180, 224643, 227242, 220051)

BASELINE_HOURS = 24
MIN_LOS_HOURS_FOR_INPATIENT_OUTCOME = 48
HBA1C_LOOKBACK_DAYS = 90
OMR_LOOKBACK_DAYS = 365
FOLLOWUP_BP_WINDOW_DAYS = (30, 365)
SMALL_CELL = 10

FEATURES = [
    "age", "sex_male", "has_htn", "has_dm", "has_ckd",
    "serum_sodium", "serum_potassium", "creatinine", "egfr",
    "hba1c", "fbs", "sbp", "dbp", "bmi",
]
TARGETS = ["sodium_sensitivity", "potassium_sensitivity", "protein_restriction", "carb_sensitivity"]


def lab_range_sql(name: str) -> str:
    lo, hi = PLAUSIBLE[name]
    return f"(itemid = {LAB_ITEMS[name]} AND valuenum BETWEEN {lo} AND {hi})"


def build(con: "duckdb.DuckDBPyConnection", mimic: Path, cache: Path) -> pd.DataFrame:
    h, icu = mimic / "hosp", mimic / "icu"

    # ---- cached single-pass filters over the two huge event tables ----
    labs_pq = cache / "labs.parquet"
    if not labs_pq.exists():
        print("  filtering labevents (single pass, slow) ...")
        where = " OR ".join(lab_range_sql(n) for n in LAB_ITEMS)
        con.execute(f"""
            COPY (
              SELECT subject_id, hadm_id, itemid, charttime, valuenum
              FROM read_csv('{h}/labevents.csv.gz', header=true,
                   types={{'subject_id':'BIGINT','hadm_id':'BIGINT','itemid':'INTEGER',
                           'charttime':'TIMESTAMP','valuenum':'DOUBLE'}})
              WHERE {where}
            ) TO '{labs_pq}' (FORMAT PARQUET)
        """)
    con.execute(f"CREATE OR REPLACE VIEW labs AS SELECT * FROM '{labs_pq}'")

    bp_pq = cache / "icu_bp.parquet"
    if not bp_pq.exists():
        print("  filtering chartevents for BP (single pass, slow) ...")
        con.execute(f"""
            COPY (
              SELECT subject_id, hadm_id, itemid, charttime, valuenum
              FROM read_csv('{icu}/chartevents.csv.gz', header=true,
                   types={{'subject_id':'BIGINT','hadm_id':'BIGINT','itemid':'INTEGER',
                           'charttime':'TIMESTAMP','valuenum':'DOUBLE'}})
              WHERE itemid IN {SBP_ITEMS + DBP_ITEMS} AND valuenum IS NOT NULL
            ) TO '{bp_pq}' (FORMAT PARQUET)
        """)
    con.execute(f"""
        CREATE OR REPLACE VIEW icu_bp AS
        SELECT subject_id, hadm_id, charttime,
               CASE WHEN itemid IN {SBP_ITEMS} THEN 'sbp' ELSE 'dbp' END AS kind, valuenum
        FROM '{bp_pq}'
        WHERE (itemid IN {SBP_ITEMS} AND valuenum BETWEEN {PLAUSIBLE['sbp'][0]} AND {PLAUSIBLE['sbp'][1]})
           OR (itemid IN {DBP_ITEMS} AND valuenum BETWEEN {PLAUSIBLE['dbp'][0]} AND {PLAUSIBLE['dbp'][1]})
    """)

    # ---- OMR (outpatient): BMI and BP, values stored as text ----
    con.execute(f"""
        CREATE OR REPLACE TABLE omr AS
        SELECT subject_id, CAST(chartdate AS DATE) AS chartdate, result_name, result_value
        FROM read_csv('{h}/omr.csv.gz', header=true, all_varchar=true)
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW omr_bmi AS
        SELECT subject_id, chartdate, TRY_CAST(result_value AS DOUBLE) AS bmi
        FROM omr WHERE result_name IN ('BMI (kg/m2)', 'BMI')
          AND TRY_CAST(result_value AS DOUBLE) BETWEEN {PLAUSIBLE['bmi'][0]} AND {PLAUSIBLE['bmi'][1]}
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW omr_bp AS
        SELECT subject_id, chartdate,
               TRY_CAST(split_part(result_value, '/', 1) AS DOUBLE) AS sbp,
               TRY_CAST(split_part(result_value, '/', 2) AS DOUBLE) AS dbp
        FROM omr WHERE result_name LIKE 'Blood Pressure%'
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW omr_bp_ok AS SELECT * FROM omr_bp
        WHERE sbp BETWEEN {PLAUSIBLE['sbp'][0]} AND {PLAUSIBLE['sbp'][1]}
          AND dbp BETWEEN {PLAUSIBLE['dbp'][0]} AND {PLAUSIBLE['dbp'][1]}
    """)

    # ---- adult admissions ----
    con.execute(f"""
        CREATE OR REPLACE TABLE adm AS
        SELECT a.subject_id, a.hadm_id,
               CAST(a.admittime AS TIMESTAMP) AS admittime,
               CAST(a.dischtime AS TIMESTAMP) AS dischtime,
               p.anchor_age + (year(CAST(a.admittime AS TIMESTAMP)) - p.anchor_year) AS age,
               CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS sex_male,
               p.anchor_year_group,
               date_diff('minute', CAST(a.admittime AS TIMESTAMP), CAST(a.dischtime AS TIMESTAMP)) / 60.0 AS los_hours
        FROM read_csv_auto('{h}/admissions.csv.gz') a
        JOIN read_csv_auto('{h}/patients.csv.gz') p USING (subject_id)
    """)
    con.execute("DELETE FROM adm WHERE age < 18 OR los_hours <= 0")

    # ---- comorbidity flags (ICD-9 and ICD-10, codes stored without dots) ----
    con.execute(f"""
        CREATE OR REPLACE TABLE flags AS
        SELECT hadm_id,
          MAX(CASE WHEN (icd_version = 10 AND regexp_matches(icd_code, '^I1[0-356]'))
                     OR (icd_version = 9  AND regexp_matches(icd_code, '^40[1-5]')) THEN 1 ELSE 0 END) AS has_htn,
          MAX(CASE WHEN (icd_version = 10 AND icd_code LIKE 'E11%')
                     OR (icd_version = 9  AND regexp_matches(icd_code, '^250[0-9][02]$')) THEN 1 ELSE 0 END) AS has_dm,
          MAX(CASE WHEN (icd_version = 10 AND icd_code LIKE 'N18%')
                     OR (icd_version = 9  AND icd_code LIKE '585%') THEN 1 ELSE 0 END) AS has_ckd
        FROM read_csv('{h}/diagnoses_icd.csv.gz', header=true,
             types={{'hadm_id':'BIGINT','icd_code':'VARCHAR','icd_version':'INTEGER'}})
        GROUP BY hadm_id
    """)

    def first_in_window(name: str, extra: str = "") -> str:
        return f"""
            SELECT a.hadm_id, arg_min(l.valuenum, l.charttime) AS v
            FROM adm a JOIN labs l ON l.subject_id = a.subject_id
            WHERE l.itemid = {LAB_ITEMS[name]}
              AND l.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {BASELINE_HOURS} HOUR
              {extra}
            GROUP BY a.hadm_id"""

    def max_after_baseline(name: str) -> str:
        return f"""
            SELECT a.hadm_id, MAX(l.valuenum) AS v
            FROM adm a JOIN labs l ON l.subject_id = a.subject_id
            WHERE l.itemid = {LAB_ITEMS[name]}
              AND l.charttime > a.admittime + INTERVAL {BASELINE_HOURS} HOUR
              AND l.charttime <= a.dischtime
            GROUP BY a.hadm_id"""

    con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        WITH
          na   AS ({first_in_window('sodium')}),
          k    AS ({first_in_window('potassium')}),
          cr   AS ({first_in_window('creatinine')}),
          -- morning draw (04:00-08:00) is the fasting proxy: MIMIC has no fasting flag
          glu  AS ({first_in_window('glucose', "AND hour(l.charttime) BETWEEN 4 AND 7")}),
          a1c  AS (
            SELECT a.hadm_id, arg_max(l.valuenum, l.charttime) AS v
            FROM adm a JOIN labs l ON l.subject_id = a.subject_id
            WHERE l.itemid = {LAB_ITEMS['hba1c']}
              AND l.charttime BETWEEN a.admittime - INTERVAL {HBA1C_LOOKBACK_DAYS} DAY
                                  AND a.admittime + INTERVAL {BASELINE_HOURS} HOUR
            GROUP BY a.hadm_id),
          bmi  AS (
            SELECT a.hadm_id, arg_max(o.bmi, o.chartdate) AS v
            FROM adm a JOIN omr_bmi o ON o.subject_id = a.subject_id
            WHERE o.chartdate BETWEEN CAST(a.admittime AS DATE) - INTERVAL {OMR_LOOKBACK_DAYS} DAY
                                  AND CAST(a.admittime AS DATE)
            GROUP BY a.hadm_id),
          obp  AS (
            SELECT a.hadm_id, arg_max(o.sbp, o.chartdate) AS sbp, arg_max(o.dbp, o.chartdate) AS dbp
            FROM adm a JOIN omr_bp_ok o ON o.subject_id = a.subject_id
            WHERE o.chartdate BETWEEN CAST(a.admittime AS DATE) - INTERVAL {OMR_LOOKBACK_DAYS} DAY
                                  AND CAST(a.admittime AS DATE)
            GROUP BY a.hadm_id),
          ibp  AS (
            SELECT a.hadm_id,
                   arg_min(b.valuenum, b.charttime) FILTER (WHERE b.kind = 'sbp') AS sbp,
                   arg_min(b.valuenum, b.charttime) FILTER (WHERE b.kind = 'dbp') AS dbp
            FROM adm a JOIN icu_bp b ON b.hadm_id = a.hadm_id
            WHERE b.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {BASELINE_HOURS} HOUR
            GROUP BY a.hadm_id),
          k_post   AS ({max_after_baseline('potassium')}),
          cr_post  AS ({max_after_baseline('creatinine')}),
          glu_post AS ({max_after_baseline('glucose')}),
          fu_bp AS (
            SELECT a.hadm_id, AVG(o.sbp) AS sbp_mean, COUNT(*) AS n
            FROM adm a JOIN omr_bp_ok o ON o.subject_id = a.subject_id
            WHERE o.chartdate BETWEEN CAST(a.dischtime AS DATE) + INTERVAL {FOLLOWUP_BP_WINDOW_DAYS[0]} DAY
                                  AND CAST(a.dischtime AS DATE) + INTERVAL {FOLLOWUP_BP_WINDOW_DAYS[1]} DAY
            GROUP BY a.hadm_id)
        SELECT a.subject_id, a.hadm_id, a.anchor_year_group, a.los_hours,
               a.age, a.sex_male,
               COALESCE(f.has_htn, 0) AS has_htn, COALESCE(f.has_dm, 0) AS has_dm, COALESCE(f.has_ckd, 0) AS has_ckd,
               na.v AS serum_sodium, k.v AS serum_potassium, cr.v AS creatinine,
               a1c.v AS hba1c, glu.v AS fbs,
               COALESCE(obp.sbp, ibp.sbp) AS sbp, COALESCE(obp.dbp, ibp.dbp) AS dbp,
               CASE WHEN obp.sbp IS NOT NULL THEN 'omr' WHEN ibp.sbp IS NOT NULL THEN 'icu' END AS bp_source,
               bmi.v AS bmi,
               k_post.v AS k_post_max, cr_post.v AS cr_post_max, glu_post.v AS glu_post_max,
               fu_bp.sbp_mean AS followup_sbp_mean, fu_bp.n AS followup_bp_n
        FROM adm a
        LEFT JOIN flags f USING (hadm_id)
        LEFT JOIN na USING (hadm_id)  LEFT JOIN k USING (hadm_id)   LEFT JOIN cr USING (hadm_id)
        LEFT JOIN glu USING (hadm_id) LEFT JOIN a1c USING (hadm_id) LEFT JOIN bmi USING (hadm_id)
        LEFT JOIN obp USING (hadm_id) LEFT JOIN ibp USING (hadm_id)
        LEFT JOIN k_post USING (hadm_id) LEFT JOIN cr_post USING (hadm_id) LEFT JOIN glu_post USING (hadm_id)
        LEFT JOIN fu_bp USING (hadm_id)
    """)
    return con.execute("SELECT * FROM cohort").df()


def ckd_epi_2021(creatinine: pd.Series, age: pd.Series, sex_male: pd.Series) -> pd.Series:
    """CKD-EPI 2021 race-free creatinine equation (mL/min/1.73 m^2)."""
    female = sex_male == 0
    kappa = female.map({True: 0.7, False: 0.9})
    alpha = female.map({True: -0.241, False: -0.302})
    ratio = creatinine / kappa
    egfr = (142 * ratio.clip(upper=1) ** alpha * ratio.clip(lower=1) ** -1.200
            * 0.9938 ** age * female.map({True: 1.012, False: 1.0}))
    return egfr


def ordinal(values: pd.Series, moderate_above: float, high_above: float) -> pd.Series:
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    known = values.notna()
    out[known] = 0
    out[known & (values > moderate_above)] = 1
    out[known & (values > high_above)] = 2
    return out


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df["egfr"] = ckd_epi_2021(df["creatinine"], df["age"], df["sex_male"])
    df["covid_era"] = (df["anchor_year_group"] == "2020 - 2022").astype(int)

    inpatient_ok = df["los_hours"] >= MIN_LOS_HOURS_FOR_INPATIENT_OUTCOME

    df["potassium_sensitivity"] = ordinal(df["k_post_max"].where(inpatient_ok), 5.0, 5.5)
    df["carb_sensitivity"] = ordinal(df["glu_post_max"].where(inpatient_ok), 180, 250)
    df["sodium_sensitivity"] = ordinal(df["followup_sbp_mean"], 140, 160)

    # KDIGO AKI: stage 1 = rise >= 0.3 mg/dL or >= 1.5x baseline; stage 2+ = >= 2x baseline
    cr0, cr1 = df["creatinine"], df["cr_post_max"].where(inpatient_ok)
    known = cr0.notna() & cr1.notna()
    renal = pd.Series(pd.NA, index=df.index, dtype="Int64")
    renal[known] = 0
    renal[known & (((cr1 - cr0) >= 0.3) | (cr1 / cr0 >= 1.5))] = 1
    renal[known & (cr1 / cr0 >= 2.0)] = 2
    df["protein_restriction"] = renal
    return df


def suppress(n: int):
    return n if n >= SMALL_CELL else f"<{SMALL_CELL}"


def cohort_card(df: pd.DataFrame, tag: str, mimic: Path) -> dict:
    flags = df[["has_htn", "has_dm", "has_ckd"]]
    combos = {
        "none": (flags.sum(axis=1) == 0),
        "htn_only": (df.has_htn == 1) & (df.has_dm == 0) & (df.has_ckd == 0),
        "dm_only": (df.has_htn == 0) & (df.has_dm == 1) & (df.has_ckd == 0),
        "ckd_only": (df.has_htn == 0) & (df.has_dm == 0) & (df.has_ckd == 1),
        "htn_dm": (df.has_htn == 1) & (df.has_dm == 1) & (df.has_ckd == 0),
        "htn_ckd": (df.has_htn == 1) & (df.has_dm == 0) & (df.has_ckd == 1),
        "dm_ckd": (df.has_htn == 0) & (df.has_dm == 1) & (df.has_ckd == 1),
        "htn_dm_ckd": (flags.sum(axis=1) == 3),
    }
    card = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": tag,
        "source_path": str(mimic),
        "n_admissions": int(len(df)),
        "n_patients": int(df.subject_id.nunique()),
        "comorbidity_combinations": {k: suppress(int(v.sum())) for k, v in combos.items()},
        "feature_missingness_pct": {f: round(float(df[f].isna().mean() * 100), 1) for f in FEATURES},
        "bp_source_pct": {str(k): round(float(v * 100), 1)
                          for k, v in df["bp_source"].value_counts(normalize=True, dropna=False).items()},
        "targets": {},
        "covid_era_pct": round(float(df["covid_era"].mean() * 100), 1),
        "definitions": {
            "baseline_window_hours": BASELINE_HOURS,
            "min_los_hours_for_inpatient_outcomes": MIN_LOS_HOURS_FOR_INPATIENT_OUTCOME,
            "hba1c_lookback_days": HBA1C_LOOKBACK_DAYS,
            "omr_lookback_days": OMR_LOOKBACK_DAYS,
            "followup_bp_window_days": FOLLOWUP_BP_WINDOW_DAYS,
            "fbs": "first morning (04:00-07:59) plasma glucose within the baseline window; fasting proxy",
            "egfr": "CKD-EPI 2021 race-free creatinine equation",
            "potassium_sensitivity": "max serum K after baseline: <=5.0 low, 5.1-5.5 moderate, >5.5 high",
            "protein_restriction": "KDIGO AKI vs baseline creatinine: stage1 moderate, stage2+ high",
            "carb_sensitivity": "max glucose after baseline: <=180 low, 181-250 moderate, >250 high",
            "sodium_sensitivity": "mean outpatient SBP 30-365 d post-discharge: <=140 low, 141-160 moderate, >160 high",
            "plausible_ranges": PLAUSIBLE,
            "small_cell_suppression": f"counts below {SMALL_CELL} reported as '<{SMALL_CELL}'",
        },
    }
    for t in TARGETS:
        y = df[t]
        card["targets"][t] = {
            "n_labelled": int(y.notna().sum()),
            "class_counts": {lbl: suppress(int((y == i).sum())) for i, lbl in enumerate(["low", "moderate", "high"])},
        }
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-dir", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "derived")
    ap.add_argument("--memory-limit", default="10GB")
    ap.add_argument("--temp-dir", type=Path, help="DuckDB spill directory (default: inside the cache dir)")
    args = ap.parse_args()

    import duckdb  # extraction-only dependency; FEATURES/TARGETS import without it

    cache = args.out / f"cache_{args.tag}"
    cache.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET temp_directory='{args.temp_dir or cache / 'tmp'}'")
    con.execute("SET enable_progress_bar=false")

    print(f"Extracting cohort from {args.mimic_dir} ...")
    df = add_derived(build(con, args.mimic_dir.resolve(), cache))

    out_pq = args.out / f"cohort_{args.tag}.parquet"
    df.to_parquet(out_pq, index=False)

    card = cohort_card(df, args.tag, args.mimic_dir)
    card_path = args.out / f"cohort_card_{args.tag}.json"
    card_path.write_text(json.dumps(card, indent=2))

    print(f"  wrote {out_pq.name} and {card_path.name}")
    print(json.dumps({k: card[k] for k in ("n_admissions", "n_patients", "comorbidity_combinations",
                                            "feature_missingness_pct", "targets", "covid_era_pct")}, indent=2))


if __name__ == "__main__":
    main()
