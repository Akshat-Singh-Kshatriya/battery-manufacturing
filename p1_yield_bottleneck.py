"""
Project 1: Cell line yield & bottleneck analytics
Stack: PySpark (stage yield/SPC), Python ML (batch-failure model), SQL (Spark SQL), Power BI export
Data: synthetic MES-style batch data (replace read step with your Azure Data Lake path for real data)
"""
import numpy as np, pandas as pd
from pyspark.sql import SparkSession, functions as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

OUT = os.environ.get("OUT", "p1_out"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ---------- 1. Synthetic MES data: 6 stages x 3 lines x 400 batches ----------
stages = ["mixing", "coating", "calendering", "assembly", "formation", "aging"]
base_fail = {"mixing": .015, "coating": .05, "calendering": .02, "assembly": .03, "formation": .035, "aging": .01}
rows = []
for s in stages:
    for line in ["L1", "L2", "L3"]:
        n = 400
        moisture = rng.normal(180, 40, n)            # ppm
        thick_dev = rng.normal(0, 2.0, n)            # um deviation from target
        temp = rng.normal(95, 4, n)                  # drying temp C
        press = rng.normal(250, 15, n)               # calender pressure
        cycle = rng.normal(42, 5, n)                 # minutes
        # failure probability driven by moisture, thickness and temp for coating/formation
        z = (np.log(base_fail[s] / (1 - base_fail[s]))
             + (0.03 * (moisture - 180) if s in ("coating", "formation") else 0)
             + (0.5 * np.abs(thick_dev) if s == "coating" else 0)
             + (0.15 * np.abs(temp - 95) if s == "coating" else 0)
             - 1.6)
        p = 1 / (1 + np.exp(-z))
        fail = rng.random(n) < p
        units = 1000
        rows.append(pd.DataFrame(dict(stage=s, line=line, batch=range(n), moisture_ppm=moisture,
                                      thick_dev_um=thick_dev, dry_temp_c=temp, calender_press=press,
                                      cycle_min=cycle, units_in=units,
                                      units_scrap=np.where(fail, rng.integers(150, 400, n), rng.integers(0, 8, n)),
                                      failed=fail.astype(int))))
mes = pd.concat(rows, ignore_index=True)

# ---------- 2. PySpark: stage yield, rolled throughput yield, bottleneck ----------
spark = SparkSession.builder.master("local[2]").appName("yield").config("spark.ui.enabled", "false").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sdf = spark.createDataFrame(mes); sdf.createOrReplaceTempView("mes")

stage_yield = spark.sql("""
    SELECT stage,
           ROUND(1 - SUM(units_scrap) / SUM(units_in), 4) AS stage_yield,
           ROUND(AVG(cycle_min), 2)                        AS avg_cycle_min,
           ROUND(SUM(units_scrap) / 1000.0, 1)             AS scrap_k_units
    FROM mes GROUP BY stage
""").toPandas().set_index("stage").loc[stages]
rty = stage_yield["stage_yield"].prod()
stage_yield["yield_loss_pct_pts"] = (1 - stage_yield["stage_yield"]) * 100
bottleneck = stage_yield["yield_loss_pct_pts"].idxmax()
stage_yield.to_csv(f"{OUT}/stage_yield.csv")

# ---------- 3. SPC: Cpk and out-of-control rate per stage/parameter ----------
def cpk(x, lsl, usl):
    m, s = x.mean(), x.std()
    return min((usl - m) / (3 * s), (m - lsl) / (3 * s))
coat = mes[mes.stage == "coating"]
spc = pd.DataFrame({
    "parameter": ["moisture_ppm", "thick_dev_um", "dry_temp_c"],
    "cpk": [cpk(coat.moisture_ppm, 60, 300), cpk(coat.thick_dev_um, -6, 6), cpk(coat.dry_temp_c, 85, 105)],
    "out_of_spec_pct": [((coat.moisture_ppm > 300) | (coat.moisture_ppm < 60)).mean() * 100,
                        (coat.thick_dev_um.abs() > 6).mean() * 100,
                        ((coat.dry_temp_c > 105) | (coat.dry_temp_c < 85)).mean() * 100]}).round(2)
spc.to_csv(f"{OUT}/spc_coating.csv", index=False)

# ---------- 4. ML: predict batch failure (coating) ----------
feats = ["moisture_ppm", "thick_dev_um", "dry_temp_c", "calender_press", "cycle_min"]
X = coat[feats].assign(thick_dev_um=coat.thick_dev_um.abs(), dry_temp_c=(coat.dry_temp_c - 95).abs())
y = coat["failed"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=1, stratify=y)
clf = GradientBoostingClassifier(random_state=1).fit(Xtr, ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
imp = pd.Series(clf.feature_importances_, feats).sort_values(ascending=False)

# ---------- 5. Policy simulation on held-out batches ----------
# Hold/reprocess the top 15% highest-risk batches before they run (cost: reprocess = 25% of a failed batch's scrap)
risk = clf.predict_proba(Xte)[:, 1]
thr = np.quantile(risk, .85)
flag = risk >= thr
fails_caught = (yte.values[flag]).sum() / yte.sum()
scrap_units = coat.loc[Xte.index, "units_scrap"].values
scrap_base = scrap_units.sum()
scrap_after = scrap_units[~flag].sum() + 0.25 * scrap_units[flag].sum()
scrap_cut = 1 - scrap_after / scrap_base
coat_yield_base = 1 - scrap_base / (len(Xte) * 1000)
coat_yield_after = 1 - scrap_after / (len(Xte) * 1000)
rty_after = rty / stage_yield.loc["coating", "stage_yield"] * coat_yield_after

pd.DataFrame({"metric": ["RTY baseline", "RTY after policy", "Coating scrap reduction %", "Failed batches caught %", "Model AUC"],
              "value": [round(rty * 100, 2), round(rty_after * 100, 2), round(scrap_cut * 100, 1),
                        round(fails_caught * 100, 1), round(auc, 3)]}).to_csv(f"{OUT}/kpi_summary.csv", index=False)
mes.to_csv(f"{OUT}/mes_batches.csv", index=False)   # load into Power BI

print("Stage yield:\n", stage_yield[["stage_yield", "yield_loss_pct_pts"]])
print(f"\nBottleneck stage: {bottleneck}")
print(f"Rolled throughput yield baseline: {rty*100:.2f}%  -> after policy: {rty_after*100:.2f}%")
print("\nSPC coating:\n", spc)
print(f"\nModel AUC: {auc:.3f}"); print("Feature importance:\n", imp.round(3))
print(f"Batches flagged: 15% | failures caught: {fails_caught*100:.1f}% | coating scrap cut: {scrap_cut*100:.1f}%")
spark.stop()
