#!/usr/bin/env python3
"""
Quick data scan for candidate predictors of depth / vertical velocity.

Upgraded version:
 - Automatically uses ALL numeric columns (except timestamp + depth)
 - Skips constant / all-NaN columns (no warnings)
 - Safe correlation wrapper removes ConstantInputWarning
 - Prints ranked correlations
 - Heatmap, lag plots, Granger tests remain unchanged

Dependencies:
    pip install numpy pandas matplotlib seaborn scipy statsmodels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate
import os, math

CSV = r"C:\Users\suhan\Downloads\alvin_all.csv"
TIME_COL = "timestamp"
DEPTH_COL = "depth_m"

# params
RESAMPLE_DT = 1.0
MAX_LAG_SECONDS = 60
GRANGER_MAX_LAG = 10


# ---------- helpers ----------
def parse_time(df):
    if TIME_COL in df.columns:
        col = df[TIME_COL]
        if np.issubdtype(col.dtype, np.number):
            t = col.to_numpy().astype(float)
            return t - t.min()
        else:
            ts = pd.to_datetime(col, errors="coerce")
            return (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    raise ValueError("timestamp column missing")


def resample_regular(t, arr, t_grid):
    return np.interp(t_grid, t, arr)


def compute_vz(z, t):
    return np.gradient(z, t)


### UPDATED — safe_corr to avoid ConstantInputWarnings
def safe_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan

    x = x[mask]
    y = y[mask]

    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan

    try:
        pear = pearsonr(x, y)[0]
    except Exception:
        pear = np.nan

    try:
        spear = spearmanr(x, y)[0]
    except Exception:
        spear = np.nan

    return pear, spear


def lagged_corr(x, y, t_grid, max_lag_sec=60, step=1.0):
    dt = t_grid[1] - t_grid[0]
    maxlag = int(round(max_lag_sec / dt))
    n = len(t_grid)

    lags = np.arange(-maxlag, maxlag + 1) * dt
    cors = []

    x0 = x - np.nanmean(x)
    y0 = y - np.nanmean(y)

    for lag in range(-maxlag, maxlag + 1):
        if lag < 0:
            xa = x0[:n + lag]
            ya = y0[-lag:]
        elif lag > 0:
            xa = x0[lag:]
            ya = y0[:n - lag]
        else:
            xa = x0
            ya = y0

        if len(xa) < 10:
            cors.append(0.0)
            continue

        denom = np.nanstd(xa) * np.nanstd(ya)
        if denom == 0:
            cors.append(0.0)
        else:
            cors.append(np.nanmean(xa * ya) / denom)

    return lags, np.array(cors)


# ---------- main ----------
df = pd.read_csv(CSV, low_memory=False)
t_raw = parse_time(df)

if DEPTH_COL not in df.columns:
    raise SystemExit(f"{DEPTH_COL} missing")

z_raw = pd.to_numeric(df[DEPTH_COL], errors="coerce").to_numpy(float)

# ---------- UPDATED: auto-detect numeric columns ----------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# remove timestamp + depth from candidates
auto_candidates = [c for c in numeric_cols if c not in [TIME_COL, DEPTH_COL]]

# remove constant columns
CANDIDATES = []
for c in auto_candidates:
    arr = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    arr_finite = arr[np.isfinite(arr)]
    if len(arr_finite) < 10:
        continue
    if np.all(arr_finite == arr_finite[0]):
        continue
    CANDIDATES.append(c)

print("Using candidate columns:", CANDIDATES)


# ---------- resample ----------
t0, t1 = float(np.nanmin(t_raw)), float(np.nanmax(t_raw))
t_grid = np.arange(t0, t1 + 1e-8, RESAMPLE_DT)

z = resample_regular(t_raw, z_raw, t_grid)
vz = compute_vz(z, t_grid)

# candidate signals
X = {}
for c in CANDIDATES:
    arr = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    X[c] = resample_regular(t_raw, arr, t_grid)

# ---------- correlations ----------
rows = []
for c, arr in X.items():
    pear_z, spear_z = safe_corr(arr, z)
    pear_vz, spear_vz = safe_corr(arr, vz)
    rows.append((c, pear_z, spear_z, pear_vz, spear_vz))

res_df = pd.DataFrame(rows, columns=["col", "pear_z", "spear_z", "pear_vz", "spear_vz"])
res_df = res_df.sort_values(by="pear_vz", key=lambda s: np.abs(s), ascending=False)

print("\nCorrelation summary (top abs corr |vz|):")
print(res_df.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

# ---------- heatmap ----------
pear_matrix = res_df[["pear_z", "pear_vz"]].to_numpy()
labels = res_df["col"].tolist()

plt.figure(figsize=(6, max(2, len(labels)*0.35)))
sns.heatmap(
    pear_matrix,
    annot=True,
    yticklabels=labels,
    xticklabels=["pear_z", "pear_vz"],
    cmap="coolwarm",
    center=0,
)
plt.title("Pearson corr with depth & vz")
plt.tight_layout()
plt.savefig("pearson_heatmap.png", dpi=200)
print("Saved pearson_heatmap.png")

# ---------- lag scans ----------
os.makedirs("lag_plots", exist_ok=True)
top_candidates = res_df["col"].tolist()[:4]

for c in top_candidates:
    arr = X[c]
    lags, cors = lagged_corr(arr, vz, t_grid, MAX_LAG_SECONDS)
    peak = np.nanargmax(np.abs(cors))
    print(f"\n{c}: peak lag = {lags[peak]:.1f}s, corr = {cors[peak]:.3f}")

    plt.figure(figsize=(8, 3))
    plt.plot(lags, cors)
    plt.axvline(0, color="k", linestyle="--")
    plt.grid(True)
    plt.xlabel("lag (s) (neg → candidate leads vz)")
    plt.ylabel("corr")
    plt.title(f"{c} vs vz (peak={cors[peak]:.3f} @ {lags[peak]:.1f}s)")
    plt.tight_layout()
    plt.savefig(f"lag_plots/lag_{c}.png", dpi=200)
    print(f"Saved lag_plots/lag_{c}.png")

# ---------- Granger ----------
try:
    import statsmodels.tsa.stattools as ts
    print("\nRunning Granger causality (candidate → z)...")

    for c in top_candidates:
        series = np.vstack([z, X[c]]).T
        mask = ~np.isnan(series).any(axis=1)
        series = series[mask]

        if len(series) < 50:
            print(f"{c}: too few points")
            continue

        maxlag = min(GRANGER_MAX_LAG, len(series)//10)
        gr = ts.grangercausalitytests(series, maxlag=maxlag, verbose=False)

        pvals = [gr[k][0]["ssr_ftest"][1] for k in sorted(gr.keys())]
        best = int(np.argmin(pvals)) + 1
        print(f"{c}: best lag = {best}, p = {pvals[best-1]:.3e}")

except Exception as e:
    print("Granger failed or statsmodels missing:", e)

print("\nDone. Inspect pearson_heatmap.png and lag_plots/ .")
