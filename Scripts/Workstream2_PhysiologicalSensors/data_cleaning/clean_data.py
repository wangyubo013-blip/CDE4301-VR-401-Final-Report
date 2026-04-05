"""
MIST Experiment Data Cleaning Script
=====================================

A three-stage pipeline for cleaning physiological sensor data (HR via PPG, GSR)
collected during a MIST (Montreal Imaging Stress Task) stress induction experiment.

Input (per participant):
    - Baseline sensor CSV:  timestamp, HR, GSR recorded during resting baseline
    - MIST sensor CSV:      timestamp, HR, GSR recorded during MIST task
    - MIST log CSV:         event log from MIST program with timestamps

Output (per participant):
    - {name}_mist_cleaned.csv:  cleaned sensor data, trimmed to experiment window only

Pipeline:
    Stage 1 — Temporal trimming
        Extract EXPERIMENT_START and EXPERIMENT_END timestamps from MIST log.
        Discard all sensor rows outside this window.

    Stage 2 — HR artifact removal (threshold + shoulder expansion)
        Mark HR=0 as invalid (sensor not ready).
        Flag all HR > 150 bpm as artifact core (Altini et al., 2017).
        Expand each artifact cluster forward/backward to include ramp shoulders
        (samples exceeding local normal level + 15 bpm).
        Set flagged samples to NaN (no interpolation — preserves data integrity).

    Stage 3 — GSR adaptive cleaning (median filter + isolated spike removal)
        Diagnose noise type via sign-change rate of consecutive differences.
        If sign_change_rate > 0.55 and relative jump amplitude > 5%:
            Apply median filter (kernel=5) to suppress high-frequency zigzag
            while preserving slow tonic drift (BIOPAC EDA guidelines; iMotions).
        Then scan for isolated spikes: any sample deviating > 4σ from its
        4-neighbor median is replaced with that median (Hernandez et al., 2018).
        Both strategies may apply sequentially to the same signal.

References:
    [1] Altini, M. (2017). Artifact removal for PPG-based HRV analysis.
        Based on Plews et al., Int J Sports Physiol Perform.
    [2] Matsumura, K. et al. (2014). Motion artifact cancellation and outlier
        rejection for clip-type PPG-based heart rate sensor. PubMed: 26736684.
    [3] BIOPAC Systems. EDA Data Analysis & Correction FAQ.
        https://www.biopac.com/eda-faq-data/
    [4] iMotions. EDA Peak Detection signal processing flow.
        https://imotions.com/blog/learning/research-fundamentals/eda-peak-detection/
    [5] Hernandez, J. et al. (2018). Efficient wavelet-based artifact removal
        for electrodermal activity in real-world applications.
        Biomedical Signal Processing and Control, 42, 45-52.

Usage:
    python clean_data.py <subject>  (e.g., python clean_data.py A)

    Expects three files in the current directory:
        <subject>_baseline.csv
        <subject>_mist.csv
        <subject>_mist_log.csv

    Outputs:
        <subject>_mist_cleaned.csv
"""

import pandas as pd
import numpy as np
from scipy.signal import medfilt
from pathlib import Path

OUTPUT_DIR = Path('.')


# ============================================================
# I/O helpers
# ============================================================
def parse_sensor_csv(path):
    """Read sensor CSV, handling BOM and leading-quote timestamps."""
    df = pd.read_csv(path, encoding='utf-8-sig')
    ts_col = df.columns[0]
    df[ts_col] = df[ts_col].astype(str).str.lstrip("'")
    df[ts_col] = pd.to_datetime(df[ts_col])
    df.rename(columns={ts_col: 'timestamp'}, inplace=True)
    return df


def parse_mist_log(path):
    """Extract experiment start/end and per-level boundaries from MIST log."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    exp_start = exp_end = None
    levels = {}

    for _, row in df.iterrows():
        evt = row['event']
        if evt == 'EXPERIMENT_START':
            exp_start = row['timestamp']
        elif evt == 'EXPERIMENT_END':
            exp_end = row['timestamp']
        elif evt == 'LEVEL_START':
            levels[int(row['level'])] = {'start': row['timestamp']}
        elif evt == 'LEVEL_END':
            lvl = int(row['level'])
            if lvl in levels:
                levels[lvl]['end'] = row['timestamp']

    return exp_start, exp_end, levels


def save_cleaned_csv(df, path):
    """Save cleaned data in original format (leading-quote timestamp)."""
    out = df.copy()
    out['Timestamp_Beijing'] = "'" + out['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    out['HR'] = out['HR'].fillna(0).astype(int)
    out['GSR'] = out['GSR'].round().astype(int)
    out[['Timestamp_Beijing', 'HR', 'GSR']].to_csv(path, index=False, encoding='utf-8-sig')


# ============================================================
# Stage 1: Temporal trimming
# ============================================================
def trim_to_experiment(df, exp_start, exp_end):
    """Keep only rows within [EXPERIMENT_START, EXPERIMENT_END]."""
    n_before = len(df)
    df = df[(df['timestamp'] >= exp_start) & (df['timestamp'] <= exp_end)].reset_index(drop=True)
    n_after = len(df)
    return df, n_before, n_after


# ============================================================
# Stage 2: HR artifact removal
# ============================================================
HR_ARTIFACT_THRESHOLD = 150       # bpm — values above this are artifact cores
HR_SHOULDER_MARGIN    = 15        # bpm — ramp shoulder detection margin above local normal
HR_SHOULDER_SCAN      = 30        # samples — max distance to scan for shoulders
HR_CLUSTER_GAP        = 5         # samples — gap to separate artifact clusters

def clean_hr(hr_series):
    """
    PPG motion artifact removal via threshold filtering + shoulder expansion.

    1. HR=0 → NaN (sensor not ready)
    2. HR > 150 bpm → artifact core
    3. Expand cores outward to include ramp-up/down shoulders
    4. All flagged samples → NaN (not interpolated)

    Returns: (cleaned_series, stats_dict)
    """
    hr = hr_series.copy().astype(float)
    original_valid = int((hr > 0).sum())

    # Step 1: sensor-not-ready
    hr[hr == 0] = np.nan

    # Step 2: find artifact cores
    artifact_mask = hr > HR_ARTIFACT_THRESHOLD
    if artifact_mask.sum() == 0:
        return hr, {'original_valid': original_valid, 'artifacts_removed': 0,
                     'final_valid': int(hr.notna().sum()), 'pct_removed': 0.0}

    # Group consecutive artifact indices into clusters
    indices = np.where(artifact_mask)[0]
    clusters = []
    cl_start = cl_prev = indices[0]
    for idx in indices[1:]:
        if idx - cl_prev > HR_CLUSTER_GAP:
            clusters.append((cl_start, cl_prev))
            cl_start = idx
        cl_prev = idx
    clusters.append((cl_start, cl_prev))

    # Step 3: expand each cluster to include ramp shoulders
    expanded = artifact_mask.copy()
    for cs, ce in clusters:
        # Estimate local normal level from pre-artifact region
        pre_region = hr[max(0, cs - 20):cs]
        pre_clean = pre_region[pre_region.notna() & (pre_region < HR_ARTIFACT_THRESHOLD)]
        normal_level = pre_clean.median() if len(pre_clean) > 0 else 90
        threshold = normal_level + HR_SHOULDER_MARGIN

        # Scan backward for ramp-up shoulder
        i = cs - 1
        while i >= max(0, cs - HR_SHOULDER_SCAN):
            val = hr.iloc[i]
            if pd.isna(val) or val <= threshold:
                break
            expanded.iloc[i] = True
            i -= 1

        # Scan forward for ramp-down shoulder
        i = ce + 1
        while i < min(len(hr), ce + HR_SHOULDER_SCAN):
            val = hr.iloc[i]
            if pd.isna(val) or val <= threshold:
                break
            expanded.iloc[i] = True
            i += 1

    # Step 4: mark all flagged samples as invalid
    n_removed = int(expanded.sum())
    hr[expanded] = np.nan

    return hr, {
        'original_valid': original_valid,
        'artifacts_removed': n_removed,
        'final_valid': int(hr.notna().sum()),
        'pct_removed': round(n_removed / original_valid * 100, 2)
    }


# ============================================================
# Stage 3: GSR adaptive cleaning
# ============================================================
GSR_ZIGZAG_SCR_THRESHOLD  = 0.55   # sign-change rate above which zigzag is detected
GSR_ZIGZAG_JR_THRESHOLD   = 0.05   # relative jump ratio threshold
GSR_MEDIAN_KERNEL          = 5      # median filter window size
GSR_SPIKE_NEIGHBOR_RANGE   = 2      # samples on each side for neighbor calculation
GSR_SPIKE_SIGMA_THRESHOLD  = 4      # number of σ for spike detection
GSR_SPIKE_MIN_STD          = 3      # minimum std to avoid false positives in flat regions

def clean_gsr(gsr_series):
    """
    Adaptive GSR cleaning: median filter for zigzag + isolated spike removal.

    Diagnosis phase:
        Compute sign-change rate (SCR) of consecutive differences.
        Compute relative jump ratio (mean |diff| / median signal).

    If SCR > 0.55 and JR > 0.05:
        Apply median filter (k=5) to suppress high-frequency oscillation
        while preserving slow tonic drift.

    Then for all signals:
        Scan for isolated spikes deviating > 4σ from 4-neighbor median.
        Replace with neighbor median.

    Returns: (cleaned_series, stats_dict)
    """
    gsr = gsr_series.copy().astype(float)
    gsr[gsr == 0] = np.nan
    original_valid = int(gsr.notna().sum())

    valid_vals = gsr[gsr.notna()].values
    if len(valid_vals) < 10:
        return gsr, {'method': 'insufficient_data', 'n_modified': 0, 'pct_modified': 0.0}

    # Diagnosis: characterize noise type
    diffs = np.diff(valid_vals)
    sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
    scr = sign_changes / max(1, len(diffs) - 1)
    jr = np.mean(np.abs(diffs)) / max(1, np.median(valid_vals))

    method_parts = []
    n_modified = 0

    # Sub-stage 3a: median filter for high-frequency zigzag
    if scr > GSR_ZIGZAG_SCR_THRESHOLD and jr > GSR_ZIGZAG_JR_THRESHOLD:
        valid_mask = gsr.notna()
        valid_idx = np.where(valid_mask)[0]
        raw_vals = gsr[valid_mask].values
        filtered = medfilt(raw_vals, kernel_size=GSR_MEDIAN_KERNEL)
        n_mod = int(np.sum(np.abs(filtered - raw_vals) > 0.5))
        gsr.iloc[valid_idx] = filtered
        n_modified += n_mod
        method_parts.append(
            f'median_filter(k={GSR_MEDIAN_KERNEL}, scr={scr:.2f}, jr={jr:.3f})'
        )

    # Sub-stage 3b: isolated spike removal
    valid_mask = gsr.notna()
    vals = gsr[valid_mask].values
    idx = np.where(valid_mask)[0]
    k = GSR_SPIKE_NEIGHBOR_RANGE
    spike_count = 0

    for i in range(k, len(vals) - k):
        neighbors = np.concatenate([vals[i-k:i], vals[i+1:i+k+1]])
        med = np.median(neighbors)
        std = max(np.std(neighbors), GSR_SPIKE_MIN_STD)

        if abs(vals[i] - med) > GSR_SPIKE_SIGMA_THRESHOLD * std:
            gsr.iloc[idx[i]] = med
            spike_count += 1

    if spike_count > 0:
        n_modified += spike_count
        method_parts.append(f'spike_removal(n={spike_count}, threshold={GSR_SPIKE_SIGMA_THRESHOLD}σ)')

    method = ' + '.join(method_parts) if method_parts else 'no_cleaning_needed'

    return gsr, {
        'method': method,
        'sign_change_rate': round(scr, 3),
        'jump_ratio': round(jr, 4),
        'n_modified': n_modified,
        'pct_modified': round(n_modified / original_valid * 100, 2) if original_valid else 0.0
    }


# ============================================================
# Main pipeline
# ============================================================
def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python clean_data.py <subject>")
        print("Example: python clean_data.py A")
        sys.exit(1)

    name = sys.argv[1]
    baseline_path = OUTPUT_DIR / f'{name}_baseline.csv'
    mist_path     = OUTPUT_DIR / f'{name}_mist.csv'
    log_path      = OUTPUT_DIR / f'{name}_mist_log.csv'

    for p in [baseline_path, mist_path, log_path]:
        if not p.exists():
            print(f"Error: file not found: {p}")
            sys.exit(1)

    print("=" * 70)
    print(f"  MIST Data Cleaning — Subject {name}")
    print(f"  Stages: Temporal Trim → HR Artifact Removal → GSR Adaptive Clean")
    print("=" * 70)

    # ── Stage 1: Temporal trimming ──
    exp_start, exp_end, levels = parse_mist_log(log_path)
    df = parse_sensor_csv(mist_path)
    df, n_before, n_after = trim_to_experiment(df, exp_start, exp_end)

    print(f"\n  [Stage 1] Temporal trimming")
    print(f"    Window:  {exp_start} → {exp_end}")
    print(f"    Rows:    {n_before} → {n_after}  (removed {n_before - n_after})")

    # ── Stage 2: HR artifact removal ──
    hr_cleaned, hr_stats = clean_hr(df['HR'])
    df['HR'] = hr_cleaned

    print(f"\n  [Stage 2] HR artifact removal  (threshold={HR_ARTIFACT_THRESHOLD}bpm)")
    print(f"    Valid samples:     {hr_stats['original_valid']}")
    print(f"    Artifacts removed: {hr_stats['artifacts_removed']}  ({hr_stats['pct_removed']}%)")
    print(f"    Final valid:       {hr_stats['final_valid']}")

    # ── Stage 3: GSR adaptive cleaning ──
    gsr_cleaned, gsr_stats = clean_gsr(df['GSR'])
    df['GSR'] = gsr_cleaned

    print(f"\n  [Stage 3] GSR adaptive cleaning")
    print(f"    Noise diagnosis:  scr={gsr_stats.get('sign_change_rate','—')}, "
          f"jr={gsr_stats.get('jump_ratio','—')}")
    print(f"    Method applied:   {gsr_stats['method']}")
    print(f"    Samples modified: {gsr_stats['n_modified']}  ({gsr_stats['pct_modified']}%)")

    # ── Save ──
    out_path = OUTPUT_DIR / f'{name}_mist_cleaned.csv'
    save_cleaned_csv(df, out_path)

    print(f"\n  ✓ Saved: {out_path.name}  ({n_after} rows)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
