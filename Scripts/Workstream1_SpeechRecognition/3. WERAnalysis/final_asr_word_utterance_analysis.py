#!/usr/bin/env python3
"""
final_asr_word_utterance_analysis.py

Purpose
-------
Read a final WER CSV and export:
1. model-level S/D/I + corpus WER summary
2. stage/comparison summary, including Azure -> Final
3. utterance-level WER/S/D/I details
4. word-level inspection by model
5. word improvement lists for each stage/comparison
6. aviation vs non-aviation category summary

Expected CSV columns
--------------------
Reference column:
    Real

Hypothesis/model columns:
    Azure
    Raw prediction
    LoRA prediction
    LoRA+Prompt prediction

The first column may contain either:
- utterance indices, with marker rows like "Sample 03"
- or some other ID column

This script will try to create an utterance ID such as 03-01, 03-02, etc.

Usage
-----
python final_asr_word_utterance_analysis.py -i WER.csv -o analysis_out

Optional:
python final_asr_word_utterance_analysis.py -i WER.csv -o analysis_out --top-n 50
"""

import argparse
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import pandas as pd

# ----------------------------
# Editable aviation term list
# ----------------------------
AVIATION_TERMS = {
    # avionics / modes
    "lnav", "vnav", "fms", "cdu", "fcu", "ils", "vor", "loc", "glideslope",
    "autopilot", "autothrottle", "thrust", "speedbrake", "spoiler", "flaps",
    "checklist", "landing", "gear", "runway", "tower", "downwind", "captain",
    "roger", "busan", "singapore", "route", "direct", "activate", "activated",
    "available", "green", "light", "clearance", "altitude", "heading",
    "exterior", "below", "line", "obtained", "obtain",

    # common tokenized numbers / IDs found in this dataset
    "18r", "ab18r", "616", "700", "30", "2",

    # likely phraseology words that are operationally meaningful
    "pitch", "high", "whites", "check",
}

PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
SAMPLE_MARKER_RE = re.compile(r"^\s*sample\s+(\d+)\s*$", flags=re.IGNORECASE)


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def normalize_text(s: str) -> str:
    s = safe_text(s).lower()
    s = PUNCT_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []


def align_ops(ref: List[str], hyp: List[str]) -> List[Tuple[str, str, str, int, int]]:
    """
    Return alignment ops with indices:
      (op, ref_word, hyp_word, ref_idx, hyp_idx)

    ref_idx/hyp_idx are -1 when not applicable.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                sub_cost = dp[i - 1][j - 1]
                sub_op = "OK"
            else:
                sub_cost = dp[i - 1][j - 1] + 1
                sub_op = "S"

            del_cost = dp[i - 1][j] + 1
            ins_cost = dp[i][j - 1] + 1

            best = sub_cost
            op = sub_op
            if del_cost < best:
                best = del_cost
                op = "D"
            if ins_cost < best:
                best = ins_cost
                op = "I"

            dp[i][j] = best
            bt[i][j] = op

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in ("OK", "S"):
            ops.append((op, ref[i - 1], hyp[j - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == "D":
            ops.append(("D", ref[i - 1], "", i - 1, -1))
            i -= 1
        elif op == "I":
            ops.append(("I", "", hyp[j - 1], -1, j - 1))
            j -= 1
        else:
            break

    ops.reverse()
    return ops


def parse_csv_with_sample_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cleaned dataframe with:
      utt_id, ref, model columns...
    Marker rows like 'Sample 03' in the first column are removed.
    """
    first_col = df.columns[0]
    current_sample = None
    current_counter = 0
    rows = []

    for _, row in df.iterrows():
        first_val = safe_text(row[first_col]).strip()
        marker = SAMPLE_MARKER_RE.match(first_val)
        if marker:
            current_sample = int(marker.group(1))
            current_counter = 0
            continue

        current_counter += 1
        if current_sample is None:
            # fall back to sample 1 if header itself encoded sample and no marker row yet
            hdr_match = SAMPLE_MARKER_RE.match(safe_text(first_col).strip())
            current_sample = int(hdr_match.group(1)) if hdr_match else 1

        utt_id = f"{current_sample:02d}-{current_counter:02d}"
        new_row = row.copy()
        new_row["utt_id"] = utt_id
        rows.append(new_row)

    out = pd.DataFrame(rows).reset_index(drop=True)
    return out


def compute_model_stats(refs: List[str], hyps: List[str]) -> Dict[str, Any]:
    total_s = total_d = total_i = 0
    total_ref_words = 0
    utt_wers = []

    for ref_text, hyp_text in zip(refs, hyps):
        ref_tok = tokenize(ref_text)
        hyp_tok = tokenize(hyp_text)
        ops = align_ops(ref_tok, hyp_tok)

        s = sum(1 for op, *_ in ops if op == "S")
        d = sum(1 for op, *_ in ops if op == "D")
        i = sum(1 for op, *_ in ops if op == "I")
        n = len(ref_tok)

        total_s += s
        total_d += d
        total_i += i
        total_ref_words += n

        if n > 0:
            utt_wers.append((s + d + i) / n)
        else:
            utt_wers.append(0.0 if (s + d + i) == 0 else 1.0)

    total_errors = total_s + total_d + total_i
    corpus_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0
    avg_utt_wer = sum(utt_wers) / len(utt_wers) if utt_wers else 0.0

    return {
        "S": total_s,
        "D": total_d,
        "I": total_i,
        "total_errors": total_errors,
        "ref_words": total_ref_words,
        "corpus_wer": corpus_wer,
        "avg_utt_wer": avg_utt_wer,
    }


def per_utterance_metrics(ref_text: str, hyp_text: str) -> Dict[str, Any]:
    ref_tok = tokenize(ref_text)
    hyp_tok = tokenize(hyp_text)
    ops = align_ops(ref_tok, hyp_tok)

    s = sum(1 for op, *_ in ops if op == "S")
    d = sum(1 for op, *_ in ops if op == "D")
    i = sum(1 for op, *_ in ops if op == "I")
    n = len(ref_tok)
    wer = ((s + d + i) / n) if n > 0 else (0.0 if (s + d + i) == 0 else 1.0)

    sub_pairs = [f"{r}->{h}" for op, r, h, _, _ in ops if op == "S"]
    del_tokens = [r for op, r, h, _, _ in ops if op == "D"]
    ins_tokens = [h for op, r, h, _, _ in ops if op == "I"]

    return {
        "WER": wer,
        "S": s,
        "D": d,
        "I": i,
        "Substitutions": " | ".join(sub_pairs),
        "Deletions": " | ".join(del_tokens),
        "Insertions": " | ".join(ins_tokens),
        "ops": ops,
    }


def build_word_error_counts(refs: List[str], hyps: List[str]) -> Tuple[Counter, Counter]:
    """
    Returns:
      ref_count[token] = how many times token appears in reference
      error_count[token] = how many times token appears as S or D on reference side
    """
    ref_count = Counter()
    error_count = Counter()

    for ref_text, hyp_text in zip(refs, hyps):
        ref_tok = tokenize(ref_text)
        hyp_tok = tokenize(hyp_text)
        ops = align_ops(ref_tok, hyp_tok)

        for tok in ref_tok:
            ref_count[tok] += 1

        for op, r, h, _, _ in ops:
            if op in ("S", "D") and r:
                error_count[r] += 1

    return ref_count, error_count


def stage_utterance_comparison(df: pd.DataFrame, prev_col: str, next_col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    labels = []
    for _, row in df.iterrows():
        prev_wer = row[f"WER__{prev_col}"]
        next_wer = row[f"WER__{next_col}"]
        if next_wer < prev_wer:
            label = "Improvement"
        elif next_wer > prev_wer:
            label = "Worsen"
        else:
            label = "Same"
        labels.append(label)

    counts = Counter(labels)
    comp = pd.DataFrame({
        "utt_id": df["utt_id"],
        "prev_wer": df[f"WER__{prev_col}"],
        "next_wer": df[f"WER__{next_col}"],
        "delta_wer": df[f"WER__{next_col}"] - df[f"WER__{prev_col}"],
        "comparison": labels,
        "ref": df["Real"],
        "prev_text": df[prev_col],
        "next_text": df[next_col],
    })
    return comp, {
        "Improvement": counts.get("Improvement", 0),
        "Same": counts.get("Same", 0),
        "Worsen": counts.get("Worsen", 0),
    }


def make_word_stage_df(
    ref_count: Counter,
    prev_error_count: Counter,
    next_error_count: Counter,
    aviation_terms: set
) -> pd.DataFrame:
    tokens = sorted(ref_count.keys())
    rows = []
    for tok in tokens:
        rc = ref_count[tok]
        pe = prev_error_count.get(tok, 0)
        ne = next_error_count.get(tok, 0)
        rows.append({
            "token": tok,
            "ref_count": rc,
            "aviation_term": 1 if tok in aviation_terms else 0,
            "prev_error_count": pe,
            "next_error_count": ne,
            "error_reduction": pe - ne,
            "prev_error_rate": (pe / rc) if rc else 0.0,
            "next_error_rate": (ne / rc) if rc else 0.0,
            "error_rate_reduction": ((pe / rc) - (ne / rc)) if rc else 0.0,
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["error_reduction", "error_rate_reduction", "ref_count", "token"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)
    return out


def category_summary(
    ref_count: Counter,
    prev_error_count: Counter,
    next_error_count: Counter,
    aviation_terms: set
) -> pd.DataFrame:
    categories = {
        "all_terms": lambda t: True,
        "aviation_terms": lambda t: t in aviation_terms,
        "non_aviation_terms": lambda t: t not in aviation_terms,
    }
    rows = []
    for name, pred in categories.items():
        toks = [t for t in ref_count if pred(t)]
        rc = sum(ref_count[t] for t in toks)
        pe = sum(prev_error_count.get(t, 0) for t in toks)
        ne = sum(next_error_count.get(t, 0) for t in toks)
        rows.append({
            "category": name,
            "ref_count": rc,
            "prev_error_count": pe,
            "next_error_count": ne,
            "error_reduction": pe - ne,
            "prev_error_rate": (pe / rc) if rc else 0.0,
            "next_error_rate": (ne / rc) if rc else 0.0,
            "relative_error_reduction": ((pe - ne) / pe) if pe else 0.0,
        })
    return pd.DataFrame(rows)


def save_top_and_bottom(df: pd.DataFrame, path_prefix: str, top_n: int) -> None:
    best = df.sort_values(
        ["error_reduction", "error_rate_reduction", "ref_count", "token"],
        ascending=[False, False, False, True]
    ).head(top_n)
    worst = df.sort_values(
        ["error_reduction", "error_rate_reduction", "ref_count", "token"],
        ascending=[True, True, False, True]
    ).head(top_n)

    best.to_csv(f"{path_prefix}_best_improved_words.csv", index=False)
    worst.to_csv(f"{path_prefix}_worst_worsened_words.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to WER.csv")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory")
    ap.add_argument("--top-n", type=int, default=30, help="Top N rows to export for best/worst lists")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw_df = pd.read_csv(args.input)
    df = parse_csv_with_sample_ids(raw_df)

    required_cols = ["Real", "Azure", "Raw prediction", "LoRA prediction", "LoRA+Prompt prediction"]
    for col in required_cols:
        if col not in df.columns:
            raise SystemExit(f"Required column missing: {col}. Available columns: {df.columns.tolist()}")

    model_cols = ["Azure", "Raw prediction", "LoRA prediction", "LoRA+Prompt prediction"]
    refs = df["Real"].apply(safe_text).tolist()

    # ---------------------------------
    # 1) Model-level summary
    # ---------------------------------
    model_rows = []
    per_model_error_counts = {}
    per_model_ref_counts = None

    for col in model_cols:
        hyps = df[col].apply(safe_text).tolist()
        stats = compute_model_stats(refs, hyps)
        model_rows.append({
            "model": col,
            **stats
        })
        rc, ec = build_word_error_counts(refs, hyps)
        per_model_error_counts[col] = ec
        if per_model_ref_counts is None:
            per_model_ref_counts = rc

    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(os.path.join(args.outdir, "model_error_summary.csv"), index=False)

    # ---------------------------------
    # 2) Utterance-level details
    # ---------------------------------
    out_df = df[["utt_id", "Real", "Azure", "Raw prediction", "LoRA prediction", "LoRA+Prompt prediction"]].copy()

    for col in model_cols:
        wers, ss, ds, ins, subs_txt, dels_txt, ins_txt = [], [], [], [], [], [], []
        for _, row in df.iterrows():
            met = per_utterance_metrics(row["Real"], row[col])
            wers.append(met["WER"])
            ss.append(met["S"])
            ds.append(met["D"])
            ins.append(met["I"])
            subs_txt.append(met["Substitutions"])
            dels_txt.append(met["Deletions"])
            ins_txt.append(met["Insertions"])
        out_df[f"WER__{col}"] = wers
        out_df[f"S__{col}"] = ss
        out_df[f"D__{col}"] = ds
        out_df[f"I__{col}"] = ins
        out_df[f"Subs__{col}"] = subs_txt
        out_df[f"Dels__{col}"] = dels_txt
        out_df[f"Ins__{col}"] = ins_txt

    # comparisons
    comparisons = [
        ("Large_v3_stage", "Azure", "Raw prediction"),
        #("LoRA_stage", "Raw prediction", "LoRA prediction"),
        ("Prompt_stage", "LoRA prediction", "LoRA+Prompt prediction"),
        ("Azure_to_LoRA", "Azure", "LoRA prediction"),
        ("Azure_to_Final", "Azure", "LoRA+Prompt prediction"),
    ]

    summary_rows = []
    for name, prev_col, next_col in comparisons:
        prev_model = model_summary.loc[model_summary["model"] == prev_col].iloc[0]
        next_model = model_summary.loc[model_summary["model"] == next_col].iloc[0]
        comp_df, comp_counts = stage_utterance_comparison(out_df, prev_col, next_col)

        out_df[f"delta_WER__{name}"] = out_df[f"WER__{next_col}"] - out_df[f"WER__{prev_col}"]
        out_df[f"comparison__{name}"] = comp_df["comparison"]

        summary_rows.append({
            "comparison": name,
            "prev_model": prev_col,
            "next_model": next_col,
            "S_prev": int(prev_model["S"]),
            "S_next": int(next_model["S"]),
            "S_reduction": int(prev_model["S"] - next_model["S"]),
            "D_prev": int(prev_model["D"]),
            "D_next": int(next_model["D"]),
            "D_reduction": int(prev_model["D"] - next_model["D"]),
            "I_prev": int(prev_model["I"]),
            "I_next": int(next_model["I"]),
            "I_reduction": int(prev_model["I"] - next_model["I"]),
            "total_errors_prev": int(prev_model["total_errors"]),
            "total_errors_next": int(next_model["total_errors"]),
            "total_error_reduction": int(prev_model["total_errors"] - next_model["total_errors"]),
            "Improvement": comp_counts["Improvement"],
            "Same": comp_counts["Same"],
            "Worsen": comp_counts["Worsen"],
        })

        comp_df.to_csv(os.path.join(args.outdir, f"{name}_utterance_comparison.csv"), index=False)

    out_df.to_csv(os.path.join(args.outdir, "utterance_error_details.csv"), index=False)
    comparison_summary = pd.DataFrame(summary_rows)
    comparison_summary.to_csv(os.path.join(args.outdir, "comparison_summary.csv"), index=False)

    # ---------------------------------
    # 3) Word-level inspection by model
    # ---------------------------------
    word_rows = []
    for tok in sorted(per_model_ref_counts.keys()):
        row = {
            "token": tok,
            "ref_count": per_model_ref_counts[tok],
            "aviation_term": 1 if tok in AVIATION_TERMS else 0,
        }
        for col in model_cols:
            ec = per_model_error_counts[col].get(tok, 0)
            row[f"errors__{col}"] = ec
            row[f"error_rate__{col}"] = ec / per_model_ref_counts[tok] if per_model_ref_counts[tok] else 0.0
        word_rows.append(row)

    word_model_df = pd.DataFrame(word_rows)
    word_model_df.to_csv(os.path.join(args.outdir, "word_inspection_by_model.csv"), index=False)

    # ---------------------------------
    # 4) Word improvement lists by stage
    # ---------------------------------
    category_rows = []
    for name, prev_col, next_col in comparisons:
        stage_word_df = make_word_stage_df(
            per_model_ref_counts,
            per_model_error_counts[prev_col],
            per_model_error_counts[next_col],
            AVIATION_TERMS,
        )
        stage_word_df.to_csv(os.path.join(args.outdir, f"{name}_word_improvement.csv"), index=False)
        save_top_and_bottom(stage_word_df, os.path.join(args.outdir, name), args.top_n)

        cat_df = category_summary(
            per_model_ref_counts,
            per_model_error_counts[prev_col],
            per_model_error_counts[next_col],
            AVIATION_TERMS,
        )
        cat_df.insert(0, "comparison", name)
        cat_df.to_csv(os.path.join(args.outdir, f"{name}_aviation_category_summary.csv"), index=False)
        category_rows.append(cat_df)

    aviation_summary_all = pd.concat(category_rows, ignore_index=True)
    aviation_summary_all.to_csv(os.path.join(args.outdir, "aviation_category_summary_all.csv"), index=False)

    # ---------------------------------
    # 5) Console summary
    # ---------------------------------
    print("\n=== Saved files ===")
    print(os.path.join(args.outdir, "model_error_summary.csv"))
    print(os.path.join(args.outdir, "comparison_summary.csv"))
    print(os.path.join(args.outdir, "utterance_error_details.csv"))
    print(os.path.join(args.outdir, "word_inspection_by_model.csv"))
    print(os.path.join(args.outdir, "aviation_category_summary_all.csv"))
    print("\nPer-comparison files also saved:")
    for name, _, _ in comparisons:
        print(f"  {name}_utterance_comparison.csv")
        print(f"  {name}_word_improvement.csv")
        print(f"  {name}_best_improved_words.csv")
        print(f"  {name}_worst_worsened_words.csv")
        print(f"  {name}_aviation_category_summary.csv")


if __name__ == "__main__":
    main()
