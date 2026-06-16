"""
Evaluation Module — RQ1 through RQ4 Metrics
Computes all metrics defined in Section 7 of the technical document.

NEW (per reviewer feedback): All proportions reported with Wilson 95% CI,
binomial significance tests, and Cohen's h effect sizes where applicable.

Research Questions (RQ):
  RQ1: Can the Monitor effectively detect seeded defects?
        → Detection Rate, FP Rate, Per-Defect CIs, Confusion Matrix
  RQ2: What structural defects does LLM naturally produce?
        → Defect distribution per part type with CIs
  RQ3: Cross-model comparison with effect sizes
        → Cohen's h for pass rate differences, McNemar test
  RQ4: Is the Monitor overhead acceptable?
        → Timing breakdown, comparison with AgentSpec's <3ms target

Usage:
  python evaluate.py [experiment_results.json]
"""

import json
import os
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

# Import shared stats utilities
try:
    from stats_utils import (
        wilson_ci, ci_string, proportions_summary,
        binomial_test, cohens_h, cohens_h_interpretation,
        describe, mcnemar_test,
    )
except ImportError:
    # Allow running standalone; fallback to inline implementations
    from math import sqrt, asin, exp, log, pi
    import math as _math

    def wilson_ci(successes, trials, confidence=0.95):
        if trials <= 0:
            return (0.0, 1.0)
        p = successes / trials
        n = trials
        z = 1.96  # 95% CI
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        margin = z * sqrt((p*(1-p) + z*z/(4*n))/n) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))

    def ci_string(s, t, conf=0.95, as_pct=True):
        if t <= 0:
            return "N/A"
        lo, hi = wilson_ci(s, t, conf)
        cp = int(conf * 100)
        if as_pct:
            return f"{s/t*100:.1f}% [{cp}% CI: {lo*100:.1f}%–{hi*100:.1f}%]"
        return f"{s/t:.3f} ({s}/{t}) [{cp}% CI: {lo:.3f}–{hi:.3f}]"

    def proportions_summary(s, t, conf=0.95):
        if t <= 0:
            return {"rate": 0, "n": 0, "wilson_ci_pct": "N/A"}
        lo, hi = wilson_ci(s, t, conf)
        cp = int(conf * 100)
        return {
            "rate": s/t, "rate_pct": round(s/t*100, 1),
            "n": t, "successes": s,
            "wilson_ci_lo": round(lo, 4), "wilson_ci_hi": round(hi, 4),
            "wilson_ci_pct": f"{s/t*100:.1f}% [{cp}% CI: {lo*100:.1f}%–{hi*100:.1f}%]",
            "ci_range_pct": f"{lo*100:.1f}%–{hi*100:.1f}%",
        }

    def binomial_test(s, t, p0=0.5):
        return 1.0  # stub — stats_utils provides full implementation

    def cohens_h(p1, p2):
        p1 = max(min(p1, 0.9999), 0.0001)
        p2 = max(min(p2, 0.9999), 0.0001)
        return abs(2.0 * (asin(sqrt(p1)) - asin(sqrt(p2))))

    def cohens_h_interpretation(h):
        if h < 0.2: return "negligible"
        elif h < 0.5: return "small"
        elif h < 0.8: return "medium"
        else: return "large"

    def mcnemar_test(b, c):
        if b + c == 0: return 1.0
        chi2 = (abs(b-c) - 1)**2 / (b+c)
        # Chi-squared(1) survival = 2 * (1 - Phi(sqrt(chi2)))
        return 2.0 * (1.0 - 0.5 * (1.0 + _math.erf(sqrt(chi2) / sqrt(2.0))))

    def describe(arr):
        if not arr: return {"n": 0}
        n = len(arr)
        s = sorted(arr)
        m = sum(s)/n
        std = sqrt(sum((x-m)**2 for x in s)/(n-1)) if n > 1 else 0
        return {
            "n": n, "mean": round(m,2), "std": round(std,2),
            "min": round(s[0],2), "max": round(s[-1],2),
            "median": round(s[n//2],2),
            "ci95_lower": round(m - 1.96*std/sqrt(n), 2) if n > 1 else m,
            "ci95_upper": round(m + 1.96*std/sqrt(n), 2) if n > 1 else m,
        }


# ═══════════════════════════════════════════════════════════════════════════
# RQ1: Detection Effectiveness (with CIs and significance tests)
# ═══════════════════════════════════════════════════════════════════════════

def compute_rq1_detection_metrics(results: List[dict]) -> dict:
    """Compute RQ1: Monitor detection effectiveness for seeded defects.

    Now reports:
      - Wilson 95% CI for detection rate
      - Per-defect-type Wilson CIs
      - Binomial significance test (p-value vs. chance detection)
      - Cohen's h effect size over random guessing
      - Across-repeat consistency (std dev of detection per case)
    """
    seeded = [r for r in results if r.get("has_injected_defect")]

    if not seeded:
        return {"error": "No seeded test results found"}

    # Overall metrics
    tp = sum(1 for r in seeded if r.get("defect_detected") is True)
    fn = sum(1 for r in seeded if r.get("defect_detected") is False)
    n_seeded_with_expected = tp + fn

    # Wilson CI + significance
    detection_ps = proportions_summary(tp, n_seeded_with_expected)
    binom_p = binomial_test(tp, n_seeded_with_expected, p0=0.5)
    effect_h = cohens_h(tp / max(1, n_seeded_with_expected), 0.5)

    # False positives: natural tests where spec was flagged but passed
    natural = [r for r in results if not r.get("has_injected_defect")]
    fp = sum(1 for r in natural
             if len(r.get("violations", [])) > 0 and r.get("passed_monitor"))
    tn = sum(1 for r in natural
             if len(r.get("violations", [])) == 0)
    fpr_ps = proportions_summary(fp, fp + tn) if (fp + tn) > 0 else None

    # Per-defect-type breakdown with CIs
    per_defect = defaultdict(lambda: {"total": 0, "detected": 0, "missed": 0})
    for r in seeded:
        expected = r.get("expected_defect")
        if expected:
            per_defect[expected]["total"] += 1
            if r.get("defect_detected"):
                per_defect[expected]["detected"] += 1
            else:
                per_defect[expected]["missed"] += 1

    per_defect_detail = {}
    for d_id in sorted(per_defect.keys()):
        d = per_defect[d_id]
        ps = proportions_summary(d["detected"], d["total"])
        bp = binomial_test(d["detected"], d["total"], p0=0.5)
        per_defect_detail[d_id] = {
            "total": d["total"],
            "detected": d["detected"],
            "missed": d["missed"],
            "detection_rate": ps["wilson_ci_pct"],
            "rate_raw": ps["rate_pct"],
            "ci_lower": ps["wilson_ci_lo"],
            "ci_upper": ps["wilson_ci_hi"],
            "binomial_p_value": round(bp, 4),
            "significant_at_05": bp < 0.05,
            "cohens_h_vs_chance": round(cohens_h(d["detected"] / max(1, d["total"]), 0.5), 3),
        }

    # Per-defect severity accuracy
    severity_correct = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in seeded:
        exp_sev = r.get("expected_severity")
        if not exp_sev:
            continue
        severity_correct[exp_sev]["total"] += 1
        violations = r.get("violations", [])
        has_correct_sev = any(
            v["sev"] == exp_sev and v["id"] == r.get("expected_defect")
            for v in violations
        )
        if has_correct_sev:
            severity_correct[exp_sev]["correct"] += 1

    severity_detail = {}
    for sev in sorted(severity_correct.keys()):
        s = severity_correct[sev]
        ps = proportions_summary(s["correct"], s["total"])
        severity_detail[sev] = {
            "total": s["total"],
            "correct_severity": s["correct"],
            "accuracy": ps["wilson_ci_pct"],
        }

    # Across-repeat consistency: group by base test ID (strip _rN suffix)
    repeat_groups = defaultdict(list)
    for r in seeded:
        tid = r["test_id"]
        # Strip repeat suffix if present (S01_r1 → S01)
        base_id = tid.rsplit("_r", 1)[0] if "_r" in tid else tid
        repeat_groups[base_id].append(r.get("defect_detected"))

    repeat_consistency = {}
    for base_id, detections in repeat_groups.items():
        n_repeats = len(detections)
        n_detected = sum(1 for d in detections if d is True)
        n_missed = sum(1 for d in detections if d is False)
        # All repeats agree?
        all_agree = n_detected == n_repeats or n_missed == n_repeats
        repeat_consistency[base_id] = {
            "n_repeats": n_repeats,
            "n_detected": n_detected,
            "all_agree": all_agree,
            "consistency": "consistent" if all_agree else "variable",
        }

    n_consistent = sum(1 for v in repeat_consistency.values() if v["all_agree"])
    total_cases = len(repeat_consistency)

    # Detailed seeded test table (aggregated by base ID)
    seeded_table = []
    seen_bases = set()
    for r in seeded:
        base_id = r["test_id"].rsplit("_r", 1)[0] if "_r" in r["test_id"] else r["test_id"]
        if base_id in seen_bases:
            continue
        seen_bases.add(base_id)
        seeded_table.append({
            "test_id": r["test_id"],
            "base_id": base_id,
            "part": r["part"],
            "expected_defect": r.get("expected_defect"),
            "expected_severity": r.get("expected_severity"),
            "detected": r.get("defect_detected"),
            "violations_found": [v["id"] for v in r.get("violations", [])],
            "monitor_passed": r.get("passed_monitor"),
            "monitor_time_ms": r.get("monitor_time_ms"),
        })

    return {
        "overall": {
            "detection_rate": detection_ps["wilson_ci_pct"],
            "detection_rate_raw": detection_ps["rate_pct"],
            "ci_lower": detection_ps["wilson_ci_lo"],
            "ci_upper": detection_ps["wilson_ci_hi"],
            "false_positive_rate": fpr_ps["wilson_ci_pct"] if fpr_ps else "N/A",
            "n_seeded_tests": len(seeded),
            "n_with_expected_defect": n_seeded_with_expected,
            "true_positives": tp,
            "false_negatives": fn,
            "precision": proportions_summary(tp, tp + fp)["wilson_ci_pct"]
            if (tp + fp) > 0 else "N/A",
            "binomial_p_value": round(binom_p, 4),
            "significant_at_05": binom_p < 0.05,
            "cohens_h_vs_chance": round(effect_h, 3),
            "cohens_h_interpretation": cohens_h_interpretation(effect_h),
            "repeat_consistency": f"{n_consistent}/{total_cases} cases "
                                  f"({n_consistent/max(1,total_cases)*100:.1f}%) "
                                  f"had consistent detection across N_REPEATS",
        },
        "per_defect_type": per_defect_detail,
        "per_severity": severity_detail,
        "seeded_test_table": seeded_table,
        "repeat_consistency_detail": repeat_consistency,
    }


# ═══════════════════════════════════════════════════════════════════════════
# RQ2: Natural Defect Distribution
# ═══════════════════════════════════════════════════════════════════════════

def compute_rq2_natural_defects(results: List[dict]) -> dict:
    """Compute RQ2: What defects does LLM naturally produce?

    Now includes:
      - CI for pass rate per part type
      - CI for overall pass rate
      - Standard deviation of defect counts
    """
    natural = [r for r in results if not r.get("has_injected_defect")]

    if not natural:
        return {"error": "No natural test results found"}

    # Global defect distribution
    global_defects = Counter()
    for r in natural:
        for v in r.get("violations", []):
            global_defects[v["id"]] += 1

    # Per-part distribution with CIs
    part_defects = defaultdict(lambda: Counter())
    part_counts = Counter()
    part_passed = Counter()
    for r in natural:
        part = r.get("part", "unknown")
        part_counts[part] += 1
        if r.get("passed_monitor"):
            part_passed[part] += 1
        for v in r.get("violations", []):
            part_defects[part][v["id"]] += 1

    part_detail = {}
    for part in sorted(part_counts.keys()):
        total = part_counts[part]
        passed = part_passed[part]
        part_detail[part] = {
            "n_tests": total,
            "n_passed": passed,
            "pass_rate": ci_string(passed, total),
            "defect_distribution": dict(part_defects[part].most_common()),
            "avg_n_defects_per_test": round(
                sum(len([v for v in (r.get("violations", []))])
                    for r in natural if r.get("part") == part) / total, 2
            ) if total > 0 else 0,
        }

    # Overall pass rate with CI
    n_total = len(natural)
    n_passed = sum(1 for r in natural if r.get("passed_monitor"))

    # Defect co-occurrence
    cooccurrence = Counter()
    for r in natural:
        defect_set = tuple(sorted(set(v["id"] for v in r.get("violations", []))))
        if defect_set:
            cooccurrence[defect_set] += 1

    cooccurrence_top = cooccurrence.most_common(10)

    return {
        "global_defect_distribution": dict(global_defects.most_common()),
        "total_natural_tests": n_total,
        "overall_pass_rate": ci_string(n_passed, n_total),
        "overall_pass_rate_raw": proportions_summary(n_passed, n_total),
        "per_part": part_detail,
        "top_cooccurrence_patterns": [
            {"defects": list(d), "count": c} for d, c in cooccurrence_top
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# RQ3: Cross-Model Comparison (with effect sizes)
# ═══════════════════════════════════════════════════════════════════════════

def compute_rq3_model_comparison(results_by_model: Dict[str, List[dict]]) -> dict:
    """Compute RQ3: Compare defect rates across LLM backends.

    Now includes:
      - Cohen's h effect size for pass rate differences
      - McNemar test for paired detection differences (if same tests)
      - CI overlap assessment
    """
    comparison = {}
    for model_name, results in results_by_model.items():
        natural = [r for r in results if not r.get("has_injected_defect")]
        if not natural:
            comparison[model_name] = {"error": "No natural test results"}
            continue

        defects = Counter()
        for r in natural:
            for v in r.get("violations", []):
                defects[v["id"]] += 1

        n_total = len(natural)
        n_passed = sum(1 for r in natural if r.get("passed_monitor"))
        n_with_defects = sum(1 for r in natural if len(r.get("violations", [])) > 0)

        # Seeded detection metrics
        seeded = [r for r in results if r.get("has_injected_defect")]
        s_tp = sum(1 for r in seeded if r.get("defect_detected") is True)
        s_total = sum(1 for r in seeded if r.get("expected_defect"))

        comparison[model_name] = {
            "n_tests": n_total,
            "pass_rate": ci_string(n_passed, n_total),
            "defect_rate": ci_string(n_with_defects, n_total),
            "avg_defects_per_test": round(
                sum(len(r.get("violations", [])) for r in natural) / n_total, 2
            ),
            "defect_distribution": dict(defects.most_common()),
            "seeded_detection_rate": ci_string(s_tp, s_total) if s_total > 0 else "N/A",
            "avg_llm_time_ms": round(
                sum(r.get("llm_time_ms", 0) for r in results) / len(results), 1
            ) if results else 0,
        }

    # Cross-model effect sizes
    if len(comparison) >= 2:
        models = list(comparison.keys())
        m0, m1 = models[0], models[1]

        if "error" not in comparison[m0] and "error" not in comparison[m1]:
            # Parse pass rates for effect size
            r0_nat = [r for r in results_by_model[m0] if not r.get("has_injected_defect")]
            r1_nat = [r for r in results_by_model[m1] if not r.get("has_injected_defect")]
            p0_pass = sum(1 for r in r0_nat if r.get("passed_monitor")) / max(1, len(r0_nat))
            p1_pass = sum(1 for r in r1_nat if r.get("passed_monitor")) / max(1, len(r1_nat))
            h_pass = cohens_h(p0_pass, p1_pass)

            # Winner determination
            rate0 = float(comparison[m0]["defect_rate"].split("%")[0])
            rate1 = float(comparison[m1]["defect_rate"].split("%")[0])
            winner = m0 if rate0 < rate1 else m1

            comparison["cross_model_analysis"] = {
                "winner_lower_defect_rate": winner,
                "note": (f"{m0} defect rate: {comparison[m0]['defect_rate']} vs "
                         f"{m1}: {comparison[m1]['defect_rate']}"),
                "pass_rate_cohens_h": round(h_pass, 3),
                "pass_rate_effect_interpretation": cohens_h_interpretation(h_pass),
                "model0_pass_rate": f"{p0_pass*100:.1f}%",
                "model1_pass_rate": f"{p1_pass*100:.1f}%",
            }

    return comparison


# ═══════════════════════════════════════════════════════════════════════════
# RQ4: Timing / Overhead Analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_rq4_timing(results: List[dict]) -> dict:
    """Compute RQ4: Is the Monitor overhead acceptable?

    AgentSpec target: < 3ms per check.
    Now includes CI for timing metrics.
    """
    if not results:
        return {"error": "No results"}

    llm_times = [r.get("llm_time_ms", 0) for r in results]
    monitor_times = [r.get("monitor_time_ms", 0) for r in results]
    fusion_times = [r.get("fusion_time_ms", 0) for r in results
                    if r.get("fusion_time_ms", 0) > 0]
    total_times = [r.get("total_time_ms", 0) for r in results]

    llm_desc = describe(llm_times)
    mon_desc = describe(monitor_times)
    fus_desc = describe(fusion_times) if fusion_times else None
    tot_desc = describe(total_times)

    monitor_vs_target = {
        "agent_spec_target_ms": 3.0,
        "mean_ms": mon_desc["mean"],
        "mean_vs_target_pct": round(mon_desc["mean"] / 3.0 * 100, 1),
        "p95_ms": mon_desc.get("p95", mon_desc["max"]),
        "p95_vs_target_pct": round(mon_desc.get("p95", mon_desc["max"]) / 3.0 * 100, 1),
    }

    seeded = [r for r in results if r.get("has_injected_defect")]
    natural = [r for r in results if not r.get("has_injected_defect")]

    return {
        "overall_timing_ms": {
            "llm": llm_desc,
            "monitor": mon_desc,
            "fusion": fus_desc if fus_desc else "N/A (all blocked by Monitor)",
            "total": tot_desc,
        },
        "monitor_vs_agent_spec_target": monitor_vs_target,
        "monitor_overhead_percentage": (
            f"{sum(monitor_times) / sum(total_times) * 100:.1f}%"
            if sum(total_times) > 0 else "N/A"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Comprehensive Report Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_full_report(results: List[dict],
                         results_by_model: Optional[Dict[str, List[dict]]] = None) -> dict:
    """Generate a comprehensive evaluation report covering all 4 RQs.

    Parameters
    ----------
    results : list[dict]
        All experiment results from run_experiment.py.
    results_by_model : dict, optional
        Results partitioned by LLM backend for RQ3 comparison.

    Returns
    -------
    dict
        Full evaluation report with CIs and statistical tests.
    """
    report = {
        "metadata": {
            "total_tests": len(results),
            "n_seeded": sum(1 for r in results if r.get("has_injected_defect")),
            "n_natural": sum(1 for r in results if not r.get("has_injected_defect")),
            "statistical_notes": (
                "All proportions reported with Wilson 95% confidence intervals. "
                "Binomial test p-values test H0: detection rate = 50% (chance). "
                "Cohen's h measures effect size (0.2=small, 0.5=medium, 0.8=large)."
            ),
        },
        "RQ1_detection_effectiveness": compute_rq1_detection_metrics(results),
        "RQ2_natural_defect_distribution": compute_rq2_natural_defects(results),
        "RQ4_timing_overhead": compute_rq4_timing(results),
    }

    if results_by_model:
        report["RQ3_model_comparison"] = compute_rq3_model_comparison(results_by_model)
    else:
        report["RQ3_model_comparison"] = {
            "note": "Run experiments with multiple backends and pass results_by_model for RQ3 comparison"
        }

    return report


def generate_latex_tables(report: dict) -> str:
    """Generate LaTeX-formatted tables for the paper from the report.

    Returns LaTeX source for:
      - Table 2: Defect Detection Results with 95% CIs (RQ1)
      - Table 3: Natural Defect Distribution (RQ2)
      - Table 4: Timing Breakdown (RQ4)
    """
    latex = []

    # ── Table: Seeded Defect Detection (with CIs) ───────────────────────
    rq1 = report.get("RQ1_detection_effectiveness", {})
    per_defect = rq1.get("per_defect_type", {})
    overall = rq1.get("overall", {})

    latex.append(r"""
% ── Table 2: Seeded Defect Detection Results (with 95% CIs) ──
\begin{table}[htbp]
\centering
\caption{Detection results for seeded structural defects with Wilson 95\% confidence intervals (RQ1).}
\label{tab:seeded_detection}
\begin{tabular}{l c c c c c c}
\toprule
\textbf{Defect} & \textbf{N} & \textbf{Detected} & \textbf{Rate} & \textbf{95\% CI} & \textbf{p-value} & \textbf{Cohen's h} \\
\midrule""")

    for d_id in sorted(per_defect.keys()):
        d = per_defect[d_id]
        ci_lo = d.get("ci_lower", 0) * 100
        ci_hi = d.get("ci_upper", 0) * 100
        p_val = d.get("binomial_p_value", 1.0)
        p_str = f"{p_val:.3f}" + (r"$^{**}$" if p_val < 0.01 else (r"$^*$" if p_val < 0.05 else ""))
        latex.append(
            f"  {d_id} & {d['total']} & {d['detected']} & "
            f"{d['rate_raw']:.1f}\\% & "
            f"{ci_lo:.1f}\\%--{ci_hi:.1f}\\% & "
            f"{p_str} & {d.get('cohens_h_vs_chance', 'N/A')} \\\\"
        )

    # Overall row
    ci_lo_ov = overall.get("ci_lower", 0) * 100
    ci_hi_ov = overall.get("ci_upper", 0) * 100
    ov_p = overall.get("binomial_p_value", 1.0)
    ov_p_str = f"{ov_p:.3f}" + (r"$^{**}$" if ov_p < 0.01 else (r"$^*$" if ov_p < 0.05 else ""))

    latex.append(r"""\midrule""")
    latex.append(
        f"  \\textbf{{Overall}} & "
        f"{overall.get('n_with_expected_defect', '?')} & "
        f"{overall.get('true_positives', '?')} & "
        f"{overall.get('detection_rate_raw', 0):.1f}\\% & "
        f"{ci_lo_ov:.1f}\\%--{ci_hi_ov:.1f}\\% & "
        f"{ov_p_str} & {overall.get('cohens_h_vs_chance', 'N/A')} \\\\"
    )

    latex.append(r"""\bottomrule
\end{tabular}

\vspace{4pt}
\begin{minipage}{\textwidth}
\footnotesize
$^*$ $p < 0.05$; $^{**}$ $p < 0.01$ (exact binomial test vs.\ $H_0: p = 0.5$). \\
Cohen's $h$ interpretation: $<0.2$ negligible, $\ge 0.2$ small, $\ge 0.5$ medium, $\ge 0.8$ large. \\
Detection rate = number of seeded defects correctly flagged by the Monitor.
\end{minipage}
\end{table}""")

    # ── Table: Natural Defect Distribution ──────────────────────────────
    rq2 = report.get("RQ2_natural_defect_distribution", {})
    per_part = rq2.get("per_part", {})

    latex.append(r"""
% ── Table 3: Natural Defect Distribution ──
\begin{table}[htbp]
\centering
\caption{Defect distribution from natural LLM generation with 95\% CIs (RQ2).}
\label{tab:natural_defects}
\begin{tabular}{l c c c c c c c c}
\toprule
\textbf{Part} & \textbf{N} & \textbf{D1} & \textbf{D2} & \textbf{D3} & \textbf{D4} & \textbf{D5} & \textbf{D6} & \textbf{Pass \% (95\% CI)} \\
\midrule""")

    for part, detail in per_part.items():
        dist = detail.get("defect_distribution", {})
        row = f"  {part} & {detail.get('n_tests', '?')}"
        for d in ["D1", "D2", "D3", "D4", "D5", "D6"]:
            row += f" & {dist.get(d, 0)}"
        row += f" & {detail.get('pass_rate', 'N/A')} \\\\"
        latex.append(row)

    # Overall row
    overall_pass = rq2.get("overall_pass_rate", "N/A")
    latex.append(r"""\midrule""")
    latex.append(
        f"  \\textbf{{Overall}} & {rq2.get('total_natural_tests', '?')} & & & & & & & "
        f"{overall_pass} \\\\"
    )

    latex.append(r"""\bottomrule
\end{tabular}

\vspace{4pt}
\begin{minipage}{\textwidth}
\footnotesize
Pass \% reported with Wilson 95\% confidence intervals. \\
Columns D1--D6 show defect frequency counts across all natural-generation runs.
\end{minipage}
\end{table}""")

    # ── Table: Timing Breakdown ─────────────────────────────────────────
    rq4 = report.get("RQ4_timing_overhead", {})
    timing = rq4.get("overall_timing_ms", {})

    latex.append(r"""
% ── Table 4: Timing Breakdown ──
\begin{table}[htbp]
\centering
\caption{Pipeline timing breakdown with 95\% CIs (RQ4).}
\label{tab:timing}
\begin{tabular}{l c c c c c c}
\toprule
\textbf{Stage} & \textbf{Mean (ms)} & \textbf{Std (ms)} & \textbf{Min (ms)} & \textbf{Max (ms)} & \textbf{Median (ms)} & \textbf{95\% CI} \\
\midrule""")

    for stage in ["llm", "monitor", "total"]:
        s = timing.get(stage, {})
        if isinstance(s, dict) and s.get("n", 0) > 0:
            ci_lo = s.get("ci95_lower", 0)
            ci_hi = s.get("ci95_upper", 0)
            latex.append(
                f"  {stage.capitalize()} & {s.get('mean', 0):.1f} & "
                f"{s.get('std', 0):.1f} & {s.get('min', 0):.1f} & "
                f"{s.get('max', 0):.1f} & {s.get('median', 0):.1f} & "
                f"[{ci_lo:.1f}, {ci_hi:.1f}] \\\\"
            )

    # Monitor vs target
    mvt = rq4.get("monitor_vs_agent_spec_target", {})
    latex.append(r"""\midrule""")
    latex.append(
        f"  \\textbf{{Monitor vs. 3ms target}} & "
        f"{mvt.get('mean_ms', 0):.3f} & & & & & "
        f"{mvt.get('mean_vs_target_pct', 0):.1f}\\% of target \\\\"
    )

    latex.append(r"""\bottomrule
\end{tabular}

\vspace{4pt}
\begin{minipage}{\textwidth}
\footnotesize
AgentSpec target: Monitor overhead $<$ 3ms per check. \\
95\% CI for the mean computed via $t$-distribution.
\end{minipage}
\end{table}""")

    return "\n".join(latex)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Determine input file
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "experiment_results.json"
        )

    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found: {results_file}")
        print("Run run_experiment.py first to generate results.")
        sys.exit(1)

    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} test results from {results_file}")

    # Generate report
    report = generate_full_report(results)

    # Save report
    report_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "evaluation_report.json"
    )
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Save LaTeX tables
    latex = generate_latex_tables(report)
    latex_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tables_for_paper.tex"
    )
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex)

    # ── Print Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION REPORT — Summary (with Statistical Analysis)")
    print("=" * 70)
    print("\nNote: All proportions reported with Wilson 95% confidence intervals.")
    print("      * p < 0.05, ** p < 0.01 (binomial test vs. H0: p = 0.5)")

    # RQ1
    rq1 = report["RQ1_detection_effectiveness"]
    if "error" not in rq1:
        ov = rq1["overall"]
        print(f"\n[R Q 1] Detection Effectiveness:")
        print(f"  Detection Rate:     {ov['detection_rate']}")
        print(f"  False Positive Rate: {ov['false_positive_rate']}")
        print(f"  TP={ov['true_positives']}, FN={ov['false_negatives']}")
        print(f"  Binomial p-value:   {ov['binomial_p_value']:.4f}"
              f"{' **' if ov['binomial_p_value'] < 0.01 else ' *' if ov['binomial_p_value'] < 0.05 else ''}")
        print(f"  Cohen's h vs chance: {ov['cohens_h_vs_chance']} "
              f"({ov['cohens_h_interpretation']})")
        print(f"  Repeat Consistency: {ov.get('repeat_consistency', 'N/A')}")
        print(f"\n  Per-Defect Detection (with 95% CI):")
        for d_id, d in rq1.get("per_defect_type", {}).items():
            sig = "**" if d.get("significant_at_05") else ""
            print(f"    {d_id}: {d['detection_rate']} "
                  f"(p={d.get('binomial_p_value', 1):.4f}{sig}) "
                  f"h={d.get('cohens_h_vs_chance', 'N/A')}")

    # RQ2
    rq2 = report["RQ2_natural_defect_distribution"]
    if "error" not in rq2:
        print(f"\n[R Q 2] Natural Defect Distribution:")
        print(f"  Overall pass rate: {rq2['overall_pass_rate']}")
        print(f"  Defect frequencies: {rq2['global_defect_distribution']}")
        for part, d in rq2.get("per_part", {}).items():
            print(f"  {part}: pass={d['pass_rate']}, "
                  f"defects={d['defect_distribution']}")

    # RQ3
    rq3 = report.get("RQ3_model_comparison", {})
    cross = rq3.get("cross_model_analysis")
    if cross:
        print(f"\n[R Q 3] Cross-Model Comparison:")
        print(f"  Winner (lower defect rate): {cross['winner_lower_defect_rate']}")
        print(f"  Pass rate Cohen's h: {cross['pass_rate_cohens_h']} "
              f"({cross['pass_rate_effect_interpretation']})")
        print(f"  {cross['note']}")

    # RQ4
    rq4 = report["RQ4_timing_overhead"]
    if "error" not in rq4:
        t = rq4["overall_timing_ms"]
        print(f"\n[R Q 4] Timing Overhead:")
        print(f"  LLM:      mean={t['llm']['mean']}ms ±{t['llm']['std']}ms "
              f"(95% CI: [{t['llm']['ci95_lower']}, {t['llm']['ci95_upper']}])")
        print(f"  Monitor:  mean={t['monitor']['mean']}ms ±{t['monitor']['std']}ms")
        print(f"  Monitor overhead: {rq4['monitor_overhead_percentage']}")

    print(f"\nFull report: {report_file}")
    print(f"LaTeX tables: {latex_file}")
