"""Compute detailed metrics from experiment_results.json with statistical rigor.

Now supports the expanded test matrix (60 unique seeded × N_REPEATS),
and outputs confidence intervals for all rate metrics.

Usage:
  python compute_metrics.py [experiment_results.json] [output_folder]
"""
import json
import os
import sys
from collections import Counter, defaultdict

# Try importing shared stats; fall back to inline
try:
    from stats_utils import ci_string, proportions_summary
except ImportError:
    from math import sqrt
    def wilson_ci(s, t, conf=0.95):
        if t <= 0: return (0.0, 1.0)
        p, n, z = s/t, t, 1.96
        d = 1 + z*z/n
        c = (p + z*z/(2*n))/d
        m = z*sqrt((p*(1-p) + z*z/(4*n))/n)/d
        return (max(0, c-m), min(1, c+m))
    def ci_string(s, t, conf=0.95, as_pct=True):
        if t <= 0: return "N/A"
        lo, hi = wilson_ci(s, t, conf)
        cp = int(conf*100)
        if as_pct:
            return f"{s/t*100:.1f}% [{cp}% CI: {lo*100:.1f}%–{hi*100:.1f}%]"
        return f"{s/t:.3f} [{cp}% CI: {lo:.3f}–{hi:.3f}]"
    def proportions_summary(s, t, conf=0.95):
        if t <= 0: return {"rate": 0, "n": 0, "wilson_ci_pct": "N/A"}
        lo, hi = wilson_ci(s, t, conf)
        cp = int(conf*100)
        return {
            "rate": s/t, "rate_pct": round(s/t*100, 1),
            "n": t, "successes": s,
            "wilson_ci_lo": round(lo, 4), "wilson_ci_hi": round(hi, 4),
            "wilson_ci_pct": f"{s/t*100:.1f}% [{cp}% CI: {lo*100:.1f}%–{hi*100:.1f}%]",
            "ci_range_pct": f"{lo*100:.1f}%–{hi*100:.1f}%",
        }


# Determine paths
if len(sys.argv) > 1:
    results_path = sys.argv[1]
else:
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "experiment_results.json")

if len(sys.argv) > 2:
    out_folder = sys.argv[2]
else:
    out_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "generated_specs", "metrics")

os.makedirs(out_folder, exist_ok=True)

# Load
with open(results_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Separate
seeded = [r for r in data if r.get('has_injected_defect')]
natural = [r for r in data if not r.get('has_injected_defect')]

print(f"Loaded {len(data)} results ({len(seeded)} seeded, {len(natural)} natural)")

# ── Seeded detection metrics (per defect type with CI) ──
by_defect = defaultdict(lambda: {'total': 0, 'detected': 0, 'nil': 0})
for r in seeded:
    exp = r.get('expected_defect', '?')
    by_defect[exp]['total'] += 1
    se_hist = r.get('self_examine_history')
    if se_hist and len(se_hist) > 0:
        if exp in se_hist[0].get('defect_types', []):
            by_defect[exp]['detected'] += 1
    else:
        # Direct detection check
        if r.get('defect_detected'):
            by_defect[exp]['detected'] += 1
        else:
            by_defect[exp]['nil'] += 1

# ── SE stats ──
se_tests = [r for r in data if r.get('self_examine_history')]
direct_pass = sum(1 for r in se_tests
                  if len(r['self_examine_history']) == 1
                  and r['self_examine_history'][0]['passed'])
corrected = sum(1 for r in se_tests
                if len(r['self_examine_history']) > 1
                and r['passed_monitor'])
failed = sum(1 for r in se_tests
             if not r['passed_monitor']
             and len(r['self_examine_history']) > 1)

# ── Defect distribution (attempt-0) ──
all_d = Counter()
for r in data:
    if r.get('self_examine_history'):
        for d in r['self_examine_history'][0].get('defect_types', []):
            all_d[d] += 1

# ── Per-part breakdown ──
by_part = defaultdict(lambda: {
    'total': 0, 'fusion_ok': 0, 'llm_fail': 0,
    'seeded_det': 0, 'seeded_eval': 0
})
for r in data:
    p = r['part']
    by_part[p]['total'] += 1
    if r.get('fusion_success'):
        by_part[p]['fusion_ok'] += 1
    if r.get('llm_error'):
        by_part[p]['llm_fail'] += 1
    if r.get('has_injected_defect'):
        se_hist = r.get('self_examine_history')
        if se_hist:
            by_part[p]['seeded_eval'] += 1
            if r.get('expected_defect') in se_hist[0].get('defect_types', []):
                by_part[p]['seeded_det'] += 1

# ── Build report ──
total_evaluable = sum(v['total'] - v['nil'] for v in by_defect.values())
total_detected = sum(v['detected'] for v in by_defect.values())
total_nil = sum(v['nil'] for v in by_defect.values())

# By-defect detail with CIs
by_defect_detail = {}
for d, info in sorted(by_defect.items()):
    ev = info['total'] - info['nil']
    if ev > 0:
        ps = proportions_summary(info['detected'], ev)
        rate_str = ps['wilson_ci_pct']
    else:
        rate_str = 'N/A'
    by_defect_detail[d] = {
        'total': info['total'],
        'detected': info['detected'],
        'nil_parse_fail': info['nil'],
        'evaluable': ev,
        'detection_rate': rate_str,
        'proportion_summary': proportions_summary(info['detected'], ev) if ev > 0 else None,
    }

# Defect distribution detail
defect_dist_detail = {}
total_defects = sum(all_d.values())
if total_defects > 0:
    for d, c in all_d.most_common():
        defect_dist_detail[d] = {
            'frequency': c,
            'proportion': f"{c/total_defects*100:.1f}%"
        }

# Per-part detail
per_part_detail = {}
for p, info in sorted(by_part.items()):
    sd = f"{info['seeded_det']}/{info['seeded_eval']}" if info['seeded_eval'] > 0 else 'N/A'
    per_part_detail[p] = {
        'total_tests': info['total'],
        'fusion_success': info['fusion_ok'],
        'fusion_rate': ci_string(info['fusion_ok'], info['total']),
        'llm_failures': info['llm_fail'],
        'seeded_detection': sd,
    }

# Overall detection with CI
detection_ps = proportions_summary(total_detected, total_evaluable)

# Natural test pass/defect rates with CI
n_natural = len(natural)
n_pass = sum(1 for r in natural if r['passed_monitor'])
n_defect = sum(1 for r in natural if len(r['violations']) > 0)

# Timing
llm_times = [r.get('llm_time_ms', 0) for r in data if r.get('llm_time_ms', 0) > 0]
mon_times = [r.get('monitor_time_ms', 0) for r in data if r.get('monitor_time_ms', 0) > 0]

from math import sqrt
def desc(arr):
    if not arr: return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    n = len(arr)
    m = sum(arr)/n
    s = sqrt(sum((x-m)**2 for x in arr)/(n-1)) if n>1 else 0
    sa = sorted(arr)
    return {
        'mean': round(m, 1), 'std': round(s, 1),
        'min': round(sa[0], 1), 'max': round(sa[-1], 1),
        'median': round(sa[n//2], 1),
    }

# Detect model info
model_name = data[0].get('spec', {}).get('metadata', {}).get('model', 'unknown') if data else 'unknown'
backend = os.environ.get('LLM_BACKEND', 'unknown')

metrics = {
    'model': backend,
    'experiment_date': '2026-06',
    'sample_size_note': (
        f"Seeded: {len(seeded)} runs across "
        f"{len(set(r.get('expected_defect') for r in seeded if r.get('expected_defect')))} defect types. "
        f"Natural: {len(natural)} runs across "
        f"{len(set(r['part'] for r in natural))} part types."
    ),
    'total_tests': len(data),
    'total_seeded': len(seeded),
    'total_natural': len(natural),
    'seeded_tests': {
        'n_total': len(seeded),
        'n_unique_cases': len(set(
            r['test_id'].rsplit('_r', 1)[0] if '_r' in r.get('test_id', '')
            else r['test_id']
            for r in seeded
        )),
        'n_evaluable': total_evaluable,
        'n_llm_parse_failures': total_nil,
        'n_detected': total_detected,
        'detection_rate': detection_ps['wilson_ci_pct'],
        'detection_rate_ci': detection_ps['ci_range_pct'],
        'detection_rate_raw': detection_ps['rate_pct'],
        'by_defect_type': by_defect_detail,
    },
    'self_examine': {
        'n_tests_with_se': len(se_tests),
        'direct_pass': direct_pass,
        'corrected_pass': corrected,
        'failed_after_retries': failed,
        'total_llm_attempts': sum(len(r['self_examine_history']) for r in se_tests),
        'avg_attempts_per_test': round(
            sum(len(r['self_examine_history']) for r in se_tests) /
            max(1, len(se_tests)), 1
        ),
        'direct_pass_rate': ci_string(
            direct_pass,
            direct_pass + corrected + failed
        ) if (direct_pass + corrected + failed) > 0 else 'N/A',
    },
    'natural_tests': {
        'n_total': n_natural,
        'n_passed_monitor': n_pass,
        'n_with_defects': n_defect,
        'pass_rate': ci_string(n_pass, n_natural),
        'defect_rate': ci_string(n_defect, n_natural),
    },
    'fusion360': {
        'build_success': sum(1 for r in data if r.get('fusion_success')),
        'total': len(data),
        'build_rate': ci_string(
            sum(1 for r in data if r.get('fusion_success')),
            len(data)
        ),
    },
    'defect_distribution_attempt0': defect_dist_detail,
    'timing': {
        'total_pipeline_time_s': round(
            sum(r.get('total_time_ms', 0) for r in data) / 1000, 1
        ),
        'llm_time_ms': desc(llm_times),
        'monitor_time_ms': desc(mon_times),
    },
    'per_part': per_part_detail,
    'llm_parse_failures': [
        {'test_id': r['test_id'], 'part': r['part'],
         'error': r.get('llm_error', '')[:120]}
        for r in data if r.get('llm_error')
    ],
}

# ── Write ──
outpath = os.path.join(out_folder, 'DETAILED_METRICS.json')
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"\nMetrics saved to: {outpath}")
print(f"\n=== KEY RESULTS ===")
print(f"Seeded Detection Rate: {metrics['seeded_tests']['detection_rate']}")
print(f"  ({total_detected}/{total_evaluable} detected, {total_nil} parse failures)")
print(f"Natural Pass Rate:     {metrics['natural_tests']['pass_rate']}")
print(f"Natural Defect Rate:   {metrics['natural_tests']['defect_rate']}")
print(f"Self-Examine:          {direct_pass} direct, {corrected} corrected, {failed} failed")
print(f"Avg LLM time:          {metrics['timing']['llm_time_ms']['mean']}ms")
print(f"Avg Monitor time:      {metrics['timing']['monitor_time_ms']['mean']}ms")
print(f"\nPer-Defect Type:")
for d_id, detail in sorted(by_defect_detail.items()):
    print(f"  {d_id}: {detail['detection_rate']} ({detail['detected']}/{detail['evaluable']})")
