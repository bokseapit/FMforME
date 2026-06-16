"""Compute Flash metrics from newer experiment_results.json (now backed up to dschat folder)."""
import json
import os
from collections import Counter

# Read the current experiment_results.json which is the new Flash run
results_path = r'C:\Users\32506\Desktop\FMforME\experiment_results.json'
with open(results_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

seeded = [r for r in data if r.get('has_injected_defect')]
natural = [r for r in data if not r.get('has_injected_defect')]

# ── Seeded detection from attempt-0 (correct logic) ──
by_defect = {}
for r in seeded:
    exp = r.get('expected_defect', '?')
    if exp not in by_defect:
        by_defect[exp] = {'total': 0, 'detected': 0, 'nil': 0}
    by_defect[exp]['total'] += 1
    se_hist = r.get('self_examine_history')
    if se_hist and len(se_hist) > 0:
        if exp in se_hist[0].get('defect_types', []):
            by_defect[exp]['detected'] += 1
    else:
        by_defect[exp]['nil'] += 1

# ── SE stats ──
se_tests = [r for r in data if r.get('self_examine_history')]
direct_pass = sum(1 for r in se_tests if len(r['self_examine_history']) == 1 and r['self_examine_history'][0]['passed'])
corrected = sum(1 for r in se_tests if len(r['self_examine_history']) > 1 and r['passed_monitor'])
failed = sum(1 for r in se_tests if not r['passed_monitor'] and len(r['self_examine_history']) > 1)

# ── Defect distribution (attempt-0) ──
all_d = Counter()
for r in data:
    if r.get('self_examine_history'):
        for d in r['self_examine_history'][0].get('defect_types', []):
            all_d[d] += 1

# ── Per-part ──
by_part = {}
for r in data:
    p = r['part']
    if p not in by_part:
        by_part[p] = {'total': 0, 'fusion_ok': 0, 'llm_fail': 0, 'seeded_det': 0, 'seeded_eval': 0}
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

# ── Natural by part ──
natural_by_part = {}
for r in natural:
    p = r['part']
    if p not in natural_by_part:
        natural_by_part[p] = {'total': 0, 'passed': 0, 'defects': 0}
    natural_by_part[p]['total'] += 1
    if r['passed_monitor']:
        natural_by_part[p]['passed'] += 1
    if len(r['violations']) > 0:
        natural_by_part[p]['defects'] += 1

# ── Correction attempt distribution ──
attempt_dist = Counter()
for r in se_tests:
    attempt_dist[len(r['self_examine_history'])] += 1

# ── Build report ──
total_evaluable = sum(v['total'] - v['nil'] for v in by_defect.values())
total_detected = sum(v['detected'] for v in by_defect.values())
total_nil = sum(v['nil'] for v in by_defect.values())

by_defect_detail = {}
for d, info in sorted(by_defect.items()):
    ev = info['total'] - info['nil']
    rate_str = "{}/{} ({:.1f}%)".format(info['detected'], ev, info['detected']/ev*100) if ev > 0 else 'N/A'
    by_defect_detail[d] = {
        'total': info['total'], 'detected': info['detected'],
        'nil_parse_fail': info['nil'], 'evaluable': ev, 'rate': rate_str
    }

total_defects = sum(all_d.values())
defect_dist_detail = {}
for d, c in all_d.most_common():
    defect_dist_detail[d] = {'frequency': c, 'proportion': "{:.1f}%".format(c/total_defects*100) if total_defects > 0 else '0%'}

per_part_detail = {}
for p, info in sorted(by_part.items()):
    sd = "{}/{}".format(info['seeded_det'], info['seeded_eval']) if info['seeded_eval'] > 0 else 'N/A'
    per_part_detail[p] = {
        'total_tests': info['total'], 'fusion_success': info['fusion_ok'],
        'llm_failures': info['llm_fail'], 'seeded_detection': sd,
    }

nat_by_part_detail = {}
for p, info in sorted(natural_by_part.items()):
    nat_by_part_detail[p] = {
        'total': info['total'],
        'passed': "{}/{}".format(info['passed'], info['total']),
        'defect_rate': "{}/{} ({:.0f}%)".format(info['defects'], info['total'], info['defects']/info['total']*100),
    }

llm_times = [r['llm_time_ms'] for r in data if r.get('llm_time_ms', 0) > 0]
monitor_times = [r['monitor_time_ms'] for r in data if r.get('monitor_time_ms', 0) > 0]

metrics = {
    'model': 'DeepSeek-V4-Flash',
    'experiment_date': '2026-06',
    'total_tests': 33,
    'seeded_tests': {
        'n_total': 18, 'n_evaluable': total_evaluable,
        'n_llm_parse_failures': total_nil, 'n_detected': total_detected,
        'detection_rate': "{}/{} ({:.1f}%)".format(total_detected, total_evaluable, total_detected/total_evaluable*100),
        'by_defect_type': by_defect_detail,
    },
    'self_examine': {
        'n_tests_with_se': len(se_tests),
        'direct_pass': direct_pass,
        'corrected_pass': corrected,
        'failed_after_retries': failed,
        'total_llm_attempts': sum(len(r['self_examine_history']) for r in se_tests),
        'avg_attempts_per_test': round(sum(len(r['self_examine_history']) for r in se_tests)/len(se_tests), 1),
        'attempt_distribution': {str(k): v for k, v in sorted(attempt_dist.items())},
    },
    'natural_tests': {
        'n_total': len(natural),
        'n_passed_monitor': sum(1 for r in natural if r['passed_monitor']),
        'n_failed_monitor': sum(1 for r in natural if not r['passed_monitor']),
        'n_with_defects': sum(1 for r in natural if len(r['violations']) > 0),
        'pass_rate': "{}/{} ({:.0f}%)".format(
            sum(1 for r in natural if r['passed_monitor']), len(natural),
            sum(1 for r in natural if r['passed_monitor'])/len(natural)*100),
        'defect_rate': "{}/{} ({:.1f}%)".format(
            sum(1 for r in natural if len(r['violations'])>0), len(natural),
            sum(1 for r in natural if len(r['violations'])>0)/len(natural)*100),
        'by_part': nat_by_part_detail,
    },
    'fusion360': {
        'build_success': sum(1 for r in data if r.get('fusion_success')),
        'total': len(data),
        'build_rate': "{}/{} ({:.1f}%)".format(
            sum(1 for r in data if r.get('fusion_success')), len(data),
            sum(1 for r in data if r.get('fusion_success'))/len(data)*100),
    },
    'defect_distribution_attempt0': defect_dist_detail,
    'timing': {
        'total_pipeline_time_s': round(sum(r.get('total_time_ms',0) for r in data)/1000, 1),
        'avg_llm_time_ms': round(sum(llm_times)/len(llm_times), 1) if llm_times else 0,
        'avg_monitor_time_ms': round(sum(monitor_times)/len(monitor_times), 3) if monitor_times else 0,
    },
    'per_part': per_part_detail,
    'llm_parse_failures': [
        {'test_id': r['test_id'], 'part': r['part'], 'error': r.get('llm_error','')[:120]}
        for r in data if r.get('llm_error')
    ],
}

outpath = r'C:\Users\32506\Desktop\FMforME\generated_specs\dschat\V4FLASH_METRICS.json'
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("Flash metrics saved to:", outpath)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
