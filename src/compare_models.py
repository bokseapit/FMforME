"""Generate side-by-side comparison of Flash vs Pro metrics."""
import json

flash_path = r'C:\Users\32506\Desktop\FMforME\generated_specs\dschat\V4FLASH_METRICS.json'
pro_path = r'C:\Users\32506\Desktop\FMforME\generated_specs\dsV4pro\V4PRO_METRICS.json'

with open(flash_path, 'r', encoding='utf-8') as f:
    flash = json.load(f)
with open(pro_path, 'r', encoding='utf-8') as f:
    pro = json.load(f)

print("=" * 80)
print("CROSS-MODEL COMPARISON: DeepSeek-V4-Flash vs DeepSeek-V4-Pro")
print("=" * 80)

print("\n### SEEDED DEFECT DETECTION ###")
print(f"{'Metric':40s} {'V4-Flash':>15s} {'V4-Pro':>15s}")
print("-" * 70)
fs = flash['seeded_tests']
ps = pro['seeded_tests']
print(f"{'Detection Rate':40s} {fs['detection_rate']:>15s} {ps['detection_rate']:>15s}")
print(f"{'LLM Parse Failures':40s} {str(fs['n_llm_parse_failures']):>15s} {str(ps['n_llm_parse_failures']):>15s}")
print()
print("Per defect type:")
for d in sorted(set(list(fs['by_defect_type'].keys()) + list(ps['by_defect_type'].keys()))):
    fr = fs['by_defect_type'].get(d, {}).get('rate', 'N/A')
    pr = ps['by_defect_type'].get(d, {}).get('rate', 'N/A')
    print(f"  {d:5s} {fr:>15s} {pr:>15s}")

print("\n### SELF-EXAMINE ###")
fse = flash['self_examine']
pse = pro['self_examine']
print(f"{'Metric':40s} {'V4-Flash':>15s} {'V4-Pro':>15s}")
print("-" * 70)
print(f"{'Tests with SE data':40s} {str(fse['n_tests_with_se']):>15s} {str(pse['n_tests_with_se']):>15s}")
print(f"{'Direct Pass':40s} {str(fse['direct_pass']):>15s} {str(pse['direct_pass']):>15s}")
print(f"{'Corrected Pass':40s} {str(fse['corrected_pass']):>15s} {str(pse['corrected_pass']):>15s}")
print(f"{'Failed After Retries':40s} {str(fse['failed_after_retries']):>15s} {str(pse['failed_after_retries']):>15s}")
print(f"{'Avg Attempts/Test':40s} {str(fse['avg_attempts_per_test']):>15s} {str(pse['avg_attempts_per_test']):>15s}")
print(f"{'Attempt Distribution':40s} {str(fse['attempt_distribution']):>15s} {str(pse.get('attempt_distribution','N/A')):>15s}")

print("\n### NATURAL TESTS ###")
fn = flash['natural_tests']
pn = pro['natural_tests']
print(f"{'Metric':40s} {'V4-Flash':>15s} {'V4-Pro':>15s}")
print("-" * 70)
print(f"{'Pass Rate':40s} {fn['pass_rate']:>15s} {pn['pass_rate']:>15s}")
print(f"{'Defect Rate (WARNs)':40s} {fn['defect_rate']:>15s} {pn['defect_rate']:>15s}")
print()
for p in sorted(set(list(fn.get('by_part',{}).keys()) + list(pn.get('by_part',{}).keys()))):
    fpb = fn.get('by_part', {}).get(p, {})
    ppb = pn.get('by_part', {}).get(p, {})
    # For Pro, compute nat defect rate manually — not in metrics but we can pull from per_part
    print(f"  {p:20s} defects: {fpb.get('defect_rate','?'):>10s} vs {ppb.get('defect_rate','?'):>10s}")

print("\n### FUSION 360 BUILD ###")
ff = flash['fusion360']
pf = pro['fusion360']
print(f"{'Metric':40s} {'V4-Flash':>15s} {'V4-Pro':>15s}")
print("-" * 70)
print(f"{'Build Success':40s} {ff['build_rate']:>15s} {pf['build_rate']:>15s}")

print("\n### DEFECT DISTRIBUTION (attempt-0) ###")
fd = flash['defect_distribution_attempt0']
pd = pro['defect_distribution_attempt0']
print(f"{'Defect':10s} {'V4-Flash Freq':>15s} {'V4-Flash %':>12s} {'V4-Pro Freq':>13s} {'V4-Pro %':>10s}")
print("-" * 65)
for d in sorted(set(list(fd.keys()) + list(pd.keys()))):
    ffreq = str(fd.get(d, {}).get('frequency', 0))
    fpct = fd.get(d, {}).get('proportion', '0%')
    pfreq = str(pd.get(d, {}).get('frequency', 0))
    ppct = pd.get(d, {}).get('proportion', '0%')
    print(f"  {d:5s} {ffreq:>15s} {fpct:>12s} {pfreq:>13s} {ppct:>10s}")

print("\n### TIMING ###")
ft = flash['timing']
pt = pro['timing']
print(f"{'Metric':40s} {'V4-Flash':>15s} {'V4-Pro':>15s}")
print("-" * 70)
print(f"{'Total Pipeline Time':40s} {str(ft['total_pipeline_time_s'])+'s':>15s} {str(pt['total_pipeline_time_s'])+'s':>15s}")
print(f"{'Avg LLM Time':40s} {str(ft['avg_llm_time_ms'])+'ms':>15s} {str(pt['avg_llm_time_ms'])+'ms':>15s}")
print(f"{'Avg Monitor Time':40s} {str(ft['avg_monitor_time_ms'])+'ms':>15s} {str(pt['avg_monitor_time_ms'])+'ms':>15s}")

print("\n### PER-PART ###")
fp = flash['per_part']
pp = pro['per_part']
print(f"{'Part':20s} {'Metric':20s} {'V4-Flash':>15s} {'V4-Pro':>15s}")
print("-" * 70)
for p in sorted(set(list(fp.keys()) + list(pp.keys()))):
    fpi = fp.get(p, {})
    ppi = pp.get(p, {})
    print(f"{p:20s} {'Fusion Success':20s} {str(fpi.get('fusion_success','?')):>15s} {str(ppi.get('fusion_success','?')):>15s}")
    print(f"{'':20s} {'LLM Failures':20s} {str(fpi.get('llm_failures','?')):>15s} {str(ppi.get('llm_failures','?')):>15s}")
    print(f"{'':20s} {'Seeded Detection':20s} {str(fpi.get('seeded_detection','?')):>15s} {str(ppi.get('seeded_detection','?')):>15s}")

print("\n### LLM PARSE FAILURES ###")
print(f"V4-Flash: {len(flash['llm_parse_failures'])} failures")
print(f"V4-Pro:   {len(pro['llm_parse_failures'])} failures")
for f in pro['llm_parse_failures']:
    print(f"  {f['test_id']} ({f['part']}): {f['error'][:80]}")
