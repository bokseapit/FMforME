# FMforME — Fusion Modeling for Mechanical Engineering

A three-layer runtime constraint verification framework with self-correction for AI-generated parametric CAD models.

## Overview

FMforME addresses the reliability gap between LLM-generated parametric CAD specifications and physically valid FEA-ready models. The framework consists of:

- **Layer 1 — LLM Generation**: Structured system-prompt-driven ModelSpec JSON generation via DeepSeek or OpenAI-compatible APIs
- **Layer 2 — Runtime Monitor**: Six standard-grounded defect detection predicates (D1–D6) with Wilson 95% confidence intervals
- **Layer 3 — CAD Execution**: Fusion 360 Python add-in for parametric solid modeling with STEP/F3D export

A core component is the **Self-Examine** subsystem, which iteratively feeds structured engineering feedback (field paths, actual values, standard references) back to the LLM upon detecting violations.

## Repository Structure

```
FMforME/
├── src/                          # Python source code (10 modules, ~3,500 lines)
│   ├── run.py                    # One-click experiment entry point
│   ├── run_experiment.py         # Experiment orchestration
│   ├── layer1_llm.py             # LLM API interface (multi-backend)
│   ├── layer2_monitor.py         # D1–D6 defect detection predicates
│   ├── layer2_self_examine.py    # Self-Examine correction loop
│   ├── layer3_fusion_addin.py    # Fusion 360 parametric modeling add-in
│   ├── layer3_fea_sim.py         # FEA simulation interface
│   ├── model_spec_schema.py      # ModelSpec JSON schema
│   ├── stats_utils.py            # Wilson CI, binomial tests, Cohen's h
│   ├── evaluate.py               # Experiment evaluation (RQ1–RQ4)
│   ├── compute_metrics.py        # Metric computation pipeline
│   ├── compare_models.py         # Cross-model comparison
│   └── compute_flash_metrics.py  # Flash-specific metrics
│
├── config/
│   └── structural_rules.yaml     # Declarative YAML-based defect rules (DSL)
│
├── fusion_addin/                 # Fusion 360 add-in deployment package
│   ├── FMforME_AddIn.py          # Add-in entry point
│   └── layer3_fusion_addin.py    # Main add-in module (copy from src/)
│
├── data/
│   ├── experiment_results/       # Aggregated results per model
│   │   ├── deepseek_v4_flash_results.json
│   │   ├── deepseek_v4_flash_summary.json
│   │   ├── deepseek_v4_pro_results.json
│   │   ├── deepseek_v4_pro_summary.json
│   │   ├── qwen_flash_results.json
│   │   └── qwen_flash_summary.json
│   │
│   └── generated_specs/          # Generated ModelSpec JSON files
│       ├── dschat/               # DeepSeek-V4-Flash (old batch, 33 tests)
│       ├── dsV4pro/              # DeepSeek-V4-Pro (old batch, 25 tests)
│       └── qwen_flash/           # Qwen 3.6 Flash (current batch, 141 tests)
│
│
└── README.md                     # This file
```

## Quick Start

### Prerequisites

- Python 3.9+
- Fusion 360 (for Layer 3 CAD execution)
- API keys for at least one LLM backend

### Installation

```bash
git clone https://github.com/your-org/FMforME.git
cd FMforME
pip install -r requirements.txt
```

### Running Experiments

```powershell
# DeepSeek-V4-Flash (default)
$env:DEEPSEEK_API_KEY="sk-your-key"
$env:LLM_BACKEND="deepseek"
python src/run.py

# DeepSeek-V4-Pro
$env:DEEPSEEK_API_KEY="sk-your-key"
$env:LLM_BACKEND="deepseek-pro"
python src/run.py

# Qwen 3.6 Flash (via Alibaba Cloud DashScope)
$env:OPENAI_API_KEY="your-dashscope-key"
$env:LLM_BACKEND="openai_compatible"
$env:LLM_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen-flash"
python src/run.py
```

### Fusion 360 Add-in Installation

1. Copy the `fusion_addin/` folder to:
   ```
   %APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\FMforME\
   ```
2. In Fusion 360: **Tools → Add-Ins → FMforME → Run**
3. The add-in monitors `~/fusion_bridge/clean_spec.json` and builds models automatically

## Key Results

Systematic evaluation across 141 test cases per model (96 seeded defect injection, 45 natural generation) on three LLM backends:

| Model | Seeded Detection | Natural Pass | LLM Latency (ms) | Total Time (s) |
|---|---|---|---|---|
| Qwen 3.6 Flash | 95.8% [89.8–98.4] | 100% [92.1–100] | 5,286 | 1,779.6 |
| DeepSeek-V4-Flash | 91.7% [84.4–95.7] | 100% [92.1–100] | 4,395 | 1,811.7 |
| DeepSeek-V4-Pro | 90.6% [83.1–95.0] | 75.6% [61.3–85.8] | 48,011 | 8,015.8 |

Wilson 95% confidence intervals shown in brackets. Monitor overhead: ~0.1 ms per check (2.9–4.1% of 3 ms target).

## Defect Taxonomy (D1–D6)

| ID | Defect | Standard Reference |
|---|---|---|
| D1 | Unconstrained DOF | ASME V&V 10-2019 §4 |
| D2 | Negative Stiffness | Bathe FEM §2.3 |
| D3 | Stress Singularity | Knupp 2001 |
| D4 | Load-BC Conflict | ASME V&V 10 §5.1 |
| D5 | Material Violation | ISO 286 / ASTM |
| D6 | Mesh Topology | Verdict Library (Stimpson 2007) |

## Citation

If you use FMforME in your research, please cite:

```bibtex
waiting for accept
```

## License

This project is available under the MIT License. See LICENSE file for details.

## Contact

Timothy Lee — timothylee@intl.zju.edu.cn
Zhejiang University — University of Illinois at Urbana-Champaign Institute
