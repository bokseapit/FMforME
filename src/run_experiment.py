"""
Main Experiment Script
Orchestrates the full three-layer pipeline across all test cases.

Test Matrix (60 seeded + 90 natural = 150 tests total):
  S1-S60: Seeded defects — 10 per defect category (D1-D6),
          distributed evenly across 3 part types (bolt, bracket, bearing)
  N1-N90: Natural generation — 30 runs × 3 part types

Each seeded test runs N_REPEATS times (default 3) to measure
per-test variance across LLM calls, enabling standard deviation
and confidence interval reporting.

Pipeline Flow:
  User NL → Layer 1 (LLM) → ModelSpec JSON
  → Layer 2 (Monitor) → check D1-D6 → ViolationReport
  → Layer 3 (Fusion) → parametric CAD → result.json

Output:
  experiment_results.json — raw per-test results
  experiment_summary.json — aggregated metrics with CIs
"""

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer1_llm import LLMGenerator
from layer2_monitor import StructuralMonitor
from layer2_self_examine import SelfExamineMonitor
from layer3_fusion_addin import execute_directly


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# LLM backend: "deepseek", "deepseek-pro", "gemini", or "openai_compatible"
LLM_BACKEND = os.environ.get("LLM_BACKEND", "deepseek")
LLM_API_KEY = (os.environ.get("DEEPSEEK_API_KEY") or
               os.environ.get("GEMINI_API_KEY") or
               os.environ.get("OPENAI_API_KEY"))

# Output paths
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.json")
SUMMARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_summary.json")
SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_specs")
os.makedirs(SPECS_DIR, exist_ok=True)

# Number of natural-generation runs per part for RQ2/RQ3
NATURAL_RUNS_PER_PART = int(os.environ.get("NATURAL_RUNS", "15"))

# Number of repeated runs per seeded test case (to measure variance)
N_REPEATS = int(os.environ.get("N_REPEATS", "2"))

# Self-Examine correction loop flag
SELF_EXAMINE = os.environ.get("SELF_EXAMINE", "true").lower() == "true"
SELF_EXAMINE_MAX_RETRIES = int(os.environ.get("SELF_EXAMINE_RETRIES", "3"))

# Scenario mode: "seeded" (default) or "four_scenarios" (A/B/C/D comparison)
SCENARIO_MODE = os.environ.get("SCENARIO_MODE", "seeded")
FOUR_SCENARIO_RUNS = int(os.environ.get("FOUR_SCENARIO_RUNS", "3"))

# Scenario prompts (minimal vs guided)
UNGUIDED_PROMPT = "You are a CAD assistant. Generate a ModelSpec JSON. Output ONLY valid JSON."
GUIDED_PROMPT = (
    "You are a mechanical FEA assistant. Generate ModelSpec JSON. "
    "Rules: Poisson ratio in (0,0.5), Young modulus in GPa (>0), density in kg/m3 (>0). "
    "Include: dimensions, material, boundary_conditions (list of objects with node_id, dof, type), loads (list), mesh. "
    "Output ONLY valid JSON."
)


# ═══════════════════════════════════════════════════════════════════════════
# Test Case Definitions — 48 Seeded Tests (8 per defect category D1-D6)
# ═══════════════════════════════════════════════════════════════════════════
#
# Each D* has 8 cases across 3 part types. With N_REPEATS=2: 16 data points
# per defect — narrow enough Wilson CI for reviewer acceptance.
#
# Per model: 48 × 2 + 15 natural/part × 3 = 96 + 45 = 141 tests
# Three models: ~423 runs total (DS Flash + DS Pro + GLM cross-validation)

SEEDED_TEST_CASES = [
    # ═══════════════════════════════════════════════════════════════════════
    # D4: Load-Boundary Condition Conflict (8 tests)
    # Standard: ASME V&V-10 §5.1
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "S01", "part": "l_bracket",
        "desc": "50x50x6mm steel L-bracket with 5mm fillet radius per JIS B",
        "inject": {"path": "loads", "value": [
            {"node_id": 1, "direction": "-ty", "magnitude": 500.0}
        ]},
        "expected_defect": "D4", "expected_severity": "WARN",
        "notes": "Load at BC node 1 — absorbed load",
    },
    {
        "id": "S02", "part": "bolt_iso4762",
        "desc": "M10x40 steel hex bolt, axial tensile loading",
        "inject": {"path": "loads", "value": [
            {"node_id": 1, "direction": "tz", "magnitude": 1000.0}
        ]},
        "expected_defect": "D4", "expected_severity": "WARN",
        "notes": "Axial load at BC node",
    },
    {
        "id": "S03", "part": "ball_bearing",
        "desc": "6205 deep groove ball bearing, radial load test",
        "inject": {"path": "loads", "value": [
            {"node_id": 1, "direction": "-ty", "magnitude": 2000.0}
        ]},
        "expected_defect": "D4", "expected_severity": "WARN",
        "notes": "Radial load at inner ring BC node",
    },
    {
        "id": "S04", "part": "l_bracket",
        "desc": "Steel bracket 60x60x8mm, multiple load nodes",
        "inject": {"path": "loads", "value": [
            {"node_id": 1, "direction": "-tz", "magnitude": 300.0},
            {"node_id": 2, "direction": "-ty", "magnitude": 400.0},
        ]},
        "expected_defect": "D4", "expected_severity": "WARN",
        "notes": "Mixed loads with one at BC node 1",
    },
    {
        "id": "S05", "part": "bolt_iso4762",
        "desc": "M8x25 stainless bolt, shear loading at head",
        "inject": [
            {"path": "boundary_conditions", "value": [
                {"node_id": 1, "dof": ["tx","ty","tz","rx","ry","rz"], "type": "fixed"},
                {"node_id": 3, "dof": ["ty"], "type": "sliding"}
            ]},
            {"path": "loads", "value": [
                {"node_id": 3, "direction": "-ty", "magnitude": 800.0}
            ]},
        ],
        "expected_defect": "D4", "expected_severity": "WARN",
        "notes": "Load at sliding BC node 3",
    },
    {
        "id": "S06", "part": "ball_bearing",
        "desc": "6004 bearing, outer ring BC + load overlap",
        "inject": [
            {"path": "boundary_conditions", "value": [
                {"node_id": 2, "dof": ["tx","ty","tz"], "type": "fixed"}
            ]},
            {"path": "loads", "value": [
                {"node_id": 2, "direction": "tx", "magnitude": 500.0}
            ]},
        ],
        "expected_defect": "D4", "expected_severity": "WARN",
        "notes": "Load and BC at same outer-ring node",
    },
    {
        "id": "S07", "part": "bolt_iso4762",
        "desc": "M6x16 bolt, load applied at fixed shank node",
        "inject": [
            {"path": "boundary_conditions", "value": [
                {"node_id": 1, "dof": ["tx","ty","tz","rx","ry","rz"], "type": "fixed"},
                {"node_id": 2, "dof": ["tx","ty"], "type": "sliding"}
            ]},
            {"path": "loads", "value": [
                {"node_id": 2, "direction": "-tz", "magnitude": 400.0}
            ]},
        ],
        "expected_defect": "D4", "expected_severity": "WARN",
        "notes": "Load at sliding BC node 2 — absorbed reaction",
    },
    {
        "id": "S08", "part": "l_bracket",
        "desc": "Steel bracket 80x80x10mm, load on fixed mounting face",
        "inject": [
            {"path": "boundary_conditions", "value": [
                {"node_id": 1, "dof": ["tx","ty","tz","rx","ry","rz"], "type": "fixed"}
            ]},
            {"path": "loads", "value": [
                {"node_id": 1, "direction": "-ty", "magnitude": 2000.0},
                {"node_id": 3, "direction": "-tz", "magnitude": 1500.0},
            ]},
        ],
        "expected_defect": "D4", "expected_severity": "WARN",
        "notes": "Multiple loads, one at fixed BC node 1",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # D5: Material Property Violation (8 tests)
    # Standard: ISO 286 / ASTM Material Standards
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "S11", "part": "bolt_iso4762",
        "desc": "M8x30 stainless steel hex socket bolt per ISO 4762",
        "inject": {"path": "material.poisson_ratio", "value": 0.6},
        "expected_defect": "D5", "expected_severity": "ERROR",
        "notes": "Poisson ratio > 0.5",
    },
    {
        "id": "S12", "part": "bolt_iso4762",
        "desc": "M8x30 bolt, negative Young's modulus",
        "inject": {"path": "material.youngs_modulus", "value": -200},
        "expected_defect": "D5", "expected_severity": "ERROR",
        "notes": "Negative Young's modulus",
    },
    {
        "id": "S13", "part": "ball_bearing",
        "desc": "6204 bearing, density near-zero",
        "inject": {"path": "material.density", "value": 0.01},
        "expected_defect": "D5", "expected_severity": "ERROR",
        "notes": "Density 0.01 kg/m³",
    },
    {
        "id": "S14", "part": "l_bracket",
        "desc": "L-bracket 50x50x6mm, Poisson ratio 0.8",
        "inject": {"path": "material.poisson_ratio", "value": 0.8},
        "expected_defect": "D5", "expected_severity": "ERROR",
        "notes": "Poisson 0.8 — LLM hallucination",
    },
    {
        "id": "S15", "part": "bolt_iso4762",
        "desc": "M8 bolt, yield > tensile",
        "inject": [
            {"path": "material.yield_strength", "value": 600},
            {"path": "material.tensile_strength", "value": 500},
        ],
        "expected_defect": "D5", "expected_severity": "ERROR",
        "notes": "Yield(600) > Tensile(500)",
    },
    {
        "id": "S16", "part": "ball_bearing",
        "desc": "6205 bearing, yield/tensile inversion",
        "inject": [
            {"path": "material.yield_strength", "value": 800},
            {"path": "material.tensile_strength", "value": 700},
        ],
        "expected_defect": "D5", "expected_severity": "ERROR",
        "notes": "Yield(800) > Tensile(700)",
    },
    {
        "id": "S17", "part": "l_bracket",
        "desc": "Aluminum bracket, density out of bounds (high)",
        "inject": {"path": "material.density", "value": 50000},
        "expected_defect": "D5", "expected_severity": "ERROR",
        "notes": "Density 50000 kg/m³ — exceeds osmium",
    },
    {
        "id": "S18", "part": "bolt_iso4762",
        "desc": "M8 bolt, Poisson exactly zero",
        "inject": {"path": "material.poisson_ratio", "value": 0.0},
        "expected_defect": "D5", "expected_severity": "WARN",
        "notes": "Poisson=0 — physically impossible for metals",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # D1: Unconstrained Degrees of Freedom (8 tests)
    # Standard: ASME V&V-10-2019 §4
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "S21", "part": "l_bracket",
        "desc": "50x50x6mm steel L-bracket, sliding-only BC",
        "inject": {"path": "boundary_conditions", "value": [
            {"node_id": 1, "dof": ["ty", "tz"], "type": "sliding"}
        ]},
        "expected_defect": "D1", "expected_severity": "WARN",
        "notes": "Only ty,tz — tx,rx,ry,rz free (4 DOF)",
    },
    {
        "id": "S22", "part": "bolt_iso4762",
        "desc": "M8x30 bolt, no boundary conditions",
        "inject": {"path": "boundary_conditions", "value": []},
        "expected_defect": "D1", "expected_severity": "ERROR",
        "notes": "Empty BC → 6 unconstrained DOF",
    },
    {
        "id": "S23", "part": "ball_bearing",
        "desc": "6204 bearing, single pinned support only",
        "inject": {"path": "boundary_conditions", "value": [
            {"node_id": 1, "dof": ["tz"], "type": "pinned"}
        ]},
        "expected_defect": "D1", "expected_severity": "WARN",
        "notes": "Only tz — 5 rigid body DOF",
    },
    {
        "id": "S24", "part": "bolt_iso4762",
        "desc": "M10 bolt, translational-only BC (no rotational)",
        "inject": {"path": "boundary_conditions", "value": [
            {"node_id": 1, "dof": ["tx", "ty", "tz"], "type": "fixed"}
        ]},
        "expected_defect": "D1", "expected_severity": "WARN",
        "notes": "tx,ty,tz only — rx,ry,rz free",
    },
    {
        "id": "S25", "part": "l_bracket",
        "desc": "Bracket, single node pinned in 2 DOF",
        "inject": {"path": "boundary_conditions", "value": [
            {"node_id": 1, "dof": ["tx", "ty"], "type": "pinned"}
        ]},
        "expected_defect": "D1", "expected_severity": "WARN",
        "notes": "Only tx,ty — 4 DOF free",
    },
    {
        "id": "S26", "part": "ball_bearing",
        "desc": "6204 bearing, completely free — no BCs",
        "inject": {"path": "boundary_conditions", "value": []},
        "expected_defect": "D1", "expected_severity": "ERROR",
        "notes": "No BC — 6 DOF free rigid body",
    },
    {
        "id": "S27", "part": "bolt_iso4762",
        "desc": "M8 bolt, BC node with empty DOF array",
        "inject": {"path": "boundary_conditions", "value": [
            {"node_id": 1, "dof": [], "type": "fixed"}
        ]},
        "expected_defect": "D1", "expected_severity": "WARN",
        "notes": "Empty DOF — constrains nothing",
    },
    {
        "id": "S28", "part": "l_bracket",
        "desc": "Bracket, symmetry BC only (single DOF)",
        "inject": {"path": "boundary_conditions", "value": [
            {"node_id": 1, "dof": ["tx"], "type": "symmetry"}
        ]},
        "expected_defect": "D1", "expected_severity": "WARN",
        "notes": "Single symmetry BC — 5 DOF free",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # D2: Negative Stiffness (8 tests)
    # Standard: Bathe FEM Textbook §2.3
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "S29", "part": "ball_bearing",
        "desc": "Deep groove ball bearing — negative bore",
        "inject": {"path": "dimensions.bore_diameter", "value": -25.0},
        "expected_defect": "D2", "expected_severity": "ERROR",
        "notes": "Negative bore → negative stiffness",
    },
    {
        "id": "S30", "part": "bolt_iso4762",
        "desc": "M8x30 bolt — zero nominal diameter",
        "inject": {"path": "dimensions.nominal_diameter", "value": 0.0},
        "expected_defect": "D2", "expected_severity": "ERROR",
        "notes": "Zero diameter → singular stiffness",
    },
    {
        "id": "S31", "part": "l_bracket",
        "desc": "50x50x6mm L-bracket — negative thickness",
        "inject": {"path": "dimensions.thickness", "value": -3.0},
        "expected_defect": "D2", "expected_severity": "ERROR",
        "notes": "Negative thickness → negative bending stiffness",
    },
    {
        "id": "S32", "part": "bolt_iso4762",
        "desc": "M8 bolt — negative length",
        "inject": {"path": "dimensions.length", "value": -30.0},
        "expected_defect": "D2", "expected_severity": "ERROR",
        "notes": "Negative bolt length",
    },
    {
        "id": "S33", "part": "ball_bearing",
        "desc": "6204 bearing — negative outer diameter",
        "inject": {"path": "dimensions.outer_diameter", "value": -40.0},
        "expected_defect": "D2", "expected_severity": "ERROR",
        "notes": "Negative OD → negative radial stiffness",
    },
    {
        "id": "S34", "part": "l_bracket",
        "desc": "Bracket — zero width (degenerate plate)",
        "inject": {"path": "dimensions.width", "value": 0.0},
        "expected_defect": "D2", "expected_severity": "WARN",
        "notes": "Zero width → degenerate cross-section",
    },
    {
        "id": "S35", "part": "bolt_iso4762",
        "desc": "M10 bolt — zero thread pitch",
        "inject": {"path": "dimensions.thread_pitch", "value": 0.0},
        "expected_defect": "D2", "expected_severity": "WARN",
        "notes": "Zero thread pitch — no engagement",
    },
    {
        "id": "S36", "part": "ball_bearing",
        "desc": "6205 bearing — zero width (degenerate ring)",
        "inject": {"path": "dimensions.width", "value": 0.0},
        "expected_defect": "D2", "expected_severity": "ERROR",
        "notes": "Zero bearing width → zero volume",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # D3: Stress Singularity (8 tests)
    # Standard: Knupp 2001, SIAM J. Sci. Comput.
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "S37", "part": "ball_bearing",
        "desc": "6204 bearing — ball diameter near radial gap",
        "inject": {"path": "dimensions.ball_diameter", "value": 13.5},
        "expected_defect": "D3", "expected_severity": "WARN",
        "notes": "Ball near radial gap → thin raceway",
    },
    {
        "id": "S38", "part": "l_bracket",
        "desc": "50x50x6mm L-bracket — zero fillet",
        "inject": {"path": "dimensions.fillet_radius", "value": 0.0},
        "expected_defect": "D3", "expected_severity": "WARN",
        "notes": "Zero fillet → unbounded stress",
    },
    {
        "id": "S39", "part": "ball_bearing",
        "desc": "6205 bearing — ball diameter > radial gap",
        "inject": [
            {"path": "dimensions.bore_diameter", "value": 20},
            {"path": "dimensions.outer_diameter", "value": 47},
            {"path": "dimensions.ball_diameter", "value": 15.0},
        ],
        "expected_defect": "D3", "expected_severity": "ERROR",
        "notes": "Ball(15) > radial gap(13.5)",
    },
    {
        "id": "S40", "part": "ball_bearing",
        "desc": "Bearing — extreme mesh AR 100:1",
        "inject": {"path": "mesh.max_aspect_ratio", "value": 100.0},
        "expected_defect": "D3", "expected_severity": "WARN",
        "notes": "AR=100 >> 50 — severely degraded",
    },
    {
        "id": "S41", "part": "l_bracket",
        "desc": "Bracket — mesh AR 30 (degraded)",
        "inject": {"path": "mesh.max_aspect_ratio", "value": 30.0},
        "expected_defect": "D3", "expected_severity": "WARN",
        "notes": "AR=30 > 20 — element quality degraded",
    },
    {
        "id": "S42", "part": "ball_bearing",
        "desc": "Bearing — bore >= outer (impossible)",
        "inject": [
            {"path": "dimensions.bore_diameter", "value": 50},
            {"path": "dimensions.outer_diameter", "value": 47},
        ],
        "expected_defect": "D3", "expected_severity": "ERROR",
        "notes": "Bore(50) > Outer(47) → impossible",
    },
    {
        "id": "S43", "part": "ball_bearing",
        "desc": "Bearing — zero ball count",
        "inject": {"path": "dimensions.ball_count", "value": 0},
        "expected_defect": "D3", "expected_severity": "ERROR",
        "notes": "Zero balls → no load transfer",
    },
    {
        "id": "S44", "part": "l_bracket",
        "desc": "Bracket — negative fillet radius",
        "inject": {"path": "dimensions.fillet_radius", "value": -2.0},
        "expected_defect": "D3", "expected_severity": "ERROR",
        "notes": "Negative fillet → meaningless geometry",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # D6: Mesh Topology Errors (8 tests)
    # Standard: Verdict Library (Stimpson 2007)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "id": "S45", "part": "bolt_iso4762",
        "desc": "M8x30 bolt — negative Jacobian (inverted)",
        "inject": {"path": "mesh.min_jacobian", "value": -0.5},
        "expected_defect": "D6", "expected_severity": "ERROR",
        "notes": "Jacobian < 0 → inverted elements",
    },
    {
        "id": "S46", "part": "bolt_iso4762",
        "desc": "M8 bolt — marginal Jacobian (DIANA threshold)",
        "inject": {"path": "mesh.min_jacobian", "value": 0.2},
        "expected_defect": "D6", "expected_severity": "WARN",
        "notes": "Jacobian 0.2 < 0.5 — marginal",
    },
    {
        "id": "S47", "part": "ball_bearing",
        "desc": "6204 bearing — zero Jacobian (degenerate)",
        "inject": {"path": "mesh.min_jacobian", "value": 0.0},
        "expected_defect": "D6", "expected_severity": "ERROR",
        "notes": "Jacobian=0 → zero-volume elements",
    },
    {
        "id": "S48", "part": "l_bracket",
        "desc": "Bracket — severely negative Jacobian",
        "inject": {"path": "mesh.min_jacobian", "value": -1.0},
        "expected_defect": "D6", "expected_severity": "ERROR",
        "notes": "Jacobian=-1 → all elements inside-out",
    },
    {
        "id": "S49", "part": "bolt_iso4762",
        "desc": "M8 bolt — unknown element type",
        "inject": {"path": "mesh.element_type", "value": "quad9"},
        "expected_defect": "D6", "expected_severity": "WARN",
        "notes": "Invalid element type",
    },
    {
        "id": "S50", "part": "l_bracket",
        "desc": "Bracket — negative max_aspect_ratio",
        "inject": {"path": "mesh.max_aspect_ratio", "value": -2.0},
        "expected_defect": "D6", "expected_severity": "ERROR",
        "notes": "Negative AR → impossible mesh",
    },
    {
        "id": "S51", "part": "ball_bearing",
        "desc": "6205 bearing — Jacobian 0.1 (near-degenerate)",
        "inject": {"path": "mesh.min_jacobian", "value": 0.1},
        "expected_defect": "D6", "expected_severity": "WARN",
        "notes": "Jacobian 0.1 < 0.3 — critically low",
    },
    {
        "id": "S52", "part": "bolt_iso4762",
        "desc": "M8 bolt — Jacobian 0.25 (critically low)",
        "inject": {"path": "mesh.min_jacobian", "value": 0.25},
        "expected_defect": "D6", "expected_severity": "WARN",
        "notes": "Jacobian 0.25 < 0.3 — critically low",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Natural Generation Test Cases (for RQ2/RQ3)
# ═══════════════════════════════════════════════════════════════════════════

NATURAL_TEST_PARTS = [
    {
        "part": "bolt_iso4762",
        "desc": "M8x30 stainless steel hex socket head bolt per ISO 4762, with partial thread, chamfered tip, and knurled head edge",
        "label": "bolt",
    },
    {
        "part": "l_bracket",
        "desc": "50x50x6mm steel L-bracket with 5mm inside fillet radius, two 6mm mounting holes per leg with countersinks, and 2mm edge rounds per JIS B standard",
        "label": "bracket",
    },
    {
        "part": "ball_bearing",
        "desc": "6204 deep groove ball bearing, 20mm bore, 47mm OD, 14mm width, 8 chrome steel balls, stamped steel cage, double-shielded, per ISO 15",
        "label": "bearing",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Single Test Runner
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_spec_fields(spec: dict) -> dict:
    """Normalize LLM-generated field names to match expected schema.

    Handles common LLM quirks:
      - 'type' instead of 'part_type'
      - 'material_properties' instead of 'material'
      - Variant part_type spellings
    """
    # type -> part_type
    if "type" in spec and "part_type" not in spec:
        raw = str(spec.pop("type")).lower().replace("-", "_").replace(" ", "_")
        valid = {"bolt_iso4762", "l_bracket", "stepped_shaft", "ball_bearing"}
        spec["part_type"] = raw if raw in valid else "unknown"
    if "part_type" not in spec:
        spec["part_type"] = "unknown"

    # material_properties / properties -> material
    for alt in ("material_properties", "properties"):
        if alt in spec and "material" not in spec:
            spec["material"] = spec.pop(alt)

    # Normalize part_type values
    pt = spec.get("part_type", "")
    pt_map = {
        "l-bracket": "l_bracket", "lbracket": "l_bracket", "bracket": "l_bracket",
        "bolt": "bolt_iso4762", "iso4762": "bolt_iso4762", "m8_bolt": "bolt_iso4762",
        "steppedshaft": "stepped_shaft", "shaft": "stepped_shaft", "step_shaft": "stepped_shaft",
        "ballbearing": "ball_bearing", "bearing": "ball_bearing", "deep_groove_bearing": "ball_bearing",
    }
    if pt.lower() in pt_map:
        spec["part_type"] = pt_map[pt.lower()]

    return spec


def run_single_test(tc: dict, generator: LLMGenerator, monitor: StructuralMonitor) -> dict:
    """Execute one complete test case through all 3 layers.

    Parameters
    ----------
    tc : dict
        Test case definition with 'id', 'part', 'desc', optional 'inject'.
    generator : LLMGenerator
        Layer 1 LLM generator.
    monitor : StructuralMonitor
        Layer 2 Runtime Monitor.

    Returns
    -------
    dict
        Complete result record.
    """
    result = {
        "test_id": tc["id"],
        "part": tc["part"],
        "description": tc["desc"],
        "has_injected_defect": tc.get("inject") is not None,
        "injected_defect": tc.get("inject"),
        "expected_defect": tc.get("expected_defect"),
        "expected_severity": tc.get("expected_severity"),
    }

    # ── Layer 1: LLM Generation ───────────────────────────────────────
    t0 = time.time()

    SYSTEM_PROMPT = (
        "You are a CAD/FEA pre-processing assistant specializing in precision mechanical components. "
        "Generate a DETAILED, REALISTIC ModelSpec JSON for the described mechanical part. "
        "Include rich dimensional details — do NOT use minimal/default values. "
        "For bolts: include head_diameter, head_height, nominal_diameter, length, thread_pitch, "
        "head_chamfer_angle, socket_size. "
        "For brackets: include width, height, thickness, fillet_radius, hole_diameter, "
        "hole_count, edge_round_radius. "
        "For bearings: include bore_diameter, outer_diameter, width, ball_diameter, ball_count, "
        "raceway_groove_radius, cage_type, shield_type. "
        "CRITICAL: Use EXACTLY these JSON field names (case-sensitive): "
        "part_type (one of: bolt_iso4762, l_bracket, ball_bearing), "
        "dimensions (object with numeric values in mm), "
        "material (object with: name, youngs_modulus in GPa, poisson_ratio in [0,0.5], "
        "density in kg/m3, yield_strength in MPa, tensile_strength in MPa), "
        "boundary_conditions (array of objects with: node_id, dof array, type), "
        "loads (array of objects with: node_id, direction, magnitude in N), "
        "mesh (object with: element_type, min_jacobian, max_aspect_ratio). "
        "ALL values in standard SI-derived units (mm, MPa, GPa, kg/m3, N). "
        "Output ONLY valid JSON, no markdown, no extra text."
    )

    try:
        if SELF_EXAMINE:
            # ── Self-Examine Mode: iterative correction loop ──────
            se_monitor_obj = SelfExamineMonitor(
                generator, monitor,
                max_retries=SELF_EXAMINE_MAX_RETRIES,
                verbose=True,
            )
            # FIXED: pass inject_defect into the correction loop so it stays
            # in sync with the Monitor report
            spec, report, correction_history = se_monitor_obj.generate_with_correction(
                tc["desc"],
                system_prompt=SYSTEM_PROMPT,
                inject_defect=tc.get("inject"),
                metadata={
                    "test_id": tc["id"],
                    "experiment_timestamp": time.time(),
                    "has_injected_defect": tc.get("inject") is not None,
                },
            )
            result["self_examine_history"] = correction_history
            result["self_examine_stats"] = se_monitor_obj.get_stats()
        else:
            # ── Standard Mode: single-shot generation ──────────
            spec = generator.generate(
                tc["desc"],
                system_prompt=SYSTEM_PROMPT,
                inject_defect=tc.get("inject"),
                metadata={
                    "test_id": tc["id"],
                    "experiment_timestamp": time.time(),
                    "has_injected_defect": tc.get("inject") is not None,
                }
            )

        t1 = time.time()
        result["llm_time_ms"] = round((t1 - t0) * 1000, 1)

        # Save the full generated spec for traceability
        result["spec"] = spec

        # Inject test_id metadata into spec so the Fusion add-in
        # can name exported files correctly
        spec["_meta"] = {
            "test_id": tc["id"],
            "experiment_timestamp": time.time(),
            "has_injected_defect": tc.get("inject") is not None,
            "self_examine": SELF_EXAMINE,
        }

        # Normalize LLM field names to match expected schema
        _normalize_spec_fields(spec)

        # Save individual spec JSON file
        part_type = spec.get("part_type", "unknown")
        spec_filename = f"{tc['id']}_{part_type}.json"
        spec_path = os.path.join(SPECS_DIR, spec_filename)
        with open(spec_path, "w", encoding="utf-8") as sf:
            json.dump(spec, sf, ensure_ascii=False, indent=2)

        result["spec_summary"] = {
            "part_type": part_type,
            "has_material": bool(spec.get("material")),
            "has_bc": bool(spec.get("boundary_conditions")),
            "has_loads": bool(spec.get("loads")),
            "has_mesh": bool(spec.get("mesh")),
        }
    except Exception as e:
        t1 = time.time()
        result["llm_time_ms"] = round((t1 - t0) * 1000, 1)
        result["llm_error"] = str(e)
        result["passed_monitor"] = False
        result["violations"] = [{"id": "SYSTEM", "sev": "ERROR", "desc": f"LLM generation failed: {e}"}]
        result["n_violations"] = len(result["violations"])
        result["n_errors"] = 1
        result["n_warnings"] = 0
        result["defect_detected"] = None
        result["fusion_success"] = False
        result["fusion_error"] = "Skipped: LLM failed"
        result["monitor_time_ms"] = 0.0
        return result

    # ── Layer 2: Monitor Check ────────────────────────────────────────
    t2 = time.time()
    report = monitor.check(spec)
    t3 = time.time()
    result["monitor_time_ms"] = round((t3 - t2) * 1000, 3)

    result["violations"] = [
        {
            "id": v.defect_id,
            "sev": v.severity,
            "desc": v.description,
            "field_path": v.field_path,
            "standard": v.standard,
        }
        for v in report.violations
    ]
    result["passed_monitor"] = report.passed
    result["n_violations"] = len(report.violations)
    result["n_errors"] = len(report.errors())
    result["n_warnings"] = len(report.warnings())

    # Check if expected defect was detected — use attempt-0 (pre-correction)
    # when Self-Examine is enabled, because post-correction the defect has
    # been removed by design.
    expected = tc.get("expected_defect")
    if expected and result.get("self_examine_history"):
        # Use the FIRST attempt (pre-correction) for detection scoring
        h0 = result["self_examine_history"][0]
        result["defect_detected"] = expected in h0.get("defect_types", [])
    elif expected:
        detected_defects = report.defect_types()
        result["defect_detected"] = expected in detected_defects
    else:
        result["defect_detected"] = None

    # ── Layer 3: Fusion Modeling (only if monitor passes) ──────────────
    if report.passed:
        t4 = time.time()
        fusion_result = execute_directly(spec)
        t5 = time.time()
        result["fusion_time_ms"] = round((t5 - t4) * 1000, 1)
        result["fusion_success"] = fusion_result.get("success", False)
        if not fusion_result.get("success"):
            result["fusion_error"] = fusion_result.get("error", "Unknown Fusion error")
        result["model_info"] = fusion_result.get("model_info")
        result["exports"] = fusion_result.get("exports")
    else:
        result["fusion_time_ms"] = 0
        result["fusion_success"] = False
        result["fusion_error"] = "Blocked by Monitor"

    # ── Total pipeline time ───────────────────────────────────────────
    result["total_time_ms"] = round(
        result["llm_time_ms"] + result["monitor_time_ms"] + result.get("fusion_time_ms", 0), 1
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_all_tests(generator: LLMGenerator, monitor: StructuralMonitor) -> List[dict]:
    """Run ALL test cases: seeded (with repeats) + natural tests.

    Each unique seeded test case runs N_REPEATS times to measure
    per-test LLM variance, enabling standard deviation and CI reporting.

    Results are appended to experiment_results.json after each test,
    so interrupting mid-run does not lose completed results.

    Returns
    -------
    list[dict]
        All test results in order (seeded first, then natural).
    """
    all_results = []

    # Helper: append a result and checkpoint to disk
    def save_result(r: dict):
        all_results.append(r)
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── Phase 1: Seeded Tests (with N_REPEATS per case) ────────────────
    total_seeded_runs = len(SEEDED_TEST_CASES) * N_REPEATS
    print("=" * 70)
    print(f"PHASE 1: SEEDED DEFECT TESTS "
          f"({len(SEEDED_TEST_CASES)} unique × {N_REPEATS} repeats "
          f"= {total_seeded_runs} runs)")
    print("=" * 70)

    run_counter = 0
    for tc in SEEDED_TEST_CASES:
        inj = tc.get("inject")
        inj_desc = ""
        if inj:
            if isinstance(inj, list):
                parts = [f"{i['path']}={i['value']}" for i in inj]
                inj_desc = f" [inject: {', '.join(parts)}]"
            else:
                inj_desc = f" [inject: {inj['path']}={inj['value']}]"

        for rep in range(N_REPEATS):
            run_counter += 1
            # Create repeat-aware test case (unique ID per run)
            tc_repeat = dict(tc)
            tc_repeat["id"] = f"{tc['id']}_r{rep+1}"
            tc_repeat["repeat_index"] = rep

            if N_REPEATS > 1:
                label = f"[{run_counter}/{total_seeded_runs}]"
            else:
                label = ""

            print(f"\n{label} {tc['id']} (rep {rep+1}/{N_REPEATS}): "
                  f"{tc['part']}{inj_desc}")
            print(f"  Expected: {tc.get('expected_defect', '?')} "
                  f"[{tc.get('expected_severity', '?')}]")

            r = run_single_test(tc_repeat, generator, monitor)

            status = "PASS" if r["passed_monitor"] else "FAIL"
            detected = r.get("defect_detected")
            det_str = ("[DETECTED]" if detected is True else
                      ("[MISSED]" if detected is False else "N/A"))
            print(f"  -> Monitor: {status} | Violations: {r['n_violations']} "
                  f"({r['n_errors']}E, {r['n_warnings']}W) | {det_str}")
            print(f"  -> Times: LLM={r['llm_time_ms']}ms | "
                  f"Monitor={r['monitor_time_ms']}ms "
                  f"| Fusion={r.get('fusion_time_ms', 0)}ms")

            if not r["passed_monitor"] and r.get("n_errors", 0) > 0:
                for v in r["violations"]:
                    if v["sev"] == "ERROR":
                        print(f"     [ERROR] {v['desc'][:100]}")

            save_result(r)

    # ── Phase 2: Natural Generation Tests (RQ2/RQ3) ────────────────────
    total_natural = NATURAL_RUNS_PER_PART * len(NATURAL_TEST_PARTS)
    print("\n" + "=" * 70)
    print(f"PHASE 2: NATURAL GENERATION TESTS "
          f"({NATURAL_RUNS_PER_PART} runs × {len(NATURAL_TEST_PARTS)} parts "
          f"= {total_natural} runs)")
    print("=" * 70)

    nat_count = 0
    for part_info in NATURAL_TEST_PARTS:
        print(f"\n--- {part_info['label']} ({NATURAL_RUNS_PER_PART} runs) ---")
        for run_idx in range(NATURAL_RUNS_PER_PART):
            nat_count += 1
            tc = {
                "id": f"N{nat_count}",
                "part": part_info["part"],
                "desc": part_info["desc"],
                "inject": None,
                "expected_defect": None,
                "expected_severity": None,
                "notes": f"Natural generation run {run_idx+1}/{NATURAL_RUNS_PER_PART} "
                         f"for {part_info['label']}",
            }
            r = run_single_test(tc, generator, monitor)
            status = "PASS" if r["passed_monitor"] else "FAIL"
            print(f"  {tc['id']}: {status} | "
                  f"violations={[v['id'] for v in r['violations']]}")
            save_result(r)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Four-Scenario Comparison Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_four_scenario_comparison(generator, monitor):
    """Run A/B/C/D four-scenario comparison."""
    PARTS = [
        ("bolt_iso4762", "M8x30 hex socket bolt per ISO 4762, with chamfered tip and knurled head"),
        ("l_bracket", "50x50x6mm steel L-bracket with mounting holes and edge rounds"),
        ("ball_bearing", "6204 deep groove ball bearing, 20mm bore, 47mm OD, 14mm width, 8 balls"),
    ]

    all_results = {}
    se_mon = SelfExamineMonitor(generator, monitor, max_retries=SELF_EXAMINE_MAX_RETRIES, verbose=True)

    # ═══ SCENARIO D: SELF-EXAMINE ═══
    print("=" * 70)
    print("SCENARIO D: MONITOR + SELF-EXAMINE CORRECTION")
    print("=" * 70)
    for part_type, desc in PARTS:
        for run_idx in range(FOUR_SCENARIO_RUNS):
            tid = f"D_{part_type}_r{run_idx+1}"
            print(f"\n--- {tid} ---")
            t0 = time.time()
            spec, report, history = se_mon.generate_with_correction(
                desc, system_prompt=UNGUIDED_PROMPT)
            _normalize_spec_fields(spec)
            t1 = time.time()
            fusion_result = execute_directly(spec)

            all_results[tid] = {
                "scenario": "D_SELF_EXAMINE", "part": part_type,
                "n_attempts": len(history), "final_passed": report.passed,
                "final_errors": len(report.errors()), "final_warnings": len(report.warnings()),
                "final_defect_types": list(report.defect_types()),
                "correction_history": history,
                "fusion_success": fusion_result.get("success", False),
                "total_time_s": round(t1 - t0, 2),
            }
            print(f"  -> {'PASS' if report.passed else 'FAIL'} | "
                  f"{len(history)} attempts | {t1-t0:.1f}s")

    # ═══ SCENARIO A, B, C ═══
    for label, prompt, prefix, use_monitor in [
        ("A_UNCONSTRAINED", UNGUIDED_PROMPT, "A", False),
        ("B_GUIDED", GUIDED_PROMPT, "B", False),
        ("C_MONITORED", GUIDED_PROMPT, "C", True),
    ]:
        print(f"\n{'='*70}")
        print(f"SCENARIO {label}")
        print("=" * 70)
        for part_type, desc in PARTS:
            for run_idx in range(FOUR_SCENARIO_RUNS):
                tid = f"{prefix}_{part_type}_r{run_idx+1}"
                print(f"\n--- {tid} ---")
                t0 = time.time()
                spec = generator.generate(desc, system_prompt=prompt)
                _normalize_spec_fields(spec)
                report = monitor.check(spec)
                t1 = time.time()

                can_fuse = report.passed if use_monitor else True
                fusion_result = execute_directly(spec) if can_fuse else {
                    "success": False, "error": "Blocked by Monitor"}

                all_results[tid] = {
                    "scenario": label, "part": part_type,
                    "passed": report.passed,
                    "errors": len(report.errors()), "warnings": len(report.warnings()),
                    "defect_types": list(report.defect_types()),
                    "monitor_action": "ALLOWED" if can_fuse else "BLOCKED",
                    "fusion_success": fusion_result.get("success", False),
                    "time_s": round(t1 - t0, 2),
                }
                action = "FUSION" if can_fuse else "BLOCKED"
                print(f"  -> {'PASS' if report.passed else 'FAIL'} | "
                      f"{action} | {t1-t0:.1f}s")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from stats_utils import ci_string, wilson_ci, proportions_summary

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Runtime Constraint Verification Experiment               ║")
    print("║  AI-Assisted Structural Modeling (FMforME)                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nConfiguration:")
    print(f"  LLM Backend:      {LLM_BACKEND}")
    print(f"  Scenario mode:    {SCENARIO_MODE}")
    print(f"  Self-Examine:     {'ENABLED' if SELF_EXAMINE else 'disabled'} "
          f"(max {SELF_EXAMINE_MAX_RETRIES} retries)")
    print(f"  Seeded tests:     {len(SEEDED_TEST_CASES)} cases × {N_REPEATS} repeats "
          f"= {len(SEEDED_TEST_CASES) * N_REPEATS} runs")
    print(f"  Natural runs:     {NATURAL_RUNS_PER_PART} per part × {len(NATURAL_TEST_PARTS)} parts "
          f"= {NATURAL_RUNS_PER_PART * len(NATURAL_TEST_PARTS)} runs")
    total_seeded_runs = len(SEEDED_TEST_CASES) * N_REPEATS
    total_natural_runs = NATURAL_RUNS_PER_PART * len(NATURAL_TEST_PARTS)
    print(f"  Total tests:      {total_seeded_runs} + {total_natural_runs} "
          f"= {total_seeded_runs + total_natural_runs}")

    print("\nInitializing components...")
    if not LLM_API_KEY:
        print("ERROR: No API key set. Set DEEPSEEK_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.")
        sys.exit(1)

    generator = LLMGenerator(backend=LLM_BACKEND, api_key=LLM_API_KEY)
    monitor = StructuralMonitor(strict_mode=False)
    print(f"  Generator: {generator.backend} (model={generator.model}, "
          f"base_url={generator.base_url})")
    print(f"  Monitor:   StructuralMonitor (D1-D6, 6 predicates)")
    print()

    t_start = time.time()

    if SCENARIO_MODE == "four_scenarios":
        all_results = run_four_scenario_comparison(generator, monitor)
    else:
        all_results = run_all_tests(generator, monitor)

    t_total = time.time() - t_start

    # Save raw results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # ── Quick Summary with Confidence Intervals ───────────────────────────
    seeded = [r for r in all_results if r.get("has_injected_defect")]
    natural = [r for r in all_results if not r.get("has_injected_defect")]

    seeded_detected = sum(1 for r in seeded if r.get("defect_detected") is True)
    seeded_total_with_expected = sum(1 for r in seeded if r.get("expected_defect"))

    natural_passed = sum(1 for r in natural if r["passed_monitor"])
    natural_with_defects = sum(1 for r in natural if len(r["violations"]) > 0)

    # Self-Examine metrics
    se_tests = [r for r in all_results if r.get("self_examine_history")]
    se_direct_passes = sum(1 for r in se_tests
                          if len(r["self_examine_history"]) == 1
                          and r["self_examine_history"][0]["passed"])
    se_corrected = sum(1 for r in se_tests
                       if len(r["self_examine_history"]) > 1
                       and r["passed_monitor"])
    se_failed = sum(1 for r in se_tests
                    if not r["passed_monitor"]
                    and len(r["self_examine_history"]) > 1)
    se_total_attempts = sum(len(r["self_examine_history"]) for r in se_tests)

    # Per-defect-type detection with CIs
    per_defect_summary = {}
    for d_id in sorted(set(r.get("expected_defect") for r in seeded if r.get("expected_defect"))):
        d_results = [r for r in seeded if r.get("expected_defect") == d_id]
        d_detected = sum(1 for r in d_results if r.get("defect_detected") is True)
        d_total = len(d_results)
        per_defect_summary[d_id] = proportions_summary(d_detected, d_total)

    summary = {
        "experiment_config": {
            "llm_backend": LLM_BACKEND,
            "llm_model": generator.model,
            "natural_runs_per_part": NATURAL_RUNS_PER_PART,
            "n_repeats": N_REPEATS,
            "total_tests": len(all_results),
            "n_seeded": len(seeded),
            "n_seeded_unique": len(SEEDED_TEST_CASES),
            "n_natural": len(natural),
            "self_examine_enabled": SELF_EXAMINE,
        },
        "self_examine": {
            "n_tests": len(se_tests),
            "direct_passes": se_direct_passes,
            "corrected_passes": se_corrected,
            "failed_after_retries": se_failed,
            "total_llm_attempts": se_total_attempts,
            "avg_attempts_per_test": round(se_total_attempts / max(1, len(se_tests)), 1),
        } if SELF_EXAMINE else {},
        "seeded_tests": {
            "n_total": len(seeded),
            "n_with_expected_defect": seeded_total_with_expected,
            "n_detected": seeded_detected,
            "detection_rate": ci_string(seeded_detected, seeded_total_with_expected),
            "detection_rate_unformatted": proportions_summary(seeded_detected, seeded_total_with_expected),
        },
        "seeded_per_defect": per_defect_summary,
        "natural_tests": {
            "n_total": len(natural),
            "n_passed": natural_passed,
            "n_with_defects": natural_with_defects,
            "defect_rate": ci_string(natural_with_defects, len(natural)),
            "pass_rate": ci_string(natural_passed, len(natural)),
            "defect_rate_unformatted": proportions_summary(natural_with_defects, len(natural)),
            "pass_rate_unformatted": proportions_summary(natural_passed, len(natural)),
        },
        "timing": {
            "total_experiment_time_s": round(t_total, 2),
            "avg_llm_time_ms": round(sum(r["llm_time_ms"] for r in all_results) / len(all_results), 1),
            "avg_monitor_time_ms": round(sum(r["monitor_time_ms"] for r in all_results) / len(all_results), 3),
        },
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── Print Final Summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {summary['timing']['total_experiment_time_s']:.2f}s")
    print(f"Total tests: {summary['experiment_config']['total_tests']}")
    print(f"\nSeeded Tests ({len(seeded)} runs from {len(SEEDED_TEST_CASES)} unique cases):")
    print(f"  Detection Rate: {summary['seeded_tests']['detection_rate']}")
    print(f"  Detected: {seeded_detected}/{seeded_total_with_expected}")
    print(f"\n  Per-Defect Detection Rates (with 95% CI):")
    for d_id in sorted(per_defect_summary.keys()):
        ps = per_defect_summary[d_id]
        print(f"    {d_id}: {ps['wilson_ci_pct']} "
              f"(p={ps['binomial_p_value']:.4f}{'*' if ps['significant_at_05'] else ''})")
    print(f"\nNatural Tests ({len(natural)} tests):")
    print(f"  Pass Rate:   {summary['natural_tests']['pass_rate']}")
    print(f"  Defect Rate: {summary['natural_tests']['defect_rate']}")
    if SELF_EXAMINE and se_tests:
        se_s = summary["self_examine"]
        print(f"\nSelf-Examine Correction:")
        print(f"  Direct Pass:    {se_s['direct_passes']}")
        print(f"  Corrected:      {se_s['corrected_passes']}")
        print(f"  Failed:         {se_s['failed_after_retries']}")
        print(f"  Total Attempts: {se_s['total_llm_attempts']}")
        print(f"  Avg Attempts:   {se_s['avg_attempts_per_test']}")

    print(f"\nTiming:")
    print(f"  Avg LLM time:    {summary['timing']['avg_llm_time_ms']}ms")
    print(f"  Avg Monitor time: {summary['timing']['avg_monitor_time_ms']}ms")

    print(f"\nResults saved to:")
    print(f"  {RESULTS_FILE}")
    print(f"  {SUMMARY_FILE}")
    print(f"  {SPECS_DIR}/   ({len(os.listdir(SPECS_DIR))} spec files)")
    print(f"\n  * = significant at p < 0.05 (binomial test vs p0=0.5)")
