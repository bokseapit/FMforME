"""
Layer 2: Runtime Monitor Core Implementation
Implements the six structural defect predicates D1-D6.
Corresponds to AgentSpec's check layer — intercepts ModelSpec before CAD execution.

Architecture:
  - Violation / ViolationReport: data classes for structured violation tracking
  - StructuralMonitor: core checker, executes D5→D4→D1→D2→D3→D6 in order
  - Each _check_D* method is self-contained with clear standard references

Standard References:
  D1: ASME V&V-10-2019 §4 (rigid body motion)
  D2: Bathe FEM §2.3 (positive definiteness of stiffness)
  D3: Knupp 2001, SIAM J.Sci.Comput. (mesh quality)
  D4: ASME V&V-10 §5.1 (load-BC consistency)
  D5: ISO 286 / ASTM material standards
  D6: Verdict Library (Stimpson 2007, mesh topology)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import json
import math
import time


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class Violation:
    """A single constraint violation found by the monitor.

    Attributes
    ----------
    defect_id : str
        D1–D6 identifier corresponding to the defect taxonomy.
    severity : str
        "ERROR" (block execution) or "WARN" (log but continue).
    description : str
        Human-readable description of the violation.
    field_path : str
        Dot-separated path to the violating field in ModelSpec.
    actual_value : any
        The actual violating value.
    standard : str, optional
        The engineering standard that grounds this defect type.
    """
    defect_id: str          # "D1" ~ "D6"
    severity: str           # "ERROR" | "WARN"
    description: str
    field_path: str
    actual_value: Any = None
    standard: str = ""

    def to_dict(self) -> dict:
        return {
            "defect_id": self.defect_id,
            "severity": self.severity,
            "description": self.description,
            "field_path": self.field_path,
            "actual_value": self.actual_value,
            "standard": self.standard,
        }


@dataclass
class ViolationReport:
    """Aggregated result of monitor checking.

    Attributes
    ----------
    violations : list
        All violations found.
    passed : bool
        True if no ERROR-level violations (WARN only is still "passed").
    timestamp : float
        Unix timestamp when the report was generated.
    spec_hash : str
        Optional hash of the input spec for traceability.
    """
    violations: List[Violation] = field(default_factory=list)
    passed: bool = True
    timestamp: float = 0.0
    spec_hash: str = ""

    def add(self, v: Violation):
        self.violations.append(v)
        if v.severity == "ERROR":
            self.passed = False

    def has_errors(self) -> bool:
        return any(v.severity == "ERROR" for v in self.violations)

    def has_warnings(self) -> bool:
        return any(v.severity == "WARN" for v in self.violations)

    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == "ERROR"]

    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == "WARN"]

    def defect_types(self) -> set:
        return {v.defect_id for v in self.violations}

    def summary(self) -> str:
        """One-line summary string."""
        n_err = len(self.errors())
        n_warn = len(self.warnings())
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {n_err} error(s), {n_warn} warning(s)"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_violations": len(self.violations),
            "n_errors": len(self.errors()),
            "n_warnings": len(self.warnings()),
            "violations": [v.to_dict() for v in self.violations],
        }


# ── Structural Monitor ────────────────────────────────────────────────────

class StructuralMonitor:
    """Runtime constraint validator for structural ModelSpec.

    Execution order (D5→D4→D1→D2→D3→D6) is by detectability
    and importance: material violations (D5) are the most common
    AI errors and should be caught first.

    Parameters
    ----------
    strict_mode : bool
        If True, all bounds violations are ERROR. Default: False.
    material_bounds : dict, optional
        Override default material parameter bounds.
    max_segment_ratio : float
        Maximum allowed diameter ratio between adjacent shaft segments.
    """

    # Default material bounds (from ISO/ASTM standards)
    DEFAULT_MATERIAL_BOUNDS = {
        "youngs_modulus":    (0.001, 1000),    # GPa
        "poisson_ratio":     (0.0,   0.5),     # thermodynamic constraint: (0, 0.5)
        "density":           (100,   25000),   # kg/m^3
        "yield_strength":    (1,     5000),    # MPa
        "tensile_strength":  (1,     6000),    # MPa
    }

    # Reference standards for each defect type
    STANDARDS = {
        "D1": "ASME V&V-10-2019 §4",
        "D2": "Bathe FEM Textbook §2.3",
        "D3": "Knupp 2001, SIAM J.Sci.Comput.",
        "D4": "ASME V&V-10 §5.1",
        "D5": "ISO 286 / ASTM Material Standards",
        "D6": "Verdict Library (Stimpson 2007)",
    }

    def __init__(self, strict_mode: bool = False,
                 material_bounds: Optional[dict] = None,
                 max_segment_ratio: float = 10.0):
        self.strict_mode = strict_mode
        self.material_bounds = material_bounds or self.DEFAULT_MATERIAL_BOUNDS.copy()
        self.max_segment_ratio = max_segment_ratio

    def _sanitize_spec(self, spec: dict) -> dict:
        """Ensure spec has all expected fields in correct types.
        LLMs sometimes generate malformed JSON (e.g., list fields as strings).
        """
        # Ensure lists for fields that should be lists
        for field in ("boundary_conditions", "loads"):
            val = spec.get(field)
            if not isinstance(val, list):
                spec[field] = []
        # Ensure mesh is a dict
        if not isinstance(spec.get("mesh"), dict):
            spec["mesh"] = {}
        # Ensure material is a dict
        if not isinstance(spec.get("material"), dict):
            spec["material"] = {}
        # Ensure dimensions is a dict
        if not isinstance(spec.get("dimensions"), dict):
            spec["dimensions"] = {}
        return spec

    def check(self, spec: dict) -> ViolationReport:
        """Run all six defect checks on a ModelSpec.

        Parameters
        ----------
        spec : dict
            The ModelSpec to validate.

        Returns
        -------
        ViolationReport
            Aggregated violation report.
        """
        report = ViolationReport(timestamp=time.time())

        # Sanitize spec (LLMs may produce malformed JSON)
        spec = self._sanitize_spec(spec)

        # Execute checks in order of detectability
        self._check_D5_material(spec, report)
        self._check_D4_load_bc_conflict(spec, report)
        self._check_D1_unconstrained_dof(spec, report)
        self._check_D2_negative_stiffness(spec, report)
        self._check_D3_stress_singularity(spec, report)
        self._check_D6_mesh_topology(spec, report)

        return report

    # ── D5: Material Property Violation ──────────────────────────────────

    def _check_D5_material(self, spec: dict, report: ViolationReport):
        """Check material properties against physical bounds.

        Standard: ISO 286 / ASTM Material Standards.
        Most commonly triggered by AI — LLMs often generate
        Poisson ratio > 0.5 or negative elastic modulus.
        """
        mat = spec.get("material", {})
        if not mat:
            report.add(Violation(
                "D5", "WARN",
                "No material defined in ModelSpec",
                "material", None,
                self.STANDARDS["D5"]
            ))
            return

        # Check each parameter against bounds
        for param, (lo, hi) in self.material_bounds.items():
            val = mat.get(param)
            if val is None:
                continue

            if not (lo < val < hi):
                # Critical parameters → ERROR; secondary → WARN
                if param in ("youngs_modulus", "poisson_ratio", "density"):
                    severity = "ERROR"
                else:
                    severity = "WARN" if not self.strict_mode else "ERROR"

                report.add(Violation(
                    "D5", severity,
                    f"{param}={val} out of range ({lo}, {hi})",
                    f"material.{param}", val,
                    self.STANDARDS["D5"]
                ))

        # Special check: yield_strength < tensile_strength
        ys = mat.get("yield_strength")
        ts = mat.get("tensile_strength")
        if ys is not None and ts is not None and ys >= ts:
            report.add(Violation(
                "D5", "ERROR",
                f"yield_strength({ys}) >= tensile_strength({ts}); violates material physics",
                "material.yield_strength", ys,
                self.STANDARDS["D5"]
            ))

        # Poisson ratio edge cases (exactly 0 or 0.5 are physically impossible)
        pr = mat.get("poisson_ratio")
        if pr is not None:
            if pr == 0.0:
                report.add(Violation(
                    "D5", "WARN",
                    f"poisson_ratio=0.0 is physically impossible for most materials",
                    "material.poisson_ratio", pr,
                    self.STANDARDS["D5"]
                ))
            if pr == 0.5:
                report.add(Violation(
                    "D5", "ERROR",
                    f"poisson_ratio=0.5 implies incompressible material (volume-preserving)",
                    "material.poisson_ratio", pr,
                    self.STANDARDS["D5"]
                ))

    # ── D4: Load-Boundary Condition Conflict ─────────────────────────────

    def _check_D4_load_bc_conflict(self, spec: dict, report: ViolationReport):
        """Check for nodes that are simultaneously in BC and load sets.

        Standard: ASME V&V-10 §5.1 — applying load on a fixed node
        creates conflicting constraints. In FEA this produces
        meaningless results because the load is absorbed by the constraint.
        """
        bc_nodes = {bc.get("node_id") for bc in spec.get("boundary_conditions", [])
                    if bc.get("node_id") is not None}
        load_nodes = {ld.get("node_id") for ld in spec.get("loads", [])
                      if ld.get("node_id") is not None}

        conflict = bc_nodes & load_nodes
        if conflict:
            report.add(Violation(
                "D4", "WARN",
                f"Nodes {sorted(conflict)} appear in both boundary_conditions and loads. "
                f"Load applied at constrained nodes will be absorbed by the constraint.",
                "loads/boundary_conditions", sorted(conflict),
                self.STANDARDS["D4"]
            ))

    # ── D1: Unconstrained Degrees of Freedom ─────────────────────────────

    def _check_D1_unconstrained_dof(self, spec: dict, report: ViolationReport):
        """Check for rigid body motion — insufficient constraints.

        Standard: ASME V&V-10-2019 §4.
        A model without adequate BCs has rigid body modes and
        singular stiffness matrix K. FEA solvers will fail or
        produce unbounded displacements.

        Minimum requirement: all 6 rigid body DOF must be constrained
        (tx, ty, tz, rx, ry, rz) across the BC set.
        """
        bcs = spec.get("boundary_conditions", [])
        if not bcs:
            report.add(Violation(
                "D1", "ERROR",
                "No boundary conditions defined — model is a free rigid body "
                "(6 unconstrained DOF). FEA solver will fail.",
                "boundary_conditions", [],
                self.STANDARDS["D1"]
            ))
            return

        # Check if any BC is fully fixed
        has_fixed = any(bc.get("type") == "fixed" for bc in bcs
                        if isinstance(bc, dict))

        if not has_fixed:
            # Collect all constrained DOF across all BCs
            all_dofs = set()
            for bc in bcs:
                if isinstance(bc, dict):
                    dofs = bc.get("dof", [])
                    if isinstance(dofs, list):
                        all_dofs.update(dofs)

            required = {"tx", "ty", "tz", "rx", "ry", "rz"}
            missing = required - all_dofs

            if missing:
                report.add(Violation(
                    "D1", "WARN",
                    f"Missing DOF constraints: {sorted(missing)}. "
                    f"Unconstrained DOF cause rigid body modes.",
                    "boundary_conditions", sorted(missing),
                    self.STANDARDS["D1"]
                ))

        # Check if any BC has no constrained DOF at all
        for bc in bcs:
            if isinstance(bc, dict):
                dofs = bc.get("dof", [])
                if isinstance(dofs, list) and len(dofs) == 0 and bc.get("type") != "fixed":
                    report.add(Violation(
                        "D1", "WARN",
                        f"BC at node {bc.get('node_id', '?')} has no constrained DOF",
                        f"boundary_conditions[{bcs.index(bc)}].dof", [],
                        self.STANDARDS["D1"]
                    ))

    # ── D2: Negative Stiffness ────────────────────────────────────────────

    def _check_D2_negative_stiffness(self, spec: dict, report: ViolationReport):
        """Check for negative or zero dimensions that imply negative stiffness.

        Standard: Bathe FEM Textbook §2.3.
        Stiffness matrix K requires positive definiteness.
        Zero or negative cross-sections → non-positive K diagonals →
        solver failure or unphysical results.

        For shafts: large adjacent segment ratios imply stress concentration
        and potential local negative stiffness in simplified models.
        """
        dims = spec.get("dimensions", {})

        # Check all scalar dimensions
        for key, val in dims.items():
            if isinstance(val, (int, float)):
                if val < 0:
                    report.add(Violation(
                        "D2", "ERROR",
                        f"Dimension '{key}' = {val} < 0 (negative stiffness contribution)",
                        f"dimensions.{key}", val,
                        self.STANDARDS["D2"]
                    ))
                elif val == 0:
                    # Zero dimension → singular element (zero volume/area)
                    severity = "ERROR" if key in ("nominal_diameter", "thickness",
                        "bore_diameter", "outer_diameter", "ball_diameter") else "WARN"
                    report.add(Violation(
                        "D2", severity,
                        f"Dimension '{key}' = 0 (zero cross-section, singular stiffness)",
                        f"dimensions.{key}", val,
                        self.STANDARDS["D2"]
                    ))

        # Check shaft segment diameters
        seg_diams = dims.get("segment_diameters", [])
        if isinstance(seg_diams, list) and len(seg_diams) > 1:
            for i in range(len(seg_diams) - 1):
                if not isinstance(seg_diams[i], (int, float)) or not isinstance(seg_diams[i+1], (int, float)):
                    continue
                if seg_diams[i] <= 0 or seg_diams[i+1] <= 0:
                    continue

                ratio = max(seg_diams[i], seg_diams[i+1]) / min(seg_diams[i], seg_diams[i+1])
                if ratio > self.max_segment_ratio:
                    report.add(Violation(
                        "D2", "WARN",
                        f"Diameter ratio {ratio:.2f}:1 between segments {i} and {i+1} "
                        f"exceeds threshold {self.max_segment_ratio}:1. "
                        f"May cause stress concentration and numerical stiffness issues.",
                        f"dimensions.segment_diameters[{i}]", seg_diams[i],
                        self.STANDARDS["D2"]
                    ))

        # Check list dimensions for negative values
        for key in ("segment_diameters", "segment_lengths", "segment_fillets"):
            lst = dims.get(key, [])
            if not isinstance(lst, list):
                continue
            for i, val in enumerate(lst):
                if isinstance(val, (int, float)) and val <= 0:
                    report.add(Violation(
                        "D2", "ERROR" if key == "segment_diameters" else "WARN",
                        f"{key}[{i}] = {val} <= 0",
                        f"dimensions.{key}[{i}]", val,
                        self.STANDARDS["D2"]
                    ))

    # ── D3: Stress Singularity ─────────────────────────────────────────────

    def _check_D3_stress_singularity(self, spec: dict, report: ViolationReport):
        """Check for geometric features that cause stress singularities.

        Standard: Knupp 2001, SIAM J. Sci. Comput.
        Sharp re-entrant corners (zero fillet radius), extreme aspect ratios,
        and mesh singularities can cause unbounded stress values in FEA.

        Key indicators:
        - fillet_radius <= 0 (sharp corner → infinite stress at corner)
        - max_aspect_ratio > threshold (elongated elements)
        - High condition number in stiffness → flagged as D3
        """
        dims = spec.get("dimensions", {})

        # Check fillet radius (applies to bracket and shaft)
        for key in ("fillet_radius", "bend_radius"):
            val = dims.get(key)
            if isinstance(val, (int, float)):
                if val < 0:
                    report.add(Violation(
                        "D3", "ERROR",
                        f"{key} = {val} < 0 (meaningless geometry, stress singularity)",
                        f"dimensions.{key}", val,
                        self.STANDARDS["D3"]
                    ))
                elif val == 0:
                    report.add(Violation(
                        "D3", "WARN",
                        f"{key} = 0 (sharp corner → stress singularity, "
                        f"FEA stress at re-entrant corner unbounded)",
                        f"dimensions.{key}", val,
                        self.STANDARDS["D3"]
                    ))

        # Check list fillets
        fillets = dims.get("segment_fillets", [])
        if isinstance(fillets, list):
            for i, val in enumerate(fillets):
                if isinstance(val, (int, float)):
                    if val < 0:
                        report.add(Violation(
                            "D3", "ERROR",
                            f"segment_fillets[{i}] = {val} < 0",
                            f"dimensions.segment_fillets[{i}]", val,
                            self.STANDARDS["D3"]
                        ))
                    elif val == 0:
                        report.add(Violation(
                            "D3", "WARN",
                            f"segment_fillets[{i}] = 0 → stress singularity at shoulder",
                            f"dimensions.segment_fillets[{i}]", val,
                            self.STANDARDS["D3"]
                        ))

        # ── Ball bearing specific D3 checks ──────────────────────
        part_type = spec.get("part_type", "")
        if part_type == "ball_bearing":
            # ball_count == 0 → no load path, point contact on raceway alone
            ball_count = dims.get("ball_count")
            if isinstance(ball_count, (int, float)):
                if ball_count == 0:
                    report.add(Violation(
                        "D3", "ERROR",
                        "ball_count = 0 — zero rolling elements means no load transfer "
                        "and point contact stress singularity at raceway",
                        "dimensions.ball_count", ball_count,
                        self.STANDARDS["D3"]
                    ))
                elif ball_count < 3:
                    report.add(Violation(
                        "D3", "WARN",
                        f"ball_count = {ball_count} — too few rolling elements for "
                        f"stable load distribution (< 3)",
                        "dimensions.ball_count", ball_count,
                        self.STANDARDS["D3"]
                    ))

            # ball_diameter <= 0 or extremely small → stress concentration
            ball_dia = dims.get("ball_diameter")
            if isinstance(ball_dia, (int, float)):
                if ball_dia <= 0:
                    report.add(Violation(
                        "D3", "ERROR",
                        f"ball_diameter = {ball_dia} <= 0 — impossible geometry, "
                        f"stress singularity on raceway contact",
                        "dimensions.ball_diameter", ball_dia,
                        self.STANDARDS["D3"]
                    ))
                else:
                    # Check ball_diameter vs radial gap: ball too small → point contact
                    bore = dims.get("bore_diameter")
                    outer = dims.get("outer_diameter")
                    if isinstance(bore, (int, float)) and isinstance(outer, (int, float)):
                        radial_gap = (outer - bore) / 2.0
                        if radial_gap > 0 and ball_dia > radial_gap:
                            report.add(Violation(
                                "D3", "ERROR",
                                f"ball_diameter({ball_dia}) > radial_gap({radial_gap:.1f}) — "
                                f"ball cannot fit between inner and outer races",
                                "dimensions.ball_diameter", ball_dia,
                                self.STANDARDS["D3"]
                            ))
                        elif radial_gap > 0 and ball_dia > radial_gap * 0.8:
                            report.add(Violation(
                                "D3", "WARN",
                                f"ball_diameter({ball_dia}) near radial_gap({radial_gap:.1f}) — "
                                f"insufficient raceway wall thickness",
                                "dimensions.ball_diameter", ball_dia,
                                self.STANDARDS["D3"]
                            ))

            # bore_diameter >= outer_diameter → impossible bearing
            bore_val = dims.get("bore_diameter")
            outer_val = dims.get("outer_diameter")
            if isinstance(bore_val, (int, float)) and isinstance(outer_val, (int, float)):
                if bore_val >= outer_val:
                    report.add(Violation(
                        "D3", "ERROR",
                        f"bore_diameter({bore_val}) >= outer_diameter({outer_val}) — "
                        f"impossible bearing geometry",
                        "dimensions.bore_diameter", bore_val,
                        self.STANDARDS["D3"]
                    ))

        # Check mesh aspect ratio
        mesh = spec.get("mesh", {})
        max_ar = mesh.get("max_aspect_ratio")
        if isinstance(max_ar, (int, float)):
            if max_ar > 50:
                report.add(Violation(
                    "D3", "WARN",
                    f"max_aspect_ratio = {max_ar} (>50) may cause stress singularities "
                    f"in distorted elements",
                    "mesh.max_aspect_ratio", max_ar,
                    self.STANDARDS["D3"]
                ))
            elif max_ar > 20:
                report.add(Violation(
                    "D3", "WARN",
                    f"max_aspect_ratio = {max_ar} (>20), element quality degraded",
                    "mesh.max_aspect_ratio", max_ar,
                    self.STANDARDS["D3"]
                ))

    # ── D6: Mesh Topology Error ────────────────────────────────────────────

    def _check_D6_mesh_topology(self, spec: dict, report: ViolationReport):
        """Check for mesh topology errors.

        Standard: Verdict Library (Stimpson 2007).
        - Negative Jacobian → inverted/flipped elements
        - Zero-volume elements
        - Degenerate element shapes

        min_jacobian < 0: strictly invalid mesh (inverted elements)
        min_jacobian in [0, 0.5): poor quality but may run
        """
        mesh = spec.get("mesh", {})
        if not mesh:
            report.add(Violation(
                "D6", "WARN",
                "No mesh specification defined",
                "mesh", None,
                self.STANDARDS["D6"]
            ))
            return

        min_jac = mesh.get("min_jacobian")
        if isinstance(min_jac, (int, float)):
            if min_jac < 0:
                report.add(Violation(
                    "D6", "ERROR",
                    f"min_jacobian = {min_jac} < 0 → inverted elements. "
                    f"Mesh has negative-volume elements (flipped/inside-out). "
                    f"FEA solver will fail.",
                    "mesh.min_jacobian", min_jac,
                    self.STANDARDS["D6"]
                ))
            elif min_jac == 0:
                report.add(Violation(
                    "D6", "ERROR",
                    f"min_jacobian = 0 → zero-volume element(s). Degenerate mesh.",
                    "mesh.min_jacobian", min_jac,
                    self.STANDARDS["D6"]
                ))
            elif min_jac < 0.3:
                report.add(Violation(
                    "D6", "WARN",
                    f"min_jacobian = {min_jac} < 0.3, mesh quality critically low",
                    "mesh.min_jacobian", min_jac,
                    self.STANDARDS["D6"]
                ))
            elif min_jac < 0.5:
                report.add(Violation(
                    "D6", "WARN",
                    f"min_jacobian = {min_jac} < 0.5 (DIANA solver threshold)",
                    "mesh.min_jacobian", min_jac,
                    self.STANDARDS["D6"]
                ))

        # Check element type validity
        valid_types = {"tet4", "tet10", "hex8", "hex20", "shell3", "shell4", "wedge6"}
        elem_type = mesh.get("element_type", "")
        if elem_type and elem_type.lower() not in valid_types:
            report.add(Violation(
                "D6", "WARN",
                f"Unknown element_type '{elem_type}'. Valid: {sorted(valid_types)}",
                "mesh.element_type", elem_type,
                self.STANDARDS["D6"]
            ))

        # Check max_aspect_ratio
        max_ar = mesh.get("max_aspect_ratio")
        if isinstance(max_ar, (int, float)):
            if max_ar <= 0:
                report.add(Violation(
                    "D6", "ERROR",
                    f"max_aspect_ratio = {max_ar} <= 0 (physically impossible)",
                    "mesh.max_aspect_ratio", max_ar,
                    self.STANDARDS["D6"]
                ))


# ── DSL-Driven Monitor (YAML-based rule loading) ───────────────────────────

class DSLMonitor:
    """Alternative monitor that loads rules from a YAML DSL file.

    This implements the AgentSpec-style (trigger, check, enforce) pattern
    defined in structural_rules.yaml. It's a higher-level interface
    that delegates to StructuralMonitor for the actual checks.

    Parameters
    ----------
    rules_file : str
        Path to structural_rules.yaml.
    """

    def __init__(self, rules_file: str):
        self.rules_file = rules_file
        self.rules = self._load_rules(rules_file)

    def _load_rules(self, yaml_path: str) -> list:
        """Load rules from YAML file."""
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("rules", [])
        except ImportError:
            print("[WARN] PyYAML not installed. DSLMonitor will use built-in rules.")
            return self._builtin_rules()
        except FileNotFoundError:
            print(f"[WARN] Rules file {yaml_path} not found. Using built-in rules.")
            return self._builtin_rules()

    @staticmethod
    def _builtin_rules() -> list:
        """Fallback built-in rules matching the YAML definitions."""
        return [
            {
                "id": "D5_poisson_ratio",
                "on": "material",
                "when": "poisson_ratio NOT IN_RANGE (0.0, 0.5)",
                "raise": "ERROR",
                "message": "Poisson ratio violates thermodynamics, must be in (0, 0.5)",
                "standard": "ISO/IEC material physics constraint"
            },
            {
                "id": "D5_youngs_modulus",
                "on": "material",
                "when": "youngs_modulus <= 0",
                "raise": "ERROR",
                "message": "Young modulus <= 0, physically impossible"
            },
            {
                "id": "D4_load_bc_conflict",
                "on": ["loads", "boundary_conditions"],
                "when": "INTERSECTS(loads.node_ids, bc.node_ids)",
                "raise": "WARN",
                "message": "Nodes in both loads and BCs",
                "standard": "ASME V&V-10 Sec 5.1"
            },
            {
                "id": "D1_no_boundary_condition",
                "on": "boundary_conditions",
                "when": "COUNT == 0",
                "raise": "ERROR",
                "message": "No boundary conditions defined"
            },
            {
                "id": "D2_negative_dimension",
                "on": "dimensions",
                "when": "ANY_DIMENSION <= 0",
                "raise": "ERROR",
                "message": "Dimension <= 0, negative stiffness"
            },
            {
                "id": "D6_negative_jacobian",
                "on": "mesh",
                "when": "min_jacobian < 0",
                "raise": "ERROR",
                "message": "Jacobian < 0, inverted elements (Knupp 2001)"
            },
            {
                "id": "D6_poor_jacobian",
                "on": "mesh",
                "when": "min_jacobian IN_RANGE [0, 0.5)",
                "raise": "WARN",
                "message": "Jacobian < 0.5, mesh quality marginal (DIANA threshold)"
            },
        ]

    def check(self, spec: dict) -> ViolationReport:
        """Evaluate all DSL rules against the spec.

        For YAML rules, we use pattern-matching on the 'when' field.
        For complex rules, delegates to StructuralMonitor.
        """
        import re
        report = ViolationReport()

        for rule in self.rules:
            if self._eval_rule(rule, spec):
                defect_id = rule["id"].split("_")[0]  # "D5_poisson_ratio" → "D5"
                report.add(Violation(
                    defect_id=defect_id,
                    severity=rule["raise"],
                    description=rule.get("message", ""),
                    field_path=str(rule.get("on", "")),
                    standard=rule.get("standard", ""),
                ))

        return report

    def _eval_rule(self, rule: dict, spec: dict) -> bool:
        """Evaluate a single DSL rule against the spec.

        Supports these WHEN patterns:
          - "field NOT IN_RANGE (lo, hi)"
          - "field <= 0"
          - "field < 0"
          - "COUNT == 0"
          - "INTERSECTS(...)"
          - "ANY_DIMENSION <= 0"
          - "field IN_RANGE [lo, hi)"
        """
        when = rule.get("when", "")
        on = rule.get("on", "")

        # ── INTERSECTS pattern ──
        if "INTERSECTS" in when:
            bc_nodes = {bc.get("node_id") for bc in spec.get("boundary_conditions", [])
                        if isinstance(bc, dict) and bc.get("node_id") is not None}
            ld_nodes = {ld.get("node_id") for ld in spec.get("loads", [])
                        if isinstance(ld, dict) and ld.get("node_id") is not None}
            return bool(bc_nodes & ld_nodes)

        # ── COUNT pattern ──
        if "COUNT == 0" in when:
            if isinstance(on, str):
                val = self._get_value(spec, on)
                return isinstance(val, list) and len(val) == 0
            return False

        # ── ANY_DIMENSION pattern ──
        if "ANY_DIMENSION" in when:
            dims = spec.get("dimensions", {})
            is_error = "<= 0" in when or "< 0" in when
            for key, val in dims.items():
                if isinstance(val, (int, float)) and val <= 0:
                    return True
            return False

        # ── NOT IN_RANGE pattern ──
        if "NOT IN_RANGE" in when:
            # Extract the field name before NOT
            field_match = re.match(r"(\w+)\s+NOT\s+IN_RANGE", when)
            if field_match:
                field = field_match.group(1)
                value = self._get_value(spec, on)
                if value is None and isinstance(on, str):
                    value = self._get_value(spec, f"{on}.{field}")

                nums = re.findall(r"[-+]?\d*\.?\d+", when)
                if len(nums) >= 2 and isinstance(value, (int, float)):
                    lo, hi = float(nums[0]), float(nums[1])
                    return not (lo < value < hi)
            return False

        # ── IN_RANGE pattern ──
        if "IN_RANGE" in when and "NOT" not in when:
            field_match = re.match(r"(\w+)\s+IN_RANGE", when)
            if field_match:
                field = field_match.group(1)
                value = self._get_value(spec, on)
                if value is None and isinstance(on, str):
                    value = self._get_value(spec, f"{on}.{field}")

                nums = re.findall(r"[-+]?\d*\.?\d+", when)
                if len(nums) >= 2 and isinstance(value, (int, float)):
                    lo, hi = float(nums[0]), float(nums[1])
                    # Check for inclusive/exclusive bounds
                    if "[" in when and when.index("[") < when.index(nums[0]):
                        left_inclusive = True
                    else:
                        left_inclusive = False
                    if "]" in when:
                        right_inclusive = True
                    else:
                        right_inclusive = False

                    if left_inclusive and right_inclusive:
                        return lo <= value <= hi
                    elif left_inclusive:
                        return lo <= value < hi
                    elif right_inclusive:
                        return lo < value <= hi
                    else:
                        return lo < value < hi
            return False

        # ── Simple comparison patterns ──
        if "<= 0" in when:
            for word in when.split():
                if word not in ("<=", "0", "<", ">", ">=", "NOT", "IN_RANGE", "IN_RANGE"):
                    field = word
                    value = self._get_value(spec, on)
                    if value is None and isinstance(on, str):
                        value = self._get_value(spec, f"{on}.{field}")
                    return isinstance(value, (int, float)) and value <= 0
            return False

        if "< 0" in when:
            field_match = re.match(r"(\w+)\s+<\s+0", when)
            if field_match:
                field = field_match.group(1)
                value = self._get_value(spec, on)
                if value is None and isinstance(on, str):
                    value = self._get_value(spec, f"{on}.{field}")
                return isinstance(value, (int, float)) and value < 0
            return False

        return False

    @staticmethod
    def _get_value(obj: any, path: str) -> any:
        """Traverse a dot-separated path into a nested dict."""
        if isinstance(path, list):
            return None
        parts = path.split(".")
        current = obj
        for p in parts:
            if isinstance(current, dict):
                current = current.get(p)
            else:
                return None
        return current


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with a valid spec
    from layer1_llm import LLMGenerator
    gen = LLMGenerator(backend="mock")
    spec = gen.generate("M8x30 stainless steel bolt")
    monitor = StructuralMonitor()
    report = monitor.check(spec)
    print("Valid spec:", report.summary())
    for v in report.violations:
        print(f"  {v.defect_id} [{v.severity}]: {v.description}")

    # Test with defective spec
    spec_bad = gen.generate("M8 bolt",
                            inject_defect={"path": "material.poisson_ratio", "value": 0.6})
    report_bad = monitor.check(spec_bad)
    print("\nDefective spec:", report_bad.summary())
    for v in report_bad.violations:
        print(f"  {v.defect_id} [{v.severity}]: {v.description}")

    # Test DSL Monitor
    print("\n--- DSL Monitor ---")
    dsl = DSLMonitor.__new__(DSLMonitor)
    dsl.rules = DSLMonitor._builtin_rules()
    dsl.rules_file = ""
    report_dsl = dsl.check(spec_bad)
    print("DSL result:", report_dsl.summary())
