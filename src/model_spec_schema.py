"""
ModelSpec JSON Schema Definition
Core data structure flowing across the three-layer architecture.
Defines the contract between LLM generation, Runtime Monitor, and CAD execution.
"""

from typing import TypedDict, List


# ── JSON Schema (Python-typed version) ─────────────────────────────────────

class Dimensions(TypedDict, total=False):
    """Part dimensions - varies by part_type."""
    nominal_diameter: float      # mm, bolt
    length: float                # mm, bolt/shaft
    head_diameter: float         # mm, bolt
    head_height: float           # mm, bolt
    thread_pitch: float          # mm, bolt
    width: float                 # mm, bracket
    height: float                # mm, bracket
    thickness: float             # mm, bracket
    fillet_radius: float         # mm, bracket
    bend_radius: float           # mm, bracket
    segment_diameters: List[float]  # mm, shaft
    segment_lengths: List[float]    # mm, shaft
    segment_fillets: List[float]    # mm, shaft


class Material(TypedDict, total=False):
    name: str
    youngs_modulus: float        # GPa, range (0, inf)
    poisson_ratio: float         # range (0, 0.5)
    density: float               # kg/m^3, range (0, inf)
    yield_strength: float        # MPa, must be < tensile_strength
    tensile_strength: float      # MPa


class BoundaryCondition(TypedDict, total=False):
    node_id: int
    dof: List[str]               # e.g. ["tx", "ty", "tz", "rx", "ry", "rz"]
    type: str                    # "fixed", "pinned", "sliding"


class Load(TypedDict, total=False):
    node_id: int
    direction: str               # e.g. "tx", "-ty", "tz"
    magnitude: float             # N


class Mesh(TypedDict, total=False):
    element_type: str            # "tet4", "hex8", "shell3", etc.
    min_jacobian: float          # must be > 0; < 0 = inverted elements
    max_aspect_ratio: float      # dimensionless


class ModelSpec(TypedDict, total=False):
    part_type: str               # "bolt_iso4762" | "l_bracket" | "stepped_shaft"
    standard: str                # "ISO 4762" | "JIS B" | "ISO 286"
    dimensions: Dimensions
    material: Material
    boundary_conditions: List[BoundaryCondition]
    loads: List[Load]
    mesh: Mesh
    metadata: dict               # reserved for experiment tracking


# ── JSON Schema for LLM prompt (human-readable) ────────────────────────────

MODEL_SPEC_SCHEMA_DOC = """
{
  "part_type": "bolt_iso4762 | l_bracket | stepped_shaft",
  "standard": "ISO 4762 | JIS B | ISO 286",
  "dimensions": {
    // For bolt_iso4762:
    "nominal_diameter": 8.0,     // mm
    "length": 30.0,              // mm
    "head_diameter": 13.0,       // mm
    "head_height": 8.0,          // mm
    "thread_pitch": 1.25,        // mm

    // For l_bracket:
    "width": 50.0,               // mm
    "height": 50.0,              // mm
    "thickness": 6.0,            // mm
    "fillet_radius": 5.0,        // mm
    "bend_radius": 2.0,          // mm

    // For stepped_shaft:
    "segment_diameters": [30.0, 40.0, 30.0],   // mm
    "segment_lengths": [50.0, 60.0, 50.0],     // mm
    "segment_fillets": [2.0, 2.0]              // mm (N-1 entries)
  },
  "material": {
    "name": "Steel 304",
    "youngs_modulus": 200.0,     // GPa, MUST be > 0
    "poisson_ratio": 0.3,        // MUST be in (0, 0.5) exclusive
    "density": 7850.0,           // kg/m^3, MUST be > 0
    "yield_strength": 250.0,     // MPa, MUST be < tensile_strength
    "tensile_strength": 505.0    // MPa
  },
  "boundary_conditions": [
    {
      "node_id": 1,
      "dof": ["tx", "ty", "tz", "rx", "ry", "rz"],
      "type": "fixed"
    }
  ],
  "loads": [
    {
      "node_id": 2,
      "direction": "-ty",
      "magnitude": 1000.0         // N
    }
  ],
  "mesh": {
    "element_type": "tet4",
    "min_jacobian": 0.95,
    "max_aspect_ratio": 3.0
  }
}
"""

# ── Material bounds from standards ─────────────────────────────────────────

MATERIAL_BOUNDS = {
    "youngs_modulus":    (0.001, 1000),    # GPa — open upper bound for exotic materials
    "poisson_ratio":     (0.0,   0.5),     # strict thermodynamic constraint
    "density":           (100,   25000),   # kg/m^3 — covers polymers to tungsten
    "yield_strength":    (1,     5000),    # MPa
    "tensile_strength":  (1,     6000),    # MPa
}

# ── ISO 4762 M8 bolt standard dimensions (ground truth) ────────────────────

ISO4762_M8 = {
    "nominal_diameter": 8.0,
    "head_diameter": 13.0,
    "head_height": 8.0,
    "thread_pitch": 1.25,
}

# ── ISO 286 standard shaft tolerances ──────────────────────────────────────

ISO286_SHAFT = {
    "max_segment_ratio": 10.0,   # max diameter ratio between adjacent segments
    "min_diameter": 1.0,         # mm
    "min_fillet_radius": 0.5,    # mm
}

# ── MCP tool definitions (for Future Work / Discussion) ────────────────────

FUSION_TOOLS_DEFINITION = {
    "create_sketch":     {"description": "Create a 2D sketch on a plane"},
    "extrude_profile":   {"description": "Extrude a sketch profile to create/join/cut body"},
    "apply_material":    {"description": "Assign material to a body"},
    "set_boundary_cond": {"description": "Set boundary condition on nodes/faces"},
    "apply_load":        {"description": "Apply force/pressure/moment load"},
}
