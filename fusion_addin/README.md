# FMforME Fusion 360 Add-in

Receives validated ModelSpec JSON from the Monitor and drives parametric solid modeling in Autodesk Fusion 360.

## Supported Part Types

- **ISO 4762 Hex Socket Bolt**: Hexagonal head with chamfer, cylindrical shank
- **JIS B L-Bracket**: L-shaped profile with inside-corner fillet, four M6 mounting holes
- **ISO 286 Stepped Shaft**: Multi-segment cylindrical shaft with diameter transitions

## Installation

1. Copy this entire `FMforME` folder into the Fusion 360 Add-Ins directory:

   ```
   %APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\FMforME\
   ```

   The final path should be:
   ```
   %APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\FMforME\FMforME_AddIn.py
   %APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\FMforME\layer3_fusion_addin.py
   ```

2. Launch Fusion 360

3. Open **Tools** → **Add-Ins** (or press `Shift+S` and search "Add-Ins")

4. Find **FMforME** in the list, select it, and click **Run**

5. (Optional) Check "Run on Startup" to auto-start the add-in when Fusion 360 opens

## How It Works

The add-in monitors a bridge directory for new validated specifications:

```
~/fusion_bridge/
├── clean_spec.json    ← Monitor writes validated spec here
└── result.json        ← Add-in writes build result here
```

### Data Flow

1. **Layer 2 Monitor** validates the LLM-generated ModelSpec JSON
2. If passed, the spec is written to `~/fusion_bridge/clean_spec.json`
3. The add-in detects the new file and dispatches to the appropriate part builder
4. The parametric model is constructed using the `adsk.fusion` Python API
5. Results (including STEP and F3D export paths) are written to `result.json`

### Model Construction (centimeter-scale)

| Part | Key Features | API Methods |
|---|---|---|
| Bolt | 6-sided polygon head + cylindrical shank + 45° chamfer | `Sketch.addByCenterStartEnd`, `ExtrudeFeatureInput`, `ChamferFeatureInput` |
| L-Bracket | 6-point L-profile + corner fillet + 4 M6 holes + edge rounds | `Sketch.addByTwoPointRectangle`, `HoleFeatureInput`, `FilletFeatureInput` |
| Shaft | Multi-segment cylindrical extrusions with diameter transitions | `ExtrudeFeatureInput`, multiple segment construction |

### Material Assignment

Materials are assigned via fuzzy name matching against the Fusion material library:
- Stainless Steel 316 → "Stainless Steel"
- Steel 304 → "Steel"
- Aluminum 6061 → "Aluminum"

### Export Formats

- **STEP** (ISO 10303-21): Neutral CAD exchange format
- **F3D**: Fusion 360 native archive format

Both are saved to `~/fusion_bridge/exports/<part_type>_<timestamp>.*`

## Bridge Protocol

### Input (`clean_spec.json`)

```json
{
  "part_type": "bolt_iso4762",
  "dimensions": {
    "nominal_diameter": 10.0,
    "head_diameter": 16.0,
    "head_height": 10.0,
    "shank_length": 50.0,
    "thread_length": 26.0,
    "chamfer_angle": 45.0
  },
  "material": {
    "name": "Stainless Steel 316",
    "youngs_modulus": 193.0,
    "poisson_ratio": 0.30,
    "density": 8000.0,
    "yield_strength": 290.0,
    "tensile_strength": 580.0
  },
  "boundary_conditions": [
    {"type": "fixed", "node_set": [1, 2, 3, 4]}
  ],
  "loads": [
    {"type": "pressure", "node_set": [10, 11, 12], "value": 5.0, "unit": "MPa"}
  ],
  "mesh": {
    "element_type": "tet10",
    "element_size": 2.0,
    "min_jacobian": 0.72,
    "max_aspect_ratio": 12.3
  }
}
```

### Output (`result.json`)

```json
{
  "success": true,
  "part_type": "bolt_iso4762",
  "step_path": "~/fusion_bridge/exports/bolt_iso4762_20260617_120000.step",
  "f3d_path": "~/fusion_bridge/exports/bolt_iso4762_20260617_120000.f3d",
  "build_time_ms": 2340.5,
  "error": null
}
```

## Threading Model

The add-in uses a two-thread architecture to respect Fusion 360's API requirements:

1. **Background thread**: Polls `~/fusion_bridge/` every 1 second for new `clean_spec.json`
2. **Main thread**: All `adsk.fusion` API calls are dispatched via `CustomEvent` to the main thread

This ensures all geometric operations execute on Fusion 360's main thread as required by the API.

## Requirements

- Autodesk Fusion 360 (any license tier with API access)
- Python 3.7+ (bundled with Fusion 360)
- No additional Python packages required (uses only Fusion 360 built-in modules)

## Troubleshooting

| Issue | Solution |
|---|---|
| Add-in not appearing | Verify folder is in `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\` |
| "Failed to create sketch" | Check that dimensions in `clean_spec.json` are positive |
| "Material not found" | Verify material name matches a known Fusion material (Steel, Aluminum, etc.) |
| Build timeout | Check that `max_aspect_ratio` ≤ 50 and `min_jacobian` ≥ 0.3 |
| STEP export failed | Ensure write permissions to `~/fusion_bridge/exports/` |
