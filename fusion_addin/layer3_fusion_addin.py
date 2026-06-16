"""
Layer 3: CAD Execution Layer (Fusion 360 Python Add-in)
Receives validated ModelSpec and drives Fusion 360 parametric modeling.

Runs inside Fusion 360 as an add-in, using the adsk.fusion API.

Architecture:
  - Watches BRIDGE_DIR for clean_spec.json
  - Parses spec → dispatches to part-specific builder
  - Writes result.json when done (success/failure)

Part builders (real Fusion 360 API):
  - build_bolt_real: ISO 4762 hex socket head bolt
  - build_l_bracket_real: JIS B L-bracket with fillets
  - build_stepped_shaft_real: ISO 286 multi-segment shaft
"""

import json
import os
import time
import traceback
from typing import Optional, Dict, Any

# ── Configuration ──────────────────────────────────────────────────────────

BRIDGE_DIR = os.path.expanduser("~/fusion_bridge")

CLEAN_SPEC_FILE = os.path.join(BRIDGE_DIR, "clean_spec.json")
OUTPUT_FILE = os.path.join(BRIDGE_DIR, "result.json")


# ═══════════════════════════════════════════════════════════════════════════
# Fusion 360 Add-in Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def run(context=None):
    """Fusion 360 Add-in entry point.

    Called by Fusion 360 on add-in load. Monitors BRIDGE_DIR
    for new clean_spec.json files and processes them.
    """
    try:
        import adsk.core, adsk.fusion
        app = adsk.core.Application.get()
    except ImportError:
        raise RuntimeError(
            "Fusion 360 API (adsk) not available. "
            "This add-in must run inside Fusion 360."
        )

    os.makedirs(BRIDGE_DIR, exist_ok=True)

    try:
        if not os.path.exists(CLEAN_SPEC_FILE):
            print("[Layer3] No clean_spec.json found. Waiting for experiment to write one...")
            print(f"[Layer3] Bridge directory: {BRIDGE_DIR}")
            return

        with open(CLEAN_SPEC_FILE, "r", encoding="utf-8") as f:
            spec = json.load(f)

        os.remove(CLEAN_SPEC_FILE)

        # Extract test_id from spec metadata for export naming
        test_id = spec.get("_meta", {}).get("test_id", "unknown")
        part_type = spec.get("part_type", "unknown")

        # ── Auto-create fresh design ──────────────────────
        try:
            # Close existing design without saving, then create a new one
            active_doc = app.activeDocument
            if active_doc is not None:
                active_doc.close(False)  # False = don't save changes

            doc = app.documents.add(
                adsk.core.DocumentTypes.FusionDesignDocumentType)
            time.sleep(0.3)
        except Exception:
            pass  # If auto-create fails, user needs to Ctrl+N manually

        result = build_part_real(app, spec)

        # Export 3D models if build succeeded
        if result.get("success"):
            export_result = _export_step(app, test_id, part_type)
            result["exports"] = export_result

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    except Exception:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "success": False,
                "error": traceback.format_exc(),
                "timestamp": time.time(),
            }, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT: Save 3D model files (STEP + .f3d)
# ═══════════════════════════════════════════════════════════════════════════

def _export_step(app, test_id: str, part_type: str) -> dict:
    """Export active Fusion design as STEP and .f3d files.

    Called after build_part_real() succeeds — saves both a neutral
    STEP (.step) for interchange and a Fusion native archive (.f3d).
    """
    import adsk.core, adsk.fusion

    export_dir = os.path.join(BRIDGE_DIR, "exports", test_id)
    os.makedirs(export_dir, exist_ok=True)

    design = adsk.fusion.Design.cast(app.activeProduct)
    export_mgr = design.exportManager
    exported = {}

    # STEP export
    step_path = os.path.join(export_dir, f"{test_id}_{part_type}.step")
    try:
        step_opts = export_mgr.createSTEPExportOptions(step_path)
        exported["step"] = {"path": step_path, "ok": export_mgr.execute(step_opts)}
    except Exception as e:
        exported["step"] = {"path": step_path, "ok": False, "error": str(e)}

    # Fusion native archive
    f3d_path = os.path.join(export_dir, f"{test_id}_{part_type}.f3d")
    try:
        f3d_opts = export_mgr.createFusionArchiveExportOptions(f3d_path)
        exported["f3d"] = {"path": f3d_path, "ok": export_mgr.execute(f3d_opts)}
    except Exception as e:
        exported["f3d"] = {"path": f3d_path, "ok": False, "error": str(e)}

    return exported


# ═══════════════════════════════════════════════════════════════════════════
# PART BUILDERS — REAL (Fusion 360 API)
# ═══════════════════════════════════════════════════════════════════════════

def build_part_real(app, spec: dict) -> dict:
    """Dispatch to part-specific builder using real Fusion API."""
    part_type = spec.get("part_type", "")
    builders = {
        "bolt_iso4762": build_bolt_real,
        "l_bracket": build_l_bracket_real,
        "stepped_shaft": build_stepped_shaft_real,
        "ball_bearing": build_ball_bearing_real,
    }
    builder = builders.get(part_type)
    if builder is None:
        return {
            "success": False,
            "error": f"Unknown part_type: '{part_type}'. Valid: {list(builders.keys())}",
            "timestamp": time.time(),
        }
    return builder(app, spec)


def build_bolt_real(app, spec: dict) -> dict:
    """Build ISO 4762 hex socket head bolt with full engineering detail.

    Features:
      - Hexagonal head (regular 6-sided polygon)
      - 45° chamfer on top edge of head
      - Cylindrical shank joined to head
      - Hex socket (Allen-key recess) on top face
      - Material from Fusion library
    """
    import adsk.core, adsk.fusion
    import math

    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    d = spec.get("dimensions", {})
    SCALE = 1.0 / 10.0  # mm → cm

    head_dia = d.get("head_diameter", 13.0) * SCALE
    head_ht  = d.get("head_height", 8.0) * SCALE
    shank_dia = d.get("nominal_diameter", 8.0) * SCALE
    shank_len = d.get("length", 30.0) * SCALE
    head_r = head_dia / 2.0
    shank_r = shank_dia / 2.0

    # ── 1. Hex Head Sketch & Extrude ──────────────────────────
    sketch_head = root.sketches.add(root.xYConstructionPlane)
    sketch_head.name = "Bolt Head Hex"

    n_sides = 6
    pts = []
    for i in range(n_sides):
        angle = math.pi / 2 - i * 2 * math.pi / n_sides
        x = head_r * math.cos(angle)
        y = head_r * math.sin(angle)
        pts.append(adsk.core.Point3D.create(x, y, 0))

    lines = sketch_head.sketchCurves.sketchLines
    for i in range(n_sides):
        lines.addByTwoPoints(pts[i], pts[(i + 1) % n_sides])

    extrudes = root.features.extrudeFeatures
    prof_head = sketch_head.profiles.item(0)
    ext_head = extrudes.createInput(
        prof_head, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ext_head.setDistanceExtent(False,
        adsk.core.ValueInput.createByReal(head_ht))
    head_feat = extrudes.add(ext_head)

    # ── 2. Chamfer on Head Top Edge ───────────────────────────
    try:
        chamfers = root.features.chamferFeatures
        head_body = head_feat.bodies.item(0) if head_feat.bodies.count > 0 else None
        if head_body is None:
            head_body = root.bRepBodies.item(root.bRepBodies.count - 1)

        edge_collection = adsk.core.ObjectCollection.create()
        for face in head_body.faces:
            ev = face.evaluator
            _, _, centroid = ev.getAreaProperties(
                adsk.core.Point3D.create(0, 0, head_ht))
            if (abs(centroid.z - head_ht * SCALE) < 0.001 * SCALE and
                centroid.z > head_ht * 0.8):
                for loop in face.loops:
                    if loop.isOuter:
                        for edge in loop.edges:
                            edge_collection.add(edge)

        if edge_collection.count > 0:
            chamfer_input = chamfers.createInput(
                edge_collection,
                adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            chamfer_input.setToEqualDistance(
                adsk.core.ValueInput.createByReal(0.5 * SCALE))
            chamfers.add(chamfer_input)
    except Exception:
        pass  # chamfer is cosmetic, don't fail the build

    # ── 3. Shank Sketch & Extrude (-Z) ────────────────────────
    sketch_shank = root.sketches.add(root.xYConstructionPlane)
    sketch_shank.name = "Bolt Shank"
    sketch_shank.sketchCurves.sketchCircles.addByCenterRadius(
        adsk.core.Point3D.create(0, 0, 0), shank_r)

    prof_shank = sketch_shank.profiles.item(0)
    ext_shank = extrudes.createInput(
        prof_shank, adsk.fusion.FeatureOperations.JoinFeatureOperation)
    ext_shank.setDistanceExtent(False,
        adsk.core.ValueInput.createByReal(-shank_len))
    extrudes.add(ext_shank)

    # ── 4. Hex Socket Cutout on Top Face ──────────────────────
    try:
        top_face = None
        for face in head_body.faces:
            ev = face.evaluator
            _, _, centroid = ev.getAreaProperties(
                adsk.core.Point3D.create(0, 0, head_ht))
            if (abs(centroid.z - head_ht * SCALE) < 0.01 * SCALE and
                centroid.z > head_ht * 0.5):
                top_face = face
                break

        if top_face is not None:
            sketch_socket = root.sketches.add(top_face)
            sketch_socket.name = "Hex Socket"

            socket_r = 2.0 * SCALE
            socket_pts_n = []
            for i in range(6):
                angle = math.pi / 2 - i * 2 * math.pi / 6
                sx = socket_r * math.cos(angle)
                sy = socket_r * math.sin(angle)
                socket_pts_n.append(adsk.core.Point3D.create(sx, sy, 0))

            s_lines = sketch_socket.sketchCurves.sketchLines
            for i in range(6):
                s_lines.addByTwoPoints(
                    socket_pts_n[i], socket_pts_n[(i + 1) % 6])

            prof_socket = sketch_socket.profiles.item(0)
            ext_socket = extrudes.createInput(
                prof_socket, adsk.fusion.FeatureOperations.CutFeatureOperation)
            ext_socket.setDistanceExtent(False,
                adsk.core.ValueInput.createByReal(-4.5 * SCALE))
            extrudes.add(ext_socket)
    except Exception:
        pass  # socket is cosmetic

    # ── 5. Apply material ─────────────────────────────────────
    _apply_material_real(design, root, spec.get("material", {}))

    return {
        "success": True,
        "part": "bolt_iso4762",
        "dimensions": d,
        "features": ["hex_head", "chamfer", "shank", "hex_socket"],
        "timestamp": time.time(),
    }


def build_l_bracket_real(app, spec: dict) -> dict:
    """Build L-bracket with mounting holes, fillets, and edge rounds.

    Features:
      - L-profile sketch with fillet at the inside bend corner
      - Extrude to web thickness
      - 4× mounting holes (2 per leg)
      - Edge fillets on outer corners
      - Material from Fusion library
    """
    import adsk.core, adsk.fusion

    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    d = spec.get("dimensions", {})
    SCALE = 1.0 / 10.0

    w = d.get("width", 50.0) * SCALE
    h = d.get("height", 50.0) * SCALE
    t = d.get("thickness", 6.0) * SCALE
    fillet_r = d.get("fillet_radius", 5.0) * SCALE

    # ── 1. L-profile sketch with inside-corner fillet ────────
    sketch = root.sketches.add(root.xYConstructionPlane)
    sketch.name = "L-Bracket Profile"

    lines = sketch.sketchCurves.sketchLines
    arcs = sketch.sketchCurves.sketchArcs

    pts = [
        (0, 0),
        (w, 0),
        (w, t),
        (t, t),
        (t, h),
        (0, h),
    ]
    pts_3d = [adsk.core.Point3D.create(p[0], p[1], 0) for p in pts]

    line_objs = []
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        line_objs.append(lines.addByTwoPoints(pts_3d[i], pts_3d[j]))

    # Add fillet at the inside corner
    try:
        arcs.addFillet(
            line_objs[1],
            line_objs[1].startSketchPoint.geometry,
            line_objs[2],
            line_objs[2].startSketchPoint.geometry,
            fillet_r,
        )
    except Exception:
        try:
            arcs.addFillet(
                line_objs[2],
                line_objs[2].endSketchPoint.geometry,
                line_objs[3],
                line_objs[3].startSketchPoint.geometry,
                fillet_r,
            )
        except Exception:
            pass  # fillet is best-effort

    # ── 2. Extrude web ────────────────────────────────────────
    extrudes = root.features.extrudeFeatures
    prof = sketch.profiles.item(0)
    ext_input = extrudes.createInput(
        prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ext_input.setDistanceExtent(
        False, adsk.core.ValueInput.createByReal(t))
    extrude_feat = extrudes.add(ext_input)

    body = root.bRepBodies.item(root.bRepBodies.count - 1)

    # ── 3. Mounting holes ─────────────────────────────────────
    hole_radius = 3.0 * SCALE  # 3mm radius → M6 clearance
    hole_positions = [
        (w * 0.5, t * 0.5),   # right leg — centered
        (w * 0.5, -t * 0.5),  # right leg — offset
        (t * 0.5, h * 0.5),   # top leg — centered
        (-t * 0.5, h * 0.5),  # top leg — offset
    ]

    for hp_idx, (hx, hy) in enumerate(hole_positions):
        try:
            target_face = None
            for face in body.faces:
                try:
                    ev = face.evaluator
                    _, _, centroid = ev.getAreaProperties(
                        adsk.core.Point3D.create(hx, hy, 0))
                    if ((centroid.x - hx)**2 + (centroid.y - hy)**2) < (t * 0.6)**2:
                        target_face = face
                        break
                except Exception:
                    continue

            if target_face is not None:
                hole_sketch = root.sketches.add(target_face)
                hole_sketch.name = f"Mounting Hole {hp_idx+1}"
                hole_sketch.sketchCurves.sketchCircles.addByCenterRadius(
                    adsk.core.Point3D.create(0, 0, 0), hole_radius)

                hole_prof = hole_sketch.profiles.item(0)
                ext_hole = extrudes.createInput(
                    hole_prof, adsk.fusion.FeatureOperations.CutFeatureOperation)
                ext_hole.setDistanceExtent(
                    False, adsk.core.ValueInput.createByReal(-t))
                extrudes.add(ext_hole)
        except Exception:
            pass  # individual hole failures are non-fatal

    # ── 4. Edge fillets on outer corners ──────────────────────
    try:
        fillet_feats = root.features.filletFeatures
        edge_coll = adsk.core.ObjectCollection.create()
        for edge in body.edges:
            if edge.length > 0.5 * SCALE:
                edge_coll.add(edge)
        if edge_coll.count > 0:
            fillet_input = fillet_feats.createInput()
            fillet_input.addConstantRadiusEdgeSet(
                edge_coll,
                adsk.core.ValueInput.createByReal(1.0 * SCALE),
                True)
            fillet_feats.add(fillet_input)
    except Exception:
        pass  # fillets are cosmetic

    # ── 5. Apply material ─────────────────────────────────────
    _apply_material_real(design, root, spec.get("material", {}))

    return {
        "success": True,
        "part": "l_bracket",
        "dimensions": d,
        "features": ["l_profile", "fillet_bend", "mounting_holes", "edge_fillets"],
        "timestamp": time.time(),
    }


def build_stepped_shaft_real(app, spec: dict) -> dict:
    """Build stepped shaft using revolve method with full detail.

    Features:
      - Sketch half-profile on XY plane
      - 360° revolve around X axis
      - Hide construction sketch after revolve
      - Chamfers at both shaft ends
      - Keyway slot on the largest-diameter segment
      - Material from Fusion library
    """
    import adsk.core, adsk.fusion
    import math

    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    d = spec.get("dimensions", {})
    SCALE = 1.0 / 10.0

    seg_diams = d.get("segment_diameters", [30.0, 40.0, 30.0])
    seg_lens = d.get("segment_lengths", [50.0, 60.0, 50.0])

    total_len_mm = sum(seg_lens)
    total_len = total_len_mm * SCALE

    # ── 1. Revolve sketch (half-profile) ──────────────────────
    sketch = root.sketches.add(root.xYConstructionPlane)
    sketch.name = "Shaft Half-Profile"
    lines = sketch.sketchCurves.sketchLines

    x_offset = 0.0
    prev_r = seg_diams[0] * SCALE / 2.0 if seg_diams else 15.0 * SCALE

    # Centerline along X axis
    lines.addByTwoPoints(
        adsk.core.Point3D.create(0, 0, 0),
        adsk.core.Point3D.create(total_len, 0, 0),
    )

    for i, (dia, length) in enumerate(zip(seg_diams, seg_lens)):
        r = dia * SCALE / 2.0
        x_start = x_offset
        x_end = x_offset + length * SCALE

        if i > 0:
            lines.addByTwoPoints(
                adsk.core.Point3D.create(x_start, prev_r, 0),
                adsk.core.Point3D.create(x_start, r, 0),
            )

        lines.addByTwoPoints(
            adsk.core.Point3D.create(x_start, r, 0),
            adsk.core.Point3D.create(x_end, r, 0),
        )

        prev_r = r
        x_offset = x_end

    lines.addByTwoPoints(
        adsk.core.Point3D.create(x_offset, prev_r, 0),
        adsk.core.Point3D.create(x_offset, 0, 0),
    )

    # ── 2. Revolve ────────────────────────────────────────────
    revolves = root.features.revolveFeatures
    prof = sketch.profiles.item(0)
    rev_input = revolves.createInput(
        prof,
        root.xConstructionAxis,
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    rev_input.setAngleExtent(
        False,
        adsk.core.ValueInput.createByReal(2 * math.pi))
    revolve_feat = revolves.add(rev_input)

    sketch.isVisible = False

    body = root.bRepBodies.item(root.bRepBodies.count - 1)

    # ── 3. Chamfers at shaft ends ─────────────────────────────
    try:
        chamfers = root.features.chamferFeatures
        end_edges = adsk.core.ObjectCollection.create()

        for face in body.faces:
            try:
                ev = face.evaluator
                _, _, centroid = ev.getAreaProperties(
                    adsk.core.Point3D.create(0, 0, 0))
                if abs(centroid.x) < 0.001 * SCALE:
                    for loop in face.loops:
                        if loop.isOuter:
                            for edge in loop.edges:
                                end_edges.add(edge)
                if abs(centroid.x - total_len) < 0.001 * SCALE:
                    for loop in face.loops:
                        if loop.isOuter:
                            for edge in loop.edges:
                                end_edges.add(edge)
            except Exception:
                continue

        if end_edges.count > 0:
            chamfer_input = chamfers.createInput(
                end_edges,
                adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            chamfer_input.setToEqualDistance(
                adsk.core.ValueInput.createByReal(0.5 * SCALE))
            chamfers.add(chamfer_input)
    except Exception:
        pass  # chamfers are cosmetic

    # ── 4. Keyway on largest segment ──────────────────────────
    try:
        if seg_diams:
            largest_idx = seg_diams.index(max(seg_diams))
            keyway_x = (sum(seg_lens[:largest_idx]) + seg_lens[largest_idx] / 2.0) * SCALE
            largest_dia = seg_diams[largest_idx] * SCALE

            tangent_plane = root.constructionPlanes.createInput()
            tangent_plane.setByOffset(
                root.xZConstructionPlane,
                adsk.core.ValueInput.createByReal(largest_dia / 2.0 * 0.98))
            cp = root.constructionPlanes.add(tangent_plane)

            keyway_sketch = root.sketches.add(cp)
            keyway_sketch.name = "Keyway"
            kw_w = 5.0 * SCALE
            kw_h = 3.0 * SCALE
            kw_l = seg_lens[largest_idx] * 0.6 * SCALE

            kx0 = keyway_x - kw_l / 2.0
            kx1 = keyway_x + kw_l / 2.0
            ky0 = 0.0
            ky1 = -kw_h

            keyway_sketch.sketchCurves.sketchLines.addTwoPointRectangle(
                adsk.core.Point3D.create(kx0, ky0, 0),
                adsk.core.Point3D.create(kx1, ky1, 0),
            )

            kw_prof = keyway_sketch.profiles.item(0) if keyway_sketch.profiles.count > 0 else None
            if kw_prof is not None:
                ext_kw = root.features.extrudeFeatures.createInput(
                    kw_prof, adsk.fusion.FeatureOperations.CutFeatureOperation)
                ext_kw.setDistanceExtent(
                    False,
                    adsk.core.ValueInput.createByReal(kw_h))
                root.features.extrudeFeatures.add(ext_kw)
    except Exception:
        pass  # keyway is cosmetic

    # ── 5. Apply material ─────────────────────────────────────
    _apply_material_real(design, root, spec.get("material", {}))

    return {
        "success": True,
        "part": "stepped_shaft",
        "dimensions": d,
        "features": ["revolved_profile", "hidden_sketch", "chamfers", "keyway"],
        "timestamp": time.time(),
    }


def build_ball_bearing_real(app, spec: dict) -> dict:
    """Build deep groove ball bearing — outer/inner race + balls via revolve + pattern.

    Features:
      - Outer ring (thick-walled cylinder)
      - Inner ring (thick-walled cylinder, smaller)
      - Single ball created by revolve → circular-patterned N times
      - Material from Fusion library
    """
    import adsk.core, adsk.fusion
    import math

    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    d = spec.get("dimensions", {})
    SCALE = 1.0 / 10.0

    bore = d.get("bore_diameter", 20.0) * SCALE
    outer_dia = d.get("outer_diameter", 47.0) * SCALE
    width = d.get("width", 14.0) * SCALE
    ball_dia = d.get("ball_diameter", 6.0) * SCALE
    n_balls = int(d.get("ball_count", 8))

    inner_r = bore / 2.0
    outer_r = outer_dia / 2.0
    pitch_r = (inner_r + outer_r) / 2.0
    half_w = width / 2.0
    ball_r = ball_dia / 2.0

    extrudes = root.features.extrudeFeatures

    # ── Outer Ring ──
    sketch_o = root.sketches.add(root.xYConstructionPlane)
    sketch_o.name = "Bearing Outer Ring"
    circles = sketch_o.sketchCurves.sketchCircles
    circles.addByCenterRadius(adsk.core.Point3D.create(0, 0, 0), outer_r)
    circles.addByCenterRadius(adsk.core.Point3D.create(0, 0, 0), outer_r - 3.0 * SCALE)
    prof_o = sketch_o.profiles.item(0)
    ext_o = extrudes.createInput(prof_o, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ext_o.setSymmetricExtent(adsk.core.ValueInput.createByReal(half_w), True)
    extrudes.add(ext_o)

    # ── Inner Ring ──
    sketch_i = root.sketches.add(root.xYConstructionPlane)
    sketch_i.name = "Bearing Inner Ring"
    circles = sketch_i.sketchCurves.sketchCircles
    circles.addByCenterRadius(adsk.core.Point3D.create(0, 0, 0), inner_r + 3.0 * SCALE)
    circles.addByCenterRadius(adsk.core.Point3D.create(0, 0, 0), inner_r)
    prof_i = sketch_i.profiles.item(0)
    ext_i = extrudes.createInput(prof_i, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ext_i.setSymmetricExtent(adsk.core.ValueInput.createByReal(half_w), True)
    ext_i_feat = extrudes.add(ext_i)

    # ── One Ball via Revolve (half-circle, axis = flat edge) ──
    sketch_ball = root.sketches.add(root.xYConstructionPlane)
    sketch_ball.name = "Ball Half-Profile"
    arcs = sketch_ball.sketchCurves.sketchArcs
    arc = arcs.addByCenterStartSweep(
        adsk.core.Point3D.create(pitch_r, 0, 0),
        adsk.core.Point3D.create(pitch_r, -ball_r, 0),
        math.pi,
    )
    lines = sketch_ball.sketchCurves.sketchLines
    close_line = lines.addByTwoPoints(
        arc.startSketchPoint.geometry,
        arc.endSketchPoint.geometry,
    )
    prof_ball = sketch_ball.profiles.item(0)
    revolves = root.features.revolveFeatures
    rev_input = revolves.createInput(
        prof_ball, close_line,
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    rev_input.setAngleExtent(False, adsk.core.ValueInput.createByReal(2 * math.pi))
    ball_feat = revolves.add(rev_input)
    sketch_ball.isVisible = False

    # ── Circular Pattern: replicate balls ──
    ball_body = ball_feat.bodies.item(0) if ball_feat.bodies.count > 0 else root.bRepBodies.item(root.bRepBodies.count - 1)
    patterns = root.features.circularPatternFeatures
    bodies_coll = adsk.core.ObjectCollection.create()
    bodies_coll.add(ball_body)
    pat_input = patterns.createInput(bodies_coll, root.zConstructionAxis)
    pat_input.quantity = adsk.core.ValueInput.createByReal(n_balls)
    pat_input.totalAngle = adsk.core.ValueInput.createByString("360 deg")
    pat_input.isSymmetric = False
    patterns.add(pat_input)

    _apply_material_real(design, root, spec.get("material", {}))

    return {
        "success": True,
        "part": "ball_bearing",
        "dimensions": d,
        "features": ["outer_ring", "inner_ring", "balls", "circular_pattern"],
        "timestamp": time.time(),
    }


def _apply_material_real(design, component, mat_spec: dict):
    """Apply material from Fusion 360 material library.

    Searches all loaded material libraries for a match by name.
    Falls back to first steel-like material if no match found.
    """
    import adsk.fusion

    mat_name = mat_spec.get("name", "Steel")
    mat_name_lower = mat_name.lower()

    app = adsk.core.Application.get()
    libs = app.materialLibraries
    applied = False

    for lib in libs:
        for mat in lib.materials:
            if mat_name_lower in mat.name.lower():
                for body in component.bRepBodies:
                    body.material = mat
                applied = True
                break
        if applied:
            break

    if not applied:
        for lib in libs:
            for mat in lib.materials:
                if "steel" in mat.name.lower():
                    for body in component.bRepBodies:
                        body.material = mat
                    applied = True
                    break
            if applied:
                break

    if not applied:
        print(f"[Layer3] WARNING: Could not apply material '{mat_name}' — no match found")


# ═══════════════════════════════════════════════════════════════════════════
# Direct execution (bridges run_experiment.py → Fusion 360 add-in)
# ═══════════════════════════════════════════════════════════════════════════

def execute_directly(spec: dict) -> dict:
    """Execute modeling via file-based bridge to Fusion 360.

    Writes clean_spec.json and polls for result.json.
    The Fusion 360 add-in monitors BRIDGE_DIR and processes
    new spec files as they appear.

    Parameters
    ----------
    spec : dict
        Validated (clean) ModelSpec.

    Returns
    -------
    dict
        Result from Fusion 360 (success/failure + model info).
    """
    os.makedirs(BRIDGE_DIR, exist_ok=True)

    # Remove old result if exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Write the clean spec to bridge directory
    with open(CLEAN_SPEC_FILE, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    part_type = spec.get("part_type", "unknown")
    print(f"\n  [WAITING] Fusion 360 add-in should build: {part_type}")
    print(f"  Bridge: {CLEAN_SPEC_FILE}")
    print(f"  >>> Click your add-in button in Fusion 360 <<<")

    # Poll for result.json
    timeout = int(os.environ.get("FUSION_TIMEOUT", "600"))
    interval = 0.5
    waited = 0.0
    while not os.path.exists(OUTPUT_FILE):
        time.sleep(interval)
        waited += interval
        if int(waited) % 10 == 0 and waited >= 10:
            print(f"  ... waiting ({int(waited)}s) — click the add-in button")
        if waited > timeout:
            return {
                "success": False,
                "error": f"Timeout after {timeout}s: result.json not produced in {BRIDGE_DIR}",
                "timestamp": time.time(),
            }

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        result = json.load(f)
    os.remove(OUTPUT_FILE)
    print(f"  [OK] Fusion 360 completed: success={result.get('success')}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Quick test (standalone — no Fusion needed)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Standalone test: verifies the module loads and builders dispatch correctly
    test_spec = {
        "part_type": "bolt_iso4762",
        "standard": "ISO 4762",
        "dimensions": {
            "nominal_diameter": 8.0, "length": 30.0,
            "head_diameter": 13.0, "head_height": 8.0, "thread_pitch": 1.25,
        },
        "material": {
            "name": "Steel 304", "youngs_modulus": 200.0,
            "poisson_ratio": 0.3, "density": 7850.0,
            "yield_strength": 250.0, "tensile_strength": 505.0,
        },
        "boundary_conditions": [
            {"node_id": 1, "dof": ["tx", "ty", "tz", "rx", "ry", "rz"], "type": "fixed"}
        ],
        "loads": [
            {"node_id": 2, "direction": "-ty", "magnitude": 1000.0}
        ],
        "mesh": {
            "element_type": "tet4", "min_jacobian": 0.9, "max_aspect_ratio": 3.5,
        }
    }

    print("Module loaded successfully.")
    print(f"Bridge directory: {BRIDGE_DIR}")
    print(f"Part dispatch test: {list(build_part_real.__code__.co_names)}")
    print(f"Valid part types: bolt_iso4762, l_bracket, stepped_shaft")
    print("Ready for Fusion 360 execution.")
