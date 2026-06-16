"""
Layer 2 Extension: LLM Self-Examine / Auto-Correction Loop
Implements the llm_self_examine enforcement mode from AgentSpec.
When Monitor detects violations, feeds them back to the LLM for
automatic correction, looping until the spec passes or max retries.

This is the key differentiator: Monitor doesn't just block,
it actively improves the modeling success rate through iterative feedback.

Architecture:
  Monitor.check(spec) → ViolationReport
  → If violations: format_feedback(report) → LLM.regenerate(feedback) → re-check
  → Loop until PASS or max_retries exhausted
"""

import json
import copy
from typing import Optional, Dict, Any, List, Tuple

from layer2_monitor import StructuralMonitor, ViolationReport, Violation


# ── Feedback Formatter ──────────────────────────────────────────────

def format_violation_feedback(report: ViolationReport) -> str:
    """Convert violations into actionable LLM feedback.

    Produces a structured feedback string that the LLM can use
    to understand exactly what went wrong and how to fix it.
    """
    errors = report.errors()
    warnings = report.warnings()

    lines = []
    lines.append("YOUR PREVIOUS ModelSpec JSON WAS REJECTED by the Runtime Monitor.")
    lines.append("")
    lines.append("The following issues were found:")

    if errors:
        lines.append(f"\n## CRITICAL ERRORS ({len(errors)} issues — MUST fix):")
        for i, v in enumerate(errors, 1):
            lines.append(f"  {i}. [{v.defect_id}] {v.description}")
            lines.append(f"     Field: {v.field_path}")
            lines.append(f"     Actual value: {v.actual_value}")
            if v.standard:
                lines.append(f"     Standard: {v.standard}")

    if warnings:
        lines.append(f"\n## WARNINGS ({len(warnings)} issues — should fix for best results):")
        for i, v in enumerate(warnings, 1):
            lines.append(f"  {i}. [{v.defect_id}] {v.description}")
            lines.append(f"     Field: {v.field_path}")

    lines.append("")
    lines.append("## FIX INSTRUCTIONS:")
    lines.append("1. Generate a COMPLETELY NEW ModelSpec JSON that fixes ALL of the above issues.")
    lines.append("2. For D5 (Material): ensure poisson_ratio is in (0, 0.5), youngs_modulus > 0 in GPa, density > 0 in kg/m3.")
    lines.append("3. For D1 (Unconstrained DOF): include at least one boundary_condition with type='fixed' covering all 6 DOF [tx,ty,tz,rx,ry,rz].")
    lines.append("4. For D6 (Mesh): ensure min_jacobian > 0.5 and max_aspect_ratio > 0.")
    lines.append("5. For D4 (Load-BC Conflict): loads and boundary_conditions must use DIFFERENT node_ids.")
    lines.append("6. For D2 (Negative Stiffness): ALL dimension values must be positive (> 0).")
    lines.append("7. For D3 (Stress Singularity): fillet_radius and segment_fillets must be > 0.")
    lines.append("")
    lines.append("Output ONLY the corrected JSON. No markdown, no commentary.")

    return "\n".join(lines)


# ── Self-Examine Monitor ────────────────────────────────────────────

class SelfExamineMonitor:
    """Monitor with automatic LLM-driven correction loop.

    Corresponds to AgentSpec's 'llm_self_examine' enforcement mode.
    When violations are found, feeds structured feedback back to the LLM
    and requests regeneration. Loops until clean or max retries exhausted.

    Parameters
    ----------
    generator : LLMGenerator
        Layer 1 LLM generator instance (must support generate() with
        user_description and system_prompt parameters).
    monitor : StructuralMonitor
        Layer 2 Monitor instance for checking specs.
    max_retries : int
        Maximum correction attempts before giving up (default: 3).
    verbose : bool
        Print correction progress to stdout.
    """

    def __init__(self, generator, monitor: StructuralMonitor,
                 max_retries: int = 3, verbose: bool = True):
        self.generator = generator
        self.monitor = monitor
        self.max_retries = max_retries
        self.verbose = verbose

        # Statistics
        self.stats = {
            "total_attempts": 0,
            "direct_pass": 0,
            "corrected_pass": 0,
            "failed_after_retries": 0,
        }

    def generate_with_correction(self, user_description: str,
                                  system_prompt: str,
                                  inject_defect: Optional[dict] = None,
                                  metadata: Optional[dict] = None
                                  ) -> Tuple[dict, ViolationReport, List[dict]]:
        """Generate a ModelSpec with automatic correction.

        Full pipeline:
          1. LLM generates initial spec
          2. Monitor checks it
          3. If violations → format feedback → LLM regenerates → re-check
          4. Repeat until PASS or max_retries exhausted

        Parameters
        ----------
        user_description : str
            Natural language part description.
        system_prompt : str
            Initial system prompt for the LLM.
        inject_defect : dict, optional
            Seeded defect to inject AFTER each generation attempt.
        metadata : dict, optional
            Extra metadata for the spec.

        Returns
        -------
        tuple
            (final_spec, final_report, correction_history)
            correction_history: list of dicts tracking each attempt
        """
        correction_history = []
        current_prompt = system_prompt

        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            self.stats["total_attempts"] += 1

            if self.verbose:
                label = "INITIAL" if attempt == 0 else f"CORRECTION #{attempt}"
                print(f"  [{label}] Generating...")

            # Generate spec (with defect injection for the initial attempt only;
            # correction attempts should try to fix naturally)
            spec = self.generator.generate(
                user_description,
                system_prompt=current_prompt,
                inject_defect=inject_defect if attempt == 0 else None,
                metadata={
                    **(metadata or {}),
                    "attempt": attempt,
                    "stage": "initial" if attempt == 0 else "correction",
                }
            )

            # Check
            report = self.monitor.check(spec)

            history_entry = {
                "attempt": attempt,
                "n_errors": len(report.errors()),
                "n_warnings": len(report.warnings()),
                "passed": report.passed,
                "defect_types": list(report.defect_types()),
                "defect_details": [
                    {"id": v.defect_id, "sev": v.severity, "desc": v.description[:150]}
                    for v in report.violations
                ],
            }
            correction_history.append(history_entry)

            if self.verbose:
                status = "PASS" if report.passed else "FAIL"
                print(f"  [{label}] Result: {status} | "
                      f"{history_entry['n_errors']}E/{history_entry['n_warnings']}W | "
                      f"defects={history_entry['defect_types']}")

            # If PASS → we're done
            if report.passed:
                if attempt == 0:
                    self.stats["direct_pass"] += 1
                else:
                    self.stats["corrected_pass"] += 1
                return spec, report, correction_history

            # If FAIL and not last attempt → generate feedback, retry
            if attempt < self.max_retries:
                feedback = format_violation_feedback(report)
                # Build correction prompt: original system prompt + violation feedback
                current_prompt = (
                    system_prompt + "\n\n"
                    "══════════════════════════════════════════\n"
                    + feedback
                )

        # Exhausted retries
        self.stats["failed_after_retries"] += 1
        return spec, report, correction_history

    def get_stats(self) -> dict:
        """Return correction statistics."""
        total = self.stats["total_attempts"]
        success = self.stats["direct_pass"] + self.stats["corrected_pass"]
        return {
            **self.stats,
            "success_rate": f"{success}/{max(1, total//2)} specs passed",
            "correction_efficiency": (
                f"{self.stats['corrected_pass']}/{self.stats['corrected_pass'] + self.stats['failed_after_retries']} "
                f"corrected specs passed"
            ) if (self.stats['corrected_pass'] + self.stats['failed_after_retries']) > 0 else "N/A",
        }


# ═══════════════════════════════════════════════════════════════════════
# Quick Test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from layer1_llm import LLMGenerator
    import os

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    gen = LLMGenerator(backend="deepseek" if api_key else "mock", api_key=api_key)
    monitor = StructuralMonitor()

    sem = SelfExamineMonitor(gen, monitor, max_retries=2, verbose=True)

    MINIMAL_PROMPT = "You are a CAD assistant. Generate a ModelSpec JSON. Output ONLY valid JSON."

    print("=" * 60)
    print("Self-Examine Correction Test")
    print("=" * 60)

    spec, report, history = sem.generate_with_correction(
        "M8x30 hex socket bolt per ISO 4762",
        system_prompt=MINIMAL_PROMPT,
    )

    print(f"\nFinal result: {'PASS' if report.passed else 'FAIL'}")
    print(f"Attempts: {len(history)}")
    print(f"Correction history: {[(h['attempt'], h['passed']) for h in history]}")
    print(f"\nStats: {json.dumps(sem.get_stats(), indent=2)}")
