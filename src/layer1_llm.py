"""
Layer 1: LLM Generation Layer
Calls LLM APIs to convert natural language → structured ModelSpec JSON.
Supports seeded defect injection for controlled experiments.

Supported backends:
  - deepseek / deepseek-pro: DeepSeek models (via OpenAI-compatible endpoint)
  - gemini: Google Gemini models (via Google Generative AI SDK)
  - openai_compatible: ANY OpenAI-compatible API (OpenAI, Groq, Together AI,
    Fireworks, Anyscale, local vLLM/Ollama, etc.) — configure via env vars

Environment Variables (for openai_compatible backend):
  LLM_API_BASE_URL  — API base URL (default: https://api.openai.com/v1)
  LLM_MODEL         — Model name (default: gpt-4o)
  LLM_TEMPERATURE   — Sampling temperature (default: 1.0)
  LLM_MAX_TOKENS    — Max output tokens (default: 2048)

Usage:
    from layer1_llm import LLMGenerator
    # DeepSeek
    gen = LLMGenerator(backend="deepseek", api_key="sk-...")
    # OpenAI / any compatible provider
    gen = LLMGenerator(backend="openai_compatible", api_key="sk-...")
    spec = gen.generate("M8x30 stainless steel bolt")
"""

import json
import os
import time
import copy
from typing import Optional, Dict, Any


# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are a mechanical design assistant specializing in FEA
(Finite Element Analysis) pre-processing. Your task is to generate a DETAILED,
REALISTIC ModelSpec JSON object representing a parametric CAD part.

CRITICAL — Physical validity rules (you MUST follow exactly):
1. Poisson's ratio MUST be strictly between 0 and 0.5 (exclusive): 0 < ν < 0.5
   - Steel: ν = 0.3, Aluminum: ν = 0.33
2. Young's modulus MUST be in GPa (NOT Pa): E > 0 GPa
   - Steel: E = 200 (GPa), Aluminum: E = 69 (GPa)
   - IMPORTANT: Output 200 NOT 200000000000 (that would be Pa, not GPa)
3. Density MUST be in kg/m³: ρ > 0
   - Steel: ρ = 7850, Aluminum: ρ = 2700
4. Yield strength MUST be less than tensile strength: σ_y < σ_uts
5. All dimensions MUST be positive (no zero or negative values)
6. boundary_conditions MUST be a LIST of objects, each with node_id (int), dof (list of strings), type (string)
7. loads MUST be a LIST of objects, each with node_id (int), direction (string), magnitude (float)
8. Boundary condition nodes MUST be DIFFERENT from load application nodes
   - Typically: BC at node 1, load at node 2
9. At least one boundary condition with type="fixed" covering all 6 DOF
   - ["tx","ty","tz","rx","ry","rz"]
10. mesh must have min_jacobian > 0 and max_aspect_ratio > 0

DIMENSIONAL DETAIL — Include rich, manufacturable dimensions:
- For bolts: head_diameter, head_height, nominal_diameter, length, thread_pitch,
  head_chamfer_angle, socket_size (hex socket across-flats)
- For brackets: width, height, thickness, fillet_radius (inside corner),
  hole_diameter, hole_count, edge_round_radius
- For bearings: bore_diameter, outer_diameter, width, ball_diameter, ball_count,
  raceway_groove_radius, cage_type, shield_type

Part types: bolt_iso4762, l_bracket, ball_bearing

Output ONLY valid JSON in this exact structure — no markdown fences, no commentary:
{
  "part_type": "bolt_iso4762",
  "standard": "ISO 4762",
  "dimensions": {...},
  "material": {...},
  "boundary_conditions": [{"node_id": 1, "dof": [...], "type": "fixed"}],
  "loads": [{"node_id": 2, "direction": "...", "magnitude": 1000.0}],
  "mesh": {"element_type": "tet4", "min_jacobian": 0.9, "max_aspect_ratio": 3.5}
}"""


# ── LLM Generator Class ────────────────────────────────────────────────────

class LLMGenerator:
    """Generates ModelSpec JSON from natural language using LLM APIs.

    Supports DeepSeek, Gemini, and any OpenAI-compatible API (OpenAI,
    Groq, Together AI, Fireworks, vLLM, Ollama, etc.).

    Parameters
    ----------
    backend : str
        One of "deepseek", "deepseek-pro", "gemini", or "openai_compatible".
    api_key : str, optional
        API key for the LLM service. If omitted, reads from environment:
        DEEPSEEK_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.
    model : str, optional
        Model name override. If omitted, uses default for the backend.
    base_url : str, optional
        Override API base URL (for openai_compatible or custom endpoints).
    temperature : float, optional
        Override sampling temperature. Default 1.0; reads LLM_TEMPERATURE env var.
    max_tokens : int, optional
        Override max output tokens. Default 2048; reads LLM_MAX_TOKENS env var.
    """

    def __init__(self, backend: str = "deepseek", api_key: Optional[str] = None,
                 model: Optional[str] = None, base_url: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        self.backend = backend.lower()

        # Resolve API key: explicit arg > backend-specific env > generic OPENAI_API_KEY
        if api_key:
            self.api_key = api_key
        elif self.backend in ("deepseek", "deepseek-pro"):
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        elif self.backend == "gemini":
            self.api_key = os.environ.get("GEMINI_API_KEY", "")
        elif self.backend == "openai_compatible":
            self.api_key = (os.environ.get("OPENAI_API_KEY") or
                           os.environ.get("DEEPSEEK_API_KEY") or
                           os.environ.get("GEMINI_API_KEY") or "")
        else:
            self.api_key = ""

        if self.backend not in ("deepseek", "deepseek-pro", "gemini", "openai_compatible"):
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                f"Use 'deepseek', 'deepseek-pro', 'gemini', or 'openai_compatible'."
            )

        # Model resolution: explicit arg > LLM_MODEL env > backend default
        if model:
            self.model = model
        elif os.environ.get("LLM_MODEL"):
            self.model = os.environ.get("LLM_MODEL")
        elif self.backend == "deepseek":
            self.model = "deepseek-chat"
        elif self.backend == "gemini":
            self.model = "gemini-2.5-flash"
        elif self.backend == "deepseek-pro":
            self.model = "deepseek-v4-pro"
        elif self.backend == "openai_compatible":
            self.model = "gpt-4o"

        # Base URL: explicit arg > LLM_API_BASE_URL env > backend default
        if base_url:
            self.base_url = base_url
        elif os.environ.get("LLM_API_BASE_URL"):
            self.base_url = os.environ.get("LLM_API_BASE_URL")
        elif self.backend in ("deepseek", "deepseek-pro"):
            self.base_url = "https://api.deepseek.com"
        elif self.backend == "openai_compatible":
            self.base_url = "https://api.openai.com/v1"
        else:
            self.base_url = None

        # Temperature and max_tokens: explicit arg > env var > default
        if temperature is not None:
            self.temperature = temperature
        elif os.environ.get("LLM_TEMPERATURE"):
            self.temperature = float(os.environ.get("LLM_TEMPERATURE"))
        else:
            self.temperature = 1.0

        if max_tokens is not None:
            self.max_tokens = max_tokens
        elif os.environ.get("LLM_MAX_TOKENS"):
            self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS"))
        else:
            self.max_tokens = 2048

        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate API client. Raises on failure."""
        if self.backend in ("deepseek", "deepseek-pro", "openai_compatible"):
            from openai import OpenAI
            if not self.api_key:
                raise ValueError(
                    f"API key is required for '{self.backend}' backend. "
                    f"Set the appropriate environment variable "
                    f"(DEEPSEEK_API_KEY, OPENAI_API_KEY, etc.) or pass api_key= explicitly."
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

        elif self.backend == "gemini":
            import google.generativeai as genai
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is required for Gemini backend")
            genai.configure(api_key=self.api_key)
            self._client = genai

    def generate(self, user_description: str,
                 system_prompt: Optional[str] = None,
                 inject_defect: Optional[Dict[str, Any]] = None,
                 metadata: Optional[dict] = None) -> dict:
        """Generate a ModelSpec from a natural language description.

        Parameters
        ----------
        user_description : str
            Natural language description of the part.
        system_prompt : str, optional
            Override system prompt.
        inject_defect : dict, optional
            Seeded defect injection. Format: {"path": "material.poisson_ratio", "value": 0.6}
            Also accepts a list of such dicts for multi-defect injection.
        metadata : dict, optional
            Extra metadata to attach to the spec.

        Returns
        -------
        dict
            ModelSpec with optional metadata.
        """
        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        if self.backend in ("deepseek", "deepseek-pro", "openai_compatible"):
            spec = self._generate_deepseek(user_description, prompt)
        elif self.backend == "gemini":
            spec = self._generate_gemini(user_description, prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Post-process: inject seeded defect(s)
        if inject_defect:
            if isinstance(inject_defect, list):
                for inj in inject_defect:
                    spec = self._inject_defect(spec, inj)
            else:
                spec = self._inject_defect(spec, inject_defect)

        # Attach metadata
        if metadata:
            spec["metadata"] = metadata

        return spec

    def _generate_deepseek(self, user_description: str, system_prompt: str) -> dict:
        """Call OpenAI-compatible API with response_format json_object.
        Retries up to 3 times on empty/invalid responses (handles API rate limits)."""
        import time as _time
        last_error = None
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate a ModelSpec for: {user_description}"}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = response.choices[0].message.content
                if not raw or not raw.strip():
                    raise ValueError("LLM returned empty response")
                return self._parse_llm_response(raw)
            except Exception as e:
                last_error = e
                if attempt < 2:
                    wait = (attempt + 1) * 3  # 3s, 6s backoff
                    print(f"    [RETRY {attempt+1}/2] LLM call failed: {e}. Waiting {wait}s...")
                    _time.sleep(wait)
        raise last_error

    def _generate_gemini(self, user_description: str, system_prompt: str) -> dict:
        """Call Gemini API with retry logic."""
        import google.generativeai as genai
        import time as _time

        # Configure response_mime_type for JSON output (Gemini 1.5+)
        gen_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "response_mime_type": "application/json",
        }

        last_error = None
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(
                    model_name=self.model or "gemini-2.5-flash",
                    system_instruction=system_prompt,
                )
                response = model.generate_content(
                    f"Generate a ModelSpec JSON for: {user_description}",
                    generation_config=gen_config,
                )
                raw = response.text
                if not raw or not raw.strip():
                    raise ValueError("Gemini returned empty response")
                return self._parse_llm_response(raw)
            except Exception as e:
                last_error = e
                if attempt < 2:
                    wait = (attempt + 1) * 3
                    print(f"    [GEMINI RETRY {attempt+1}/2] {e}. Waiting {wait}s...")
                    _time.sleep(wait)
        raise last_error

    def _parse_llm_response(self, raw: str) -> dict:
        """Parse LLM response, stripping markdown fences if present."""
        raw = raw.strip()
        # Remove markdown json fences if present
        if raw.startswith("```"):
            first_newline = raw.find("\n")
            if first_newline > 0:
                raw = raw[first_newline + 1:]
            # Remove trailing ```
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        return json.loads(raw)

    def _inject_defect(self, spec: dict, inject_defect: dict) -> dict:
        """Inject a seeded defect into the spec at the specified path."""
        spec = copy.deepcopy(spec)
        keys = inject_defect["path"].split(".")
        target = spec
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            if not isinstance(target[k], dict):
                raise ValueError(f"Cannot traverse into {k}: not a dict (value={target[k]})")
            target = target[k]
        target[keys[-1]] = inject_defect["value"]
        return spec


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Detect backend from environment
    api_key = (os.environ.get("DEEPSEEK_API_KEY") or
              os.environ.get("GEMINI_API_KEY") or
              os.environ.get("OPENAI_API_KEY"))
    backend = os.environ.get("LLM_BACKEND", "deepseek")

    if not api_key:
        print("No API key found. Set DEEPSEEK_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.")
        print("Skipping live test.")
    else:
        gen = LLMGenerator(backend=backend, api_key=api_key)
        spec = gen.generate("M8x30 stainless steel bolt")
        print(f"Backend: {gen.backend} (model={gen.model}, base_url={gen.base_url})")
        print(f"Temperature: {gen.temperature}, Max tokens: {gen.max_tokens}")
        print(json.dumps(spec, indent=2))
