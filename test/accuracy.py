"""Mide la precisión de una salida frente a las respuestas esperadas.

Uso:
    uv run python test/accuracy.py [resultados.json] [esperados.json]

Compara por posición: misma función y mismos parámetros (los números con
tolerancia). Es más estricto que la moulinette, que solo compara el
resultado de ejecutar la función (``[0-9]+`` y ``\\d+`` le dan igual).
"""

import json
import math
import sys
from typing import Any, Dict, List

DEFAULT_RESULTS = "data/output/function_calling_results.json"
DEFAULT_EXPECTED = "test/expected_results.json"


def same_value(got: Any, want: Any) -> bool:
    """Igualdad de valores; los números se comparan con tolerancia."""
    if isinstance(want, bool) or isinstance(got, bool):
        return got is want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9)
    return bool(got == want)


def same_call(got: Dict[str, Any], want: Dict[str, Any]) -> bool:
    """Misma función y mismos parámetros, sin claves de más ni de menos."""
    got_params = got.get("parameters", {})
    want_params = want["parameters"]
    return (
        got.get("name") == want["name"]
        and set(got_params) == set(want_params)
        and all(
            same_value(got_params[key], value)
            for key, value in want_params.items()
        )
    )


def main(argv: List[str]) -> int:
    """Imprime cada fallo y el porcentaje de aciertos."""
    results_path = argv[1] if len(argv) > 1 else DEFAULT_RESULTS
    expected_path = argv[2] if len(argv) > 2 else DEFAULT_EXPECTED
    try:
        with open(results_path, encoding="utf-8") as handle:
            results = json.load(handle)
        with open(expected_path, encoding="utf-8") as handle:
            expected = json.load(handle)
    except (OSError, json.JSONDecodeError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    by_prompt = {item.get("prompt"): item for item in results}
    hits = 0
    for want in expected:
        got = by_prompt.get(want["prompt"], {})
        if same_call(got, want):
            hits += 1
            continue
        print(f"FALLO: {want['prompt']}")
        print(f"   esperado: {want['name']} {want['parameters']}")
        print(f"   obtenido: {got.get('name')} {got.get('parameters')}")
    total = len(expected)
    percent = 100.0 * hits / total if total else 0.0
    print(f"Precisión: {hits}/{total} ({percent:.1f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
