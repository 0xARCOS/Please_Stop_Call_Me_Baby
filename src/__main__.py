"""Punto de entrada: uv run python -m src [opciones]."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from src.io_utils import AppError
from src.pipeline import ExecutionPipeline


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Lee los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Traduce prompts a llamadas a función con un LLM.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="JSON con las funciones disponibles",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="JSON con los prompts a procesar",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Ruta del JSON de resultados",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Ejecuta el programa y devuelve el código de salida."""
    level = logging.DEBUG if os.environ.get("CMM_DEBUG") else logging.INFO
    # Las bibliotecas (httpx, huggingface_hub...) solo muestran avisos;
    # el progreso INFO es solo el nuestro (loggers "src.*").
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s: %(message)s"
    )
    logging.getLogger("src").setLevel(level)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    args = parse_args()
    pipeline = ExecutionPipeline(
        functions_path=args.functions_definition,
        input_path=args.input,
        output_path=args.output,
    )
    try:
        count = pipeline.run()
    except AppError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        return 130
    except Exception as err:
        print(f"Error inesperado: {err}", file=sys.stderr)
        return 1
    print(f"{count} prompts procesados -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
