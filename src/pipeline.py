"""Orquesta: leer entradas, cargar el modelo, generar y escribir."""

import logging
import time
from pathlib import Path
from typing import List

from pydantic import BaseModel

from src.constrained_generator import ConstrainedGenerator
from src.io_utils import AppError, load_functions, load_prompts
from src.io_utils import write_results
from src.models import FunctionCallResult

logger = logging.getLogger(__name__)


class ExecutionPipeline(BaseModel):
    """Ejecuta el proceso completo para un juego de archivos."""

    functions_path: Path
    input_path: Path
    output_path: Path

    def run(self) -> int:
        """Procesa todos los prompts y devuelve cuántos se escribieron.

        Las entradas se validan antes de cargar el modelo: un archivo roto
        da error en un segundo, no después de cargar 600M de parámetros.
        """
        functions = load_functions(self.functions_path)
        cases = load_prompts(self.input_path)
        logger.info("Cargando el modelo...")
        generator = ConstrainedGenerator.create()
        results: List[FunctionCallResult] = []
        start = time.perf_counter()
        for number, case in enumerate(cases, start=1):
            logger.info("[%d/%d] %s", number, len(cases), case.prompt)
            try:
                result = generator.generate(case, functions)
            except (AppError, ValueError) as err:
                logger.error("Prompt %d omitido: %s", number, err)
                continue
            logger.info("    -> %s %s", result.name, result.parameters)
            results.append(result)
        logger.info(
            "Generación terminada en %.1f s", time.perf_counter() - start
        )
        write_results(self.output_path, results)
        return len(results)
