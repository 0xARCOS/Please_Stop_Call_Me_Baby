"""Lectura y escritura de los archivos JSON del programa"""
import json
import os
from pathlib import Path
from typing import Any, List
from pydantic import TypeAdapter, ValidationError
from src.models import FunctionCallResult, FunctionDefinition, PromptCase


class AppError(Exception):
    """Error esperado que se muestra al usuario sin traza."""


def read_json(path: Path) -> Any:
    """Lee un archivo JSON y traduce cualquier fallo a AppError."""
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise AppError(f"No existe el archivo: {path}") from None
    except PermissionError:
        raise AppError(f"Sin permiso de lectura: {path}") from None
    except IsADirectoryError:
        raise AppError(f"{path} es una carpeta, no un archivo") from None
    except json.JSONDecodeError as err:
        raise AppError(
            f"JSON inválido en {path} (línea {err.lineno}, "
            f"columna {err.colno}): {err.msg}"
        ) from None
    except UnicodeDecodeError:
        raise AppError(f"{path} no está codificado en UTF-8") from None
    except RecursionError:
        raise AppError(f"JSON demasiado anidado en {path}") from None
    except OSError as err:
        reason = err.strerror or err
        raise AppError(f"No se puede leer {path}: {reason}") from None


def _describe(err: ValidationError) -> str:
    """Resume el primer error de validación en una línea legible."""
    first = err.errors()[0]
    location = ".".join(str(part) for part in first["loc"]) or "raíz"
    return f"{location}: {first['msg']}"


def load_functions(path: Path) -> List[FunctionDefinition]:
    """Carga y valida functions_definition.json."""
    data = read_json(path)
    try:
        adapter = TypeAdapter(List[FunctionDefinition])
        functions = adapter.validate_python(data)
    except ValidationError as err:
        raise AppError(
            f"Definición de funciones inválida en {path}: {_describe(err)}"
        ) from None
    if not functions:
        raise AppError(f"{path} no define ninguna función.")
    names = [function.name for function in functions]
    if len(names) != len(set(names)):
        raise AppError(f"{path} contiene nombres de función repetidos.")
    return functions


def load_prompts(path: Path) -> List[PromptCase]:
    """Carga y valida function_calling_tests.json."""
    data = read_json(path)
    try:
        return TypeAdapter(List[PromptCase]).validate_python(data)
    except ValidationError as err:
        raise AppError(
            f"Archivo de prompts inválido en {path}: {_describe(err)}"
        ) from None


def write_results(path: Path, results: List[FunctionCallResult]) -> None:
    """Escribe los resultados como un array JSON, creando la carpeta.

    Escribe primero en un archivo temporal y lo renombra al final: si el
    programa se interrumpe a mitad (Ctrl+C), no queda un JSON a medias.
    """
    payload = [result.model_dump() for result in results]
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, indent=2, ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(tmp_path, path)
    except PermissionError:
        raise AppError(f"Sin permiso de escritura: {path}") from None
    except OSError as err:
        reason = err.strerror or err
        raise AppError(f"No se puede escribir {path}: {reason}") from None
    except ValueError as err:
        raise AppError(f"No se puede escribir {path}: {err}") from None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
