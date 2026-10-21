import copy
import inspect
from typing import Any, Callable, Dict, List, get_origin, get_type_hints

# Mapeo de tipos nativos de python a JSON schema
TYPE_MAP = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}


def python_type_to_json_type(py_type: Any) -> str:
    """
    Convierte un tipo de Python a su equivalente en JSON Schema.

    Los genéricos (List[int], Dict[str, int]) se reducen a su tipo base.
    bool e int son claves distintas en TYPE_MAP, así que no se confunden.
    Los tipos desconocidos (Any, sin anotar...) se tratan como "string".
    """
    base_type = get_origin(py_type) or py_type
    return TYPE_MAP.get(base_type, "string")


class FunctionRegistry:
    """
    Registro centralizado de funciones con introspección, generación de
    JSON Schema y ejecución segura.
    """

    def __init__(self) -> None:
        self._functions: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register(self, func: Callable[..., Any]) -> None:
        """
        Registra una función y extrae sus metadatos mediante inspect.
        """
        name = func.__name__
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        type_hints = get_type_hints(func)

        properties: Dict[str, Dict[str, Any]] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            param_type = type_hints.get(param_name, Any)
            param_meta: Dict[str, Any] = {
                "type": python_type_to_json_type(param_type)
            }

            if param.default is not inspect.Parameter.empty:
                param_meta["default"] = param.default
            else:
                required.append(param_name)

            properties[param_name] = param_meta

        self._functions[name] = func
        self._schemas[name] = {
            "name": name,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "returns": {
                "type": python_type_to_json_type(
                    type_hints.get("return", Any)
                )
            },
        }

    def get(self, name: str) -> Callable[..., Any]:
        """Obtiene la referencia ejecutable de una función registrada."""
        if name not in self._functions:
            raise KeyError(f"La función '{name}' no está registrada.")
        return self._functions[name]

    def get_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Devuelve una copia defensiva del catálogo completo de esquemas JSON.
        """
        return copy.deepcopy(self._schemas)

    def get_function_schema(self, name: str) -> Dict[str, Any]:
        """
        Devuelve una copia del esquema JSON de una función específica.
        """
        if name not in self._schemas:
            raise KeyError(f"La función '{name}' no está registrada.")
        return copy.deepcopy(self._schemas[name])

    def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Valida e invoca una función registrada con sus argumentos.
        """
        func = self.get(name)
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        try:
            bound_args = sig.bind(**arguments)
        except TypeError as err:
            raise TypeError(
                f"Error en los argumentos para la función '{name}': {err}"
            ) from err

        bound_args.apply_defaults()

        # Validación estricta de tipos antes de la llamada
        for param_name, value in bound_args.arguments.items():
            if sig.parameters[param_name].kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            expected_type = type_hints.get(param_name, Any)
            # List[int] -> list; Any / Union / etc. no son clases: se omiten
            base_type = get_origin(expected_type) or expected_type
            if not isinstance(base_type, type):
                continue

            # bool es subclase de int en python, se verifica explícitamente
            if isinstance(value, bool) and base_type in (int, float):
                raise TypeError(
                    f"El parámetro '{param_name}' esperaba "
                    f"{base_type.__name__} pero recibió bool ({value})."
                )
            # Aceptamos int donde se espera float (JSON no los distingue)
            if base_type is float and isinstance(value, int):
                continue
            if not isinstance(value, base_type):
                raise TypeError(
                    f"El parámetro '{param_name}' esperaba "
                    f"{base_type.__name__} pero recibió "
                    f"{type(value).__name__}."
                )

        return func(*bound_args.args, **bound_args.kwargs)
