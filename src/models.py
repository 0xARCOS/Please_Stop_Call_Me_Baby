"""Modelos Pydantic que validan la entrada y la salida del programa"""
from typing import Dict, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

ParamType = Literal["number", "integer", "string", "boolean"]
ParamValue = Union[bool, int, float, str]


class ParameterSpec(BaseModel):
    """Esquema de un parámetro: su tipo JSON."""
    type: ParamType


class ReturnSpec(BaseModel):
    """Tipo de retorno declarado (informativo, no se genera)."""
    type: str


class FunctionDefinition(BaseModel):
    """Una función que el LLM puede elegir."""

    name: str = Field(min_length=1, pattern=r'^[^"\\\s]+$')
    description: str = ""
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[ReturnSpec] = None


class PromptCase(BaseModel):
    """Una entrada de function_calling_tests.json."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Una entrada del archivo de salida: exactamente tres claves."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: Dict[str, ParamValue]
