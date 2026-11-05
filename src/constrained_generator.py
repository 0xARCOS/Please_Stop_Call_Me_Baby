"""Generación de llamadas a función con decodificación restringida.

Nosotros escribimos toda la estructura JSON (``{"name": "``, las claves de
los parámetros, las comas...) y el modelo solo decide en tres tipos de
huecos: qué función, el valor de un número y el contenido de un string.
En cada hueco se piden los logits, se pone -inf en los tokens no
permitidos y se toma el máximo (§5.3.3).
"""

import logging
from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model
from src.io_utils import AppError
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    ParamType,
    ParamValue,
    PromptCase,
)
from src.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-0.6B"
DIGITS = "0123456789"
SIMPLE_ESCAPES = {'"': '"', "\\": "\\", "/": "/"}
HEX_DIGITS = "0123456789abcdefABCDEF"
OPENING = b' "'
CLOSING_RESTS = (b"", b",", b"}", b"}}")

# Estados del autómata de strings: antes de la comilla de apertura, dentro
# del string, o dentro justo después de una barra invertida sin pareja.
StringState = Literal["open", "normal", "escaped"]


def number_next(text: str, integer: bool) -> Tuple[str, bool]:
    r"""Caracteres válidos tras ``text`` y si ``text`` ya es un número.

    Gramática JSON: -?(0|[1-9][0-9]*)(\.[0-9]+)?  (sin punto si integer).
    """
    if text == "":
        return DIGITS + "-", False
    if text == "-":
        return DIGITS, False
    if text.endswith("."):
        return DIGITS, False
    allowed = "" if text.lstrip("-") == "0" else DIGITS
    if not integer and "." not in text:
        allowed += "."
    return allowed, True


def scan_string_token(data: bytes, escaped: bool) -> Tuple[int, bool]:
    """Recorre un token que aparece dentro de un string JSON.

    ``escaped`` indica si el token empieza justo después de una barra
    invertida sin emparejar. Devuelve ``(cierre, escaped_final)``:
    ``cierre`` es la posición de la comilla que cierra el string, o -1 si
    el token no lo cierra. Lanza ValueError si el token no puede ir dentro
    de un string (caracteres de control).
    """
    for position, byte in enumerate(data):
        if byte < 0x20:
            raise ValueError("carácter de control")
        if escaped:
            escaped = False
        elif byte == 0x5C:  # barra invertida
            escaped = True
        elif byte == 0x22:  # comilla
            return position, False
    return -1, escaped


def decode_json_string(raw: bytes) -> str:
    r"""Interpreta los escapes que haya escrito el modelo.

    ``\"``, ``\\``, ``\/`` y ``\uXXXX`` se traducen como en JSON. El resto
    (``\d``, ``\b``, ``\n``...) se conserva literalmente: en una petición
    de una línea, ``\b`` o ``\d`` casi siempre son parte de una regex o de
    una ruta, no un carácter de control.
    """
    text = raw.decode("utf-8", errors="replace")
    out: List[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "\\" or index + 1 == len(text):
            out.append(char)
            index += 1
            continue
        follower = text[index + 1]
        code = text[index + 2:index + 6]
        if follower in SIMPLE_ESCAPES:
            out.append(SIMPLE_ESCAPES[follower])
            index += 2
        elif (
            follower == "u"
            and len(code) == 4
            and all(c in HEX_DIGITS for c in code)
        ):
            out.append(chr(int(code, 16)))
            index += 6
        else:
            out.append(char + follower)
            index += 2
    return "".join(out)


def build_prompt(user_prompt: str, functions: List[FunctionDefinition]) -> str:
    """Prompt en formato de chat de Qwen con la lista de funciones.

    El prompt solo informa al modelo; la estructura la garantiza la
    máscara de logits.
    """
    lines = [
        "Convert the user's request into one function call.",
        "Pick the function that matches the request. Copy each argument "
        "value exactly as written in the request, without the quotes "
        "around it. Keep double quotes inside a value, escaped as \\\". "
        "When a pattern is needed, write a regular expression.",
        "Functions:",
    ]
    for function in functions:
        params = ", ".join(
            f"{name}: {spec.type}"
            for name, spec in function.parameters.items()
        )
        lines.append(
            f"- {function.name}({params}): {function.description}"
        )
    system = "\n".join(lines)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def key_text(name: str, kind: ParamType) -> str:
    """Texto forzado antes de un valor, con el espaciado natural de JSON.

    Se elige para que el tokenizador produzca los mismos tokens que en un
    JSON normal: ``":`` + ``Ġtrue`` para booleanos y ``":`` + ``Ġ`` +
    dígitos para números. En los strings la comilla de apertura la elige
    el modelo, porque Qwen la fusiona con el contenido (``Ġ"/``, ``Ġ"[``).
    """
    if kind in ("string", "boolean"):
        return f'"{name}":'
    return f'"{name}": '


class ConstrainedGenerator(BaseModel):
    """Elige función y argumentos token a token enmascarando logits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Small_LLM_Model
    vocab: Vocabulary
    max_number_tokens: int = 24
    max_string_tokens: int = 64

    _content: Dict[StringState, Dict[int, StringState]] = PrivateAttr(
        default_factory=dict
    )
    _closing: Dict[StringState, Dict[int, Tuple[int, bytes]]] = PrivateAttr(
        default_factory=dict
    )

    @classmethod
    def create(cls, model_name: str = MODEL_NAME) -> "ConstrainedGenerator":
        """Carga el modelo y su vocabulario.

        Un fallo aquí (sin red, sin espacio para descargar ~1,5 GB, caché
        corrupta) se traduce a AppError con una pista para resolverlo.
        """
        try:
            llm = Small_LLM_Model(model_name)
        except (OSError, ValueError, RuntimeError) as err:
            raise AppError(
                f"No se pudo cargar el modelo {model_name}: {err}\n"
                "Pista: el modelo ocupa ~1,5 GB y se descarga en "
                "~/.cache/huggingface. Si falta espacio o red, define "
                "HF_HOME en una carpeta con espacio donde ya esté el modelo."
            ) from None
        vocab = Vocabulary.from_file(llm.get_path_to_vocab_file())
        return cls(llm=llm, vocab=vocab)

    def model_post_init(self, __context: Any) -> None:
        """Precalcula qué tokens se permiten en cada estado de un string.

        Se recorre el vocabulario una sola vez por estado. Para cada token
        se guarda si es "de contenido" (no cierra el string, y en qué
        estado lo deja) o "de cierre" (contiene la comilla final y, como
        mucho, ``,`` o ``}`` detrás, como el token ``",`` de Qwen). En el
        estado ``open`` solo valen tokens que empiezan por `` "``.
        """
        states: Tuple[StringState, ...] = ("open", "normal", "escaped")
        for state in states:
            content: Dict[int, StringState] = {}
            closing: Dict[int, Tuple[int, bytes]] = {}
            for token_id, data in self.vocab.token_bytes.items():
                skip = 0
                if state == "open":
                    if not data.startswith(OPENING):
                        continue
                    skip = len(OPENING)
                elif not data:
                    continue
                try:
                    end, escaped = scan_string_token(
                        data[skip:], state == "escaped"
                    )
                except ValueError:
                    continue
                if end < 0:
                    content[token_id] = "escaped" if escaped else "normal"
                elif data[skip + end + 1:] in CLOSING_RESTS:
                    closing[token_id] = (skip + end, data[skip + end + 1:])
            self._content[state] = content
            self._closing[state] = closing

    def _encode(self, text: str) -> List[int]:
        """Texto -> ids (encode del SDK devuelve un tensor [1, n])."""
        return [int(token) for token in self.llm.encode(text)[0].tolist()]

    def _pick(self, ids: List[int], allowed: Sequence[int]) -> int:
        """Token permitido con mayor logit; el resto queda a -inf.

        Si solo hay una opción no se llama al modelo: cada llamada es un
        forward completo sobre toda la secuencia.
        """
        if not allowed:
            raise AppError("No queda ningún token permitido.")
        if len(allowed) == 1:
            return allowed[0]
        logits = np.array(
            self.llm.get_logits_from_input_ids(ids), dtype=np.float32
        )
        index = np.array(allowed, dtype=np.int64)
        masked = np.full_like(logits, -np.inf)
        masked[index] = logits[index]
        return int(np.argmax(masked))

    def _choose(self, ids: List[int], options: Dict[str, List[int]]) -> str:
        """Elige una opción entre secuencias de tokens (recorrido de trie).

        ``ids`` se amplía con los tokens elegidos.
        """
        for name, seq in options.items():
            for other, other_seq in options.items():
                if name != other and other_seq[:len(seq)] == seq:
                    raise AppError(f"{name!r} es prefijo de {other!r}")
        alive = list(options)
        position = 0
        while len(alive) > 1 or position < len(options[alive[0]]):
            allowed = sorted({options[name][position] for name in alive})
            token = self._pick(ids, allowed)
            ids.append(token)
            alive = [n for n in alive if options[n][position] == token]
            position += 1
        return alive[0]

    def _stop_ids(self, last: bool) -> List[int]:
        """Tokens que cierran un valor: ``,`` o, si es el último, ``}``."""
        if not last:
            return [self.vocab.id_of(",")]
        stops = [self.vocab.id_of("}")]
        if b"}}" in self.vocab.bytes_to_id:
            stops.append(self.vocab.bytes_to_id[b"}}"])
        return stops

    def _number(self, ids: List[int], integer: bool, last: bool) -> str:
        """Genera el texto de un número válido; el modelo decide cuándo
        termina (elige el token de parada en lugar de otro dígito).
        """
        stops = self._stop_ids(last)
        text = ""
        for _ in range(self.max_number_tokens):
            chars, complete = number_next(text, integer)
            allowed = [self.vocab.id_of(char) for char in chars]
            if complete:
                allowed += stops
            token = self._pick(ids, allowed)
            ids.append(token)
            if token in stops:
                break
            text += self.vocab.token_bytes[token].decode("ascii")
        else:
            if not number_next(text, integer)[1]:
                text += "0"
                ids.append(self.vocab.id_of("0"))
            ids.append(stops[0])
        return text

    def _string(self, ids: List[int], last: bool) -> str:
        """Genera un string completo, desde la comilla de apertura.

        En cada paso solo se permiten tokens válidos para el estado actual
        del autómata (ver ``model_post_init``). Se acumulan bytes y se
        decodifica al final, para no romper caracteres multibyte.
        """
        stops = self._stop_ids(last)
        stop_texts = [self.vocab.token_bytes[token] for token in stops]
        raw = b""
        state: StringState = "open"
        for _ in range(self.max_string_tokens):
            content = self._content[state]
            closing = {
                token: end
                for token, (end, rest) in self._closing[state].items()
                if rest == b"" or rest in stop_texts
            }
            token = self._pick(ids, list(content) + list(closing))
            ids.append(token)
            data = self.vocab.token_bytes[token]
            skip = len(OPENING) if state == "open" else 0
            if token in closing:
                raw += data[skip:closing[token]]
                if data[closing[token] + 1:] == b"":
                    ids.append(stops[0])
                break
            raw += data[skip:]
            state = content[token]
        else:
            if state == "escaped":
                raw = raw[:-1]
            ids.append(self.vocab.id_of('"'))
            ids.append(stops[0])
        return decode_json_string(raw)

    def _boolean(self, ids: List[int], last: bool) -> bool:
        """Elige entre true y false y termina con ``,`` o ``}``."""
        options = {
            "true": self._encode(" true"),
            "false": self._encode(" false"),
        }
        choice = self._choose(ids, options)
        ids.append(self._stop_ids(last)[0])
        return choice == "true"

    def _value(
        self, ids: List[int], kind: ParamType, last: bool
    ) -> ParamValue:
        """Despacha según el tipo declarado del parámetro."""
        if kind == "string":
            return self._string(ids, last)
        if kind == "boolean":
            return self._boolean(ids, last)
        text = self._number(ids, kind == "integer", last)
        return int(text) if kind == "integer" else float(text)

    def _after_name(self, function: FunctionDefinition) -> str:
        """Texto forzado entre el nombre y el primer valor."""
        if not function.parameters:
            return ' "parameters": {}}'
        first, spec = next(iter(function.parameters.items()))
        return ' "parameters": {' + key_text(first, spec.type)

    def generate(
        self, case: PromptCase, functions: List[FunctionDefinition]
    ) -> FunctionCallResult:
        """Produce la llamada a función para un prompt.

        Secuencia: prompt, ``{"name": "``, nombre elegido + ``",``,
        ``"parameters": {`` y, por cada parámetro, su clave forzada y su
        valor restringido. Todo el estado vive dentro de esta llamada.
        """
        by_name = {function.name: function for function in functions}
        ids = self._encode(build_prompt(case.prompt, functions))
        prompt_length = len(ids)
        ids += self._encode('{"name": "')
        name = self._choose(
            ids, {f.name: self._encode(f'{f.name}",') for f in functions}
        )
        function = by_name[name]
        ids += self._encode(self._after_name(function))
        values: Dict[str, ParamValue] = {}
        items = list(function.parameters.items())
        for index, (param, spec) in enumerate(items):
            if index > 0:
                ids += self._encode(" " + key_text(param, spec.type))
            last = index == len(items) - 1
            values[param] = self._value(ids, spec.type, last)
        if logger.isEnabledFor(logging.DEBUG):
            generated = self.llm.decode(ids[prompt_length:])
            logger.debug("El modelo ha escrito: %s", generated)
        return FunctionCallResult(
            prompt=case.prompt, name=name, parameters=values
        )
