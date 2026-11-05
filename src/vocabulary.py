"""Vocabulario del modelo: id de token <-> bytes que representa.

Para enmascarar logits necesitamos saber qué texto representa cada id de
token. Así podemos decidir, por ejemplo, "este token no tiene comillas" o
"este token es el dígito 7".

- llm.get_path_to_vocab_file() devuelve la ruta a vocab.json: un
  diccionario {"token": id}.
- Qwen usa BPE a nivel de byte (igual que GPT-2): cada token es una
  secuencia de bytes, pero en vocab.json los bytes aparecen "disfrazados"
  de caracteres imprimibles. Por ejemplo, el espacio (byte 32) se escribe
  Ġ y el salto de línea (byte 10) se escribe Ċ.
- La tabla de disfraces es fija y conocida (bytes_to_unicode de GPT-2):
  los bytes imprimibles se representan a sí mismos y el resto se desplaza
  a partir de chr(256). Invertimos esa tabla para recuperar los bytes
  reales.
- Guardamos bytes y no str. Un carácter no ASCII (é, emojis) puede quedar
  partido entre dos tokens. Si concatenamos bytes y decodificamos al
  final, no se pierde nada.
- Los tokens especiales (<|im_end|>, etc.) no están en vocab.json, así que
  nunca aparecerán en nuestras listas de permitidos. Eso es justo lo que
  queremos.
- Usar el vocabulario en lugar de decode() es también la base del bonus
  "tokenizador reimplementado".
"""

import json
from typing import Dict

from pydantic import BaseModel

from src.io_utils import AppError


def bytes_to_unicode() -> Dict[int, str]:
    """Tabla byte -> carácter usada por el BPE a nivel de byte.

    Los bytes imprimibles se representan a sí mismos; el resto se
    desplaza a partir de chr(256) (por eso el espacio aparece como 'Ġ').
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table = {byte: chr(byte) for byte in printable}
    shift = 0
    for byte in range(256):
        if byte not in table:
            table[byte] = chr(256 + shift)
            shift += 1
    return table


class Vocabulary(BaseModel):
    """Correspondencia entre ids de token y sus bytes."""

    token_bytes: Dict[int, bytes]
    bytes_to_id: Dict[bytes, int]

    @classmethod
    def from_file(cls, path: str) -> "Vocabulary":
        """Construye el vocabulario a partir de vocab.json."""
        char_to_byte = {
            char: byte for byte, char in bytes_to_unicode().items()
        }
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw: Dict[str, int] = json.load(handle)
        except (OSError, json.JSONDecodeError) as err:
            raise AppError(f"No se puede leer el vocabulario: {err}") from None
        token_bytes: Dict[int, bytes] = {}
        for token, token_id in raw.items():
            if all(char in char_to_byte for char in token):
                token_bytes[token_id] = bytes(char_to_byte[c] for c in token)
        inverse = {data: token_id for token_id, data in token_bytes.items()}
        return cls(token_bytes=token_bytes, bytes_to_id=inverse)

    def id_of(self, text: str) -> int:
        """Id del token que representa exactamente ``text``."""
        data = text.encode("utf-8")
        if data not in self.bytes_to_id:
            raise AppError(f"El vocabulario no tiene un token para {text!r}")
        return self.bytes_to_id[data]
