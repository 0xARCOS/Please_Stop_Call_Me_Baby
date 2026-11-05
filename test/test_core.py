"""Tests de las piezas que no necesitan el modelo real."""

import json
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pytest
from pydantic import ValidationError

from llm_sdk import Small_LLM_Model

from src.constrained_generator import (
    DIGITS,
    ConstrainedGenerator,
    build_prompt,
    decode_json_string,
    key_text,
    number_next,
    scan_string_token,
)
from src.io_utils import AppError, load_functions, load_prompts
from src.io_utils import write_results
from src.models import FunctionCallResult, FunctionDefinition, PromptCase
from src.vocabulary import Vocabulary, bytes_to_unicode

EXTRA_TOKENS = [
    b'",', b'"}', b'\\"', b" true", b" false", b"}}", b' "', b' "/', b' ""',
]


def fake_vocab() -> Vocabulary:
    """Vocabulario mínimo: un token por byte y algunos tokens fusionados."""
    token_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for offset, data in enumerate(EXTRA_TOKENS):
        token_bytes[256 + offset] = data
    inverse = {data: token_id for token_id, data in token_bytes.items()}
    return Vocabulary(token_bytes=token_bytes, bytes_to_id=inverse)


class FakeLLM:
    """Imita la API pública del SDK; ``script`` fija qué token prefiere."""

    def __init__(self, vocab: Vocabulary, script: List[bytes]) -> None:
        self.vocab = vocab
        self.script = [vocab.bytes_to_id[data] for data in script]
        self.calls = 0

    def encode(self, text: str) -> "np.ndarray[Any, Any]":
        """Un token por byte, como un tokenizador sin fusiones."""
        return np.array([list(text.encode("utf-8"))])

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """El siguiente token del guion recibe el logit más alto."""
        logits = [0.0] * (256 + len(EXTRA_TOKENS))
        logits[self.script[self.calls]] = 10.0
        self.calls += 1
        return logits


def fake_generator(script: List[bytes]) -> ConstrainedGenerator:
    """Generador con el LLM falso (model_construct omite la validación)."""
    vocab = fake_vocab()
    return ConstrainedGenerator.model_construct(
        llm=cast(Small_LLM_Model, FakeLLM(vocab, script)),
        vocab=vocab,
        max_number_tokens=24,
        max_string_tokens=64,
    )


def test_number_grammar() -> None:
    assert number_next("", False) == (DIGITS + "-", False)
    assert number_next("-", True) == (DIGITS, False)
    assert number_next("0", False) == (".", True)
    assert number_next("0", True) == ("", True)
    assert number_next("12", False) == (DIGITS + ".", True)
    assert number_next("12", True) == (DIGITS, True)
    assert number_next("1.", False) == (DIGITS, False)
    assert number_next("1.5", False) == (DIGITS, True)


def test_scan_string_token() -> None:
    assert scan_string_token(b"hello", False) == (-1, False)
    assert scan_string_token(b'",', False) == (0, False)
    assert scan_string_token(b'\\"', False) == (-1, False)
    assert scan_string_token(b'"', True) == (-1, False)
    assert scan_string_token(b"ab\\", False) == (-1, True)
    with pytest.raises(ValueError):
        scan_string_token(b"a\nb", False)


def test_decode_json_string() -> None:
    assert decode_json_string(b"hello") == "hello"
    assert decode_json_string(b"\\\\d+") == "\\d+"
    assert decode_json_string(b"\\d+") == "\\d+"
    assert decode_json_string(b"\\bcat\\b") == "\\bcat\\b"
    assert decode_json_string(b'Say \\"hi\\"') == 'Say "hi"'
    assert decode_json_string(b"C:\\\\Users") == "C:\\Users"
    assert decode_json_string(b"\\u00e9") == "é"
    assert decode_json_string("é".encode()) == "é"


def test_byte_table() -> None:
    table = bytes_to_unicode()
    assert table[32] == "Ġ"
    assert table[10] == "Ċ"
    assert table[ord("a")] == "a"
    assert len(set(table.values())) == 256


def test_vocabulary_from_file(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"Ġis": 5, "7": 9}), encoding="utf-8")
    vocab = Vocabulary.from_file(str(path))
    assert vocab.token_bytes[5] == b" is"
    assert vocab.id_of("7") == 9
    with pytest.raises(AppError):
        vocab.id_of("missing")


def test_key_text() -> None:
    assert key_text("s", "string") == '"s":'
    assert key_text("b", "boolean") == '"b":'
    assert key_text("n", "number") == '"n": '


def test_build_prompt_lists_functions() -> None:
    function = FunctionDefinition(
        name="fn_x",
        description="Does x.",
        parameters={"a": {"type": "number"}},
    )
    prompt = build_prompt("hi", [function])
    assert "- fn_x(a: number): Does x." in prompt
    assert prompt.endswith("<think>\n\n</think>\n\n")


def test_string_handles_escaped_quote() -> None:
    generator = fake_generator([b' "', b'\\"', b"a", b'\\"', b'",'])
    ids: List[int] = []
    assert generator._string(ids, last=False) == '"a"'


def test_string_opening_token_with_content() -> None:
    generator = fake_generator([b' "/', b"x", b'"}'])
    ids: List[int] = []
    assert generator._string(ids, last=True) == "/x"


def test_string_can_be_empty() -> None:
    generator = fake_generator([b' ""'])
    ids: List[int] = []
    assert generator._string(ids, last=False) == ""
    assert ids[-1] == generator.vocab.id_of(",")


def test_string_closing_token_without_stop() -> None:
    generator = fake_generator([b' "', b"h", b"i", b'"'])
    ids: List[int] = []
    assert generator._string(ids, last=True) == "hi"
    assert ids[-1] == generator.vocab.id_of("}")


def test_number_stops_when_model_chooses() -> None:
    generator = fake_generator([b"4", b"2", b","])
    ids: List[int] = []
    assert generator._number(ids, integer=True, last=False) == "42"


def test_choose_follows_logits() -> None:
    generator = fake_generator([b"b"])
    options = {"fa": [ord("f"), ord("a")], "fb": [ord("f"), ord("b")]}
    ids: List[int] = []
    assert generator._choose(ids, options) == "fb"
    assert cast(FakeLLM, generator.llm).calls == 1


def test_generate_end_to_end() -> None:
    generator = fake_generator([b"g", b' "', b"j", b"o", b"e", b'"}'])
    number = {"type": "number"}
    string = {"type": "string"}
    functions = [
        FunctionDefinition(name="fn_add", parameters={"a": number}),
        FunctionDefinition(name="fn_greet", parameters={"n": string}),
    ]
    result = generator.generate(PromptCase(prompt="Greet joe"), functions)
    assert result.name == "fn_greet"
    assert result.parameters == {"n": "joe"}


def test_result_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        FunctionCallResult.model_validate(
            {"prompt": "p", "name": "f", "parameters": {}, "extra": 1}
        )


def test_result_keeps_types() -> None:
    result = FunctionCallResult(
        prompt="p", name="f", parameters={"a": 2.0, "b": True, "c": 3}
    )
    assert isinstance(result.parameters["a"], float)
    assert result.parameters["b"] is True
    assert isinstance(result.parameters["c"], int)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AppError, match="No existe"):
        load_prompts(tmp_path / "nope.json")


def test_invalid_json(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text('[{"prompt": ', encoding="utf-8")
    with pytest.raises(AppError, match="JSON inválido"):
        load_prompts(broken)


def test_empty_functions(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(AppError, match="ninguna función"):
        load_functions(path)


def test_unsupported_type(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps([
        {"name": "f", "parameters": {"x": {"type": "array"}}}
    ]), encoding="utf-8")
    with pytest.raises(AppError):
        load_functions(path)


def test_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps([{"name": "f"}, {"name": "f"}]),
                    encoding="utf-8")
    with pytest.raises(AppError, match="repetidos"):
        load_functions(path)


def test_write_results_creates_folder(tmp_path: Path) -> None:
    path = tmp_path / "out" / "results.json"
    result = FunctionCallResult(prompt="p", name="f", parameters={"a": 1.0})
    write_results(path, [result])
    assert json.loads(path.read_text(encoding="utf-8")) == [
        {"prompt": "p", "name": "f", "parameters": {"a": 1.0}}
    ]
