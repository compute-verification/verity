"""The client-side shim of compact egress framing: OpenAI completions SSE rendered from compact records with the
registered tokenizer, so that a stock OpenAI client reads the text vLLM would have streamed.

Under `active.Proxy(compact=True)` the server sends each sampled token as a compact record (`active.compact_record`:
the request's ingress ordinal and its token field, the finish in the field's flags), and the link carries each as a
B-byte cell. The client knows each request's prompt ids (it sent them) and the registered tokenizer, and renders the
records back (`Renderer`): one completion chunk a record, its text the token's, `finish_reason` "length" on a record
marked `COMPACT_LAST` alone and "stop" with no text on one marked `COMPACT_STOP` too (its id the stop token's), then
`data: [DONE]`. It refuses, with ValueError, a record of another request, one after the finish, an id outside the
vocab, and an end before the finish.

**Detokenization** is vLLM V1's (`FastIncrementalDetokenizer`, `Detokenizer`), so each chunk's text is the bytes vLLM
streams for that token: a stream (`tokenizers.decoders.DecodeStream`) primed with the prompt's shortest suffix of 4 …
23 ids whose text holds no U+FFFD (the whole prompt if none, or if it has at most 4 ids), then stepped once per output
token. A step decodes the ids it holds; when the text is longer than the prefix's, in UTF-8 bytes, and doesn't end in
U+FFFD, it emits what follows the prefix and keeps only the ids since the prefix began, else it emits nothing, so a
token that ends inside a UTF-8 character is a chunk of no text and its bytes come with the next. A step whose text
doesn't start with the prefix restarts the stream at that token, as vLLM does. A stop token isn't decoded. Special
tokens (tokenizer.json's added tokens marked special) decode to nothing (`skip_special_tokens`, vLLM's default).

**The tokenizer** is byte-level BPE (Llama 3's `ByteLevel` decoder): a token's bytes are its characters through GPT-2's
byte-to-unicode table, and a text is the tokens' bytes decoded as UTF-8 with U+FFFD for what isn't, so the renderer
reads only tokenizer.json's vocab and added tokens (`Vocab.from_tokenizer_json`), with no tokenizer library.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from verity.core.protocols.certifier.active import compact_record_data

REPLACEMENT = "\ufffd"
DONE = b"data: [DONE]\n\n"


def _byte_decoder() -> dict[str, int]:
    """GPT-2's byte-to-unicode table, inverted: each of the 256 characters a byte-level token is spelled in, its byte."""
    keep = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars, n = list(keep), 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(keep, chars, strict=True)}


BYTE_DECODER = _byte_decoder()


class Vocab:
    """A byte-level BPE tokenizer's decoding side: each id's token string and the special ids, which decode to
    nothing."""

    def __init__(self, tokens: Mapping[int, str], special: Iterable[int] = ()) -> None:
        self.special = frozenset(special)
        self.bytes = {i: self._spell(t) for i, t in tokens.items()}

    @staticmethod
    def _spell(token: str) -> bytes:
        if all(c in BYTE_DECODER for c in token):
            return bytes(BYTE_DECODER[c] for c in token)
        return token.encode()

    @classmethod
    def from_tokenizer_json(cls, path: Path | str) -> Vocab:
        doc = json.loads(Path(path).read_text())
        if (doc.get("decoder") or {}).get("type") != "ByteLevel":
            raise ValueError(f"{path}: the renderer decodes byte-level BPE only, not {doc.get('decoder')}")
        tokens = {i: t for t, i in doc["model"]["vocab"].items()}
        tokens |= {a["id"]: a["content"] for a in doc.get("added_tokens", ())}
        return cls(tokens, (a["id"] for a in doc.get("added_tokens", ()) if a.get("special")))

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        return b"".join(self.bytes[i] for i in ids if not (skip_special and i in self.special)).decode(
            "utf-8", errors="replace")


class Detokenizer:
    """vLLM V1's incremental detokenization of one request (the module docstring's "Detokenization")."""

    def __init__(self, vocab: Vocab, prompt: Sequence[int]) -> None:
        self.vocab = vocab
        suffix = list(prompt)
        if len(prompt) > 4:
            for i in range(4, min(len(prompt) + 1, 24)):
                if REPLACEMENT not in vocab.decode(prompt[-i:]):
                    suffix = list(prompt[-i:])
                    break
        self._restart()
        for t in suffix:
            self.step(t)

    def _restart(self) -> None:
        self.ids, self.prefix, self.prefix_index = [], "", 0

    def _step(self, token_id: int) -> str | None:
        self.ids.append(token_id)
        text = self.vocab.decode(self.ids)
        if len(text.encode()) > len(self.prefix.encode()) and not text.endswith(REPLACEMENT):
            if not text.startswith(self.prefix):
                raise ValueError("invalid prefix")
            new = text[len(self.prefix) :]
            kept = len(self.ids) - self.prefix_index
            self.ids = self.ids[self.prefix_index :]
            self.prefix, self.prefix_index = self.vocab.decode(self.ids), kept
            return new
        return None

    def step(self, token_id: int) -> str:
        """The text vLLM streams for the next output token."""
        try:
            return self._step(token_id) or ""
        except ValueError:
            self._restart()
            return self._step(token_id) or ""


class Renderer:
    """One request's completion stream rendered from its compact records (the module docstring): `chunk(cell)` gives
    the SSE event of one record, a stock OpenAI completions chunk, and `done()` the stream's end. `request` is the
    request's ingress ordinal, which every record must name. `id` and `created` are the client's own (a deployment's
    shim draws the id as vLLM does)."""

    def __init__(self, vocab: Vocab, prompt: Sequence[int], model: str, *, request: int, id: str = "cmpl-compact",
                 created: int = 0) -> None:
        self.detokenizer, self.model, self.request = Detokenizer(vocab, prompt), model, request
        self.id, self.created, self.finished = id, created, None

    def text(self, cell: bytes) -> tuple[str, str | None]:
        """(text, finish) of one record or cell of this request. ValueError, with nothing rendered, on a cell that
        isn't a record of this request, a record after the finish, or an id outside the vocab."""
        got = compact_record_data(cell)
        if got is None or got[0] != self.request:
            raise ValueError("a cell that isn't a compact record of this request")
        if self.finished is not None:
            raise ValueError(f"a record after the response's finish ({self.finished})")
        _, token_id, finish = got
        if token_id not in self.detokenizer.vocab.bytes:
            raise ValueError(f"a token id outside the vocab: {token_id}")
        self.finished = finish
        return ("" if finish == "stop" else self.detokenizer.step(token_id)), finish

    def chunk(self, cell: bytes) -> bytes:
        text, finish = self.text(cell)
        doc = {"id": self.id, "object": "text_completion", "created": self.created, "model": self.model,
               "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish,
                            "stop_reason": None}]}
        return b"data: " + json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode() + b"\n\n"

    def done(self) -> bytes:
        """`data: [DONE]`, once a record marked `COMPACT_LAST` has been rendered; ValueError before."""
        if self.finished is None:
            raise ValueError("the stream ends before its finish: no record marked COMPACT_LAST")
        return DONE
