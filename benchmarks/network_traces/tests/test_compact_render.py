"""The client-side shim of compact framing (`compact_render`) renders, from the compact records the server-side shim
makes of a real vLLM stream (`active_replay.compact_records`), the text vLLM streamed, byte for byte: each chunk's,
whatever its tokens, so the held-back bytes of a character split over tokens too, and its finish."""

import json
from pathlib import Path

import pytest

import active_replay as ar
import compact_render as cr
from verity.core.protocols.certifier import active

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "vllm_llama32_1b_streams.json").read_text())
VOCAB = cr.Vocab({int(k): v for k, v in FIXTURE["vocab"].items()}, FIXTURE["special"])
EOS = frozenset(FIXTURE["eos_token_ids"])


def _sse(text: str, ids: list[int], finish: str | None) -> bytes:
    """A vLLM chunk as its server-sent event, of the fields the server-side shim reads."""
    choice = {"index": 0, "text": text, "finish_reason": finish, "token_ids": ids}
    return b"data: " + json.dumps({"choices": [choice]}, ensure_ascii=False).encode() + b"\n\n"


@pytest.mark.parametrize("cell", [8, 16])
@pytest.mark.parametrize("k", range(len(FIXTURE["streams"])))
def test_the_renderer_gives_back_vllms_text_byte_for_byte(k, cell):
    """Stream k of the fixture (15 prompts on vLLM's Llama-3.2-1B endpoint: ASCII, accents, Chinese, emoji whose
    bytes span tokens, and three that stop at EOS at once): each vLLM chunk's text is, in UTF-8 bytes, the renderer's
    texts for the chunk's records, as cells of B bytes, and the chunk's finish is the last record's, so the whole text
    and finish are vLLM's. A record after the finish, or of another request, is refused."""
    stream = FIXTURE["streams"][k]
    renderer = cr.Renderer(VOCAB, stream["prompt"], FIXTURE["model"], request=3)
    rendered, finishes = b"", []
    for text, ids, finish in stream["chunks"]:
        records = ar.compact_records(_sse(text, ids, finish), 3, EOS)
        assert len(records) == len(ids)
        got = [renderer.text(active.compact_cell(r, cell)) for r in records]
        assert "".join(t for t, _ in got).encode() == text.encode()
        assert got[-1][1] == finish and all(f is None for _, f in got[:-1])
        rendered += "".join(t for t, _ in got).encode()
        finishes.append(got[-1][1])
    assert rendered == "".join(t for t, _, _ in stream["chunks"]).encode()
    assert finishes[-1] in ("length", "stop")
    with pytest.raises(ValueError):
        renderer.text(active.compact_record(3, 11))
    with pytest.raises(ValueError):
        cr.Renderer(VOCAB, stream["prompt"], FIXTURE["model"], request=3).text(active.compact_record(4, 11))


def test_the_fixture_covers_held_back_bytes_stops_and_non_ascii():
    """What the byte-equal test leans on is in the fixture: chunks of no text that aren't a stop (a token ending inside
    a UTF-8 character), stops at EOS, and text of 2-, 3- and 4-byte characters."""
    chunks = [c for s in FIXTURE["streams"] for c in s["chunks"]]
    assert sum(t == "" and f is None for t, _, f in chunks) >= 10
    assert sum(f == "stop" for _, _, f in chunks) >= 3 and sum(f == "length" for _, _, f in chunks) >= 10
    widths = {len(c.encode()) for t, _, _ in chunks for c in t}
    assert {1, 2, 3, 4} <= widths


def test_a_rendered_chunk_is_vllms_completion_chunk():
    """The renderer's chunk is a stock completions chunk in vLLM's own spelling: given vLLM's id and created, it is
    byte for byte the line vLLM streamed, less the fields a stock request doesn't ask for (`prompt_token_ids` and
    `token_ids`, which `return_token_ids` adds, and the committing server's `usage`). `[DONE]` follows the finish
    only."""
    sample = FIXTURE["sse_sample"]
    doc = json.loads(sample["line"].removeprefix("data: "))
    stream = FIXTURE["streams"][sample["stream"]]
    _, ids, finish = stream["chunks"][0]
    renderer = cr.Renderer(VOCAB, stream["prompt"], doc["model"], request=1, id=doc["id"], created=doc["created"])
    expected = dict(doc)
    expected.pop("usage", None)
    expected["choices"] = [{k: v for k, v in c.items() if k not in ("prompt_token_ids", "token_ids")}
                           for c in doc["choices"]]
    record = ar.compact_records(_sse("", ids, finish), 1, EOS)[0]
    assert renderer.chunk(record) == b"data: " + json.dumps(expected, ensure_ascii=False,
                                                            separators=(",", ":")).encode() + b"\n\n"
    assert finish is None
    with pytest.raises(ValueError):
        renderer.done()
    renderer.text(active.compact_record(1, FIXTURE["eos_token_ids"][0], "stop"))
    assert renderer.done() == b"data: [DONE]\n\n"


def test_a_tokenizer_json_is_read_as_its_vocab_and_special_ids(tmp_path):
    """`Vocab.from_tokenizer_json` reads a byte-level tokenizer.json's vocab and added tokens (special ones decode to
    nothing) and refuses another decoder."""
    doc = {"decoder": FIXTURE["decoder"], "model": {"vocab": {t: int(i) for i, t in FIXTURE["vocab"].items()
                                                              if int(i) not in FIXTURE["special"]}},
           "added_tokens": [{"id": i, "content": FIXTURE["vocab"][str(i)], "special": True} for i in FIXTURE["special"]]}
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(doc))
    vocab = cr.Vocab.from_tokenizer_json(path)
    ids = FIXTURE["streams"][4]["prompt"]
    assert vocab.decode(ids) == VOCAB.decode(ids) and not vocab.decode(FIXTURE["special"])
    path.write_text(json.dumps(doc | {"decoder": {"type": "Metaspace"}}))
    with pytest.raises(ValueError):
        cr.Vocab.from_tokenizer_json(path)
