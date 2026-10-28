"""The eval sets' windows: a set's documents joined by blank lines (HF's perplexity recipe), tokenized with the model's
tokenizer without special tokens, cut into consecutive windows of T tokens, each led by the tokenizer's BOS if it has
one (Qwen2.5's has none). Window i holds tokens [i * body, (i + 1) * body) of the stream, body = T - len(lead).

`windows.lock.json` pins each (tokenizer, set, context) list's sha256; a run checks it."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

import methodology

HERE = Path(__file__).resolve().parent
DATASETS = json.loads((HERE / "datasets.json").read_text())
LOCK_PATH = HERE / "windows.lock.json"


def read_texts(name: str, path: Path) -> list[str]:
    """The set's documents, in order: C4's first C4_DOCS validation documents, or WikiText-2's test lines."""
    fmt = DATASETS[name]["format"]
    if fmt == "jsonl.gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line)["text"] for _, line in zip(range(methodology.C4_DOCS), f)]
    if fmt == "json":
        return json.loads(Path(path).read_text(encoding="utf-8"))
    raise ValueError(f"{name}: unknown format {fmt}")


def stream(texts: list[str], encode: Callable[[str], list[int]]) -> np.ndarray:
    return np.asarray(encode("\n\n".join(texts)), dtype=np.int64)


def cut(ids: np.ndarray, ctx: int, bos: int | None, which) -> np.ndarray:
    lead = [] if bos is None else [bos]
    body = ctx - len(lead)
    which = list(which)
    if (max(which) + 1) * body > len(ids):
        raise ValueError(f"only {len(ids) // body} windows of {ctx} tokens")
    return np.stack([np.concatenate([lead, ids[i * body:(i + 1) * body]]).astype(np.int64) for i in which])


def digest(windows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(windows, dtype=np.int64).tobytes()).hexdigest()


def lock() -> dict:
    return json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else {}


def key(name: str, ctx: int, calibration: bool = False) -> str:
    return f"{name}/{ctx}" + ("/calibration" if calibration else "")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dataset_path(name: str, data_dir: Path) -> Path:
    """The set's file under data_dir, refused unless its sha256 is the pinned one."""
    path = Path(data_dir) / DATASETS[name]["file"]
    got = file_sha256(path)
    if got != DATASETS[name]["sha256"]:
        raise ValueError(f"{path}: sha256 {got}, pinned {DATASETS[name]['sha256']} (datasets.json)")
    return path


def encoder(tokenizer) -> tuple[Callable[[str], list[int]], int | None]:
    """An HF tokenizer's encoding without special tokens, and its BOS (or None)."""
    return (lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]), tokenizer.bos_token_id


def build(name: str, ctx: int, encode: Callable[[str], list[int]], bos: int | None, data_dir: Path,
          calibration: bool = False, texts: dict | None = None) -> tuple[np.ndarray, dict]:
    """The windows of one (set, context) and their record: digest, windows, tokens in the stream. `texts` caches the
    set's token streams across contexts."""
    which = methodology.CALIBRATION["windows"] if calibration else methodology.windows(ctx)
    if calibration and (name, ctx) != (methodology.CALIBRATION["set"], methodology.CALIBRATION["ctx"]):
        raise ValueError("calibration is C4 at 2048")
    cache = {} if texts is None else texts
    if name not in cache:
        cache[name] = stream(read_texts(name, dataset_path(name, data_dir)), encode)
    ids = cache[name]
    w = cut(ids, ctx, bos, which)
    return w, {"sha256": digest(w), "windows": [which[0], which[-1]], "stream_tokens": int(len(ids)),
               "bos_lead": bos is not None}


def check(family: str, name: str, ctx: int, windows: np.ndarray, calibration: bool = False) -> None:
    """Refuse windows whose shape isn't the context's, or whose digest (of their int64 bytes) isn't the lock's for
    (tokenizer family, set, context)."""
    n = len(methodology.CALIBRATION["windows"]) if calibration else methodology.CONTEXTS[ctx]
    if windows.shape != (n, ctx):
        raise ValueError(f"{key(name, ctx, calibration)}: windows {windows.shape}, the methodology's {(n, ctx)}")
    want = lock().get(family, {}).get(key(name, ctx, calibration))
    if want is None:
        raise ValueError(f"windows.lock.json has no {family} {key(name, ctx, calibration)}: run `run.py windows --write`")
    if digest(windows) != want["sha256"]:
        raise ValueError(f"{family} {key(name, ctx, calibration)}: windows {digest(windows)}, locked {want['sha256']}")
