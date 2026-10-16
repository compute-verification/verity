"""WikiText-2 test windows for `exact_grid.py`: the test split's text joined by blank lines (HF's perplexity recipe),
tokenized with the model's tokenizer, cut into consecutive windows of the capture's length, each led by BOS as the
capture's window 0 is.  Window 0 must equal the capture's ids exactly, which checks the recipe.  With CAPTURE `-:T`
(a model without a capture) the windows are T tokens, led by the tokenizer's BOS if it has one (Qwen2.5's has none).

    python windows.py MODEL_SNAPSHOT WIKITEXT_TEST.arrow|.parquet CAPTURE_IDS.npy|-:T N OUT.npy
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow as pa
import transformers


def read_text(path: str) -> list[str]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).column("text").to_pylist()
    with pa.memory_map(path) as src:
        try:
            table = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(src).read_all()
    return table.column("text").to_pylist()


def main(argv: list[str]) -> int:
    snapshot, arrow, capture, n, out = argv[0], argv[1], argv[2], int(argv[3]), argv[4]
    tok = transformers.AutoTokenizer.from_pretrained(snapshot)
    ids = np.asarray(tok("\n\n".join(read_text(arrow)), add_special_tokens=False)["input_ids"], dtype=np.int64)
    if capture.startswith("-:"):
        ref, t = None, int(capture[2:])
        lead = [tok.bos_token_id] if tok.bos_token_id is not None else []
    else:
        ref = np.load(capture).astype(np.int64)
        t = len(ref)
        lead = [tok.bos_token_id] if ref[0] == tok.bos_token_id else []
    body = t - len(lead)
    if n * body > len(ids):
        print(f"only {len(ids) // body} windows of {t}", file=sys.stderr)
        return 1
    windows = np.stack([np.concatenate([lead, ids[i * body:(i + 1) * body]]).astype(np.int64) for i in range(n)])
    match = None if ref is None else bool((windows[0] == ref).all())
    print(json.dumps({"tokens": int(len(ids)), "windows_available": int(len(ids) // body), "n": n, "window": t,
                      "bos_lead": bool(lead), "window0_matches_capture": match}))
    if match is False:
        print("first mismatch at", int(np.argmax(windows[0] != ref)), file=sys.stderr)
        return 1
    np.save(out, windows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
