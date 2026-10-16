"""Held-out windows from C4's validation split (allenai/c4 `en/c4-validation.00000-of-00008.json.gz`): the first DOCS
documents' text joined by blank lines, tokenized with the model's tokenizer, cut into consecutive windows of T tokens, each
led by BOS (`windows.py`'s recipe on WikiText-2).  Windows START..START+N-1 go to OUT; evaluation and calibration take
disjoint ranges.

    python c4_windows.py TOKENIZER_DIR C4_VALIDATION.json.gz DOCS T START N OUT.npy
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys

import numpy as np
import transformers


def main(argv: list[str]) -> int:
    tok_dir, path, docs, t, start, n, out = argv[0], argv[1], int(argv[2]), int(argv[3]), int(argv[4]), int(argv[5]), argv[6]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for _, line in zip(range(docs), f)]
    tok = transformers.AutoTokenizer.from_pretrained(tok_dir)
    ids = np.asarray(tok("\n\n".join(texts), add_special_tokens=False)["input_ids"], dtype=np.int64)
    body = t - 1
    if (start + n) * body > len(ids):
        print(f"only {len(ids) // body} windows of {t}", file=sys.stderr)
        return 1
    windows = np.stack([np.concatenate([[tok.bos_token_id], ids[i * body:(i + 1) * body]]).astype(np.int64)
                        for i in range(start, start + n)])
    np.save(out, windows)
    print(json.dumps({"docs": docs, "tokens": int(len(ids)), "windows_available": int(len(ids) // body), "start": start,
                      "n": n, "window": t, "sha256": hashlib.sha256(windows.tobytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
