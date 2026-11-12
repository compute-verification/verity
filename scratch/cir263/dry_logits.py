"""CPU dry run of the indexer-logits capture: operands, flattening model, key table, and one compare job with the twin's words."""
import tempfile
import time
from pathlib import Path

import numpy as np

from tests.properties import glm53_dsa_capture_gpu as DS
from tests.properties import glm53_mtp_indexer_logits_capture_gpu as M

rng = np.random.default_rng(1)
L = M.lengths(M.CASES[0][1], rng)
t = time.time()
o = M.operands(L, 9000)
lens, bt = M.flattened(L, o["bt"])
print("L", L.tolist(), "lens", lens[:12].tolist(), lens[-12:].tolist(), "bt rows equal per request",
      all((bt[6 * r:6 * r + 6] == o["bt"][r]).all() for r in range(len(L))))
k, ks, first = M.key_table(o)
print("keys", k.shape, ks.shape, first.tolist(), round(time.time() - t, 1), "s")
req = np.repeat(np.arange(len(L)), M.NEXT_N)
rr = np.array([0, 5, 6, 47, 47, 47, 47], dtype=np.int64)
cc = np.array([0, 5, 6, 0, 63, 64, 4095], dtype=np.int64)
rr = np.concatenate([rr, rng.choice(np.flatnonzero(lens > 0), 4089)])
cc = np.concatenate([cc, (rng.random(4089) * lens[rr[7:]]).astype(np.int64)])
assert (cc < lens[rr]).all()
c = first[req[rr]] + cc
t = time.time()
want, nonfin = DS.logit_rows(o["q"][rr], o["w"][rr], k[c], ks[c])
print("twin 4096 entries", round(time.time() - t, 2), "s; unresolved", int(nonfin.sum()))
d = Path(tempfile.mkdtemp())
paths = DS._save(d, q=o["q"], w=o["w"], k=k, ks=ks, r=rr, c=c, key=cc, got=want)
t = time.time()
res = DS.compare({"phase": "logits", "case": "edges", "kind": "logits", "paths": paths, "keys": ["r", "c", "got"], "range": [0, 4096]})
print(res, round(time.time() - t, 1), "s")
