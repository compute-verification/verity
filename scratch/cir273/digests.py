import hashlib

from verity.circuits import codec
from verity.circuits.defs import bind
from verity.circuits.program import Program
from verity.definitions.registry import topp_stages as T

cases = [(T.GumbelTopPSelect, dict(V=64, N=2, FIRST=False, LAST=False)), (T.GumbelTopPSelect, dict(V=64, N=2, FIRST=False, LAST=True)),
         (T.GumbelTopPSelect, dict(V=64, N=2, FIRST=True, LAST=False)), (T.GumbelTopPSelect, dict(V=1000, N=32, FIRST=False, LAST=False)),
         (T.GumbelTopPSelect, dict(V=1000, N=8, FIRST=True, LAST=True)),
         (T.TopPMaskChunk, dict(V=64, N=2, FIRST=True, LAST=False)), (T.TopPMaskChunk, dict(V=64, N=2, FIRST=False, LAST=False)),
         (T.TopPMaskChunk, dict(V=64, N=2, FIRST=False, LAST=True)), (T.TopPMaskChunk, dict(V=1000, N=32, FIRST=False, LAST=False))]
for d, st in cases:
    p = Program(bind(d, **st))
    print(d.id if hasattr(d, "id") else d, st, hashlib.sha256(codec.canonical_json(codec.encode_program(p))).hexdigest()[:16])
