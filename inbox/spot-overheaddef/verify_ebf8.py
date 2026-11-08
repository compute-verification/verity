"""Check every file of ebf8's capture against run 1ccd's custody manifest: path, size and sha256 (custody verified each on the remote)."""
import collections
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

CAP = Path("/workspace/spot-capture/r20261108-092918-ebf8")
RUN = Path("/workspace/research/runs/r20261108-100548-1ccd")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


c = json.loads((RUN / ".custody").read_text())
assert c["preserved"] is True and c["error"] is None, {k: v for k, v in c.items() if k != "files"}
man = {k[len("out/capture/"):]: v for k, v in c["files"].items() if k.startswith("out/capture/")}
local = sorted(str(p.relative_to(CAP)) for p in CAP.rglob("*") if p.is_file())
links = collections.Counter(os.stat(CAP / r).st_nlink for r in local)
same_inode = sum(os.stat(CAP / r).st_ino == os.stat(RUN / "out/capture" / r).st_ino
                 for r in local if (RUN / "out/capture" / r).exists())
missing = [r for r in local if r not in man]
extra = [r for r in man if not (CAP / r).is_file()]
size_bad = [r for r in local if r in man and os.path.getsize(CAP / r) != man[r][0]]
with Pool(48) as pool:
    digests = pool.map(sha, [CAP / r for r in local], chunksize=8)
sha_bad = [r for r, d in zip(local, digests) if r in man and d != man[r][1]]
total = sum(os.path.getsize(CAP / r) for r in local)
print(f"run record {c['run_record']} preserved {c['preserved']} verified {c['verified_utc']}")
print(f"ebf8 files {len(local)} bytes {total}; manifest out/capture files {len(man)} bytes {sum(v[0] for v in man.values())}")
print(f"missing from manifest {len(missing)}, in manifest but not local {len(extra)}, "
      f"size mismatches {len(size_bad)}, sha256 mismatches {len(sha_bad)}")
print(f"link counts {dict(links)}; same inode as 1ccd's out/capture copy {same_inode}")
ok = not (missing or extra or size_bad or sha_bad) and len(local) == len(man)
print("VERIFIED" if ok else f"NOT VERIFIED {missing[:3]} {extra[:3]} {size_bad[:3]} {sha_bad[:3]}")
sys.exit(0 if ok else 1)
