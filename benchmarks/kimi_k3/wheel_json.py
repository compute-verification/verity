"""`python wheel_json.py WHEEL ARCHS [ART]`: the built wheel's record (name, sha256, size, archs, `art:` id) as JSON."""
import hashlib
import json
import sys
from pathlib import Path

whl, archs = Path(sys.argv[1]), sys.argv[2]
art = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
h = hashlib.sha256()
with whl.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(json.dumps({"wheel": whl.name, "sha256": h.hexdigest(), "bytes": whl.stat().st_size, "torch_cuda_arch_list": archs,
                  "art": art}, indent=1))
