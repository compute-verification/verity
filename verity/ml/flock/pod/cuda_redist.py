"""Unpack NVIDIA's redistributable CUDA archives of one version into DEST, without root, each archive sha256-checked against
NVIDIA's manifest: the toolkit the prover's kernels need (their clmad needs CUDA 13.3's ptxas) on a node whose host toolkit is
older, as `CUDA_TK`.

    python3 cuda_redist.py VERSION DEST      # e.g. 13.3.1 /workspace/jobs/x/cuda-13.3.1
"""
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ver, dest = sys.argv[1], Path(sys.argv[2])
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
d = json.load(urllib.request.urlopen(f"{base}redistrib_{ver}.json"))  # noqa: S310  (NVIDIA's https base)
tmp = dest.with_name(dest.name + ".tmp")
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)
for k in ("cuda_nvcc", "cuda_cudart", "cuda_crt", "libnvvm", "cccl", "libnvptxcompiler", "cuda_culibos", "cuda_cuobjdump",
          "cuda_nvdisasm"):
    x = d[k]["linux-x86_64"]
    data = urllib.request.urlopen(base + x["relative_path"]).read()  # noqa: S310
    if hashlib.sha256(data).hexdigest() != x["sha256"]:
        raise SystemExit(f"{k}: sha256 mismatch")
    arc = tmp / Path(x["relative_path"]).name
    arc.write_bytes(data)
    with tarfile.open(arc) as t:
        for m in t.getmembers():
            parts = Path(m.name).parts
            if len(parts) > 1:
                m.name = str(Path(*parts[1:]))
                t.extract(m, tmp)
    arc.unlink()
    print(k, d[k]["version"], x["sha256"], flush=True)
(tmp / "lib64").symlink_to("lib")
tmp.rename(dest)
