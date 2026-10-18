"""Run A8's value slice: the files of a vLLM capture (`verity-capture/snapshots/v1`) that a few layers' events read at a few
steps, copied into one tar with the rank's log and a snapshot index of exactly those files, for `a8.py` on a CPU machine. It
selects and copies files and computes nothing, so it runs where the capture is (a GPU node's shared disk).

    python benchmarks/one_stage/a8_slice.py --capture DIR --rank 0 --steps 0,1 --layers 0,3 --out a8-values.tar

Kept: ``log.rank<R>.jsonl.gz``; ``rank<R>/snapshots.json`` holding the entries of the chosen steps whose event is a chosen
layer's (``model.layers.<L>.``), the first layer with a DSA indexer at the last chosen step, or no layer's (embedding, final
norm, lm_head, sampler); their ``values/`` files; ``rank<R>/cos_sin_cache.npy`` (the rotary table the capture keeps beside
the snapshots, its tensor past the snapshot cap); ``slice.json``, what was chosen. With ``--checkpoint``, also
``ckpt/a8-subset.safetensors``, the checkpoint of record's tensors of the chosen layers but their routed experts, byte for byte,
with its ``config.json``: the capture snapshots activations only, and the weights are composed from these on the CPU machine.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from pathlib import Path

LAYER = re.compile(r"model\.layers\.(\d+)\.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--steps", default="0,1")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None, help="the checkpoint of record's directory: its chosen layers' tensors too")
    ap.add_argument("--ckpt-layers", default=None, help="the layers whose checkpoint tensors are kept (default --layers)")
    a = ap.parse_args()
    cap, r = Path(a.capture), a.rank
    steps = {int(s) for s in a.steps.split(",")}
    layers = {int(x) for x in a.layers.split(",")}
    log = cap / f"log.rank{r}.jsonl.gz"
    mod, step_of = {}, {}
    indexer = set()
    with gzip.open(log, "rt") as f:
        for line in f:
            e = json.loads(line)
            if "id" in e and "step" in e and "module" in e:
                mod[e["id"]], step_of[e["id"]] = e.get("module") or "", e["step"]
                m = LAYER.search(mod[e["id"]])
                if m and ".indexer" in mod[e["id"]] and e["step"] == max(steps):
                    indexer.add(int(m[1]))
    if indexer and not layers & indexer:
        layers.add(min(indexer))

    def chosen(ev: int) -> bool:
        m = LAYER.search(mod.get(ev, ""))
        return m is None or int(m[1]) in layers

    idx = json.loads((cap / f"rank{r}" / "snapshots.json").read_text())
    keep = [e for e in idx["entries"] if e.get("step") in steps and not e.get("skipped") and chosen(e["event"])]
    files = sorted({e["file"] for e in keep})
    sub = {k: v for k, v in idx.items() if k != "entries"}
    sub.update(entries=keep, slice={"steps": sorted(steps), "layers": sorted(layers)})
    out = Path(a.out)
    nbytes = 0
    with tarfile.open(out, "w") as t:
        t.add(log, arcname=log.name)
        for name in ("cos_sin_cache.npy", "cos_sin_cache.json", "versions.json"):
            if (cap / f"rank{r}" / name).exists():
                t.add(cap / f"rank{r}" / name, arcname=f"rank{r}/{name}")
        for fn in files:
            p = cap / f"rank{r}" / fn
            nbytes += p.stat().st_size
            t.add(p, arcname=f"rank{r}/{fn}")
        for name, doc in ((f"rank{r}/snapshots.json", sub),
                          ("slice.json", {"capture": str(cap), "rank": r, "steps": sorted(steps), "layers": sorted(layers),
                                          "indexer_layers": sorted(indexer), "entries": len(keep), "files": len(files),
                                          "file_bytes": nbytes})):
            p = out.parent / f".{Path(name).name}"
            p.write_text(json.dumps(doc))
            t.add(p, arcname=name)
            p.unlink()
        if a.checkpoint:
            ck = ckpt_subset(Path(a.checkpoint), {int(x) for x in a.ckpt_layers.split(",")} if a.ckpt_layers else layers,
                             out.parent / ".a8-subset.safetensors")
            t.add(out.parent / ".a8-subset.safetensors", arcname="ckpt/a8-subset.safetensors")
            (out.parent / ".a8-subset.safetensors").unlink()
            for name in ("config.json", "model.safetensors.index.json"):
                if (Path(a.checkpoint) / name).exists():
                    t.add(Path(a.checkpoint) / name, arcname=f"ckpt/{name}")
            nbytes += ck["bytes"]
    print(json.dumps({"steps": sorted(steps), "layers": sorted(layers), "indexer_layers": sorted(indexer)[:8], "entries": len(keep),
                      "files": len(files), "file_bytes": nbytes, "tar_bytes": out.stat().st_size,
                      "checkpoint": ck if a.checkpoint else None}))
    return 0


def ckpt_subset(ckpt: Path, layers: set[int], dst: Path, extra: frozenset[str] = frozenset()) -> dict:
    """One safetensors file holding the checkpoint's tensors of `layers` but their routed experts (``mlp.experts.<e>.``),
    and the tensors named in `extra` (``lm_head.weight``), byte for byte: the sources `weights_of_record.compose` reads for
    those layers' attention, dense MLP, router and shared experts."""
    want = []
    for fn in sorted(p.name for p in ckpt.glob("*.safetensors")):
        with open(ckpt / fn, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n))
        for name, meta in hdr.items():
            m = LAYER.match(name)
            if name != "__metadata__" and (name in extra or (m and int(m[1]) in layers and ".mlp.experts." not in name)):
                want.append((name, fn, 8 + n, meta))
    header, off = {}, 0
    for name, _fn, _base, meta in want:
        lo, hi = meta["data_offsets"]
        header[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "data_offsets": [off, off + hi - lo]}
        off += hi - lo
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    with open(dst, "wb") as o:
        o.write(len(hb).to_bytes(8, "little"))
        o.write(hb)
        for name, fn, base, meta in want:
            lo, hi = meta["data_offsets"]
            with open(ckpt / fn, "rb") as f:
                f.seek(base + lo)
                o.write(f.read(hi - lo))
    return {"dir": str(ckpt), "layers": sorted(layers), "tensors": len(want), "bytes": off}


if __name__ == "__main__":
    raise SystemExit(main())
