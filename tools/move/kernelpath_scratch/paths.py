"""Scratch, never to land: where the adapter's two computed paths to the PoUW kernel sources point, in the environment
that runs this file.

    python paths.py [--under-prover] [--build] [--outside DIR]

It prints one JSON object: the files `verity`, `verity_vllm` and `verity.protocols.pouw.prover` were imported from, how the
two distributions were installed (their `direct_url.json`), `row_tables_device.CUDA_DIR` and `pouw.device.HERE`, and the
files each holds. It exits 0 when the two are one existing directory of the fifteen sources. `--under-prover` also requires
that directory to be `kernels/pouw_cuda` beside the prover package's `__init__.py`; `--outside DIR` requires that nothing
was imported from under DIR (a checkout); `--build` compiles every host library from the directory into the cache under
`$HOME`, which the caller points at an empty directory, so each is a build and none a cached file."""
import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

SOURCES = 15
SUFFIXES = {".cu", ".cuh", ".cpp"}


def held(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


def installed(name: str) -> str:
    try:
        return metadata.distribution(name).read_text("direct_url.json") or "no direct_url.json: from an index"
    except metadata.PackageNotFoundError:
        return "not installed as a distribution"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--under-prover", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--outside", type=Path)
    a = ap.parse_args()

    import verity
    import verity_vllm
    from verity.protocols.pouw import prover
    from verity_vllm.commit import row_tables_device as D

    out: dict = {"python": sys.executable, "cwd": str(Path.cwd()),
                 "verity": verity.__file__, "verity_vllm": verity_vllm.__file__, "prover": prover.__file__,
                 "installed": {n: installed(n) for n in ("verity", "verity-vllm")},
                 "CUDA_DIR": str(D.CUDA_DIR), "CUDA_DIR holds": held(D.CUDA_DIR)}
    failed: list[str] = []
    try:
        from verity_vllm.protocols.pouw import device as PD
        here = PD.HERE
    except Exception as e:  # the module reads no file at import, so this is a missing dependency, said as itself
        out["HERE"] = f"verity_vllm.protocols.pouw.device did not import: {type(e).__name__}: {e}"
        failed.append("pouw.device did not import")
        here = None
    if here is not None:
        out["HERE"], out["HERE holds"] = str(here), held(here)
        if here != D.CUDA_DIR:
            failed.append("CUDA_DIR and HERE differ")
        if len(held(here)) != SOURCES:
            failed.append(f"HERE holds {len(held(here))} files, not {SOURCES}")
    names = held(D.CUDA_DIR)
    if len(names) != SOURCES or {Path(n).suffix for n in names} - SUFFIXES:
        failed.append(f"CUDA_DIR holds {len(names)} files, not the {SOURCES} sources")
    want = Path(prover.__file__).resolve().parent / "kernels" / "pouw_cuda"
    out["under the prover package"] = D.CUDA_DIR == want and here == want
    if a.under_prover and not out["under the prover package"]:
        failed.append(f"not {want}")
    if a.outside is not None:
        root = a.outside.resolve()
        inside = [f for f in (verity.__file__, verity_vllm.__file__, prover.__file__, str(D.CUDA_DIR))
                  if Path(f).resolve().is_relative_to(root)]
        out["imported from the checkout"] = inside
        if inside:
            failed.append(f"imported from under {root}")
    if a.build and not failed:
        cache = Path.home() / ".cache" / "verity"
        before = sorted(str(p.relative_to(cache)) for p in cache.rglob("*.so")) if cache.exists() else []
        from verity_vllm.commit import history_tables_device as HD
        from verity_vllm.protocols.pouw.h1 import registered_device as RD
        built = {}
        for name, load in (("pouw.device.host", PD.host), ("row_tables_device.host_lib", D.host_lib),
                           ("history_tables_device._host", HD._host), ("registered_device.host_lib", RD.host_lib),
                           ("registered_device.zy_host_lib", RD.zy_host_lib)):
            try:
                load()
                built[name] = "built and loaded"
            except Exception as e:
                built[name] = f"{type(e).__name__}: {e}"
                failed.append(f"{name} did not build")
        out["cache before"] = before
        out["host libraries"] = built
        out["cache after"] = sorted(str(p.relative_to(cache)) for p in cache.rglob("*.so")) if cache.exists() else []
        if before:
            failed.append("the cache was not empty, so a library may be a cached file")
    out["failed"] = failed
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
