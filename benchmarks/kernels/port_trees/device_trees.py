"""The card half of `port_trees.py`: one call's A, y and B rows through the device's own tree launches
(`served_y.tree_launches`, the ones served y and B's weight trees run), ``dom_b`` once Pearl+'s domain id and once the
registered value's, each root held to the processor's.

  device_trees.py PASS_DIR --kernel DIR [--cubin F] --out FILE [--call C] [--round S] [--program P] [--window R]

Refuses a cubin other than the one the pass ran (its manifest's ``cubin_sha256``). A's rows go through the same launches as
y's and B's, not the prefill epilogue's ``dom_a`` tree or decode's `pd_commit`. Writes roots and booleans only, and exits
non-zero unless all hold; deleted with `port_trees.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import port_trees as PT
from verity.protocols.resource_accounting.compute.pouw.prover import schemes as pouw_schemes
from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c import served_y as SY


def device_root(kernel, variant: str, const_keys: dict, dom, rows: list[bytes], salts: list[bytes]) -> bytes:
    """The root the kernel's tree launches give `rows` (FP32 words) under `salts`, ``dom_b`` = `dom`'s id."""
    import torch
    from verity_torch.pouw.pearl_c.host import _put, _stream
    R, dev, m, words = kernel.run, kernel.dev, len(rows), len(rows[0]) // 4
    dom_b = torch.zeros(64, dtype=torch.uint8, device="cuda")
    _put(dom_b, dom.domain_id)
    p = R.Pipeline(dev, R.BM, R.BM, words, variant, {"dom_b": dom_b.data_ptr(), **const_keys}, 0, 0,
                   bufs={"tree_b": 0, "segdig_b": 0})
    x = torch.frombuffer(bytearray(b"".join(rows)), dtype=torch.uint8).cuda()
    s = torch.frombuffer(bytearray(b"".join(salts)), dtype=torch.uint8).cuda()
    tree = torch.zeros(R.NB * (2 * R.pow2(m) - 1), dtype=torch.uint8, device="cuda")
    chunk = SY.chunk_rows(m, words)
    dig = torch.empty(64 * m, dtype=torch.uint8, device="cuda")
    seg = torch.empty(64 * (words // 256) * chunk, dtype=torch.uint8, device="cuda")
    before, dev.stream = dev.stream, _stream(x.device)
    try:
        SY.tree_launches(R, p, tree.data_ptr(), dig.data_ptr(), seg.data_ptr(), m, chunk, 0,
                         lambda r, count: x.data_ptr() + 4 * words * r, s.data_ptr(), s.numel())
        torch.cuda.synchronize()
    finally:
        dev.stream = before
    return bytes(tree[-R.NB:].cpu().numpy())


def check(base: Path, kernel_dir: str, cubin: str | None, **kw) -> dict:
    import torch
    from verity.protocols.resource_accounting.compute.pouw.prover.serving.pearl_c.kernel import Kernel
    from verity_torch.pouw.pearl_c.host import _constant_keys, _constants
    man = json.loads((base / "manifest.json").read_text())
    kernel = Kernel.load(kernel_dir, cubin)
    if kernel.cubin_sha256 != man["cubin_sha256"]:
        raise ValueError(f"cubin {kernel.cubin_sha256[:16]} is not the pass's {man['cubin_sha256'][:16]}")
    const = _constants(torch.device("cuda"))
    variant, keys = pouw_schemes.get(man["scheme"]).device.name, _constant_keys(const)
    facts, args = PT.inputs(base, **kw)
    trees = {}
    for t, (pearl_dom, want, program, value, rows, salts, window) in args.items():
        pearl = device_root(kernel, variant, keys, pearl_dom, rows, salts)
        registered, dom = PT.port_root(program, value, rows, salts, window)
        port = device_root(kernel, variant, keys, dom, rows, salts)
        trees[t] = {"value": value, "leaves": len(rows), "device_pearl_root": pearl.hex(), "manifest_root": want,
                    "device_port_root": port.hex(), "registered_root": registered.hex(),
                    "pearl_is_manifest": pearl.hex() == want, "device_is_port": port == registered}
    return {**facts, "trees": trees, "ok": all(t["pearl_is_manifest"] and t["device_is_port"] for t in trees.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pass_dir", type=Path)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--cubin")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--call", type=int)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--program")
    ap.add_argument("--window")
    a = ap.parse_args(argv)
    got = check(a.pass_dir, a.kernel, a.cubin, call=a.call, round_=a.round, program=a.program, window=a.window)
    a.out.write_text(json.dumps(got, indent=1) + "\n")
    print(json.dumps({t: [v["pearl_is_manifest"], v["device_is_port"]] for t, v in got["trees"].items()}))
    return 0 if got["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
