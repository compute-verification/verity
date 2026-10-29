"""GPU check of the top-p split pin (`engine.topp_pin`): one nucleus whatever the batch.

  kernel (GPU, torch venv):  verity-vllm topp-pin-check kernel --out DIR
  served (GPU, torch venv):  verity-vllm topp-pin-check served --workload W --out DIR

`kernel` runs vLLM's real `_apply_topp_split` on the drain's `ties` rows (`topp_split_probe.gen_ties`, whose top-p boundary falls
inside a plateau of identical logits, so the kernel's duplicate budget decides which tied lanes stay) at every drain batch size of
the device, in order and reversed, once as vLLM picks S and once pinned. A row's keep set at batch B is compared with the same row
served alone (B = 1, at the registered S): pinned, every row equals it; unpinned, rows move once S does.

`served` builds the row's engine as a Commit does (`vllm_adapter.build_engine`, the pin armed) and serves the workload's seeded
requests twice, all at once and each alone, recording every split launch's S (`commit.committer.topp_splits`): equal tokens and one
S throughout pass. Each mode writes `DIR/<mode>.json` with `ok`.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from verity_vllm.config import option, workload_arg


def kernel(out: Path) -> dict:
    import torch

    from verity_vllm.engine import topp_pin
    from verity_vllm.pipeline.topp_split_probe import NEG_INF_BITS, drain_batches, gen_ties
    from verity.ml.definitions.registry.topp_geometry import SPLIT_MAX_BATCH, registered_splits

    import vllm.v1.sample.ops.topk_topp_triton as TT

    patch = topp_pin.install(TT)
    pinned, original = TT._apply_topp_split, patch.original
    dev = torch.device("cuda")
    num_sm = int(torch.cuda.get_device_properties(0).multi_processor_count)
    S_reg = registered_splits(num_sm)
    V, k, keep_frac, seed = 128256, 24, 0.5, 20260920 + 99991
    rows = [gen_ties(seed + b, V, 16 if b % 2 == 0 else 32, k, keep_frac) for b in range(SPLIT_MAX_BATCH)]
    X = torch.from_numpy(np.stack([r[0] for r in rows]).view(np.int32).copy()).view(torch.float32).to(dev)
    P = torch.tensor([r[1] for r in rows], dtype=torch.float32, device=dev)

    def keep(entry, idx: list[int]) -> np.ndarray:
        L = X[idx].clone()
        entry(L, None, P[idx], float("-inf"), num_sm)
        torch.cuda.synchronize()
        return L.view(torch.int32).cpu().numpy().view(np.uint32) != NEG_INF_BITS

    def vllm_s(B: int) -> int:
        s = 1
        while s < TT._SPLIT_MAX_SPLITS and s * 2 * B <= num_sm:
            s *= 2
        return s

    alone = [keep(pinned, [r])[0] for r in range(SPLIT_MAX_BATCH)]
    batches, ok = [], True
    for B in drain_batches(num_sm):
        for order in ("in order", "reversed"):
            idx = list(range(B)) if order == "in order" else list(range(B))[::-1]
            rec = {"B": B, "order": order}
            for arm, entry, S in (("pinned", pinned, S_reg), ("vllm", original, vllm_s(B))):
                kb = keep(entry, idx)
                moved = [r for j, r in enumerate(idx) if not np.array_equal(kb[j], alone[r])]
                rec[arm] = {"S": S, "rows_moved": len(moved), "kept": [int(x.sum()) for x in kb[:4]]}
            ok &= rec["pinned"]["rows_moved"] == 0
            batches.append(rec)
            print(json.dumps(rec), flush=True)
    doc = {"ok": bool(ok), "num_sm": num_sm, "registered_splits": S_reg, "V": V, "rows": SPLIT_MAX_BATCH,
           "gpu": torch.cuda.get_device_name(0), "vllm": __import__("vllm").__version__,
           "vllm_moved_some_row": any(b["vllm"]["rows_moved"] for b in batches), "batches": batches}
    (out / "kernel.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def served(workload: str, out: Path) -> dict:
    from verity_vllm.engine import engine_profile as prof

    prof.apply_env()
    from verity_vllm.commit.committer.topp_splits import install_split_kernel_proxies
    from verity_vllm.engine import topp_pin, vllm_adapter as va
    from verity.ml.definitions.registry.topp_geometry import registered_splits

    wl = va.load_workload(workload)
    reqs = wl["requests"]
    eng = va.build_engine(max_num_seqs=max(8, len(reqs)), case=va.case_of(wl), engine_args=wl["engine_args_required"],
                          target=va.target_of_workload(wl), execution=wl["execution"])
    import torch
    num_sm = int(torch.cuda.get_device_properties(0).multi_processor_count)
    sink: list = []
    patches = install_split_kernel_proxies(sink)

    def serve(group: list[dict]) -> tuple[dict, set]:
        del sink[:]
        tokens, _timing = va.run_requests(eng, group, wl["sampling"])
        return tokens, {s for (_name, _grid, s) in sink}

    together, s_together = serve(reqs)
    alone, s_alone = {}, set()
    for r in reqs:
        t, s = serve([r])
        alone.update(t)
        s_alone |= s
    for p in patches.values():
        p.uninstall()
    differ = sorted(rid for rid in together if together[rid] != alone.get(rid))
    S = registered_splits(num_sm)
    doc = {"ok": not differ and s_together == s_alone == {S} and topp_pin.pinned(), "pinned": topp_pin.pinned(),
           "row": wl["row_id"], "workload_sha256": wl["sha256"], "sampling": {k: wl["sampling"][k] for k in ("temperature", "top_p")},
           "num_sm": num_sm, "registered_splits": S, "S_together": sorted(s_together), "S_alone": sorted(s_alone),
           "requests": len(reqs), "differ": differ, "tokens_together": together, "tokens_alone": alone}
    (out / "served.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if not k.startswith("tokens")}), flush=True)
    return doc


@dataclass(frozen=True, kw_only=True)
class KernelOptions:
    out: str = option("--out")
    cmd: str = field(default="kernel", init=False)


@dataclass(frozen=True, kw_only=True)
class ServedOptions:
    workload: str = option("--workload", type=workload_arg)
    out: str = option("--out")
    cmd: str = field(default="served", init=False)


COMMANDS = {"kernel": KernelOptions, "served": ServedOptions}


def main(args: KernelOptions | ServedOptions) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = kernel(out) if args.cmd == "kernel" else served(args.workload, out)
    sys.exit(0 if doc["ok"] else 1)
