"""Correspondence of one recorded speculative step against the typed programs (lane SD).

A speculative step of request `r` is two model scans over *branch* sequences whose later positions are not
(yet) committed.  TA1's correspondence runner (`verity_vllm.correspondence.core.run`) checks one request of one
program against the request's committed replay units; here each recorded step is presented to it as a
*virtual request* of its own — the branch sequence is that request's prompt and the step's replay unit (plus the
earlier replay units for the positions before the block) are its leaves:

* side `target`: program `VerifyBranch{C=B1, LP=t0+1+K, K}`; prompt = tape[0..t0] + the K candidates; checked
  positions = the K+1 verification slots t0..t0+K (rejected slots included: their leaves are the `x=<step>`
  qualified ones of the same replay unit); the `sampler.accept` leaf is the `accept` call's `appended` output.
* side `draft`: program `Serve{C=DRAFT, LP=t_bonus+1, STEPS=K-1}` (B1's own serving program with the draft's
  static configuration); prompt = tape[0..t_bonus]; checked positions = the draft's committed catch-up rows
  t0..t0+a, its bonus row t_bonus and its K-1 generated rows; its leaves are the `draft.`-prefixed ones.

Canonical ids are rewritten for the runner (`r=<virtual request>`, the `x=` field dropped, the `draft.` prefix
stripped); the authentication locations recorded are the *real* replay units' (tree roots, leaves, offsets).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verity.ir import Program, bind  # noqa: E402
from verity_vllm.correspondence.core import CaptureIdentity, CircuitMap, CommitStore, NotInProgram, _returns_component, run  # noqa: E402
from verity_vllm.registry import b1, spec  # noqa: E402

_X_RE = re.compile(r"/x=\d+$")
# the draft runs K passes per step (first pass, K-1 generated rows); the base capture disambiguates a module that
# fires again within one step by a counter on the op name (`rmsnorm1`, `gemm2`, ...): pass p's rows carry suffix p
_PASS_RE = re.compile(r"(op=[^/]*?(?:rmsnorm|gemm|add|rope|attn|act_mul|embedding|select|cast))\d+(?=/)")
DRAFT_OP = "op=draft."


class BranchStore(CommitStore):
    """One speculative step of one request, as a virtual request whose prompt is the branch sequence."""

    def __init__(self, real: CommitStore, request: str, step: int, side: str):
        assert side in ("target", "draft"), side
        self.real, self.request, self.step, self.side = real, request, step, side
        self.root = real.root
        index, _ = real.ru(request, step)
        sp = index.get("speculative")
        if not sp or not sp.get("k"):
            raise ValueError(f"ru/{request}/{step:06d} is not a speculative decode step")
        self.sp = sp
        bnd = real.boundary(request)
        tape = list(bnd["prompt_token_ids"]) + list(bnd["output_token_ids"])
        K, a, slots = int(sp["k"]), int(sp["accepted"]), [int(t) for t in sp["slots"]]
        t0 = slots[0]
        self.K, self.accepted, self.slots = K, a, slots
        if side == "target":
            self.prompt = tape[: t0 + 1] + [int(c) for c in sp["candidates"]]
            assert len(self.prompt) == t0 + 1 + K, (len(tape), t0, K)
            self.LP, self.STEPS = t0 + 1 + K, 0
            self.positions = list(slots)
            self.branch_from = t0  # positions >= this live in this step's replay unit (live or x-qualified)
        else:
            tb = sp["draft"]["bonus_position"]
            props = [int(t) for t in sp["draft"]["proposal_positions"]]
            if tb is None or not props:
                raise ValueError(f"ru/{request}/{step:06d}: no draft proposal in this step")
            assert props[0] == tb, (props, tb)
            # tape[0..t0] + the step's committed tokens (positions t0+1..t_bonus): equal to the tape where the tape
            # reaches (a step whose block was cut by termination committed tokens the tape does not hold)
            committed = [int(x) for x in sp["committed_tokens"]]
            self.prompt = tape[: t0 + 1] + committed[: a + 1]
            assert len(self.prompt) == tb + 1, (len(self.prompt), tb)
            overlap = tape[t0 + 1: tb + 1]
            assert overlap == self.prompt[t0 + 1: t0 + 1 + len(overlap)], (overlap, committed)
            self.LP, self.STEPS = tb + 1, len(props) - 1
            self.positions = [t for t in slots if t <= t0 + a] + props
            self.branch_from = t0
        self.vreq = f"{request}@{side}{step}"
        real_bnd_root = real.commitments["trees"][f"boundary/{request}"]["root"]
        self.commitments = {"trees": {**real.commitments["trees"],
                                      f"boundary/{self.vreq}": {"kind": "boundary", "request_id": self.vreq, "root": real_bnd_root,
                                                                "virtual": True, "real_request": request, "step": step, "side": side,
                                                                "prompt": "tape[0..t0] + sampler.candidates leaf of the step" if side == "target"
                                                                else "tape[0..t_bonus] (boundary prompt + output_token_ids prefix)"}}}
        we = real.weights_index["entries"]
        if side == "target":
            self.weights_index = {**real.weights_index, "entries": [e for e in we if not e["canonical_id"].startswith("w=draft.")]}
        else:
            self.weights_index = {**real.weights_index, "entries": [{**e, "canonical_id": "w=" + e["canonical_id"][len("w=draft."):]}
                                                                    for e in we if e["canonical_id"].startswith("w=draft.")]}
        self._ru = {}

    # --- the runner's view -------------------------------------------------------------------------------
    def ru(self, request: str, step: int):
        """The real replay unit, its entries' ids rewritten for this branch (the runner looks a sampled token up
        by the rewritten lm_head id)."""
        if step not in self._ru:
            index, values = self.real.ru(self.request, step)
            ents = [{**e, "canonical_id": self._rewrite(e["canonical_id"]), "real_canonical_id": e["canonical_id"]}
                    for e in index["entries"] if self._keep(e["canonical_id"], step)]
            self._ru[step] = ({**index, "entries": ents}, values)
        return self._ru[step]

    def boundary(self, request: str) -> dict:
        return {"request_id": self.vreq, "prompt_token_ids": list(self.prompt), "output_token_ids": [], "virtual": True}

    def steps(self, request: str) -> list[int]:
        return [0]

    def step_of_position(self, request: str, t: int) -> int:
        if t >= self.branch_from:
            return self.step
        return self.real.step_of_position(self.request, t)

    def leaf_words(self, request: str, step: int, entry: dict):
        return self.real.leaf_words(self.request, step, entry)

    def tree_root(self, request: str, step: int) -> str:
        return self.real.tree_root(self.request, step)

    def _keep(self, cid: str, step: int) -> bool:
        is_draft = DRAFT_OP in cid
        if (self.side == "draft") != is_draft:
            return False
        if step != self.step and _X_RE.search(cid):
            return False  # another step's uncommitted rows are not this branch
        return True

    def _rewrite(self, cid: str) -> str:
        cid = _X_RE.sub("", cid).replace(f"/r={self.request}/", f"/r={self.vreq}/", 1)
        if self.side == "draft":
            cid = _PASS_RE.sub(r"\1", cid.replace(DRAFT_OP, "op=", 1))
        return cid

    def entries_at(self, request: str, t: int) -> tuple[int, list[dict]]:
        step = self.step_of_position(request, t)
        index, _ = self.ru(request, step)
        return step, [e for e in index["entries"] if f"/t={t}/" in e["canonical_id"]]


class TargetBranchMap(CircuitMap):
    """`VerifyBranch`'s names are B1's `t=<T>/...` plus the `accept` call (this step's block, at slot 0's position)."""

    def __init__(self, program: Program, t0: int):
        super().__init__(program)
        self.t0 = t0

    def locate(self, ident: CaptureIdentity):
        if ident.op == "sampler.accept":
            if ident.t != self.t0:
                raise NotInProgram(f"t={ident.t}: an earlier step's acceptance record; this branch program verifies the block at t={self.t0}")
            c = self.named["accept"]
            return c, _returns_component(c, 2)  # appended[K+1]: accepted candidates, the bonus token, -1 for dead slots
        if ident.op == "sampler.candidates":
            raise NotInProgram("the candidates are the last K prompt tokens of the branch program (Input gates); "
                               "this leaf authenticates them (recorded as the virtual boundary's prompt source)")
        return super().locate(ident)


def run_branch(commit: Path, request: str, step: int, side: str, checkpoint: list[Path], out: Path, layers: list[int],
               coord_samples: int = 32, full: set[tuple[int, int]] | None = None, seed: int = 0,
               cos_sin_npy: Path | None = None, cos_sin_cpu_npy: Path | None = None) -> dict:
    real = CommitStore(commit)
    store = BranchStore(real, request, step, side)
    if side == "target":
        C = dict(b1.B1_CONFIG)
        K = store.K
        factory = lambda LP, STEPS: Program(bind(spec.VerifyBranch, C=C, LP=LP, K=K))  # noqa: E731
        block_t0 = store.slots[0]
        cmap = lambda program, block_t0=block_t0: TargetBranchMap(program, block_t0)  # noqa: E731  (bound now: `t0` below is a clock)
    else:
        C = dict(spec.DRAFT_CONFIG)
        factory = lambda LP, STEPS: Program(bind(spec.ServeDraft, C=C, LP=LP, STEPS=STEPS))  # noqa: E731
        cmap = lambda program: CircuitMap(program)  # noqa: E731
    print(f"[sd-corr] {store.vreq}: side={side} K={store.K} accepted={store.accepted} slots={store.slots} "
          f"LP={store.LP} STEPS={store.STEPS} positions={store.positions}", flush=True)
    t0 = time.time()
    rec = run(store, checkpoint, store.vreq, store.positions, layers, coord_samples, full or set(), out, seed=seed,
              LP=store.LP, STEPS=store.STEPS, cos_sin_npy=cos_sin_npy, cos_sin_cpu_npy=cos_sin_cpu_npy,
              config=f"b1-spec/{side}", program_factory=factory, circuit_map=cmap, static_config=C)
    rec["speculative"] = {"real_request": request, "step": step, "side": side, "k": store.K, "accepted": store.accepted,
                          "slots": store.slots, "candidates": store.sp["candidates"], "verdicts": store.sp["verdicts"],
                          "committed_tokens": store.sp["committed_tokens"], "dead": store.sp["dead"], "draft": store.sp["draft"],
                          "branch_prompt_len": store.LP, "checked_positions": store.positions,
                          "rejected_positions": [t for t in store.positions if t in set(store.sp["dead"])],
                          "seconds": round(time.time() - t0, 1)}
    per_pos: dict[str, dict] = {}
    for r in rec["records"]:
        t = CaptureIdentity.parse(r["capture_id"]).t
        d = per_pos.setdefault(str(t), {"exact": 0, "mismatch": 0, "other": 0, "rejected": t in set(store.sp["dead"])})
        st = r["check"]["status"]
        d["exact" if st == "exact" else "mismatch" if st == "mismatch" else "other"] += 1
    rec["speculative"]["per_position"] = per_pos
    out.write_text(json.dumps(rec, indent=1))
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", type=Path, required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--side", choices=["target", "draft"], required=True)
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True, help="safetensors shard(s) of the side's model")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--coord-samples", type=int, default=32)
    ap.add_argument("--full", nargs="*", default=[], help="t,L pairs whose Gemms are evaluated in full (L=-1: lm_head)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cos-sin-npy", type=Path, default=None, help="the side's CUDA-built cos_sin_cache words (uint16 [MAX_POS, D])")
    ap.add_argument("--cos-sin-cpu-npy", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    layers = a.layers if a.layers is not None else ([0, 1, 13, 27] if a.side == "target" else [0, 1, 11, 23])
    full = {tuple(int(x) for x in s.split(",")) for s in a.full}
    rec = run_branch(a.commit, a.request, a.step, a.side, a.checkpoint, a.out, layers, a.coord_samples, full, a.seed,
                     a.cos_sin_npy, a.cos_sin_cpu_npy)
    s = rec["summary"]
    print(f"[sd-corr] done: exact={s['exact']} mismatch={s['mismatch']} missing={s['missing-operand']} sampled={s['sampled']} "
          f"not_in_program={len(rec['not_in_program'])} -> {a.out}", flush=True)
    return 0 if s["mismatch"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
