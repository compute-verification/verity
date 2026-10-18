"""pd's launches for bench.py: the module's tables (`setup`), one call's launches on a `run.Pipeline`'s own buffers
(`PdCall`), and the check (`check`): pd's launches held byte for byte to Pearl-C's on the same call, every output poisoned
first, with the Python references, a poison control and a no-stray-write control."""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

import refs
import run as R

POISON = 0xA5
HERE = os.path.dirname(os.path.abspath(__file__))


SRC = open(os.path.join(HERE, "pd.cuh")).read()
#: pd.cuh's PD_VARIANT_LIST: the variants built, a launch each (bits PD_ROUNDS 1, PD_ROLLED 2, PD_CALL 4, PD_PREFETCH 8,
#: PD_PAIR 16)
VARIANTS = tuple(int(x) for x in re.findall(r"X\((\d+)\)", re.search(r"#define PD_VARIANT_LIST\(X\)(.*)", SRC).group(1)))
PD_PAIR = 16
#: a PD_PAIR launch's dynamic shared memory (pd.cuh's PD_PAIR_SMEM: two pairs of PD_PAIR_WORDS words)
PAIR_SMEM = 2 * 8 * int(re.search(r"PD_PAIR_WORDS = (\d+)", SRC).group(1))
#: the variants of the schedules with side lanes (pd_deferred_V, pd_side_tree_V); every variant gets pd_serial_V
SIDE = tuple(int(x) for x in os.environ.get("PD_SIDE", "3,19,23").split(","))
assert set(SIDE) <= set(VARIANTS), (SIDE, VARIANTS)
MAX_DYNAMIC_SHARED = 8   # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES


def smem(v: int) -> int:
    return PAIR_SMEM if v & PD_PAIR else 0
COMMIT = ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a")
#: the served steps pd's launches do: A's commitment, E_A's lines, the tile leaves
LINES = "lines_ea"
PD_STEPS = (*COMMIT, LINES, "hash_leaf")
#: what a call's hashing writes, and what the check also compares downstream of it (the call's products)
OUTPUTS = ("rowdig_a", "tree_a", "seed_a", "line_a", "seeds_a", "lkeys_a", "ea_codes", "leaves")
DOWNSTREAM = ("c", "u", "y")
#: the A side's commitment alone (what the filler-keying proposal's timing schedule still writes right)
COMMITMENT = ("rowdig_a", "tree_a", "seed_a", "line_a", "seeds_a", "lkeys_a")


def setup(pd) -> None:
    """The module's tables, and each PD_PAIR launch's dynamic shared memory allowed past the default 48 KB with its static."""
    pd.launch("s512_heads", R.cdiv(R.RS_MAX_SEGS * (R.RS_MAX_SEGS + 1) // 2 + R.RS_MAX_SEGS + 1, 128), 128)
    pd.launch("pd_heads", 1, 32)
    pd.sync()
    for v in VARIANTS:
        if v & PD_PAIR:
            for f in ("pd_commit", "pd_tree", "pd_probe"):
                pd.ck("cuFuncSetAttribute", pd.fn(f"{f}_{v}"), MAX_DYNAMIC_SHARED, PAIR_SMEM)


class PdCall:
    """One call's pd launches on `pipe`'s buffers (Pipeline.b: x, A's tree, seeds, keys, E_A's codes, C~, U and the leaves)
    and `keys` (dom_a, the unit block, the salt key, the hm96 constants); scratch its own: pd_commit's counter, zero.  E_A's
    lines are pd_commit's and pd_tree's (the served call's `lines_ea`, LINES)."""

    def __init__(self, pd, pipe: R.Pipeline, keys: dict[str, int]):
        self.pd, self.p, self.keys = pd, pipe, keys
        self.m, self.k, self.n, self.s, self.lines = pipe.m_real, pipe.k, pipe.n, pipe.k // 256, pipe.m
        self.scratch = pd.alloc(16)

    def block(self, v: int) -> int:
        """pd_commit_V's threads: the segments' warps (doubled with PD_PAIR: consumers, then producers) and two more."""
        return 32 * (R.cdiv(self.s, 32) * (2 if v & PD_PAIR else 1) + 2)

    def _p(self) -> tuple:
        b, kk = self.p.b, self.keys
        return (b["xa"], b["rowdig_a"], b["tree_a"], b["seeds_a"], b["lkeys_a"], kk["unit"], kk["dom_a"], kk["salt_key"], kk["hm"],
                self.scratch, b["seed_a"], b["line_a"], b["ea_codes"])

    def _commit(self, v: int, mode: int) -> list:
        return [lambda: self.pd.launch(f"pd_commit_{v}", self.m, self.block(v), smem(v), p=self._p(),
                                       i=(self.m, self.k, mode, self.lines))]

    def commit(self, v: int) -> list:
        return self._commit(v, 0)

    def rows(self, v: int) -> list:
        return self._commit(v, 1)

    def tree(self, v: int) -> list:
        b, kk = self.p.b, self.keys
        return [lambda: self.pd.launch(f"pd_tree_{v}", 1, 128, smem(v), p=(0, 0, b["tree_a"], 0, 0, kk["unit"], kk["dom_a"], 0, 0,
                                                                          0, b["seed_a"], b["line_a"], b["ea_codes"]),
                                       i=(self.m, 0, 0, self.lines))]

    def tiles(self, v: int) -> list:
        b = self.p.b
        return [lambda: self.pd.launch(f"pd_tiles_{v}", self.p.tiles, 32, p=(b["c"], b["u"], b["leaves"]), i=(self.m, self.n))]

    def free(self) -> None:
        self.pd.free(self.scratch)


def _read(dev, pipe, name: str) -> bytes:
    return dev.read(pipe.b[name], pipe.size[name])


def _poison(dev, pipe, names) -> None:
    for x in names:
        dev.ck("cuMemsetD8_v2", R.u64(pipe.b[x]), POISON, pipe.size[x])


def check(sh, python: bool = True) -> dict:
    """pd's call on each ring set against Pearl-C's on the same set, in each of pd's variants, whole and split (rows, then
    pd_tree), every hashing output poisoned before each; then the Python references on set 0."""
    dev, failed, compared, controls = sh.dev, [], 0, {}
    hashing = sh.call_names(True)
    rest = [x for x in hashing if x not in PD_STEPS]
    sh.want = {}
    for j, pipe in enumerate(sh.sets):
        pdc = sh.pdc[j]
        names = [x for x in OUTPUTS + DOWNSTREAM if x in pipe.b]
        for f in sh.ref(j, hashing):
            f()
        dev.sync()
        want = sh.want[j] = {x: _read(dev, pipe, x) for x in names}
        _poison(dev, pipe, OUTPUTS)
        dev.sync()
        controls[f"set{j}_poison_differs"] = all(_read(dev, pipe, x) != want[x] for x in OUTPUTS)
        other = sh.sets[1 - j]
        before = {x: hashlib.sha256(_read(dev, other, x)).hexdigest() for x in OUTPUTS}
        inputs = {x: hashlib.sha256(dev.read(sh.keys[x], len(sh.kb[x]))).hexdigest() for x in ("unit", "dom_a", "salt_key")}
        inputs["xa"] = hashlib.sha256(dev.read(pipe.b["xa"], sh.mp * 4 * sh.k)).hexdigest()
        forms = {f"v{v}": pdc.commit(v) + sh.ref(j, rest) + pdc.tiles(v) for v in VARIANTS}
        forms.update({f"split_v{v}": pdc.rows(v) + pdc.tree(v) + sh.ref(j, rest) + pdc.tiles(v) for v in VARIANTS})
        for form, fns in forms.items():
            _poison(dev, pipe, OUTPUTS)
            for f in fns:
                f()
            dev.sync()
            for x in names:
                compared += 1
                got = _read(dev, pipe, x)
                if got != want[x]:
                    bad = sum(got[i:i + 64] != want[x][i:i + 64] for i in range(0, len(got), 64))
                    failed.append(f"set{j}/{form}/{x}: {bad} of {len(got) // 64 or 1} 64-byte words differ")
            if int.from_bytes(dev.read(pdc.scratch, 4), "little"):
                failed.append(f"set{j}/{form}: the counter was left set")
        controls[f"set{j}_other_set_untouched"] = before == {x: hashlib.sha256(_read(dev, other, x)).hexdigest() for x in OUTPUTS}
        after = {x: hashlib.sha256(dev.read(sh.keys[x], len(sh.kb[x]))).hexdigest() for x in ("unit", "dom_a", "salt_key")}
        after["xa"] = hashlib.sha256(dev.read(pipe.b["xa"], sh.mp * 4 * sh.k)).hexdigest()
        controls[f"set{j}_inputs_untouched"] = after == inputs
    py = {}
    if python:
        p0, kb = sh.sets[0], sh.kb
        xa = dev.read(p0.b["xa"], sh.m16 * 4 * sh.k)
        rows = [xa[4 * sh.k * r:4 * sh.k * (r + 1)] for r in range(sh.m16)]
        ref = refs.commit(rows, kb["salt"], kb["root_b"], kb["salt_key"], sh.index, sh.mp)
        for x in ("rowdig_a", "tree_a", "seed_a", "line_a", "seeds_a", "lkeys_a", "ea_codes"):
            py[x] = _read(dev, p0, x)[:len(ref[x])] == ref[x]
        py["dom_a"] = ref["dom_a"] == kb["dom_a"]
        c = np.frombuffer(_read(dev, p0, "c"), np.uint32).reshape(sh.mp, sh.n)
        u = np.frombuffer(_read(dev, p0, "u"), np.uint32).reshape(sh.mp, sh.n)
        leaves, tn = _read(dev, p0, "leaves"), sh.n // 64
        sample = sorted({0, 1, 2, tn - 1, sh.tiles // 2, sh.tiles - 2, sh.tiles - 1})
        py["tile_leaves_sampled"] = len(sample)
        py["tile_leaves"] = all(leaves[64 * t:64 * t + 64] == refs.tile_leaf(c, u, t // tn, t % tn, sh.m16) for t in sample)
    ok = not failed and all(controls.values()) and all(v for k, v in py.items() if k != "tile_leaves_sampled")
    return {"ok": ok, "compared": compared, "failed": failed, "controls": controls, "python": py}


def check_graph(sh, run, partial: bool) -> list[str]:
    """A captured schedule's bytes as it runs, side lanes and all: both ring sets' hashing outputs poisoned, `run` (the graph,
    once: CHAIN calls), then each set's outputs and products against the reference call's (`check`'s, which runs first).
    `partial` compares the commitment alone (the filler-keying proposal's schedule, whose lines_ea reads the ring's earlier
    line_a)."""
    dev, failed = sh.dev, []
    for pipe in sh.sets:
        _poison(dev, pipe, OUTPUTS)
    dev.sync()
    run()
    for j, pipe in enumerate(sh.sets):
        for x, v in sh.want[j].items():
            if (not partial or x in COMMITMENT) and _read(dev, pipe, x) != v:
                failed.append(f"set{j}/{x}")
    return failed
