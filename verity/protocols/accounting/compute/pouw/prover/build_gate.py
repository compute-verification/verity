"""A kernel tree's build gate: its table (`sass_gate.json` beside its build.sh) through `sass_gate.py`'s `mma_gate`, on
ptxas's report and cuobjdump's SASS. Refuses (exit 1, the reasons on stderr) on a ptxas warning or serialized wgmma, a stack
frame, a spill or local memory (but `local_ok`'s), and on a kernel whose tensor-core instructions aren't the table's: the
FP8 atom `atom` in a multiple of its `want` row's ``q``, the peel `peel` exactly ``h`` times, `window`'s atoms in the kernels
it names, no other MMA. A form holding ``{n}`` is the kernel's at its row's ``n``: in a kernel with no ``n`` it is no form,
so every MMA there is refused.

  python3 build_gate.py SPEC PTXAS_TXT SASS_TXT OUT_JSON
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path


def _harness():
    name = "pouw_harness_sass_gate"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("sass_gate.py"))
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


H = _harness()


@dataclass
class Gate:
    """One tree's table. `want` maps a kernel to its row, whose fields `columns` names (``q``, ``h``, and ``n`` where the
    forms take one); `atom`, `peel` and `window` are each ``{"key", "form"}``, the record field it counts into and its SASS
    form, `window` also ``what`` and ``atoms`` (kernel: count); `local_ok` is kernel: [bytes, why]; `summary` the kernels the
    passing line names."""

    cubin: str
    arch: str
    counts: list[str]
    atom: dict
    peel: dict
    columns: list[str]
    want: dict[str, list[int]]
    summary: list[str]
    window: dict | None = None
    local_ok: dict | None = None

    def row(self, name: str) -> dict[str, int]:
        return dict(zip(self.columns, self.want.get(name, [0] * len(self.columns)), strict=True))

    def form(self, of: dict, name: str) -> str:
        return of["form"].format(n=self.row(name).get("n", 0))

    def _matches(self, of: dict, name: str, form: str) -> bool:
        return ("{n}" not in of["form"] or bool(self.row(name).get("n"))) and form == self.form(of, name)

    def _atom(self, name: str, form: str, r: dict) -> bool:
        w = self.window
        key = (self.atom["key"] if self._matches(self.atom, name, form) else self.peel["key"] if self._matches(self.peel, name, form)
               else w["key"] if w and name in w["atoms"] and form == w["form"] else None)
        if key:
            r[key] += 1
        return key is not None

    def _check(self, name: str, r: dict) -> list[str]:
        row, a, p = self.row(name), self.atom["key"], self.peel["key"]
        q, h, got, out = row["q"], row["h"], r.get(a, 0), []
        if (got < q or (q and got % q) or (not q and got)) or r.get(p, 0) != h:
            out.append(f"{name}: {got} {self.form(self.atom, name)}, {r.get(p)} {self.form(self.peel, name)} "
                       f"(want a multiple of {q}, and {h})")
        if (w := self.window) and r.get(w["key"], 0) != w["atoms"].get(name, 0):
            out.append(f"{name}: {r.get(w['key'], 0)} {w['form']} for {w['what']} (want {w['atoms'].get(name, 0)})")
        return out

    def gate(self, ptxas: str, sass: str) -> tuple[dict, list[str]]:
        keys = [self.atom["key"], self.peel["key"]] + ([self.window["key"]] if self.window else [])
        return H.mma_gate(ptxas, sass, self.want, arch=self.arch, smem=True,
                          local_ok={k: v[0] for k, v in self.local_ok.items()} if self.local_ok else None,
                          record={**dict.fromkeys(self.counts, 0), **dict.fromkeys(keys, 0), "other_mma": []},
                          counts={op: rf"\b{op}\b" for op in self.counts}, atom=self._atom,
                          allowed=lambda name: f"{self.form(self.atom, name)} and {self.form(self.peel, name)}", check=self._check)

    def main(self, ptxas: str, sass: str, out: str) -> int:
        funcs, problems = self.gate(Path(ptxas).read_text(), Path(sass).read_text())
        doc = {"want": self.want, **({"local_ok": self.local_ok} if self.local_ok else {}), "kernels": funcs, "problems": problems}
        a, p = self.atom["key"], self.peel["key"]
        return H.write_verdict(out, doc, problems, lambda: f"{self.cubin}: {len(funcs)} kernels; " + ", ".join(
            f"{n} {funcs[n]['regs']} regs {funcs[n][a]} {a}" + (f" {funcs[n][p]} {p}" if funcs[n][p] else "") for n in self.summary)
            + "; no local memory" + (f" but {sorted(self.local_ok)}'s" if self.local_ok else ""))


def load(spec: str | Path) -> Gate:
    return Gate(**json.loads(Path(spec).read_text()))


if __name__ == "__main__":
    sys.exit(load(sys.argv[1]).main(*sys.argv[2:5]))
