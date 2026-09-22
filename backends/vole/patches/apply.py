"""Apply the c-compress patch to a swanky checkout at 1b05bddab3e02c2cc4917d08c80a629286a4a4fc.

    python3 backends/vole/patches/apply.py /workspace/swanky [--fixed-lpn]

Idempotent: every edit is anchored on an exact string and skipped when already present.  Copies ``verity_v0.rs`` into
``edge/diet-mac-and-cheese/src/plugins/`` and edits: ``plugins/mod.rs`` (the ``Verity`` execution variant),
``circuit_ir.rs`` (plugin registration), ``backend_trait.rs`` (``assert_mul`` with a default implementation),
``backend.rs`` (the output-free QuickSilver triple), ``backend_multifield.rs`` (delegation + the two dispatch sites).
With ``--fixed-lpn`` also patches ``edge/svole-wykw/src/svole.rs``: the LPN matrix indices are sampled once per
sVOLE instance and cached (one public matrix per session) instead of afresh per extension.
The resulting ``git diff`` is what ``swanky-verity_v0.patch`` records.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def edit(path: Path, anchor: str, new: str, mode: str = "after", key: str | None = None) -> None:
    txt = path.read_text()
    marker = key or new.strip().splitlines()[0].strip()
    if marker in txt:
        print(f"  = {path.name}: already patched ({marker[:50]})")
        return
    if anchor not in txt:
        sys.exit(f"anchor not found in {path}: {anchor[:80]!r}")
    if txt.count(anchor) != 1:
        sys.exit(f"anchor not unique in {path}: {anchor[:80]!r}")
    if mode == "after":
        txt = txt.replace(anchor, anchor + new)
    elif mode == "before":
        txt = txt.replace(anchor, new + anchor)
    else:
        txt = txt.replace(anchor, new)
    path.write_text(txt)
    print(f"  + {path.name}: {marker[:60]}")


def main() -> None:
    root = Path(sys.argv[1])
    fixed_lpn = "--fixed-lpn" in sys.argv
    dmc = root / "edge/diet-mac-and-cheese/src"
    here = Path(__file__).resolve().parent
    shutil.copy(here / "verity_v0.rs", dmc / "plugins/verity_v0.rs")
    print("  + plugins/verity_v0.rs")

    # plugins/mod.rs
    mod = dmc / "plugins/mod.rs"
    edit(mod, "    /// The plugin implements RAM operations.\n    Ram(RamVersion),\n",
         "    /// The plugin implements the Verity checks (assert_mul, logUp range / table lookups).\n    Verity(VerityV0),\n")
    edit(mod, "            | PluginExecution::Mux(_)\n            | PluginExecution::Ram(_) => None,",
         "            | PluginExecution::Mux(_)\n            | PluginExecution::Verity(_)\n            | PluginExecution::Ram(_) => None,",
         mode="replace", key="| PluginExecution::Verity(_)\n")
    edit(mod, "            PluginExecution::Ram(plugin) => {\n                let mut mapping = TypeIdMapping::default();\n                mapping.set(plugin.type_id());\n                mapping\n            }\n",
         "            PluginExecution::Verity(plugin) => {\n                let mut mapping = TypeIdMapping::default();\n                mapping.set(plugin.type_id());\n                mapping\n            }\n")
    edit(mod, "mod ram;\npub(crate) use ram::{Ram, RamArithV0, RamArithV1, RamBoolV0, RamBoolV1, RamOp, RamVersion};\n",
         "mod verity_v0;\npub(crate) use verity_v0::VerityV0;\n")

    # circuit_ir.rs: registration
    cir = dmc / "circuit_ir.rs"
    edit(cir, "            DisjunctionV0, GaloisPolyV0, IterV0, MuxV0, MuxV1, PermutationCheckV1, VectorsV1,\n",
         "            VerityV0,\n")
    edit(cir, "            PermutationCheckV1::NAME => PermutationCheckV1::instantiate(\n",
         "            VerityV0::NAME => VerityV0::instantiate(\n                &operation,\n                &params,\n                &output_counts,\n                &input_counts,\n                type_store,\n                fun_store,\n            )?,\n",
         mode="before")

    # backend_trait.rs: assert_mul with a default
    bt = dmc / "backend_trait.rs"
    edit(bt, "    fn mul(&mut self, a: &Self::Wire, b: &Self::Wire) -> Result<Self::Wire>;\n",
         "    /// Assert `a * b = c` without committing the product (verity_v0 patch; default: multiply, subtract, zero-check).\n"
         "    fn assert_mul(&mut self, a: &Self::Wire, b: &Self::Wire, c: &Self::Wire) -> Result<()> {\n"
         "        let ab = self.mul(a, b)?;\n        let d = self.sub(&ab, c)?;\n        self.assert_zero(&d)\n    }\n")

    # backend.rs: one QuickSilver triple, no commitment
    be = dmc / "backend.rs"
    edit(be, "    monitor_mul: usize,\n", "    monitor_assert_mul: usize,\n")
    edit(be, "    pub(crate) fn incr_monitor_mulc(&mut self) {\n",
         "    pub(crate) fn incr_monitor_assert_mul(&mut self) {\n        self.tick();\n        self.monitor_assert_mul += 1;\n    }\n",
         mode="before")
    edit(be, "        if self.monitor_mul != self.monitor_zk_mult_check {",
         "        if self.monitor_mul + self.monitor_assert_mul != self.monitor_zk_mult_check {",
         mode="replace", key="self.monitor_mul + self.monitor_assert_mul !=")
    edit(be, '        info!("nb mult:   {:>11}", self.monitor_mul);\n',
         '        info!("nb amul:   {:>11}", self.monitor_assert_mul);\n')
    edit(be, "    fn add_constant(&mut self, a: &Self::Wire, b: Self::FieldElement) -> Result<Self::Wire> {\n        self.monitor.incr_monitor_addc();\n",
         "    fn assert_mul(&mut self, a: &Self::Wire, b: &Self::Wire, c: &Self::Wire) -> Result<()> {\n"
         "        self.monitor.incr_monitor_assert_mul();\n"
         "        self.mult_check_state\n            .accumulate(&(*a, *b, *c), self.fcom.get_delta());\n"
         "        if self.mult_check_state.count() >= BATCH_SIZE {\n"
         "            self.mult_check_state.finalize(\n"
         "                self.fcom.gen_mask(&mut self.channel, &mut self.rng)?,\n"
         "                &mut self.channel,\n                &mut self.rng,\n                self.fcom.get_delta(),\n            )?;\n"
         "        }\n        Ok(())\n    }\n\n",
         mode="before")

    # backend_multifield.rs: delegation in the Conv backend + the two dispatch sites
    bm = dmc / "backend_multifield.rs"
    edit(bm, "    fn mul(&mut self, a: &Self::Wire, b: &Self::Wire) -> Result<Self::Wire> {\n        self.dmc.mul(a, b)\n    }\n",
         "    fn assert_mul(&mut self, a: &Self::Wire, b: &Self::Wire, c: &Self::Wire) -> Result<()> {\n        self.dmc.assert_mul(a, b, c)\n    }\n")
    edit(bm, "            PluginExecution::Mux(plugin) => {\n                plugin.execute::<P, B>(&mut self.backend, &mut self.memory)?\n            }\n",
         "            PluginExecution::Verity(plugin) => {\n                assert_eq!(outputs.len(), 0);\n"
         "                let mut wires = Vec::new();\n                for range in inputs {\n"
         "                    wires.extend(copy_mem(&self.memory, *range).copied());\n                }\n"
         "                plugin.execute::<B>(wires, &mut self.backend)?\n            }\n")
    edit(bm, "                PluginExecution::Disjunction(plugin) => {\n                    let type_id = plugin.field() as usize;\n",
         "                PluginExecution::Verity(plugin) => {\n                    let type_id = plugin.type_id() as usize;\n"
         "                    // no callframe: the plugin reads its input wires from the caller's memory\n"
         "                    self.eval[type_id].plugin_call_gate(\n                        self.inputs.get(type_id),\n"
         "                        None,\n                        fun_store,\n                        out_ranges,\n                        in_ranges,\n"
         "                        body,\n                    )?;\n                }\n",
         mode="before", key="// no callframe: the plugin reads its input wires")

    if fixed_lpn:
        sv = root / "edge/svole-wykw/src/svole.rs"
        apply_lpn(sv)
    print("done")


def apply_lpn(sv: Path) -> None:
    """One public LPN matrix per sVOLE instance: sample the (index, coefficient) rows once, reuse per extension."""
    txt = sv.read_text()
    if "lpn_cache" in txt:
        print("  = svole.rs: already patched")
        return
    cache_ty = "LpnCache<T>"
    txt = txt.replace(
        "use swanky_rng::SwankyRng;\n",
        "use swanky_rng::SwankyRng;\nuse std::collections::HashMap;\n\n"
        "/// One public LPN matrix per (rows, cols) per sVOLE instance, sampled once from the shared `lpn_rng` in the\n"
        "/// same order the per-extension sampling used, then reused for every extension (c-compress patch).\n"
        "type LpnCache<T> = HashMap<(usize, usize), Vec<[(usize, <T as FiniteField>::PrimeField); LPN_PARAMS_D]>>;\n",
        1,
    )
    # struct fields (sender and receiver both end with the lpn_rng field)
    txt = txt.replace(
        "generating the LPN matrix.\n    lpn_rng: SwankyRng,\n}",
        f"generating the LPN matrix.\n    lpn_rng: SwankyRng,\n    lpn_cache: {cache_ty},\n}}",
    )
    assert txt.count("lpn_cache: LpnCache<T>,") == 2, "struct fields"
    # constructors: four `lpn_rng,\n        }` / `lpn_rng,\n        })` sites
    n_before = txt.count("            lpn_rng,\n        }")
    txt = txt.replace("            lpn_rng,\n        }", "            lpn_rng,\n            lpn_cache: HashMap::new(),\n        }")
    assert n_before == 4, f"constructor sites: {n_before}"
    # the matrix fetch + per-row lookup, sender and receiver
    fetch = (
        "        let key = (rows, cols);\n"
        "        if !self.lpn_cache.contains_key(&key) {\n"
        "            let m: Vec<_> = (0..cols)\n"
        "                .map(|_| lpn_mtx_indices::<T>(&distribution, &mut self.lpn_rng))\n"
        "                .collect();\n"
        "            self.lpn_cache.insert(key, m);\n"
        "        }\n"
        "        let matrix = &self.lpn_cache[&key];\n"
    )
    anchor = "        let distribution = Uniform::<u32>::try_from(0..rows.try_into().unwrap()).unwrap();\n"
    assert txt.count(anchor) == 2, "distribution sites"
    txt = txt.replace(anchor, anchor + fetch)
    old = "            let indices = lpn_mtx_indices::<T>(&distribution, &mut self.lpn_rng);\n"
    assert txt.count(old) == 2, "sampling sites"
    txt = txt.replace(old, "            let indices = &matrix[i];\n")
    sv.write_text(txt)
    print("  + svole.rs: one cached LPN matrix per (rows, cols) per instance")


if __name__ == "__main__":
    main()
