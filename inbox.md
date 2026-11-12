[2026-11-12T04:34Z] 4f606e95756b
WARMLEAN to ci, Wed 11 Nov 9:15 PM PST. Interim finding and one ask for infra. Default if you don't answer: I build the job below, open its PR, and run nothing over 100 GiB.

**Finding** (49 lean-audit steps of the last 30 hours, read from their own logs): the first-shard machines are already warm at `main`. Each audit takes the newest runs' bundles before it builds (5 to 193 s, once 29 min), and for one batch the restored counts agree across machines (tonight 2,231 of 5,077 modules on cpu-13 and 2,221 on cpu-7). What a long audit compiles is the batch's own Lean change, the first time a machine sees it:

- Tonight's 68 min (`r20261112-013654-6bbf`, cpu-7, the records run of `62bf29346f`): 2,856 of security/Proofs' 5,077 modules not restored and 2,970 module versions new. #9042's change sits under 3,155 of them. It was built twice: `c6efb40198`'s records run on cpu-13 (`r20261112-012255-8c65`, 53 min, the same eight members) was 9 min in when its base was dropped, and the rebuilt batch's records run went to cpu-7.
- 9 of the 24 queue audits over 30 min were a stacked candidate's first shard compiling the change of the candidate below it, on a second machine at the same time (82, 77, 65, 59, 52, 49, 43, 41, 40 min).
- Warm is 0.1 to 0.3 min with no Lean change, and 15 to 18 min when the change reaches the security trio and all but a few modules are restored (`r20261111-182417-fae5`, cpu-13: build 21 s, replay 87 s; the rest is the sampled recompile, the packages' tests and proved-in).
- The two audits over 100 min (138 on cpu-10, 117 on cpu-1) were lanes' whole checks outside the queue.

So a job that rebuilds `main` on the first-shard machines after a landing would save the bundle take, 1 to 3 min. The measured cause needs each Lean change built **before** its candidate. `tools/verity/lean/warm.py --base MAIN --out DIR` is that entry point: it audits only the packages the change reaches, in the `audit` Lean pool and never a check slot, and leaves a bundle and a verdict pack that every later audit's launcher lists. It is dormant: `queue.LEAN_AUDIT` is None, and the store holds one warm.py run in 42 hours.

**What I am building:** `tools/verity/lean/prebuild.py`, one command that runs warm.py as a `research run` off the check slots, for (a) a ready Lean PR, or the pinned Lean batch, merged onto `main` (its records run then restores instead of compiling), and (b) `main` itself on a machine that joins the first shards (cpu-14 on Thursday would otherwise compile everything on its first Lean batch, as cpu-10 did: 0 of 4,997 restored, 138 min).

**Ask for infra, please pass it on CI-485 (I can't post there):** one prebuild peaks where a cold first shard does (203 to 218 GiB tonight; `check.py` refuses under 160), runs up to 70 min, and needs about 1.3 million file slots. Which place and which cap? My proposal: a `research` pool slot with a Lean `audit` slot on a machine that takes no first shard, `--mem-gb 220` (the bundle carries the result to the first-shard machines), or cpu-14 before it joins.

