# Dead end: the X-capped policy model

Everything in this directory was produced under the policy model with a **per-RU runtime-input cap `X`**
(`I(R) <= X`, X = 5 MB in the sweeps) in addition to the work caps `F` (upstream) and `G` (total). That cap is
not part of the policy: it forced *both* honest inference and training into matmul-tile RUs, so the "honest
inference costs 2.4 MB/token" baseline (frontier_8b.md), every `kappa`, `rho`, penalty and U/L ratio, and the
P1/P2/P3 tables in `RESULTS.md` describe a policy nobody would run. The final report cites this directory as the
documented dead end; nothing here feeds the corrected model (SPEC §1/§6, THEORY §0.1 as rewritten: legal RU iff
`Work(R) <= G` and `Work(Up_R(g)) <= F`, no `X`; runtime input = cross-RU values + token ingress; fixed weights free).

Contents:

* `frontier_8b.{jsonl,json,md,png}` -- the (F, G) sweep of inference-dense / pretrain-dense on llama3-8b at X = 5 MB,
  Q = 4096 and 65536 (64 + 64 rows), its Pareto frontier and "operating points".
* `sweep_p1.jsonl`, `p1_fit.json` -- Q scaling of inference-dense / lora / pretrain-dense at (F = 1e12, X = 5 MB).
* `sweep_p2.jsonl`, `p3_efficiency.json` -- the P2 grid rows (partly stale after the MoE backward landed) and the
  P3 efficiency translation.
* `scale.jsonl` -- the first two rows of the (aborted) scale table at the X-capped operating points.
* `sweep_smoke.jsonl`, `plots/` -- smoke rows and the P1/P2 plots.
* `RESULTS.md` -- the previous results document (its audit / extraction-scaling / characterization sections do not
  depend on the policy and remain valid; the P1/P2/P3 numbers do not).
* `characterization.{json,md}` -- a **copy** taken when the model was retired; the detached characterization
  campaign (pid 18466 / 36127) was still appending to `results/characterization.json` and was left untouched. The
  characterization itself (ADW, MACs, P per circuit) does not depend on X and is reused by `calibration.py`.

What survives from this phase: the guarded/resumable driver machinery (`sweeps/run.py`, `guard.py`), the MAC audit
(`audit.*`, `breakdown_*.md`), the extraction-scaling measurements, and the observation that under a per-RU input
cap the honest-inference input is dominated by attention K/V re-imports as G shrinks (2.5 -> 38 MB/token from
G = None to G = 1e7 at 8B), which is what exposed the cap as an artifact.
