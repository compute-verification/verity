
## 19:00Z, proofs coordinator: proofs has dedicated Lean nodes now; use them

infra (INF-453): vy-nebius-cpu-11, cpu-12 and cpu-7 are proofs' Lean nodes, with proofs pools of 8-10 build slots each,
until Fri 13 Nov 6 PM PST; ci's hand shards may share them until about 20:30Z. cpu-1 and cpu-9 stay up until Fri 6 Nov
6 PM PST, and cpu-9 refuses runs while its disk is low. Launch new runs with `--on vy-nebius-cpu-11 --pool proofs`,
`--on vy-nebius-cpu-12 --pool proofs` or `--on vy-nebius-cpu-7 --pool proofs`, spread across the three. Move any run of yours
that hasn't started, and cancel the old copy once the new one registers. The first build on each node is cold, so
put Security/Proofs builds there early.

## 19:30Z, proofs coordinator: scope change from Daniel, the outer system is generic too

Daniel prefers the outer proof system generic as well as the inner. Extend the blueprint and the statement:
1. **An abstract outer system:** a knowledge-sound argument for the relation "openings of the ZK server's commitments exist on
   which the inner verifier circuits accept", with its extractor's interface stated explicitly (rewinding or straight-line,
   expected or strict running time, the hash calls it makes), and zero knowledge as a simulator for the outer view.
   `RecursiveZK` already takes the outer view's zero knowledge as a premise (`OuterViewZK`), so the ZK side is generic in both.
2. **An abstract ZK server:** the properties the server must have (commit to every inner message under hiding leaves, take
   live coins, release rules, refusal on a malformed request), so the theorem doesn't depend on C-Flock's message formats.
   The deployed server stays C-Flock's and is proved to meet the interface as part of the instance.
3. **The main theorem:** compile(inner, outer, server) is knowledge sound with error inner + outer + collision terms (with the
   extractors' composition explicit), and zero knowledge by `RecursiveZK`. The instance is C-Flock for both, with the
   deployed server, recovering today's bounds or better.
4. **Risks** (in your (g) section): composing a rewinding outer extractor with the inner one, the blow-up in running time and
   hash calls, and what the server interface must guarantee for ZK when the worker misbehaves (today's any-worker premise:
   the worker's values satisfy the circuit).

Same deadline for a reviewable statement: Wed 18:00Z. If the outer generalization would push it, say so in your next
checkpoint, and deliver the inner-generic statement on time with the outer as the immediate next section.

## 19:50Z, proofs coordinator: Daniel's terminology, and the sampled side

**Terminology** (Daniel's, per GLOSSARY.md on main). Use it in the blueprint's prose and in new Lean names, and refer to
existing identifiers as they are:
- sampling design, not law;
- the sample, a sampled proof block, to sample, not draw or drawn;
- proof block, not unit;
- trace for a run's values;
- ε_bind, not δ_link;
- acceptance probability P_a, not escape;
- subprotocols and composition, not components or product;
- Flock, not C-Flock;
- the ZK server, not the firewall;
- the verifier, not 'verifier of record'.

**The sampled side.** GENSAMPLE (new worker) states the sampled audit as a compiler over the same abstract proof
system. You own the shared interface: keep it in your note under a stable heading, and check GENSAMPLE's note
(`*-draft-generic-sampling.md`) for a 'For GENERIC' section asking for additions.

## 20:01Z, proofs coordinator: cpu-9 is back

cpu-9's disk is fixed (INF-452) and it takes runs again. When proofs' nodes cpu-11, cpu-12 and cpu-7 are full (they were
at 19:52Z), use `--on vy-nebius-cpu-9 --pool proofs`, or `--pool research` there, and move any run of yours that is still
waiting, cancelling the old copy once the new one registers.

## 20:54Z, proofs coordinator: heavy Lean builds go to cpu-11 and cpu-12

cpu-1 OOM-killed a Lean run at 20:45Z (five records builds at 60-100 GB each beside training jobs, under a 700 GB cap).
infra doubled check slots on cpu-11 and cpu-12 (8 each, about 735 GB free each) and cpu-7 (9). Send any Security/Proofs
or Security records build to `--on vy-nebius-cpu-11 --pool proofs` or cpu-12; use cpu-1 and cpu-9 only for small
packages. This replaces "cpu-1 or cpu-9 first".

## 21:08Z, proofs coordinator: more Lean capacity

vy-nebius-cpu-3 is back to Fri 13 Nov 6 PM PST with a 14-slot proofs pool and 5 build trees (`--on vy-nebius-cpu-3
--pool proofs`), shared with network-accounting. cpu-5 and cpu-6 research slots were idle at 20:58Z. Heavy Security
builds: cpu-11, cpu-12 or cpu-3. Move any run of yours that is still waiting.

## 21:45Z, proofs coordinator: REDGEN's verdict on the recursion

Read `~/.research/notes/lanes/proofs/20261103T2145Z-finding-redgen.md`. L1 at Flock has no blocker. Before proving:
- R1: add `K.RO x w` to `ForksBind`/`CallsLogColl`.
- R2: inner-view decoding as an `ofOC` rerun program, with extractors stated without `H` in scope.
- R3: the Flock instance keeps the `linkPlan` term, not `PrFdevV`.
- R4: state a toy outer argument for L2 and settle F8 (`LastCoinFree`).
- R5: the simulator is quantified first, with the server indexed by the witness.
- R6: give `Proto` a rerun-program extractor, as `Arg` has, or narrow the claim.
- C1, C2: B1's trial cost counted, not declared; `Sound` at Flock.
Also: the sampled compiler moves to the claims core (generalizing main's `Claims.Sampling`), so B1 becomes claims core
to claims core. Coordinate the shared rerun-program interface with GENSAMPLE through your notes.

## 21:54Z, proofs coordinator: every PR names its goal (Daniel, today)

From now on each PR body carries `Goal <id>` naming the Goals-page goal it serves (e.g. `Goal proofs/zero-knowledge`,
`Goal proofs/sampled-audits-sound`, `Goal proofs/recursion-knowledge-sound`, `Goal proofs/sampling-generic`,
`Goal proofs/recursion-generic`, `Goal proofs/terminology-clean`, `Goal proofs/zero-knowledge-any-worker`,
`Goal proofs/zk-leaks-closed`, `Goal proofs/firewall-checks-provable`). Add it to any open PR of yours the next time you
edit its body (a body edit needs no new mark). ci may soon refuse a mark on a PR that names none.

## 21:56Z, proofs coordinator: the ZK form (R5 too)

The deployed ZK server's theorem is in leakage form, Pr_real[E] <= 2 * (Pr_sim[E] + eps(x, w, V*)) (GSFLOCKZK's
`ZKBelow`, branch `cursor/proofs-gsflockzk-818d`). State Z1 in that form, with 2^l and an error reading the witness and
the verifier, and quantify the simulator first (R5), so the recursion's ZK and the sampled audit's ZK are one notion.

## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.

## 22:16Z, proofs coordinator: your issue

Your Linear issue is PRF-364 (goal label proofs/recursion-generic). Every PR you open carries `Refs PRF-364` and `Goal proofs/recursion-generic` in its body: from about Wed 1 PM PST ci gates each mark on the issue's goal label.

## 22:28Z, proofs coordinator: one ZK notion

REDZK: three ZK notions are in flight. The definition is ZKINVALID's per-outcome form: for each outcome s, Pr[out = s and Rel in E] <= Pr[S(s) in E] + eps_s(x, w, V*), with the simulator fixed before the worker and the witness. The leakage form (2^l, sum of eps) and GSFLOCKZK's `ZKBelow` are its corollaries. ZKINVALID puts it in Definitions; state your ZK against it.

## 23:17Z, proofs coordinator: pins.py --merged runs Lean builds locally

`pins.py --merged` runs pinned fixtures' generators, which `lake build` on your VM, and nothing refuses it yet (ci CI-383,
hook refusal Wed). Until then, run it only through `research run --on auto`, or recompute pins by hashing.

## 00:43Z, proofs coordinator: a definitional change (decision)

GSJOINT found that `Compiled.E` (coins per block) cannot recover `Sampling.KS` (coins per slot): a block at two slots is
retried on the same coins, which changes the laws `Bext` reads; and `Sessions.runs` pins the session's exact shape while
the deployment gives only bounds. Decision: `Compiled.E` takes per-slot coins, and the shape carries only what the coins
read, with each session's errors bounded separately (GSJOINT's note, R8 section). GSJOINT makes the change, coordinating
with GENSAMPLE, and adapts R3a/R3b/R3e. If your piece reads the coin indexing or the shape, follow GSJOINT's note.
