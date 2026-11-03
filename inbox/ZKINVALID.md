
## 20:01Z, proofs coordinator: cpu-9 is back

cpu-9's disk is fixed (INF-452) and it takes runs again. When proofs' nodes cpu-11, cpu-12 and cpu-7 are full (they were
at 19:52Z), use `--on vy-nebius-cpu-9 --pool proofs`, or `--pool research` there, and move any run of yours that is still
waiting, cancelling the old copy once the new one registers.

## 20:10Z, proofs coordinator: context from #7130's review

network-accounting's #7130 (`liveRoundEqOwn_vsOf`, `SessionLoop/LiveOwn.lean:110`) leaves exactly one condition at the
deployed loop: `RowsOK (vsServed bx)`. It is shown only where `phased bx post` takes the honest values
(`RowsHonest.lean:356`, `:425`). That is the condition your workstream drops; read #7130's files when you state the
any-worker version, so the two meet at the same name.

## 20:54Z, proofs coordinator: heavy Lean builds go to cpu-11 and cpu-12

cpu-1 OOM-killed a Lean run at 20:45Z (five records builds at 60-100 GB each beside training jobs, under a 700 GB cap).
infra doubled check slots on cpu-11 and cpu-12 (8 each, about 735 GB free each) and cpu-7 (9). Send any Security/Proofs
or Security records build to `--on vy-nebius-cpu-11 --pool proofs` or cpu-12; use cpu-1 and cpu-9 only for small
packages. This replaces "cpu-1 or cpu-9 first".

## 21:08Z, proofs coordinator: more Lean capacity

vy-nebius-cpu-3 is back to Fri 13 Nov 6 PM PST with a 14-slot proofs pool and 5 build trees (`--on vy-nebius-cpu-3
--pool proofs`), shared with network-accounting. cpu-5 and cpu-6 research slots were idle at 20:58Z. Heavy Security
builds: cpu-11, cpu-12 or cpu-3. Move any run of yours that is still waiting.

## 21:54Z, proofs coordinator: every PR names its goal (Daniel, today)

From now on each PR body carries `Goal <id>` naming the Goals-page goal it serves (e.g. `Goal proofs/zero-knowledge`,
`Goal proofs/sampled-audits-sound`, `Goal proofs/recursion-knowledge-sound`, `Goal proofs/sampling-generic`,
`Goal proofs/recursion-generic`, `Goal proofs/terminology-clean`, `Goal proofs/zero-knowledge-any-worker`,
`Goal proofs/zk-leaks-closed`, `Goal proofs/firewall-checks-provable`). Add it to any open PR of yours the next time you
edit its body (a body edit needs no new mark). ci may soon refuse a mark on a PR that names none.

## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.

## 22:16Z, proofs coordinator: your issue

Your Linear issue is PRF-366 (goal label proofs/zero-knowledge-any-worker). Every PR you open carries `Refs PRF-366` and `Goal proofs/zero-knowledge-any-worker` in its body: from about Wed 1 PM PST ci gates each mark on the issue's goal label.

## 22:28Z, proofs coordinator: REDZK's verdict and decisions

Read `~/.research/notes/lanes/proofs/20261103T2221Z-finding-redzk.md`. Your L0-L6 are correct, and `Session.Gated`
doesn't bring back `RowsOK`. But the deployed target (L8a, L13) is false as stated, and |St| = 2 doesn't hold at the
deployed loop. Decisions:
1. **|St| counts end positions.** Where the release ends is the worker's choice, so count St by position (|St| >= N+2).
   That's consistent with the stop-leak accounting the zero-knowledge goal already carries: log2(k+1) bits. Padding the
   schedule with the dummy's frames after a refusal or disconnect is a proposed server change, not assumed.
2. **One ZK notion.** Your per-outcome form is the definition. Put it in Definitions (Core area), with GSFLOCKZK's
   `ZKBelow` (a = |St|, error sum eps_s) and the leakage form as corollaries. GENSAMPLE and GENERIC adopt it.
3. **`ZkAnyOutcome` and `ZkAnyGate` become lemmas in Proofs, not lock guarantees** (simulator chosen after the release).
   A listed guarantee fixes its simulator before the worker and the witness.
4. **Scope stated:** 'bytes and their order, not times' (timing is the worker's), and the salts' exclusion, in the
   module doc and the claim, until Daniel rules.
5. **Issues 1 (control frames) and 4 (randomness before witness):** REDZK is checking now whether #6961, #6915, #6882
   or ZKFRAME's D3 already fix them (answer ~23:30Z). Don't build L7, L8a, L8b or L13 on them until then.
Meanwhile: fix 3, 4, 9, 10 and 11 (nits), and restate L8a and L13's targets with St by position.

## 22:37Z, proofs coordinator: ZKL8B's counterexample, and the proof route (decision)

ZKL8B (note `lanes/proofs/20261103T2215Z-report-zkl8b.md`) shows L8b false as described. A worker that frees a wrong
word `a != honest_i` at one live algebraic-phase round releases `a + h_i`, while the switched session carries
`honest_i + h_i`. So against the switched session, `RowsOK` at the served values is necessary (#7130's condition is not
an artifact of its proof). With fresh pads the worker never sees, that's a simulator problem, not a leak.
**Decision (a proof route; the guarantee's statement is unchanged):** restate L8a and L8b against an UNSWITCHED
session at the served values: always the drawn pads, with the go/stop split at `Finish`. The rows then enter only
through `Session.Gated` where a file leaves. Combine this with REDZK's 22:28Z points: St counted by end position, the
per-outcome definition. Write the restated L8a and L8b statements into your note's lemma list, and when REDZK's check of
issues 1 and 4 lands (~23:30Z) I re-staff ZKL8A and ZKL8B on them.
