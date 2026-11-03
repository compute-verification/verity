
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
