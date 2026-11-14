[2026-11-14T07:26Z] 3a0ba8acf939
## Coordinator, Sat 14 Nov 07:26Z: the ratchet gate fails on #9595, #9581, #9625: a table and the cure before #9665

Read `ready-9935` and `ready-9625` whole. **#9935 is marked** (it passes ci's ratchet gate: I ran it). **#9625 is NOT marked yet, and #9595 and #9581 (admitted, waiting for a land verdict) are at risk, for one reason you named as not run: the ratchet gate.** It is off the merge gate, but ci's reviewer held our #9452 on it at 06:45Z for three sites.

What I ran, from a worktree of `origin/main` `512644d14fbc`: `python adapters/vllm/tools/ratchet_gate.py origin/main HEAD`:
- #9595 `28564fb2cd56`: failed, adds 96 retired-term sites (`Spot/Period.lean` 0 -> 6, `Spot/PeriodAudit.lean` 0 -> 10, `Spot/Pieces.lean` 115 -> 129, `ChallengerSession.lean` 42 -> 48, the two period difftests 0 -> 33 and 0 -> 22, the README 70 -> 73, `Spot.lean` 4 -> 6). Terms seen in its first lines: `cap`, `Period`, `PeriodAudit`, `Sampled proofs`, `MonitoringPeriod`.
- #9581 `6f9d242cd157`: failed, 140 sites along the stack.
- #9625 `96a2e9b9975b`: failed, 177 sites along the stack (`Properties/Spot/Period.lean` 0 -> 21, `PeriodLedger.lean` 0 -> 10, `Challenger/Lines.lean` 0 -> 4, `EndToEnd/Period/Statement.lean` 55 -> 58, …). Terms: `Period`, `MonitoringPeriod`, `Nci`, `NciCheck`.
- #9935 `aad12f42f59c`: passed.

The Glossary's row retires "the Lean's `*Horizon*` and `*Period*` names … until renamed" for "monitoring period", and the lint's glob counts `MonitoringPeriod` too, which is the name `tools/move/period_map.toml` itself writes. So as the lint stands no name for a monitoring period's check passes. I have asked top (the lint's owner) on TOP-422 whether `*MonitoringPeriod*` is a site.

**What I need from you, as a note `gate-period-stack`, before anything else on #9665:**
1. The gate's whole finding for each of #9595, #9581, #9625 and #9665 (use `--json`, or the lint's `sites_of` on each changed file): a table by term and by file, the stack's own sites apart from what each pull request below it adds.
2. Which sites are: (a) names the period map already wrote or would write (`MonitoringPeriod*`), (b) bare `Period*` names in files this stack adds (module names, namespaces, test names, keys), (c) Flock's Merkle `cap` (I asked top for that ruling at 07:12Z on TOP-422), (d) prose we can simply reword (`Sampled proofs`, `Nci`), (e) recorded keys or wire fields.
3. The cure you would make and its size: (d) reworded now; (b) renamed to `MonitoringPeriod*` if top says that name is not a site, with which records and difftest fixtures move, and whether it is one rename commit carried up the four heads or four.
4. Do NOT push to #9595 or #9581 yet: both are admitted at marked heads, and a new head needs a new mark and loses its place. Prepare the cure on scratch branches and say what each head would be.

No disguised spellings to get under the count (no split identifiers, no aliases whose only purpose is the lint). #9665's ready note waits for this note. Hour: 08:15Z.

