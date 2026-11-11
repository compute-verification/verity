# WIRES: two checks in the two-stage verifier, as one small pull request today (verifier code: Lean of record, Python mirror, difftest)

**The checks** (both only refuse more, so they land ahead of any proof):
1. **No wire reads an output the replay block returns.** Such an output is committed in the boundary record; today the second stage neither ties a unit's read of it to that record nor refuses it.
2. **A wire's writer's part is before its reader's part.** Today the design check accepts two templates wired to each other (`main`'s own difftest case `test_round_two_holds_a_wired_port_to_its_writers_output` has it and expects no refusal).

**The diff exists.** In `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, the entry `## EXTFORM (PRF-528), Wed 11 Nov 14:44Z`, item 3, carries both lines as one diff of `verity/protocols/spot/verifier/lean/Spot/SecondStage.lean` (`posOf`, `wiresOk`, `WIRES_WHY`) with the Python mirror and the changed test expectation; its runs: Lean `r20261111-135838-f64d`, difftest `r20261111-140533-f5a2` and `r20261111-143153-777d`, quick tier `r20261111-141708-d305`. Its commits were never pushed, so re-apply the diff from the entry on `main`.

**Also in this pull request if it is as small as it reads, else a second one today:** Linear issue PRF-559, "the linked read at a root the boundary record does not hold" (the toy is `tools/` in the notes of `ATTACK`, entry 13:13Z and note `lanes/proofs/20261111T0826Z-report-attack-second-reading.md`: cases A and B pass and should not). Refuse it, with a difftest case that fails on `main`.

**And one answer I owe the vLLM lane at 20:00Z (PRF-477):** in your first entry, write in five plain lines what the vLLM lane's second-stage command (`two_stage.py second`, pull request #8875, head `5dd58e0400d7`) must do so that a read of an output the replay block returns is accepted: which port, committed in which record, under which name (`returnLinks`, `portOf`, `readOk`'s link case, `linkedRow`; the entry `## TWOPROVE, Wed 11 Nov 08:27Z`, point 2, has a first answer), and what changes for them once check 1 lands.

**Not yours:** worker GATHER extends the wires the verifier derives; do not touch `nodeWires`. Tell GATHER's file ranges apart from yours in your first entry.

**Deadlines (UTC):** 18:50Z first entry (the five lines for PRF-477, and the pull request's plan); 20:00Z pull request open with its runs out; 22:00Z marked (you mark it: it only refuses more; say in the body which accepted shapes it now refuses and that no served record has one, with the run that shows it).
