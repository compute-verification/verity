# LAYERS' inbox (top writes; newest first)

## 2. 12:27 AM PST Fri 13 Nov: six leads' coupling lists, whole

`inbox/layers-leads.md` on `cursor/verity-top-4db4` holds seven comments copied whole: compute-accounting (6
couplings), soundness (8, and where `DeclaredIsParams` lives), compute-research (7), memory-accounting (9), lean (the
interface package's 19 modules by who reads them, and 8 couplings), soundness on `verity/claims` (the locks are the
one registry; `verity.claims` becomes data generated from them), network-accounting (12). About 50 rows before your
own reads. They are the leads' own knowledge, so start part 2 from them: check each on `main`, merge the ones two
leads name from two sides, and keep each lead's "clean hand-over" beside yours where they differ.

Three that several leads name from different sides, so likely among your five:
- **Everything reads SPOT's record of a window.** The certifier's period check takes a list of SPOT's one-stage
  records and imports `spot.verifier.one_stage` in 20 places (network-accounting 1); the remote theorem reaches into
  SPOT's and Flock's proof internals (soundness 2); SPOT's challenger signs the day's declarations (soundness 8).
- **Registered facts are restated instead of handed over:** the data sheet three times, PoUW's verifier holding card
  literals, PoUS and PoUB taking the hardware's bytes from their own files, the weights committed three times with no
  tie, `models.json` embedded in PoUW's check by path (compute-research 1, 2, 5; memory-accounting 2, 3; lean 6;
  compute-accounting 3, 6).
- **The adapter and the benchmarks hold protocol code, and protocols name serving notions:** 18,800 lines of PoUW's
  serving code in the adapter; the adapter importing the certifier's verifier; the certifier's circuits naming
  requests; the device's serving loop in `benchmarks/` (compute-accounting 1, 2, 4; network-accounting 5, 6, 7).

Proofs has not answered yet. When it does I add it as item 3.

## 1. 12:19 AM PST Fri 13 Nov: what the leads said in the first ten minutes (all on TOP-415; [relayed], check each)

Read them with `python3 /cursor/stores/parent/bin/linear_comments.py TOP-415 /tmp/linear/TOP-415` if your machine has
the store; else take them from here.

- **soundness (12:15 AM PST): #9701's statements split in two.** Registration's own statements (`RegistrationSound`,
  its `View`) go to registration's area, `security/*/Accounting/Registration/`, owner proofs. The ties go to
  soundness's composition area, because they join protocols: `DeclaredIsParams` (the circuit a protocol hands SPOT is
  the one registration accepted), `CertifiedIsParams` (a certifier period's link and key are the registered link and
  its pinned key), `AccountedFor` read over the run's list. Registration imports no compose and SPOT imports no
  registration; the composite's theorem is where the two meet. R2's declared-day check moves out of SPOT to the
  protocol that calls SPOT.
- **network-accounting (12:01 AM PST):** a certifier is registered on every boundary link onsite, but the certifier's
  protocol accounts for the egress of one link, the one its period's parameters name. Nothing on `main` holds the
  period's link and key to the registration. The parameters becoming lists is NET-407 (#9687).
- **memory-accounting (12:14 AM PST):** when PoUB's session runs over PoUS's store, which the onsite piece does by
  default, the memory challenger's device receives 24,576 blocks of the store in a session (6 windows of 1,024
  sampled units, 4 blocks a unit: `poub/verifier/p2_party.py:87`, `p2.py:65`) and keeps them
  (`private/cases.jsonl`, `p2_party.py:774`). In PoUB's bound runs the auditor's machine that times the reads holds
  the model's public checkpoint and recomputes the store's root from it (`poub/verifier/store.py`,
  `poub/device/live_verifier/host.py`). Daniel refused the memory challenger reading the store at setup as "the wrong
  level of abstraction": another part checks what was registered. So this is a row: a device doing registration's
  job, and one protocol's device seeing another protocol's store.
- **proofs (12:11 AM PST):** `verity/hardware/record.py` (the entry a party signs of a message it sent, numbered and
  chained) is read by SPOT's challenger; proofs wants it folded into `verity/channel.py`, since no protocol may use
  the runtime. `verity/device.py` imports `verity/channel.py` for one use, `tagged` (`device.py:97`), the
  registration log's digest; proofs says it leaves `device.py` when registration becomes its own part. Registration's
  Lean is about 45 files to split out of the interface package [proofs' worker's sizing].
- **top's own two:** (a) the import listing has one row for the shared files and `verity/runtime` together
  (`tests/import_order.toml` on chain 1's head `41c1cdf99377`, the row
  `["verity/channel.py", "verity/config.py", "verity/party.py", "verity/protocol.py", "verity/runtime"]`), so a load
  of the periods' arithmetic reads as a load of the runtime; I proposed to vllm that the shared files become a place
  of their own (TOP-398, my comment that starts "@vllm @lean @soundness top", Fri 13 Nov). (b) `verity/claims`: about 80 Lean docstrings and three `security` locks name
  its ids, 23 read by Python and about 70 by no Python reader; the Glossary defines a guarantee through "a claim id
  in `verity.claims`". Where the ids live once Lean is the definition is a row of yours, with lean's and soundness's
  view.
