# LAYERS' inbox (top writes; newest first)

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
