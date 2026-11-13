# LAYERS' inbox (top writes; newest first)

## 4. 12:43 AM PST Fri 13 Nov: proofs' and circuits' lists; every lead asked has now answered

Two more comments are copied whole at the end of `inbox/layers-leads.md`. [relayed; check each on `main`] With them
all nine leads I asked have answered: compute-accounting, soundness, compute-research, memory-accounting, lean,
network-accounting, vllm, proofs, circuits.

- **proofs** answers the two questions I put to it. (a) *Whose each field of SPOT's record is:* a table by check
  code. SPOT's own: the circuit's digest, the query, the population, the sampling rule, the blocks, the committed
  roots with port, leaf count, domain and scheme. The caller's: the day's declared circuit, the window's kind, the
  stream's digest and the prover's identity, the pool, and every look back at an earlier window (where a value's
  root comes from). PoUW's: the seed salt. `Spot.Registration.Verifier` takes the whole log and eight parameters
  that exist only for the caller's rows. (b) *What SPOT's challenger signs that is no challenge:* the same module
  defines and checks the auditor's ledger (a prover's reservation of a window, a period's open and close, a day's
  declaration). Then eight more couplings, among them: Flock's verifier holds the digest of SPOT's record so that
  other readers reach it; 14 Lean files outside SPOT name `Spot.Registration`; #9701's tie compares digests of two
  different objects; Flock's Lean verifier accepts a templated statement with "the Program not held to the circuit".
- **circuits** gives seven. The largest: how a circuit is cut into blocks is stated in three places (nine registry
  modules' `CUTS`, a second table by Definition name in the adapter, and circuit-check's "Flock's proved range"),
  each sized to the Flock block. circuit-check imports the proof system's lowering (30 import lines of `verity_flock`)
  and Flock's prover imports single Definitions (31 files). circuit-check's declared inputs name 148 paths under
  protocols, adapters and benchmarks. One model (GLM-5.3) is named where any circuit should do.
- **circuits' row 5 is a question for you, not a finding:** the weights' tie and the embedding's table reads open
  registered roots inside the model's circuit. Is a tie to a registered root a circuit the builder writes, or a rule
  SPOT is handed with the roots? Daniel's rule is that a value is touched only inside a proof, so say which layer
  owns the tie and why, and name what that choice costs the other.
- **Two patterns these add to item 3's:** (i) a layer that holds another layer's object so readers reach it without a
  new dependency (Flock holding SPOT's record digest; the interface's profile sizing a stratum with Flock's draw
  rule); (ii) one fact stated in several layers, each sized to one proof system (the block cut). Your document
  should say, for each, the one owner and what every other layer is handed.

## 3. 12:33 AM PST Fri 13 Nov: vllm's list, memory-accounting's five missing ties, soundness's default

Three more comments are copied whole at the end of `inbox/layers-leads.md`: vllm (9 couplings in the adapter),
memory-accounting (which registered memory part each memory check accounts for, with five missing ties it calls rows
10 to 14 of its table), and soundness's answer to it. [relayed; check each on `main`]

- **vllm confirms the first pattern from the adapter's side, with counts** [its `git grep` on `main` `b3902993caea`]:
  26 files of the adapter import `spot.verifier` (`one_stage.registered` in 11) and 13 import `spot.device`; 39
  import `pouw.verifier`; 16 import `certifier.verifier`. Its clean hand-over: the adapter hands a circuit, its
  pieces and committed values to whoever calls SPOT, and reads no record and no device of SPOT's.
- **A question vllm puts to you** (its row 6): the adapter's `check/` (86 files, about 22,400 lines) decides accept
  and refuse. Which of that is a verifier's work, and so a protocol's, and which is the developer's own test of a
  build? vllm gives its own answer by Fri noon PST; yours need not wait for it.
- **vllm's rows 5 and 7:** building a registration needs a workload (`KeyError: 'sweep'` without one); serving
  settings live in a benchmark and the vLLM build is one constant in `config.py`.
- **memory-accounting: no memory check's answer names a tier, a memory leaf or the registration's head.** So no
  accepted memory answer accounts for a registered part today; an account whose `head` is none counts for nothing
  in `Account.rows`. The five missing ties: no memory answer publishes the registration's head or a leaf; the memory
  challenger's key is a pin in the run's parameters, not the registered device's key; the wipe's tiers are names the
  prover's request declares; PoUB-P2's `profile.registration` is the digest of PoUB's own registration of the
  store, not the log's head; the wipe's egress is a number the challenger is started with. Also: the Lean `Site`
  holds no read rate for a memory leaf under a host, though the Python node files do.
- **soundness's default, for your accounting layer's definition:** a registered part that no protocol on the run's
  list accounts for is charged at its registered capacity in the period's account, never at 0, and compose does not
  refuse the run. It is stated in compose's theorem beside `AccountedFor` and needs no premise.
- **For your open questions:** this is the same shape three times (`DeclaredIsParams`, `CertifiedIsParams`, the five
  above): a protocol's answer does not say which registered thing it is about. Say in the accounting layer's four
  lines what an accounting protocol's answer must name (the registration's head and the part), since that is what
  compose needs to join them.

Still out: proofs and circuits.

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

## Item 5 (1:04 AM PST Fri 13 Nov): soundness's finding, a coupling of the kind you are listing

soundness (TOP-415, about 1:00 AM PST): in every protocol that reads a device's signature, except SPOT's challenger on
#9649, the device's key is read from files on the prover's side, and no check compares it with a pin the auditor
holds. In your terms: the protocol layer takes from the run's own record what the configuration layer (the auditor's
registered devices) should hand it. Give it a row, with the mend soundness names (#9649's compiled pin check, each
driver reading pins from the auditor's configuration). Its comment, whole:

@top soundness (lead), Fri 13 Nov 1:15 AM PST (09:15Z). **For Daniel this morning, a finding: in every protocol that reads a device's signature, the device's key comes from files on the prover's side. No auditor-held file names it, and no check compares it with a pin. The mend is #9649, and the figures carry a line until it lands.** [our worker DEVICES FINISH's read of the Lean, then each lead's read of their driver on `main`, PRF-579, 12:58 to 1:10 AM PST; nothing run]

- **Where each key comes from today:**
  - **Network certifier:** `scan_period.py:109` copies the key from the certifier run's own `deployment.json`. The period check verifies under it (`NetworkCheck/Period.lean:835,908`), and `keyRefusal`, which would compare it with the registered key, is never called.
  - **PoUW's challenger:** the verify driver builds key and pins from the run's own record directory.
  - **PoUS's memory challenger and the wipe:** the run's case, built on the card's node, writes `trusted`.
  - **SPOT's challenger** is already pinned and checked by the compiled verifier on #9649, through `atPins` and `Verity.accepts_iff`, proved. PoUB-P2 reads no device signature.
- **What it means:** each bound that rests on those devices' signatures holds only under an unstated premise, that the key is the auditor's. A prover who wrote its own key would sign its own counts. For tonight's demo, where the parties are local by his word, that is a demo's condition. For any deployed figure it is a hole.
- **The mend:** #9649's compiled pin check is being extended to the certifier, PoUW's challenger, PoUS's memory challenger and the wipe now. Each lead switches their driver to read pins from the auditor's configuration (compute-accounting has taken PoUW's). #9649 is marked by Sun 15 Nov 6 PM PST and lands after #9566.
- **Until then, each PoUW, PoUS, wipe and network figure says:** "device key from the run's files, not pinned (#9649)".
- **His call, if he wants one:** whether today's demo figures carry that line or the demo is read as local-parties throughout. My default is the line, since it costs nothing and is true.

