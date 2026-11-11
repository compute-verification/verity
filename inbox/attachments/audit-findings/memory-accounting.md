From proofs' worker PRF-537 AUDIT-RESOURCE, Tue 10 Nov 5:10 PM PST (01:10Z). The tool signs a worker's comment with this team's name; I am a worker of proofs, not of this team. **The rows proofs' audit (PRF-523) has for memory accounting, for your table on this issue. Each is a difference between a quoted number and the number at a prover of $2^{80}$ SHA-512 compressions.**

**Why.** Daniel ordered an audit of every concrete-security number we quote (PRF-523); the bar is $2^{-100}$ at $T = 2^{80}$. These merge my rows (PRF-537), AUDIT-CRYPTO's (PRF-536) and one from vllm's lead (PRF-523, 00:15Z). Read at `2b869ed35db5`. No second reader has checked them. Where your own number differs, yours is the one to state, with why.

**Done when**, for each row: the quote is corrected, or the bound is improved, or a reply says our reading is wrong, with why. A first reply is asked for by Wed 11 Nov 6 PM PST.

**Paths.** A file named without a directory is under `verity/core/protocols/remote_bandwidth/`; `Guarantees/…` and `Assumptions/…` are under `Security/Definitions/Accounting/`, and in row 4 under `Security/Definitions/Pous/`.

**Open decisions these rows depend on (Daniel's, none decided):** whether a premise and $T$ count SHA-512 calls or compressions (the sketch's $Q$ already counts compressions); the random-oracle ruling (`SketchHashRandom`, `AuditRO`).

**1. The sketch's term at 16 rows.**
- Quoted: "At 16 rows it costs a session's line 0.4 bits of error (2^-29.95 to 2^-29.55 at 2118 passes), since AuditRO's 2^-41 a pass dominates it" (`verity/core/protocols/remote_bandwidth/timing.py:57-64`); $2^{-30.19}$ for 10 minutes (`PROTOCOL.md:660-670`).
- Evaluated: $\delta_s(Q) = Q\cdot 2^{-95.12} + (Q+P)^2/2^{257}$ (`timing.py:265-283`): $2^{-15.12}$ at $Q = 2^{80}$, $2^{-31.12}$ at $2^{64}$, $2^{-40.12}$ at $2^{55}$. At the code's own budget (8 GPUs at $7.0\times10^{9}$ compressions a second, times $F = 2^{20}$, "Provisional"): 60 s $2^{-33.51}$, 10 minutes $2^{-30.19}$, a day $2^{-23.02}$, a week $2^{-20.21}$, a month $2^{-18.11}$, a year $2^{-14.51}$. AUDIT-CRYPTO has the same $2^{-15.1}$ and $2^{-31.1}$.
- From: `Accounting.Guarantees.Remote.SketchBoundDeployed` ($Q/2^{96} + (Q+P)(\epsilon + N/2^{512}) + (Q+P)^2/2^{257}$, `Security/Definitions/Accounting/Guarantees/RemoteSketch.lean:74`); the rank tail $2^{-96.25}$ is computed by a script outside Lean (`Assumptions/RateAuditDeployed.lean:38-41`).
- How known: read in Lean and code; evaluated by me with `timing.sketch_failure`; AUDIT-CRYPTO's relayed.
- What would change it: 64 rows, a registered option, reads $2^{-97.0}$ at $2^{80}$ and $2^{-129.0}$ at $2^{64}$, set by the keys' birthday, at $\tau = 1.226$ a pass on the reader (`PROTOCOL.md:668`); AUDIT-CRYPTO adds that a 64-byte sign key then reads $2^{-135.9}$ at $2^{80}$.

**2. The published `eps` and the sketch's term.**
- Quoted: "δ + Σ draws/2^64 … not the sketch's term at units" (`remote_bandwidth/p2_party.py:93-94`); `eps` $= (2^{24} + S)/2^{64}$ (`period.py:24`); `monitoring.py:3` pools that `eps`.
- Evaluated: `eps` is about $2^{-40}$ a session, beside $\delta_s = 2^{-30.19}$ for the same 10 minutes: 10 bits apart. The one test evaluates at $F = 1$ ($2^{-43.02}$ a node-day, `tests/test_sketch_reading.py:113-140`), not at $F = 2^{20}$.
- How known: read in code; computed by me.
- What would change it: `eps` carrying $\delta_s$, or text saying it is the draw's term alone.

**3. AuditRO's slack: two values in the tree.**
- Quoted: "`q ≤ 2^40`, `λ = 40·k + 41` makes it at most `2^-41`" (`Security/Definitions/Accounting/Guarantees/Bandwidth.lean:41-42`); "AuditRO's 2^{-41} a pass dominates its δ" (`PROTOCOL.md:666, 684`; `timing.py:63`).
- Evaluated: the code runs `QUERIES_LOG2 = 64`, `SLACK_LOG2 = 67` (`audit.py:128-135`), so the term is $2^{-67}$ a pass: $2^{-50.6}$ over an hour ($87{,}379$ passes of 41.2 ms), $2^{-37.5}$ over a year ($2^{29.5}$ passes) (mine); $2^{-52.1}$ over 10 minutes at 50 passes a second (AUDIT-CRYPTO). The sketch's term is then the session line's largest, by about 22 bits. Above the cap the bound is void: $2^{80}$ calls in one pass give about $2^{16k - 67}$, above 1 from $k = 5$ receipts. One node makes $2^{31.1}$ compressions in a pass ($2^{51.1}$ with $F = 2^{20}$).
- From: `Bandwidth.SecurityProofs.AuditRO` ($((k-1)/N)^s + s/2^h + \binom{q}{k}/2^\lambda$, `Bandwidth.lean:51-55`); `RemoteSessionConservationDeployed` takes `q ≤ 2^55`, `G ≤ 2^55` as hypotheses (`RateAuditDeployed.lean:121-135`), and its query term is "`n·P·2^-41`" (`:404-406`).
- How known: read in Lean and code; computed by me and by AUDIT-CRYPTO (relayed).
- What would change it: one value in both places, and which budget (a pass, a session, a period) $T$ is read against.

**4. PoUS: the deployed guarantee's size and rounds, and the demo's.**
- Quoted: `PeriodHeldDeployed` at 105 rounds and $2^{19}$ blocks, $1\% + 2^{-128}$ an audit (`Security/Properties/Pous/PeriodDeployed.lean:53-84`); 320 challenges, $\epsilon = 2^{-20}$ (`examples/onsite_served/README.md:47`); "Nobody has checked the `ε₀` a proof would give against `N` above `2^25`" (`Security/Definitions/Pous/Assumptions/P2.lean:37-39`).
- Evaluated: the demo's store is $1038\cdot 2^{16} = 2^{26.02}$ blocks (`Guarantees/Obligation.lean:131-145`). $0.957^{105} = 0.0099$ and $0.957^{320} = 2^{-20.29}$; $2^{-100}$ takes about $1{,}577$ rounds at $p = 0.043$. A year of daily audits by union: 365% at 1%, $2^{-11.5}$ at $2^{-20}$. $T$ does not enter: a collision or forgery is excluded from the event, and $\epsilon_{max} = 2^{-128}$ is "**Provisional**" (`Accounting/Params.lean:20-33`).
- From: `periodHeldDeployed`, `ObligationWide320` (`Obligation.lean:211-216`).
- How known: read in Lean; computed by me.
- What would change it: a deployed instance at the demo's size and rounds; a period statement at a week, a month and a year.

**5. The draw's rejection term, and the terms with no bound.**
- Quoted: $\delta_s + (\delta_c + \sum_t k_t/2^{64} + \sum Adv_{PRF} + binding)$ (`Guarantees/RemoteDraw.lean:106, 132-163`).
- Evaluated: $\sum k_t/2^{64} = 2^{-49.76}$ for a session of 7 windows of 2,773 openings, $2^{-27.7}$ over a year of such sessions. `RemoteBinding.lean:40-80`'s `split_t` is "not bounded"; `ledger.binding_over` reads $2^{-153.0}$ to $2^{-135.0}$ at $q = 2^{80}$ without it. `KeyBinding`'s $\delta$ is free (`Stacking.lean:193-194, 345-346`).
- How known: read; computed by me.

**6. Rates that act as security parameters.**
- `SPEC_BYTES_PER_S` $= 8 \times 1.597\times10^{12} + 819.2\times10^{9} = 13.5952\times10^{12}$ B/s (`timing.py:68, 74, 79`). The host term is marked `placeholder` (`verity/hardware/cards/xeon-6776p-2s.toml:29, 31`): at 8800 MT/s the host is $1.1264\times10^{12}$ and the node $13.9024\times10^{12}$ B/s, 2.3% above the headline (relayed from vllm's lead; the lines read by me).
- `SquaringCeiling` is "about 1.485× the reference kernel's rate" (`Assumptions/RateAuditDeployed.lean:73-77`); `SquaringFloor` and `ChainSum` are claimed only at $\epsilon \ge 2^{-64}$ (`Assumptions/SquaringChain.lean:53`); PoUS's deadline margin is $1.995$ ($1.64$ and $1.25$ under the sync model, `Security/Properties/Pous/EndToEnd.lean:335, 350`).
- What would change it: each rate's source (data sheet, measurement) stated beside its margin.

**7. From AUDIT-CRYPTO (relayed).**
- PoUB's sign keys are 256 bits, where a prover can search for a collision: $q^2/2^{257} = 2^{-97}$ at $2^{80}$ ("unless two seeds' 256-bit sign keys collide", `PROTOCOL.md:637`; `coins.py:9, 61`; `receipts.py:22, 96`).
- PoUS's default band permutation (Feistel over SHAKE256, Keccak-f, an Even-Mansour ARX: `verity/core/protocols/pous/band.py:21, 93, 116, 142`; `primitives.py:358-368, 617-629`) is beside AGENTS.md's two named exceptions to "One hash".

**Pointers:** PRF-537 (final 00:15Z, addendum 00:17Z), `note:proofs/20261111T0014Z-finding-resource-audits-concrete-security`; PRF-536, `note:proofs/20261111T0013Z-finding-audit-crypto-primitives-at-2-80`.
