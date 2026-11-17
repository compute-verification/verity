[2026-11-17T18:20Z] d893d2a5d241
COINSRC part 1 (origin/main 342636fbb4a). Kind: (a) refusal blocking a verdict/credit, (b) truthful label only, (c) other reason.

(a) items:
A1 benchmarks/one_stage/served_verdicts.py:1793 pearlplus(): verdict = reject if reasons else accept if live else UNCLAIMED. live = served.coins.source == "challenger" (:1704-1705). In-process epoch coins with nothing failed -> verdict "unclaimed" (withheld). This is what eb71 hit.
A2 served_verdicts.py:2165-2177 piece_row(): only a Pearl+ "accept" reaches pearlplus_of_record; "unclaimed" -> row verdict "missing", why "development ... in-process challenger's ... claims no verdict" (zero credit). Consequence of A1.
A3 served_verdicts.py:2024-2025 pearlplus_of_record cond 1: coin_source != "challenger" -> missing.
A4 :2026-2035 the live session's coin check must pass and its challenger key must be the compose's pin (--key-sha512) -> missing (key/gate).
A5 :2036-2047 the ledger must hold the session's reserve by lane LANE under its own gate key (not "for") -> missing (whose key).
A6 :2049-2054 the session must have joined a Pearl-C period of the lane (gate ledger) -> missing; and :2098-2106 that period's open must commit SHA-512 of circuit-record.json -> missing (no period exists in-process).
A7 :2060 pearlplus_live_sample: the verifier's sample must be a challenger draw session's coin -> missing (node IO.getRandomBytes samples refused).
A8 served_verdicts.py:1333-1334 pouw() (H1 piece): coin_source != challenger -> reasons -> verdict reject. But an H1 piece credits nothing anyway (DEV_WHY :292, record takes no H1 piece: different reason).
A9 verity/protocols/resource_accounting/compute/pouw/prover/serving/h1/__init__.py:707 (IN_PROCESS :126): every call's credit bit false when coins_from != challenger; bits are committed to the manifest, and verifier/credit.py credits only through the challenger's signed session (trusted_session_of, pinned key, DrawWithheld). Lifting needs H1's credit to rest on something other than the challenger's ledger -> LEAVE FOR DANIEL.
A10 Lean verity/.../pouw/verifier/lean/pearlplus/PouwPearlPlus/Coins.lean:120-125 ofRecordReason: under --of-record refuses in-process epoch coins and node samples. In the lock (security/lean-audit.json:47118, security/Properties/lean-audit.json:24518). No caller on main passes --of-record (eb71's Lean ran without it and accepted). -> LEAVE (lock).
A11 served_verdicts.sh:2052-2058 and 2074-2077 preflight: SPOT/V3c local or loopback challenger refused unless DEVELOPMENT=1. Already lifted by the existing setting DEVELOPMENT=1, which zeroes nothing downstream (integrity accepted in eb71 on a local session). Sampled proofs' line, not PoUW's.

(b) labels only:
B1 tools/verity/cli/verity_cli/served_table.py:699-704 coins_of -> "simulated"/"development" on the VERDICT line; :671.
B2 tools/verity/cli/verity_cli/run.py:650-656 coins_on, coins=simulated, simulated premises; :341-350 unmet ("Faithful and Secret unmet"). Verdict untouched.
B3 served_table.py:384-388 _spot_challenger: no gate bound -> sets DEVELOPMENT=1 (enables, does not refuse).
B4 served_verdicts.py:1720-1739 coins/why texts (PEARLPLUS_COINS_IN_PROCESS, PEARLPLUS_WHY_TIED_IN_PROCESS); served_verdicts.sh:2239-2240 and 2256 COINS line ("a development piece, it credits nothing"); :259 H1 development key text; challenger_coins/sampled_line :775-812.
B5 PR heads #10663/#10672/#10698/#10730: coins_on (attester's machine else workload's -> coins simulated) labels.

(c) other reasons, not lifted:
C1 served_verdicts.sh:593-594 preflight: PEARLC_CHALLENGER_PEER set with gate empty refused (config consistency; bench_arm refuses that pair itself). Unset the peer and it runs in-process.
C2 served_verdicts.sh:2059, 2067: V3C_PEER unset / the served node's peer in a deferral: mechanical (the session listens on that peer's 127.0.0.1).
C3 PR heads' served_table Piece.development "unbound" refusals: all-or-none binding consistency, not coin source.
C4 PoUW row "missing ... no registration to tie W to" = pearlplus_of_record :2075-2076 (spot is None: the compose got no --window-tie/--sampled-proofs). A different reason (a missing compose input), reachable only with live coins (an accept).

eb71 (r20261117-063059-eb71, research data show): commit 638fbe10322f35771aef04eb4c4914053f199349, clean (dirty false), source_transport git to vy-spot-glm53. NOT on main: GitHub has no such commit (fetch "not our ref", API 422), so it was pushed only to the node; I cannot say which branch/parent it came from. From the record's local files (piece.log, verdicts.txt): window.sh, POUW=pearlplus with PEARLPLUS_TIE, class 4,27,2 (PEARL-TIE: W 003d87b0..., R d78ae5df... chained right after), Python verify ACCEPT and Lean ACCEPT, 107,318,608,059 MACs credited at eps 0.009983 delta 1/100, coins in-process, draws node IO.getRandomBytes, verdict "unclaimed" -> work row missing. Sampling law work-rate:2/N, NOT the uniform rate:NUM/DEN. Coin record only in the node's store (art:8e852ec9..., preserve failed: R2 not in that shell; pearlplus_coin treats that as no reason). Record composed "pending", deferred to vy-nebius-cpu-14; I did not find that follow-on's run. Could not fetch the remote-only files (development.json etc.): this VM's store remote is unusable.

Plan for part 2 unless you say otherwise: one setting PEARLPLUS_COINS_ANY_SOURCE (served_verdicts.sh/served_pearlplus.sh env, passed to served_verdicts.py pearlplus and record as --coins-any-source), "coins flipped in the serving process are taken for credit": pearlplus() gives accept instead of unclaimed, pearlplus_of_record skips A3-A7 (and the period-circuit part of A6) while still requiring the W/R tie, the records audit accepted under R's rule, and the Lean accept with credit at the reference eps; the row keeps coin_source/coins/why stating in-process truthfully, plus a "coins_taken" field naming the setting. Not touching A9, A10 (Daniel).

