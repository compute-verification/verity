# PoUS (`verity.core.protocols.pous`)

Proof of useful space for model weights. A GPU keeps an encoding `C` of the weights `W` in its memory and decodes `W`
from it on every use, and a timed audit certifies that it still holds more than a `ρ = 18/19` fraction of `C`: in the
ideal-permutation model, answering a random block within the deadline takes the block itself, because rebuilding it
from a smaller state takes more sequential calls than the deadline allows. The definitions, the guarantees and their
proofs are Lean (`Security/Definitions/Pous/`, `Security/Proofs/Pous/`), and so are the verifiers of record: the setup,
timed-audit, run and `p2-audit-wire/v4` checks (`check/`, the Lake package `PousCheck`) and the memory challenger's
verdicts (`verity/core/devices/memory_challenger/lean/`). This README is the outline, the signposts, the Python API and
what Lean doesn't cover yet. The specs it replaces, `PROTOCOL.md` and `p2/PROTOCOL.md` (P2 specification v4), are in the
history at commit `08da379cb780`. `DISCREPANCIES.md` lists where the Lean model and this code differ.

**The main caveat: no result gives security at a concrete hash.** Every guarantee of a chain codec (band, dense, P3)
holds in the ideal-permutation model, and the reference Feistel-SHAKE256 `Π₂` breaks it: a prover that keeps five
post-absorb states of each block's pad call (1.53% of `C`) rebuilds any label in 481 sequential Keccak-f, inside every
deployed deadline (red team, 4 Oct 2026, art:7617c8017e80). So the `feistel-indifferentiability` claim is refuted, and
every conversion of a wall-clock deadline into a count of `Π₂` calls (`call_ns`, the effective `D`, the cliff, the
duration cap) holds in that model only. A result over a monolithic random-oracle `H` holds only in its model too: with
SHAKE's narrow state, Dense regenerates from about 20% of `C` and P3 from about 39%, and indifferentiability does not
carry over to storage games. The streamed wipe needs no permutation, and P2 rests on its own assumptions.

## The protocol, in outline

- **The game.** A trusted encoder `Enc(W) → (pp, C, vk)` runs once; `W` is arbitrary and fixed before any setup coin,
  `Dec(pp, C) = W` is public and bit-exact, `|C| + |pp| ≤ 1.05·|W|` and `λ = 128`. The adversary preprocesses with no
  time limit, keeps at most `⌊ρ·|C|⌋` bits, and answers `k` uniform challenges, each a `(D, Q)`-bounded program within
  the deadline; it must pass with probability at most `δ + ε`, `δ = 1%` and `ε ≤ 2^-128`. That is
  `Meets sch reveal D Q k ε` (`Security/Definitions/Pous/Accounting/Params.lean`) over the audit game `AuditSecure`, and
  `k = 86` is the floor `ρ^k ≤ δ` (`KFloor`). The cost of encoded matmuls and of decoding on every use is not modeled.
- **Help is accounted, not excluded.** No guarantee assumes trusted prover hardware (Daniel, 12 Oct 2026), so nothing
  attests the prover's machine. The game's "no outside help" is read through the help-accounted row of console's
  actual-circuit model, cited under the id `perfect-isolation` (kept because certificates pin it): on PoUS's challenge
  schedule, some challenge tick `t_j` has `mem_X(t_j) + 2^e·h ≥ S` for the audited location `X` and the help `h` it
  received. An accepted audit bounds what `X` held plus what it was helped, never `X`'s memory alone.
- **Parties.** The server, the prover, commits to `W`, encodes `C*`, serves from it and answers. The verifier draws the
  salt after the commitment and builds `vk`. The memory challenger runs the timed rounds: the auditor's device on a
  machine the prover doesn't control at the prover's site, never a process on the prover's node (`trusted-host`). It
  commits its indices in a coin tree before the session, hashes each whole answer after reading its receipt time, and
  signs the session's records once. For the wipe, a network certifier counts the egress.
- **Lifecycle.** The server commits to `W` (`setup.commit`, tagged SHA-512 under `verity-pous/w-commitment/v2`). The
  verifier draws the 24-byte salt (`setup.draw_salt`, through `verity.ml.randomness`; a server-chosen or predictable
  salt would break "`W` before coins"). The server encodes `C*` segment by segment, and the verifier builds `vk` by
  decoding `C*` and checking it against the commitment, the public encoder and the default (6,578 `Π₂` calls, 13 deep,
  per band segment), or by encoding `W` itself. Serving decodes tensors (`layout.decode_tensor`). The audit reveals `k`
  indices one at a time from the memory challenger's own secret, never a public beacon, and accepts only if every block
  matches `vk` within `Δ = 0.5 ms` plus the RTT allowance.
- **What is public.** The verdict; the salted roots of `W` and of `C`; the salt; the setup draw and its profile; the
  openings of the memory challenger's coin tree once the session has closed. The per-round records are witness, in the
  memory challenger's signed session, so a server can't steer what the auditor sees round by round.
- **Segment tags are injective.** Segment `s` is keyed by `tag_s = salt ‖ u64 s`. With a shared domain one stored
  segment answers for every identical one and the audit accepts with probability 1 (`shared_domain_breaks`), so every
  certificate takes `segDom_inj`, `tests/test_pous_protocol.py` checks it for every scheme, and `benchmarks/pous` gates
  every implementation's tags over 448 segments.

## Schemes

| Scheme (`schemes.get`) | Module | Lean model | Certificate |
|---|---|---|---|
| `band-chain/d12/v1`, the default | `band` | `Pous.BandChain.chainBand` | `BandMultiMeetsFamily`: `k = 111`, `D = 5`, any segment count to `2^64` |
| `dense-chain/v1` | `dense` | `Pous.Sponge.chainDense` | `ChainDenseMeets64`, one segment |
| `p3/v1`, research | `scheme`, `params` | `p3Scheme` | none |
| `p2-16448/v4`, `p2-16448/v2`, experimental | `schemes.p2` | `m1p`, `p2ring` | `P2SlackFamilyUncond`, `k = 105` |

- **The band** (Daniel, 28 Sep 2026: the secure-first scheme) has 512 blocks of 64 KiB a segment. Block `v` absorbs its
  `d = 12` newest parents, newest first, and then the pad through the overwrite chain over `Π₂` (`2m + 512` bits), and
  costs `min(v, d) + 1` calls: 6,578 a segment. A 64 KiB `Π₂` call is 4,830 sequential Keccak-f, 241.5 µs at 50 ns
  (`call_ns`), so `D = 5` covers `Δ + RTT < 1.449` ms. **The cliff:** the effective `D` must not exceed `d`, or the gap
  attack (`band_gap_attack`) holds every block but each `(d + 1)`-th and passes with probability 1, so
  `band.check_deployment` refuses a deadline past `13 × 241.5 µs ≈ 3.14` ms at `d = 12`. At fewer rounds the band has
  `BandMultiMeetsFamilyD2` (`k = 97`, under 724.5 µs), `D1` (`k = 93`, under 483 µs) and `D0` (`k = 89`, under
  241.5 µs); an unset `k` stays at the first certificate's.
- **Dense** is the band with every earlier block a parent, 131,328 calls a segment; over a random-oracle `H` it has
  `DenseMeets64`, `SegMeets14` and `SegTagMeets14` (`k = 106`).
- **P3** has no certificate: `P3ConcreteMeets64`, `P3ConcreteMeets14GB` and `P3ConcreteMeets64D12` are proved for Lean's
  chain start `iv(domP3 tag L v)`, and the code's `h_0` is the instantiation's 48-byte header (`DISCREPANCIES.md` P3
  item 16).
- **P2** is the squaring-based scheme: "P2 v4" below.

A scheme implements `encode_segment` and `decode_segment`. Its `protocol.Certificate` is the `Meets` it rests on (`k`,
`rounds`, `work`, `call_ns`, `rho`, `delta`, `eps_log2`, `cliff`, `segments`, the `theorem` `Pous.SecurityProofs.X`
with its open `hypotheses`, and `claims`). `protocol.TimedAudit` refuses a `k` below it, a deadline whose effective `D`
exceeds its `rounds`, and a scheme with no certificate; `research=True` runs a refused audit anyway, short of the cliff,
recording the refusals as the `StorageProfile`'s gaps. An experimental scheme is audited only with `experimental=True`
and is never the default.

## Where each part is defined

Lean is the verifier of record, and the Python is the reference, held to it by the difftests in `check/tests/` and
`verity/core/devices/memory_challenger/lean/tests/`. The model paths are under `Security/Definitions/Pous/`.

| Part | Lean | Python |
|---|---|---|
| The game, `Meets`, `ρ`, `δ` | `Game/`, `Model/Audit.lean`, `Accounting/Params.lean` | `protocol.Certificate` |
| The band, dense and P3 labels | `Model/BandChain.lean`, `Model/Dense.lean`, `Model/P3Chain.lean` | `band`, `dense`, `scheme`, `primitives` |
| The public-encoder setup | `Model/PubEncoder.lean`; `PousCheck.Setup`, `SetupProtocol` (`pous-setup`) | `setup`, `protocol.verifier_key` |
| The timed audit's verdict and the certificate gate | `PousCheck.TimedAudit` (`pous-timed-audit`), `PousCheck.Certificate` | `audit.UnsignedRounds`, `protocol.TimedAudit` |
| The graded audit and its duration cap | `Model/Graded.lean`, `Accounting/GradedMeets.lean`, `Accounting/Rebuild.lean` | `audit.AUDIT_CAP` |
| The continuous audit | `Model/Continuous.lean` (`contAccepts`) | `continuous` |
| The memory challenger's session and verdict | `MemoryChallenger.Verdict`, `MemoryChallenger.Records` | `memory_challenger` |
| The verdict unit `PousVerdict_v1{K, M, LIMIT}` | `PousCheck.Verdict` (`verdictProg`'s gates) | `verdict_circuit` |
| A recorded run, composed | `PousCheck.Run` (`pous-run`), `PousCheck.RunProtocol` | `run.check` |
| P2's parameters, maps and keys | `PousCheck.Params`, `P2`, `Keys`, `Wire.Codec` | `schemes.p2`, `benchmarks/pous/p2_v1/reference.py` |
| The `W` commitment and the verification tree | `PousCheck.Commit`, `Merkle` | `setup.commit`, `verifier` |
| `verity.ml.randomness` v2 and P2's challenges | `PousCheck.Randomness`, `Challenge` | `verity.ml.randomness`, `reference.audit_key` |
| `p2-audit-wire/v4` and its verifier | `PousCheck.Wire`, `Audit` | `benchmarks/pous/p2_v1/wire.py` |
| The streamed-label wipe | `Model/StreamedWipe.lean`; `MemoryChallenger.StreamedWipe` (`streamed-wipe-verdict`) | `streamed_wipe` |
| The period's wipe, `verity/wipe/v0` | `MemoryChallenger.WipeProtocol` (`wipe-protocol`) | `wipe_period`, `wipe_party` |

## Guarantees

The statements are in `Security/Definitions/Pous/Guarantees.lean` and its `Guarantees/` modules, each docstring ending
with its category, and each proved one is `Pous.SecurityProofs.X` in `Security/Proofs/Pous.lean`. The lemmas (`KFloor`,
Theorem 1, the pebbling and sampled-audit results, `N10Refuted`) have the same names in
`Security/Proofs/Pous/Statements.lean`, the properties of the Lean code are in `Security/Properties/Pous/`, and
`lean/README.md` lists every module. The ones a deployment reads:

- **The band**: the family rows above, with no hypothesis, in the ideal-permutation model; `BandChainMeets64` (one
  segment) and `BandMultiMeets14` (448 segments) are instances. With an untrusted server's encoding and the exact setup
  check, `BandPubMeetsFamily`, `BandPubMeetsFamilyD2` and `BandPubMeetsFamilyD1`, through `chainBand_encDec` and
  `pubMeets_exact`.
- **The graded audit** (`BandGradedMeetsFamily`): up to `t` late answers at `(t, k) = (0, 111), (1, 160), (2, 203),
  (3, 243)`, each late round within `(D_late, Q_late)`, `5 ≤ D_late ≤ d` and `D_late ≤ 12`. Its game assumes no
  pipelining, a hard deadline and every answer past `Δ` counted late, and `BackgroundRebuildExcluded` needs the memory
  challenger to abort an audit lasting more than `1,505 · call_ns` (≈ 363.5 ms). Within that cap `W` alone still
  rebuilds a segment's first labels (49% at `d = 5`, 24% at `d = 12`), and a few held runs of `d` labels restart the
  chain: the guarantee excludes only re-encoding a whole segment with no state about `Π₂`.
- **The continuous audit** (`ContinuousAudit`, #954): `n` challenges over time, accepted with at most `f₀` misses.
  `continuous.ContinuousMemoryChallenger` enforces the model's three requirements, which
  `Security/Proofs/Pous/Continuous/Verifier.lean` proves of it; a bare `UnsignedRounds` with a late deadline credits an
  answer still in flight to the next challenge, and is not that challenger.
- **The streamed-label wipe** (`StreamedWipe`, with no named assumption): for every prover of the sequential wipe,
  `Pr[accepted ∧ z survives] ≤ (g/N)^k + k·C(N, g+1)·2^(M+E)/2^(q+(g+1)·512) + Adv_PRF`, with `M` the bits inside the
  boundary, `E` the egress and `q` the prior content's. `StreamedWipeMeets2600GB`: 2.6 TB at `k = 74,000`, `E` plus
  1.075 GB at `2^-40`. `StreamedWipeMeetsFamily`: `k = 2152`, `⌈M/19⌉ + E` at `2^-128` for every fill of a 32 MiB
  segment or more, with `StreamedWipeMeets96GiB` and `StreamedWipeMeets141GB`. `StreamedWipeMeetsReads`: any `k` with
  `(1 − 1/r)^k ≤ 2^-129`, `512·(⌈N/r⌉ − 1) + N + ⌊log₂ k⌋ + 130 + R + E` bits with `R` the memory outside the fill,
  `verity/wipe/v0`'s figure; `StreamedWipeMeets141GB5`, an H200 at `k = 5231` within 5 GB with its driver reserve. Its
  physical premises are the game's shape: no unaccounted tier, the egress certified over the whole wipe (C6), and no
  side channel (C5).
- **P2**: `P2SlackFamilyUncond` (abstract M1 on `[0, p)`, `λ = 128` at `k = 105`, every block count from `2^8` to
  `2^23`) and `P2ErrorSplitB19Uncond` (the ideal term at `2^-129`), both in `Guarantees/P2.lean`; above `2^23` blocks
  the per-period `Pous.Period.SecurityProofs.periodHeldAt`, which rests on the open conjecture `P2RingRoundSecureWide`.
  End to end (`Security/Properties/Pous/EndToEnd.lean`), `RunCheckAccepts` and `RunCertifies` are about the deployed
  code, with no assumption: when the composed check accepts, every one of the 105 recorded answers was the challenged
  block of `C*`, on time on the device's clock, or the run holds a SHA-512 collision or an LMS forgery. `EndToEndRing`
  bounds a `⌊ρ·|C|⌋`-bit server at `1% + 2^-128` in the ring model, from `P2RingAuditSecure` and `CoinCommitHides`.
- **Named assumptions**: `LemmaA` and `B1Prime` (`Assumptions.lean`); `PrimeP16448`, `P2RingAuditSecure`,
  `P2RingReducesToM1p`, `P2RingRoundSecure`, `P2RingRoundSecureWide` and `CoinCommitHides` (`Assumptions/P2.lean`).

## P2 v4

P2 (`p2/v4`) encodes block `j` as `E_j(m) = ρ(σ_K(ρ((m + t_j) mod p)))`, two modular roots around a masked involution,
and decodes it with two signed squarings. Its parameters: `w = 16,448` bits, `p = 2^16448 − 21065`
(`p2-p16448-c21065`), the profile `p2-2054B` (2,054 payload bytes a block), `B = 2^19` blocks in 8 segments of `2^16`,
`k = 105`, `Δ = 0.5 ms`, an RTT allowance of at most 354 µs, `Q = 2^20`, and SHA-512 with its whole output for every
digest. It stays experimental: the P2–M1-SGI instantiation assumption is not accepted, and the XOR-then-reduce
compression attack on it is open.

Its definition is code. `benchmarks/pous/p2_v1/reference.py` is the maps, the per-block keys (counter-mode SHA-512
over a prefix padded to whole 128-byte blocks, `t` by rejection sampling below `p`, never reduced) and every derivation;
`wire.py` beside it is `p2-audit-wire/v4` (magic `P2AU`, version `0x04`) and its reference verifier; `PousCheck` is both
in Lean, and agrees with them byte for byte on every vector (`check/interop.py --all`: 40 transcripts, 0 mismatches).
The vectors are `vectors.json`, `audit-vectors.json` and `audit-negative-vectors.json` beside `reference.py`, which pins
each file's SHA-512 and, by width, each entry's (`ENTRY_SHA512`, which `PousCheck.SetupDifftest` holds too);
`cd benchmarks/pous && uv run python -m p2_v1.vectors` regenerates them and their pins.

- **The registered scheme.** `p2-16448/v4` follows the maps, the values, the profile, the tweak's rejection, `k` and the
  hash, and its keys are the same function, but it runs the generic protocol's setup, so four things differ from the
  wire's: the setup salt (24 bytes from `setup.draw_salt`, not 32 from `verity/pous/p2-setup/v4`), the key prefix (it
  frames the 24-byte salt), the leaf (`0x00 ‖ tag_s ‖ u16le(v) ‖ R`, not `0x00 ‖ salt ‖ u64be(s) ‖ u16be(v) ‖ R`) and
  the challenges (`verity/pous/audit-challenge/v1`, not `verity/pous/p2-audit-challenge/v4`). `PousCheck.Keys` and
  `Merkle` hold both forms. `p2-16448/v2`'s ChaCha8 keys don't conform to F4; `benchmarks/pous`' `p2-gpu` arm measures
  them.
- **SeqRoot** sets `schemes.p2.ROOT_CALL_NS = 854,029`: half the fastest cooperating 16,448-bit root,
  `16,486 × 361.55 ns ÷ 3.490 ≈ 1,708.1 µs`, from the timing red team's modelled Zen 5 floor (361.55 ns a squaring) and
  the 8-core speedup measured on one Zen 4 chiplet (3.490×). `RTT_ALLOWANCE_CAP_NS = 354,000` is the largest whole µs
  with `Δ + cap < ROOT_CALL_NS`, a margin of 2.0001×, and the memory challenger refuses 355 µs and v1's 1 ms. The floor
  is a model and the speedup was measured on an untuned kernel: under the red team's sync model (20–40 ns a squaring,
  4.24× and 5.55×) the margin at the cap is 1.64× and 1.26×. Only 8-core cooperation within one chiplet is measured,
  and ASICs, FPGAs and cross-chiplet cooperation are outside the allowed hardware. Above `2^23` blocks the run protocol
  takes `p2.wide_certificate` (`p2.WIDE_CLAIMS`).
- **The error total** (F3): a prover holding at most `⌊ρ·|C|⌋` bits past `W` and the public setup passes with
  probability at most `1% + 2^-128`, the sum of the ideal-model term (`2^-129`, `P2ErrorSplitB19Uncond`), the
  instantiation term (`2^-130`, P2–M1-SGI, assumed), the challenge-derivation term (`2^-131`, `verity.ml.randomness` as
  good as uniform) and the expander term (0 under P2-EXP-IO, which counter-mode SHA-512 realizes under the named
  assumption SHA-512-CTR/P2-EXP-IO).
- **State accounting.** `S = ⌊(18/19)·B·16,448⌋` bits, measured against the code, not the payload. At each boundary,
  before round 0 and after each response before the next challenge, every non-free bit the responder can reach is
  charged, every copy counted: device and host memory, swap, disks, caches whose contents depend on `C`, other
  processes' and devices' state, any representation of `C`, and responses or decoded blocks kept across answers. The
  exact committed `W` and fixed constants are free, working memory within a round is unrestricted (as in `Prog`), and
  any other bit that reaches the audited location during an audit is help, counted in `h`. The wire enforces none of it.
- **Decisions** (29 Sep 2026, the defaults Daniel deferred to): F1, `p2-2054B` is the one deployment profile; F2, the
  operating point above (the RTT cap v1.1's, 4 Oct; the verifier off the prover's node since 12 Oct); F3, the error
  total; F4, P2-EXP-IO realized by counter-mode SHA-512, with ChaCha8 nonconforming; F5, the pins `P2SlackB19Uncond`,
  `P2SlackFamilyUncond` and `P2ErrorSplitB19Uncond`, now proved in `Security/Proofs/Pous.lean`; F6, `k = 105`; 7, `p`
  under its ECPP certificate, with no Lean-checked primality proof required; 8, P2–M1-SGI not accepted, so P2 stays
  experimental; 9, one hash, SHA-512 (Daniel, 6 Oct 2026).

## Code

Pure Python, stdlib plus `verity`, and slow by design: it is the oracle for GPU kernels. A 32 MiB band segment decodes
in about 25 s on one core, a dense one in about 8 minutes, and a 14.4 MB P3 segment in about 21 s.

~~~python
from verity.core.protocols.pous import audit, band, codec, setup
from verity.core.protocols.pous.layout import Layout, decode_tensor

layout = Layout.of(band.SECURE_FIRST, [("w1", n1), ("w2", n2)])       # tensors -> segments of W
params, prims = layout.params, band.reference(layout.params)
w = layout.pack({"w1": b1, "w2": b2})
commitment = setup.commit(params, w)                                   # server, before any salt exists
salt = setup.draw_salt(verifier_secret, commitment)                    # verifier
C = [params.codec.encode_segment(params, prims, salt, s, g)             # server: C*, per segment
     for s, g in enumerate(codec.split(params, w))]
vk = setup.verifier_setup(params, prims, commitment, salt, encoded=C)   # verifier: decodes C*, checks the commitment
blocks = [b for seg in C for b in seg]
w1 = decode_tensor(layout, prims, salt, "w1", blocks.__getitem__)      # decode one tensor
v = audit.UnsignedRounds(vk, audit.challenge_key(audit_secret, vk))     # k = 111, Δ = 0.5 ms (+ rtt_allowance_ns)
result = audit.run_timed(v, blocks.__getitem__)
~~~

For dense, use `dense.POINT_64K` and `dense.reference`; for P3, `params.OPERATING_POINT` and `primitives.reference`.
The same lifecycle through the generic protocol, which refuses any `k` or deadline its certificate doesn't cover:

~~~python
from verity.core.protocols.pous import protocol, schemes

scheme = schemes.get("band-chain/d12/v1", segments=layout.params.segments)   # or schemes.Band(layout.params)
commitment = protocol.commit(scheme, w)
salt = protocol.draw_salt(verifier_secret, commitment)
C = protocol.encode(scheme, w, salt)                                           # server
vk = protocol.verifier_key(scheme, commitment, salt, w=w)                      # PRIVATE: the verifier encodes
# vk = protocol.verifier_key(scheme, commitment, salt, mode=protocol.PUBLIC, encoded=C)   # the same guarantee
v = protocol.TimedAudit(scheme, vk, protocol.challenge_key(audit_secret, vk), rtt_allowance_ns=rtt)
v.run(protocol.responder(scheme, blocks.__getitem__))
print(v.profile().statement())                                                 # Pous.SecurityProofs.BandMultiMeetsFamily
~~~

On the deployed path the memory challenger (`memory_challenger`) runs the rounds. It commits its indices in a coin tree
(`CoinTree`) before the session, hashes each whole answer after reading its receipt time (`block_digest`, the block's
`sha512/row/v2` digest, which `answer_circuit.py` opens), and signs the session's round records once, under a header
bound to the setup's root of `C*`. A session's id is the auditor's for one audit, and its `SessionLog`, a file of its
own, refuses an id it has used. `verdict` checks a signed session in the clear and publishes only the verdict; it
accepts S1's LMS over SHA-512 at `lms.DEPLOYED` (`memory_challenger.LMS`) and nothing else. `verdict_circuit`'s
`PousVerdict_v1{K, M, LIMIT}` is that verdict as one proof block, its output bit 1 iff at most `M` of the `K` rounds
miss (`PousVerdict.verdictProg_accepts`), and `check/tests/test_verdict_difftest.py` holds its compiled circuit to
`PousCheck.Verdict.prog`.

~~~python
from verity.ml.crypto import lms
from verity.core.protocols.pous import memory_challenger as mc

signer = mc.LmsSigner(key, state_path)  # key = lms.keygen(lms.DEPLOYED, seed); lms.LeafState.create(state_path, ...) once
log = mc.SessionLog(log_path)                     # its own file (mc.SessionLog.create once), never in the key's directory
coins = mc.CoinTree.draw(challenger_secret, session_id, params.blocks, k)          # session_id: the auditor's, for this audit
c = mc.MemoryChallenger(coins, signer, log=log, limit_ns=500_000 + rtt, block_bytes=params.block_bytes,
                        binding=vk.root)                                            # the id used; the setup's root of C*
signed = c.run(blocks.__getitem__)                                                  # witness: never sent in the clear
v = mc.verdict(signed, c.commitment, c.openings(), lambda i: mc.block_digest(blocks[i]), trusted={signer.public_key})
print(v.public())                                                                   # {"accepted": ...}
~~~

A recorded run is checked whole by `run.check(trusted, setup, digests, session, session_id)`: the setup's public
outputs, the memory challenger's table of `C*`'s row digests (the harness's `DIGESTS.bin`) and its signed session,
joined at P2's deployed point, the session under the id the auditor designated and signed under a key of `trusted`, the
verifier's own list of memory-challenger keys, which never comes from the run's record. `pous-run TRUSTED.json
SETUP.json DIGESTS.bin SESSION.json SESSION_ID` (`check/`) is the same check in Lean.

**The streamed-label wipe** (`streamed_wipe`, experimental; Daniel's tier 3, 19 Oct 2026) is a proof of secure erasure:
the memory challenger fills every memory tier of one server with labels it derives itself
(`L_i = SHA-512("verity-pous/streamed-wipe/label/v1\0" ‖ K ‖ u64be i)`, `K` fresh for the server and the wipe), then,
once the fill is complete, reveals `k` indices on live coins one at a time and checks each answer against its own label
under a deadline. The first late or wrong answer ends the session, nothing is retried, and the verdict accepts only
batch size 1, the reveal the guarantee covers. It publishes the verdict and the certified egress in whole MiB
(16 bits). The verdict of record is the Lean's, `MemoryChallenger.StreamedWipe` (`streamed-wipe-verdict`), through
`wipe_verdict.judge`; `streamed_wipe.verdict` is the reference it is difftested against, and runs beside it under
`wipe_verdict.VERDICT = "both"`.

~~~python
from verity.core.protocols.pous import streamed_wipe as sw

p = sw.WipeParams(labels=n, k=sw.DEPLOYED_K, batch=1, limit_ns=limit, tiers=(("hbm", a), ("dram", b), ("nvme", c)))
c = sw.Challenger(secret, server_id, session_id, p, signer, log=log)          # the id claimed; the commitment published
for start, data in c.fill(chunk=1 << 20):                                     # the labels, streamed to the server
    send(start, data)
c.run(lambda indices: ask(indices))                                           # one batch at a time; no retry
signed = c.close(egress_bits=certified)                                        # the network certifier's count
v, sides = wipe_verdict.judge(signed, c.commitment, trusted={signer.public_key})  # wipe_verdict.py, loaded by its path
print(v.public())                                                              # {"accepted": ..., "egress_mib": ...}
~~~

A period's wipe is `wipe_period`: one streamed wipe of the node, the memory challenger a party of its own on
`period.json`'s `challenger_host` (`wipe_party`, which streams the fill and times the reads over a socket), judged as
the wipe's component of onsite Verity, `verity/wipe/v0` (`MemoryChallenger.WipeProtocol`). An accepted wipe publishes `M`
(every declared tier), the reads `k` the memory challenger chose (`wipe_party --reads`, 2152 by default, up to
`MAX_READS = 2^17`; the prover never names them), `r = ratio(k)` and the foreign content it may leave,
`StreamedWipeMeetsReads`' threshold with the egress counted, and `M` (vacuous) otherwise; an unsigned session is
undecided. `wipe_period.leg(period_dir, node, out)` is the wipe's run in the all-protocols period runner.
`wipe_parties` is the same memory challenger and prover as parties of the deployment layer (`verity.core.deploy`), the
reads the challenger's machine gives it: on the same coins, clock readings, key and reads it signs the bytes
`wipe_party.serve` signs. On a GPU, `period.json`'s `wipe` puts the fill in device memory (`wipe_store`, with host
memory as its CPU stand-in) and has the challenger stream it natively (`wipe_stream.c`, built and gated on the
challenger's host by `wipe_stream`); `benchmarks/pous/streamed_wipe/demo.sh` runs it on one H200. Its `foreign_bytes`
makes the prover the cheat that keeps prior content in place of the fill's first labels (one label past the published
figure, which must reject), and its `placement` `simulated` stands a challenger on the prover's node in for one on its
own host. The run's record names what the bound doesn't cover (`uncertified`: the links leaving the node that no
network certifier counts, the node's memory outside the declared tiers, an egress signed `UNCOUNTED`) and carries the
session's window, `t0` and `t1` on the challenger's clock, which a network certifier counts the egress over.

| Module | What it holds |
|---|---|
| `params`, `labels`, `oracle` | P3's `Params` and `OPERATING_POINT`; Lean's bit layout (`codeEquiv`), `transpose`, `tag`; oracle programs with Lean's depth and work |
| `primitives` | `OverwriteChain` (the default `H`); the backends `reference` (Feistel-SHAKE256), `heuristic_arx` (not analysed) and `ideal`; `RandomOracleKey` (Lean's `p3Model'`); the broken `XorSponge` and `MiyaguchiPreneel`, kept for attack replay |
| `codec`, `band`, `dense`, `scheme` | the codec seam and the three codecs; `band.check_deployment` (the cliff) |
| `layout`, `setup`, `verifier` | tensors to segments; the commitment, the salt and both setups; `VerifierKey` and its RFC 6962 SHA-512 tree, each leaf `SHA512(0x00 ‖ tag_s ‖ u16le(v) ‖ R(c_v))` over the block's `sha512/row/v2` digest |
| `audit`, `continuous` | `challenge_key`, `UnsignedRounds`, `run_timed`, `simulate`, `AUDIT_CAP`; `ContinuousMemoryChallenger`, `run_continuous` |
| `memory_challenger`, `run` | the memory challenger, `verdict`, `covering_certificate`; the composed check |
| `protocol`, `schemes` | the generic protocol (`Scheme`, `Certificate`, `TimedAudit`, `StorageProfile`, `claim`, `LOCAL_CLAIMS`); the registry |
| `streamed_wipe`, `wipe_period`, `wipe_party`, `wipe_store`, `wipe_stream` | the streamed-label wipe and its period run |
| `adversary` | holdings (Lean's `ColState`), `schedule`, `answer_prog`, the cryptanalysis's strategies, `measure` |
| `answer_circuit`, `decode_circuit`, `verdict_circuit`, `circuit` | the circuits the proof service proves: an answer is the challenged block by its hiding leaf, a P2 block decodes to its row of `W`, the session's verdict, P2's modular squaring |

The byte-format names a second implementation must match carry their version in their strings: `params.SALT_BITS` and
`TAG_BITS`, `primitives.CAPACITY_BITS`, `HEADER`, `PI_DOMAIN`, `P_DOMAIN` and `EM_KEY_DOMAIN`, `verifier.leaf_hash` and
`node_hash`, `setup.COMMIT_TAG` and `SALT_DOMAIN`, and `audit.CHALLENGE_DOMAIN`.

Tests: `uv run pytest verity/core/protocols/pous` (about 8 s). P3's 64 KB-label segment vector runs with
`VERITY_POUS_SLOW=1`. Regenerate the vectors with `python verity/core/protocols/pous/tests/pous_vectors.py`, the
schemes' with `tests/pous_scheme_vectors.py`, and the Lean vectors (`tests/lean/PousRefCheck.lean`,
`PousBandCheck.lean`, `PousDenseCheck.lean`) with `tests/lean/generate.sh`.

## Not in Lean yet

Each item is memory-accounting's unless it names another owner.

- **A concrete `Π₂`.** The chain codecs' guarantees hold over an ideal permutation, and the reference one is broken
  (the main caveat). The conversion of a deadline into `D` (no ASIC, at least 50 ns a Keccak-f) is outside Lean.
- **The help-accounted row's bridge.** The row is proved at `e = 0` for a location that does no arithmetic modulo `p`,
  with challenges on a fixed schedule (#1860). Its bridge to PoUS's schedule, which sends each challenge when the last
  answer arrives (Daniel, 12 Oct: the bridge is to cover it), and the `2^e` table for other locations are not written.
  Owner: console.
- **P3's certificate**: the code's `h_0` and Lean's chain start differ (`DISCREPANCIES.md` P3 item 16).
- **A graded memory challenger.** `audit.UnsignedRounds` and `protocol.TimedAudit` accept only when every answer is
  correct and on time, the `t = 0` row (`DISCREPANCIES.md` B5); the duration cap is implemented and off by default.
- **The streamed wipe.** Batch sizes above 1 are implemented and timed but uncovered, so the verdict rejects them; they
  need a direct-product compression bound. The physical premises are not yet tied to
  `Definitions.Accounting.Physical`'s `Memory` and `Links`. The inter-node stream and the deadline's inputs (the
  challenger's round trip to a server, the fastest fetch from outside the boundary) are not measured. The guarantees
  read the game's `Accepts`, not `MemoryChallenger.StreamedWipe.verdict`, and the verdict has no circuit for the proof
  service, so it is not yet proved in zero knowledge.
- **One session per audit.** The check's half is Lean (`RunCheckAccepts`: an accepted session carries the designated
  id); the device's half is its `SessionLog` (append-only, fsynced before the first challenge, never rewritten) and that
  its host never restores an earlier copy of it or of the LMS leaf state, which is `trusted-host`. Owner:
  network-accounting, which widens `trusted-host` (Daniel's decision 2).
- **P2's instantiation.** P2–M1-SGI (`sms-m1-instantiation`), `P2RingAuditSecure` and `P2RingRoundSecure` are open
  conjectures, each with its falsification tests (`Security/Proofs/Pous/P2/RoundTests.lean`), and the XOR-then-reduce
  attack is open. `EndToEndRing` reaches the deployed system only through the claims `pous-ring-device`,
  `pous-ring-speed`, `pous-ring-instantiation` and `prf/sha-512`. The error total's additive combination, and the step
  from `P2ErrorSplitB19Uncond`'s `k = 104` to the deployed 105 (one more sequential challenge only lowers acceptance),
  are paper arguments.
- **P2's setup in the generic protocol.** The registered scheme runs the generic setup (the four differences above);
  moving the MVP and its evidence path to the wire's setup is open.
- **P2's numbers.** `p`'s primality is `PrimeP16448`, a named assumption; its record is a 339-step ECPP certificate (CM
  0.4.4's `ecpp-mpi`, three independent verifications; `docs/efficient-crypto/p2-prime-certificate.md` in the POUS
  store). SeqRoot's floor is a model, and the honest RTT was measured with the verifier on the responder's node before
  the 12 Oct ruling moved it off; an off-node measurement is open.
- **Waiting on Daniel** (what a guarantee says): `|C| ≥ |W|` in `Meets` (so `|pp| ≤ 0.05·|W|`), `ε_crypto ≤ 2^-128`,
  and which reveal a graded `Meets` may use, a deadline per answer or a shared one.
- **Pending work.** A GPU codec for dense; native and GPU kernels for encode and decode, and a native responder on a
  high-priority stream; the bulk bandwidth audit that would pin storage to HBM rather than the whole server; and N10′,
  the column-aware restatement of P3's invertible-label bound (`n10prime_core` is open; N10 as first stated is false,
  `N10Refuted`).

## History

The secure-first band at `d = 12`: Daniel, 28 Sep 2026, 01:10Z (POUS `docs/band-vs-dense.md`, superseding the dense
choice). P2's decisions F1–F8, as defaults Daniel deferred to: 29 Sep. P2 v1, frozen:
art:a7a29e84eef9d2584482b7f3e9722af6b579e2984a9633352564c562ce9e2021 (relayed in
note:20261004T1836Z-report-relay-p2-spec-v1-frozen).
v1.1's RTT cap: Daniel, 4 Oct, 21:21Z, from the timing red team (note:20261004T2102Z-report-relay-p2-redteam-timing,
art:23a323f313d2193873c59ee8ab59a0f1197167a0b4c6d008346d771c3e72de57), the review that found v1's 1 ms unsafe
(art:f33b450f2cd48f6cec2057915ae08b2221a4a6060dc2f9fa38055e8e4d74b98c) and SeqRoot's measurement (run
r20260929-060834-0ec4, art:b92b6d56fa12cc8e04b3f6843991c2e76f49366adc89140a96c33842f70b41ed). The reference `Π₂` broken:
the red team, 4 Oct (art:7617c8017e80, #1122). P2 v4, one hash: Daniel, 6 Oct, 15:07Z; `trusted-host` for the memory
challenger's host the same day. Four errata from the Lean port of the timed-audit check: Daniel, 7 Oct, 21:39Z. The
interop of `wire.py` and `PousCheck`: run r20261008-174759-2f4d,
art:6e51b5bb3d65ea9bf034a704c5c810717db5fc24dbff20222f34c4b8dd818bf2. P2's keys on an RTX PRO 6000 (sm_120), 1.136·10^8
blocks/s: run r20261008-152508-ff34, art:07d65cea4534caaa8f92e364455fd4b998632b4e020302ce6f38525a50617dfa. The duration
cap: #953; the continuous audit's stale answers: #954. No trusted prover hardware, and the memory challenger off the
prover's node: Daniel, 12 Oct. The streamed wipe: Daniel's tier 3, 19 Oct, measured in run r20261019-061002-ee6b (labels
at 7.05·10⁹ a second on one GPU). The specs this README replaces, `PROTOCOL.md` and `p2/PROTOCOL.md`, are at commit
`08da379cb780`.
