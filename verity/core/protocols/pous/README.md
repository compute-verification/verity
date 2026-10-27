# PoUS (`verity.core.protocols.pous`)

Proof of useful space for model weights. A GPU keeps an encoding `C` of the weights `W` in its memory and decodes `W`
from it on every use, and a timed audit certifies that it still holds more than a `ρ = 18/19` fraction of `C`: in the
ideal-permutation model, answering a random block within the deadline takes the block itself, because rebuilding it
from a smaller state takes more sequential calls than the deadline allows. The definitions, the guarantees and their
proofs are Lean (`Security/Definitions/Pous/`, `Security/Proofs/Pous/`), and so are the verifiers of record: the setup,
timed-audit, run and `p2-audit-wire/v4` checks (`check/`, the Lake package `pous_check`, library `PousCheck`) and the
memory challenger's verdicts (`verity/core/devices/memory_challenger/lean/`). This README is the outline, the signposts
and the Python API. `DISCREPANCIES.md` lists where the Lean model and this code differ.

The Python here is the reference and the oracle for GPU kernels: pure Python, stdlib plus `verity`, and slow by design
(a 32 MiB band segment decodes in about 25 s on one core, a dense one in about 8 minutes, and a 14.4 MB P3 segment in
about 21 s).

**The main caveat: no result gives security at a concrete hash.** Every guarantee of a chain codec (band, dense, P3)
holds in the ideal-permutation model, and the reference Feistel-SHAKE256 `Π₂` breaks it: a prover that keeps five
post-absorb states of each block's pad call (1.53% of `C`) rebuilds any block in 481 sequential Keccak-f, inside every
deployed deadline. So the `feistel-indifferentiability` claim is refuted, and every conversion of a wall-clock deadline
into a count of `Π₂` calls (`call_ns`, the effective `D`, the cliff, the duration cap) holds in that model only. A result
over a monolithic random-oracle `H` holds only in its model too: with SHAKE's narrow state, dense regenerates from about
20% of `C` and P3 from about 39%, and indifferentiability does not carry over to storage games. The streamed wipe needs
no permutation, and P2 rests on its own assumptions.

## The protocol, in outline

- **The game.** A trusted encoder `Enc(W) → (pp, C, vk)` runs once; `W` is arbitrary and fixed before any setup coin,
  `Dec(pp, C) = W` is public and bit-exact, `|C| + |pp| ≤ 1.05·|W|` and `λ = 128`. The adversary preprocesses with no
  time limit, keeps at most `⌊ρ·|C|⌋` bits, and answers `k` uniform challenges, each a `(D, Q)`-bounded program within
  the deadline; it must pass with probability at most `δ + ε`, `δ = 1%` and `ε ≤ 2^-128`. That is
  `Meets sch reveal D Q k ε` (`Security/Definitions/Pous/Accounting/Params.lean`) over the audit game `AuditSecure`, and
  `k = 86` is the floor `ρ^k ≤ δ` (`KFloor`). The cost of encoded matmuls and of decoding on every use is not modeled.
- **Help is accounted, not excluded.** No guarantee assumes trusted prover hardware, so nothing attests the prover's
  machine. The game's "no outside help" is read through the help-accounted row of console's actual-circuit model, cited
  under the id `perfect-isolation` (kept because certificates pin it): on PoUS's challenge schedule, some challenge tick
  `t_j` has `mem_X(t_j) + 2^e·h ≥ S` for the audited location `X` and the help `h` it received. An accepted audit bounds
  what `X` held plus what it was helped, never `X`'s memory alone.
- **Parties.** The server, the prover, commits to `W`, encodes `C*`, serves from it and answers. The verifier draws the
  salt after the commitment and builds `vk`. The memory challenger runs the timed rounds: the auditor's device on a
  machine the prover doesn't control at the prover's site, never a process on the prover's node (`trusted-host`). It
  commits its indices in a coin tree it keeps on the device, hashes each whole answer after reading its receipt time,
  signs the session's records once, and only then releases the coin tree's root and openings. For the wipe, a network
  certifier counts the egress.
- **Lifecycle.** The server commits to `W` (`setup.commit`, tagged SHA-512 under `verity-pous/w-commitment/v2`). The
  verifier draws the 24-byte salt (`setup.draw_salt`, through `verity.ml.randomness`; a server-chosen or predictable
  salt would break "`W` before coins"). The server encodes `C*` segment by segment, and the verifier builds `vk` by
  decoding `C*` and checking it against the commitment (the public encoder, the default: 6,578 `Π₂` calls, 13 deep, per
  band segment), or by encoding `W` itself. Serving decodes tensors (`layout.decode_tensor`). The audit reveals `k`
  indices one at a time from the memory challenger's own secret, never a public beacon, and accepts only if every block
  matches `vk` within `Δ = 0.5 ms` plus the RTT allowance.
- **What is public.** The verdict; the salted roots of `W` and of `C`; the salt; the setup draw and its profile; the
  memory challenger's session commitment (its coin tree's root and key) and its openings, released once the session is
  signed. The per-round records are witness, in the memory challenger's signed session, so a server can't steer what
  the auditor sees round by round.
- **Segment tags are injective.** Segment `s` is keyed by `tag_s = salt ‖ u64 s`. With a shared domain one stored
  segment answers for every identical one and the audit accepts with probability 1 (`shared_domain_breaks`), so every
  certificate takes `segDom_inj`, `tests/test_pous_protocol.py` checks it for every scheme, and `benchmarks/pous` gates
  every implementation's tags over 448 segments.

## Schemes

| Scheme (`schemes.get`) | Module | Lean model | Certificate |
|---|---|---|---|
| `band-chain/d12/v1`, the default | `band` | `Pous.BandChain.chainBand` | `BandMultiMeetsFamily`: `k = 111`, `D = 5`, any segment count to `2^64` |
| `dense-chain/v1` | `dense` | `Pous.Sponge.chainDense` | `ChainDenseMeets64`, one segment |
| `p3/v1`, research | `scheme`, `params` | `p3SchemeC` (`Model/P3Chain.lean`) | none |
| `p2-16448/v4`, `p2-16448/v2`, experimental | `schemes.p2` | `Guarantees/P2.lean` | `P2SlackFamilyUncond`, `k = 105` |

- **The band**, the secure-first scheme, has 512 blocks of 64 KiB a segment. Block `v` absorbs its `d = 12` newest
  parents, newest first, and then the pad through the overwrite chain over `Π₂` (`2m + 512` bits), and costs
  `min(v, d) + 1` calls: 6,578 a segment. A 64 KiB `Π₂` call is 4,830 sequential Keccak-f, 241.5 µs at 50 ns
  (`call_ns`), so `D = 5` covers `Δ + RTT < 1.449` ms. **The cliff:** the effective `D` must not exceed `d`, or the gap
  attack (`band_gap_attack`) holds every block but each `(d + 1)`-th and passes with probability 1, so
  `band.check_deployment` refuses a deadline past `13 × 241.5 µs ≈ 3.14` ms at `d = 12`. At fewer rounds the band has
  `BandMultiMeetsFamilyD2` (`k = 97`), `D1` (`k = 93`) and `D0` (`k = 89`).
- **Dense** is the band with every earlier block a parent, 131,328 calls a segment; over a random-oracle `H` it has
  `DenseMeets64`, `SegMeets14` and `SegTagMeets14` (`k = 106`).
- **P3** has no certificate: `P3ConcreteMeets64`, `P3ConcreteMeets14GB` and `P3ConcreteMeets64D12` are proved for Lean's
  chain start `iv(domP3 tag L v)`, and the code's `h_0` is the instantiation's 48-byte header (`DISCREPANCIES.md` P3
  item 16).
- **P2** is the squaring-based scheme, proved in abstract M1 with its instantiation assumption not accepted: "P2" below.
  `p2-16448/v4` has counter-mode SHA-512 keys, `p2-16448/v2` ChaCha8 keys, which v4 does not allow.

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
| The band, dense and P3 blocks | `Model/BandChain.lean`, `Model/Dense.lean`, `Model/P3Chain.lean` | `band`, `dense`, `scheme`, `primitives` |
| The public-encoder setup | `Model/PubEncoder.lean`; `PousCheck.Setup`, `SetupProtocol` (`pous-setup`) | `setup`, `protocol.verifier_key` |
| The timed audit's verdict and the certificate gate | `PousCheck.TimedAudit` (`pous-timed-audit`), `PousCheck.Certificate` | `audit.UnsignedRounds`, `protocol.TimedAudit` |
| The graded audit and its duration cap | `Model/Graded.lean`, `Accounting/GradedMeets.lean`, `Accounting/Rebuild.lean` | `audit.AUDIT_CAP` |
| The continuous audit | `Model/Continuous.lean` (`contAccepts`) | `continuous` |
| The memory challenger's session and verdict | `MemoryChallenger.Verdict`, `MemoryChallenger.Records` | `memory_challenger` |
| The verdict's proof block `PousVerdict_v1{K, M, LIMIT}` | `PousCheck.Verdict` (`verdictProg`'s gates) | `verdict_circuit` |
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
the pebbling and sampled-audit results) have the same names in `Security/Proofs/Pous/Statements.lean`, the properties of
the Lean code are in `Security/Properties/Pous/`, and `lean/README.md` lists every module. The ones a deployment reads:

- **The band**: the family rows above, with no hypothesis, in the ideal-permutation model; `BandChainMeets64` (one
  segment) and `BandMultiMeets14` (448 segments) are instances. With an untrusted server's encoding and the exact setup
  check, `BandPubMeetsFamily`, `BandPubMeetsFamilyD2` and `BandPubMeetsFamilyD1`, through `chainBand_encDec` and
  `pubMeets_exact`.
- **The graded audit** (`BandGradedMeetsFamily`): up to `t` late answers at `(t, k) = (0, 111), (1, 160), (2, 203),
  (3, 243)`, each late round within `(D_late, Q_late)`, `5 ≤ D_late ≤ d` and `D_late ≤ 12`. Its game assumes no
  pipelining, a hard deadline and every answer past `Δ` counted late, and `BackgroundRebuildExcluded` needs the memory
  challenger to abort an audit lasting more than `1,505 · call_ns` (≈ 363.5 ms, `audit.AUDIT_CAP`).
- **The continuous audit** (`ContinuousAudit`): `n` challenges over time, accepted with at most `f₀` misses.
  `continuous.ContinuousMemoryChallenger` enforces the model's three requirements (no pipelining, a hard deadline,
  every late answer counted), which `Security/Proofs/Pous/Continuous/Verifier.lean` proves of it; a bare
  `UnsignedRounds` with a late deadline credits an answer still in flight to the next challenge, and is not that
  challenger.
- **The streamed-label wipe** (`StreamedWipe`, with no named assumption): what an accepted sequential wipe leaves of the
  prior content. `StreamedWipeMeets2600GB`, `StreamedWipeMeetsFamily` (with `StreamedWipeMeets96GiB` and
  `StreamedWipeMeets141GB`), `StreamedWipeMeetsReads` (`verity/wipe/v0`'s figure) and `StreamedWipeMeets141GB5`; per
  server against its assumed capacity over every tier, `StreamedWipeMeetsReadsTight`, `StreamedWipeMeetsServer`,
  `StreamedWipeMeetsServer5GB` and `StreamedWipeMeetsServer8H200` (the streamed-wipe paragraph below has their figures).
  Its physical premises are the game's shape: no unaccounted tier, the egress certified over the whole wipe, and no side
  channel.
- **P2**: "P2" below. End to end (`Security/Properties/Pous/EndToEnd.lean`), `RunCheckAccepts` and `RunCertifies` are
  about the deployed code, with no assumption: when the composed check accepts, every one of the 105 recorded answers
  was the challenged block of `C*`, on time on the device's clock, or the run holds a SHA-512 collision or an LMS
  forgery. `EndToEndRing` bounds a `⌊ρ·|C|⌋`-bit server at `1% + 2^-128` in the ring model.
- **Named assumptions**: `LemmaA` and `B1Prime` (`Assumptions.lean`); P2's in `Assumptions/P2.lean`.

## Using it

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
Everything else is the same.

The same lifecycle through the generic protocol. A scheme implements only `encode_segment` and `decode_segment`, and
the memory challenger refuses any `k` or deadline its certificate doesn't cover:

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

On the deployed path the timed rounds run on the memory challenger (`memory_challenger`), an auditor-trusted device
on a machine the prover doesn't control at the prover's site (the auditor's own VM or box, under the auditor's account),
never a process on the prover's node. It commits its indices in a coin tree before the session, hashes each whole
answer after reading its receipt time (`block_digest`, its `sha512/row/v2` digest: the inner digest of C's leaf, which
`answer_circuit.py` opens), and signs the session's round records once, under a header bound to the setup's root of `C*`.
A session's id is the auditor's for one audit, and its `SessionLog` refuses an id it has used. `verdict` checks a
signed session in the clear and publishes only the verdict. It accepts S1's LMS over SHA-512 at `lms.DEPLOYED`
(`memory_challenger.LMS`) and nothing else: a test-signed or unsigned record is refused. `verdict_circuit`'s
`PousVerdict_v1{K, M, LIMIT}` is the verdict over the rounds as one proof block, its one output bit the public verdict: 1
iff at most `M` of the `K` rounds miss, a round hitting when its record names it, is on time within `LIMIT` and holds
the expected digest of its block (`PousVerdict.verdictProg_accepts`, `Security/Proofs/Pous/Continuous/Verdict.lean`).
Its gates as Lean code, `PousCheck.Verdict.prog` (`check/PousCheck/Verdict.lean`), are `verdictProg` gate for gate
(`Security/Proofs/Pous/Continuous/VerdictCode.lean`), and `check/tests/test_verdict_difftest.py` holds the Definition's
compiled circuit to that code.

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
outputs, the memory challenger's table of `C*`'s row digests (the harness's `DIGESTS.bin`) and its signed session, joined
at P2's deployed point, the session under the id the auditor designated and signed under a key of `trusted`, the
verifier's own list of memory-challenger keys, which never comes from the run's record. `pous-run TRUSTED.json SETUP.json
DIGESTS.bin SESSION.json SESSION_ID` (`check/`) is the same check in Lean.

**The streamed-label wipe** (`streamed_wipe`, experimental) is a proof of secure erasure for a memory wipe: the memory
challenger fills every memory tier of one server with labels it derives itself (SHA-512 under a key fresh for the server
and the wipe), then reveals `k` indices drawn on live coins once the fill is complete and checks each answer against its
own label, under a deadline per batch. `Pous.SecurityProofs.StreamedWipe` bounds what an accepted wipe leaves of the
prior content, with no named assumption, for the sequential reveal, the only one the verdict accepts (batch size 1); at
2.6 TB and `k = 74,000`, the egress plus 1.075 GB at `2^-40` (`StreamedWipeMeets2600GB`); at `k = 2152`, for every
fill of a 32 MiB segment or more, the egress plus `⌈M/19⌉` bits at `2^-128` (`StreamedWipeMeetsFamily`, with
`StreamedWipeMeets96GiB` and `StreamedWipeMeets141GB`); at any `k` with `(1 − 1/r)^k ≤ 2^-129`, the egress plus
`512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130` bits and the memory outside the fill (`StreamedWipeMeetsReads`), an H200 at
`k = 5231` within 5 GB with its driver reserve (`StreamedWipeMeets141GB5`); with the count of the missing labels in
place of `n`, per server against its assumed memory capacity `M_s` over every tier (`StreamedWipeMeetsReadsTight`,
`StreamedWipeMeetsServer`), and at 5 GB a server (`StreamedWipeMeetsServer5GB`): one H200 as measured at its deployed `k = 5500`, and a whole
8×H200 server, HBM, 2 TB of RAM and 12 TB of disk, at `k = 1,268,143` with 3.9 GB outside the fill
(`StreamedWipeMeetsServer8H200`). The model and its bound are `Security/Definitions/Pous/Model/StreamedWipe.lean` and
`Guarantees/StreamedWipe.lean`. The verdict is decided by its Lean port, `MemoryChallenger.StreamedWipe`
(`streamed-wipe-verdict` in `verity/core/devices/memory_challenger/lean/`), through `wipe_verdict.judge` there;
`streamed_wipe.verdict` is the reference it is difftested against, and runs beside it under `both`.

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
`period.json`'s `challenger_host` (`wipe_party`, which streams the fill and times the reads over a socket), judged as the
wipe's component of onsite Verity, `verity/wipe/v0` (`MemoryChallenger.WipeProtocol`, `wipe-protocol` in
`verity/core/devices/memory_challenger/lean/`). An accepted wipe publishes `M` (every declared tier), the reads `k` the
memory challenger chose (`wipe_party --reads`, 2152 by default; the prover never names them), `r = ratio(k)` and the
foreign content it may leave, `StreamedWipeMeetsReads`' `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E` at `2^-128` with
the egress counted (`R` the declared memory past the fill), and `M` (vacuous) otherwise; an unsigned session is
undecided. `wipe_period.leg(period_dir, node, out)` is the wipe's leg of the all-protocols period runner.
`wipe_monitoring` reports one accepted wipe over a week, a month and a year of a deployment's machines, each wiped
every `cadence_s` seconds (a day by default), at δ = 1% and 10⁻⁶: the T wipes, at most M − 1 of them failing their bound
except with δ (`prCoin_count_le_choose`'s count bound, which the remote bandwidth audit's period shares), every one
meeting it except with T·`eps`, and what each certifies (`foreign_bits` of `memory_bits`, its tiers and devices).
`wipe_parties` is the same memory challenger and prover as parties of the deployment layer (`verity.core.deploy`), the
reads the challenger's machine gives it: on the same coins, clock readings, key and reads it signs the bytes
`wipe_party.serve` signs.
On a GPU, `period.json`'s `wipe` puts the fill in device memory (`wipe_store`: the CUDA driver through ctypes, gated
before the fill, with host memory as its CPU stand-in) and has the challenger stream it natively (`wipe_stream.c`, built
and gated on the challenger's host by `wipe_stream`); `benchmarks/pous/streamed_wipe/demo.sh` runs it on one H200. Its
`foreign_bytes` makes the prover the cheat that keeps prior content in place of the fill's first labels (`fM`: one label
past the published figure, which must reject), its `reads` the reads the leg expects, and its `placement` `simulated` names a challenger on the prover's node the stand-in for
one on its own host. The leg's record names what the bound doesn't cover (`uncertified`): the links leaving the node that
no network certifier counts (the placement's `links`: `wipe.links`, and the challenger's as the prover connected to it),
the node's memory outside the declared tier (`undeclared_tiers`: the device's past the declared tier, which a GPU's leg
declares whole, the host's DRAM, its disks), a GPU's driver reserve when NVML gave no total (the leg declares the larger
of CUDA's and NVML's totals, and NVML's counts the reserve: `wipe_store.Cuda.info`) and an egress signed `UNCOUNTED`. It
also carries the session's window, `t0` (the prover connected) and `t1` (the session signed) on the challenger's clock,
which a network certifier counts the egress over.
`wipe.disk` puts the node's disks in the same session (`wipe_store.Disk`, laid ahead of the card or host memory by
`wipe_store.Joint`; `demo.sh --disk PATH`): a fill file of the session's own in each target directory, or a raw block
device, written with `O_DIRECT` and synced before the prover acknowledges the fill. The whole disks under the targets
are declared, with `spare_bytes` for each drive's spare area and cache, so what the fill doesn't overwrite (the used
filesystem, its metadata, the reserve, the spare area) counts in `R`. That is `Pous.Guarantees.StreamedWipeMeetsDisk`.
A virtual disk's backing store, an undeclared spare area, buffered I/O and the disks outside the targets go in
`uncertified`: every block device in `/sys/block` but loop and RAM devices, mounted or not, each whole. `wipe.dram` puts the host's DRAM in that session too, after the card (`demo.sh --dram`): the fill takes
what `MemAvailable` says less a reserve, and the whole DRAM is declared (`wipe_store.HostMemory(whole=True)`), so the
rest counts in `R`.

Tests: `uv run pytest verity/core/protocols/pous` (about 8 s). P3's 64 KB-label segment vector runs with `VERITY_POUS_SLOW=1`.
Regenerate the vectors with `python verity/core/protocols/pous/tests/pous_vectors.py`, and the Lean vectors with
`tests/lean/generate.sh`.

## Stable byte formats

These names and values are the wire format a second implementation (the GPU codec) must match. They change only with a
version bump in the strings they carry.

| Name | Value |
|---|---|
| `params.SALT_BITS`, `params.TAG_BITS` | 192, 256: `tag_s = salt ‖ u64(s)` |
| `primitives.CAPACITY_BITS` | 512: `Π₂` is `2m + 512` bits |
| `primitives.HEADER` | `b"pous-p3-v1-H"`, the first 12 bytes of `h_0` |
| `primitives.PI_DOMAIN`, `primitives.P_DOMAIN` | `0x48` (`Π₂`), `0x50` (`P_T`): the first byte of every round-function input |
| `primitives.EM_KEY_DOMAIN` | `0x4B`: the heuristic ARX `P_T`'s key |
| `verifier.leaf_hash`, `verifier.node_hash` | prefixes `0x00`, `0x01` |
| `setup.COMMIT_TAG`, `setup.SALT_DOMAIN` | `"verity-pous/w-commitment/v2"`, `"verity/pous/salt/v1"` |
| `audit.CHALLENGE_DOMAIN` | `"verity/pous/audit-challenge/v1"` |

The reference backend's `Π₂` is a 10-round Feistel network on `2m + 64` bytes with round functions
`SHAKE256(0x48 ‖ u8(i) ‖ x)[0 : m + 32]`, and `P_T` one on `m` bytes with `SHAKE256(0x50 ‖ u8(i) ‖ T ‖ x)`,
`T = tag_s ‖ u8(layer) ‖ u16(v)`. `H` is the overwrite chain over `Π₂`: `h_0` is the 48-byte header (`HEADER`, `tag_s`,
layer, `v`, the number of parents) zero-filled to `m + 64` bytes; each parent, newest first, and then the pad
(`0x01 … 0x80`) goes through `(r, h) ← Π₂(B ‖ h)`, dropping the first `m` bytes `r` and chaining the last `m + 64` bytes
`h`; the key is the pad call's `r`, and a block is its key XOR the payload. `vk` is a SHA-512 Merkle tree over the
blocks with the shape of RFC 6962: a leaf is `SHA512(0x00 ‖ tag_s ‖ u16(v) ‖ R(c))`, 99 bytes, where
`R(c) = SHA512(P(8|c|) ‖ c)` is the block's `sha512/row/v2` digest (the digest `answer_circuit`'s `PousRowDigest`
checks), `tag_s` 32 bytes and `v` 2, little-endian; a node is `SHA512(0x01 ‖ l ‖ r)`. This follows the instantiation's
byte format rather than `verity.ml.commitments`' frame v3. The known-answer vectors are `tests/fixtures/vectors.json` and
`tests/fixtures/lean_vectors.json`.

## P2

P2 v4 (`p2/v4`, `schemes/p2.py`) encodes payload `m` into block `j` as `E_j(m) = ρ(σ_K(ρ((m + t_j) mod p)))` and decodes
it as `D_j(c) = (d(σ_K(d(c))) − t_j) mod p`: `d(z)` is `z² mod p`, negated when `z` is odd, `ρ` its inverse (one root,
its parity chosen), and `σ_K(r)` is `r ⊕ K` when that is below `p` and `r` otherwise, never reduced. Its point is
`w = 16,448` bits, `p = 2^16448 − 21065` (`p2-p16448-c21065`, `p ≡ 3 mod 4`), the payload profile `p2-2054B` (2,054
payload bytes in each 2,056-byte block), `B = 2^19` blocks in 8 segments of `2^16`, `k = 105` challenges revealed one at
a time, `Δ = 0.5 ms` plus a measured RTT allowance of at most 354 µs, and SHA-512 with its whole output for every digest.
Every integer is fixed-width, unsigned and big-endian. A block's `K` and `t` are expanded where they are used:
counter-mode SHA-512 over a prefix that frames the setup salt, drawn after `W` is committed, zero-padded to whole 128-byte
chunks; `K` is the stream's first `w/8` bytes and `t` its first `w/8`-byte word below `p`. The challenges are
`verity.ml.randomness` v2, bound to the epoch, the verification root and the salt. P2 is experimental: its
instantiation assumption, P2–M1-SGI (`P2RingReducesToM1p`), is not accepted.

- **Lean.** The package `pous_check` (`check/`) is P2's verifier: `PousCheck.P2`, `Keys` and `Wire.Codec` (the maps and
  keys), `Commit`, `Merkle` and `Challenge` (the commitment to `W`, the verification tree in both leaf forms, the
  challenges), `Wire.verifyAudit` and `Audit.check` (a setup and one timed audit, to its `FINAL` frame byte for byte),
  `Certificate.gate` and `assess` (the certificate gate, `deadlineRefusals` and the StorageProfile), and
  `RunProtocol.check` (a recorded run, whole). Its executables are `pous-setup`, `pous-timed-audit`, `pous-run` and
  `pous-run-protocol`, and the difftests `pous-setup-difftest`, `pous-audit-difftest` and `pous-verdict-difftest`;
  `check/interop.py --all` holds it to `wire.py` byte for byte on every vector.
- **Guarantees** (`Security/Definitions/Pous/Guarantees/P2.lean`): `P2SlackFamilyUncond` (abstract M1 on `[0, p)`, the
  `λ = 128` row at `k = 105` for every block count from `2^8` to `2^23`: the certificate), `P2SlackB19Uncond`,
  `P2ErrorSplitB19Uncond`, `P2MeetsM1pB19Uncond`, `P2MeetsM1pFamilyUncond`, `P2MeetsM1pW8192Uncond`,
  `P2SlackFamilyFreeBlocks`, `P2SlackFamilyWide`, `P2SlackFamilyFreeBlocksWide`, and the ring-model forms
  `P2RingAuditSecureOfReduces`, `P2RingAuditSecureOfRoundSecure` and `P2RingGradedOfRoundSecure`; `P2ContentAbsorbs`
  (`Guarantees/Obligation.lean`); above `2^23` blocks, `Pous.Period.SecurityProofs.periodHeldAt`. The named
  assumptions (`Assumptions/P2.lean`): `PrimeP16448`, `P2RingAuditSecure`, `P2RingReducesToM1p`, `P2RingRoundSecure`,
  and `P2RingRoundSecureWide`. `schemes.p2.P2_CLAIMS` and `WIDE_CLAIMS` are the claims a certificate
  rests on.
- **Python.** `schemes.p2`: `P2Params`, the schemes `P2` (`p2-16448/v4`) and `P2ChaCha` (`p2-16448/v2`),
  `family_certificate`, `wide_certificate`, `expand_prefix`, `stream`, `below`, `sha512_keys`, `encode_segment`,
  `decode_segment`, and `CERT_K`, `DELTA_NS`, `RTT_ALLOWANCE_CAP_NS` and `ROOT_CALL_NS` (half the fastest cooperating
  16,448-bit root; the cap is the largest whole µs with `Δ + cap < ROOT_CALL_NS`). `benchmarks/pous/p2_v1/reference.py`
  is the reference (the maps, keys, salt, tree and derivations), and `wire.py` beside it is `p2-audit-wire/v4`
  (`envelope`, `setup_messages`, `open_message`, `challenge_message`, `response_message`, `record`, `final_message`,
  `verify_path`, `parse_envelope`, `verify_audit`, `parse_final`).
- **The registered scheme and the wire.** `p2-16448/v4` runs the generic protocol's setup, so four things differ from
  `reference.py` and the wire: its setup salt (24 bytes from `setup.draw_salt`, not 32 from `verity/pous/p2-setup/v4`
  over the commitment), its key prefix (it frames that salt), its leaf (`0x00 ‖ tag_s ‖ u16le(v) ‖ R`, not
  `0x00 ‖ salt ‖ u64be(s) ‖ u16be(v) ‖ R`) and its challenges (`verity/pous/audit-challenge/v1`, not
  `verity/pous/p2-audit-challenge/v4`). `PousCheck.Keys` and `Merkle` hold both forms.

`p2-audit-wire/v4`: a frame is `"P2AU" ‖ 0x04 ‖ type ‖ u32be(len body) ‖ body`. Digests are 64 bytes; the setup salt,
the audit id and the round nonce are 32; `n = w/8`, and `h` is the block's RFC 9162 path length.

| Type | Code | Body |
|---|---|---|
| `SETUP_COMMIT` | `0x01` | `u32be(ml) ‖ manifest[ml ≤ 512] ‖ w_commitment` |
| `SETUP_SALT` | `0x02` | `w_commitment ‖ setup_salt` (96) |
| `SETUP_BLOCK` | `0x03` | `w_commitment ‖ setup_salt ‖ u64be(index) ‖ u32be(n) ‖ block[n]` |
| `SETUP_ROOT` | `0x04` | `w_commitment ‖ setup_salt ‖ u64be(B) ‖ root ‖ setup_mode` (169) |
| `AUDIT_OPEN` | `0x10` | `epoch_id ‖ audit_id ‖ mode ‖ u16be(k) ‖ u64be(Δ) ‖ u64be(rtt) ‖ flags ‖ u64be(counter)` (124) |
| `CHALLENGE` | `0x11` | `epoch_id ‖ audit_id ‖ u16be(round) ‖ u64be(index) ‖ nonce` (138) |
| `RESPONSE` | `0x12` | `epoch_id ‖ audit_id ‖ u16be(round) ‖ u64be(index) ‖ u32be(n) ‖ block[n] ‖ u16be(h) ‖ siblings[64·h]` |
| `FINAL` | `0x13` | `epoch_id ‖ audit_id ‖ verdict ‖ code ‖ u16be(rounds) ‖ u64be(Δ) ‖ u64be(rtt) ‖ transcript_hash ‖ u16be(r) ‖ records[147·r]` |

A record is `u16be(round) ‖ u64be(index) ‖ u64be(elapsed_ns) ‖ flags ‖ SHA-512(challenge frame) ‖ SHA-512(response
frame)`, its flags bit 0 authenticated, bit 1 on time, bit 2 decode performed and bit 3 decode valid. The failure codes,
in the order one is chosen when several apply: 1 malformed, 2 identity, round, index, replay or domain, 3
authentication, 4 late, 5 decode or payload comparison; 6 is transport and 0 success. The transcript hash is
`SHA-512("verity/pous/p2/transcript/v4\0" ‖ AUDIT_OPEN ‖ every challenge and response frame)`.

The vectors are `vectors.json` (schema `p2-test-vectors/v4`), `audit-vectors.json` and `audit-negative-vectors.json`
beside `reference.py`, which pins each file's SHA-512 (`VECTORS_SHA512`, `AUDIT_VECTORS_SHA512`,
`NEGATIVE_VECTORS_SHA512`) and each entry's by width (`ENTRY_SHA512`, which `PousCheck.SetupDifftest` holds too).
`cd benchmarks/pous && uv run python -m p2_v1.vectors` regenerates them and rewrites their pins.
