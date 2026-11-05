# PoUS (`verity.core.protocols.pous`)

PoUS, proof of useful space, shows that a prover holds model weights in memory. The prover stores `C*`, an encoding of
the committed weights `W` under a salt sampled after `W` is committed. The memory challenger
(`verity/core/devices/memory_challenger`) reveals indices of `C*` one at a time on live coins and accepts when each
answer is correct and on time, within `Δ` plus an RTT allowance. A scheme's guarantee is the requirement `Pous.Meets`
(`Security/Definitions/Pous/Accounting/Params.lean`): the scheme is correct, `|C| ≥ |W|`, it is within the space bound,
and every `⌊ρ|C|⌋`-bit adversary is accepted with probability at most `δ + ε`. This package also holds the
streamed-label wipe, a proof of secure erasure that the same device runs, and P2's Lean verifier (`check/`).

The Python here is a spec-level reference: it matches the Lean models' definitions and is the oracle for GPU kernels.
It is pure Python, stdlib plus `verity`, and slow by design. Four codecs sit behind one API:

- **the band scheme** (`verity.core.protocols.pous.band`, the secure-first default): one layer on the band DAG with
  in-degree 12, `H` an overwrite chain on a double-width permutation (Lean's `chainBand`). Deploy it only while the
  effective round budget stays at or below 12 calls (the cliff);
- **the dense scheme** (`verity.core.protocols.pous.dense`, selectable): the same on the complete DAG (Lean's
  `chainDense`);
- **P2** (`verity.core.protocols.pous.schemes.p2`, **experimental**): the squaring-based scheme, proved in abstract M1
  under the pinned `P2SlackFamilyUncond` (k = 105), with its instantiation assumption not accepted; never the default.
  `p2-16448/v4` has P2 v4's counter-mode SHA-512 keys, `p2-16448/v2` ChaCha8 keys, which v4 does not allow ("P2"
  below);
- **P3** (`verity.core.protocols.pous.scheme`, the fast research codec): two layers over the band graph `B(220, 12)`
  with a transpose connector and a tweakable `P` (Lean's `p3Scheme`).

**The caveat.** The band, dense and P3 schemes hold only over an ideal `Π₂`. At the reference `Π₂`, Feistel10 over
SHAKE256, a prover that keeps each block's five odd-round states of its pad call (1.53% of `C`) rebuilds any label in
481 sequential Keccak-f with no `Π₂` call: the claim `feistel-indifferentiability` is refuted (`protocol.LOCAL_CLAIMS`),
and so is every reading of a deadline as a count of whole `Π₂` calls (`call_ns`, the cliff, `audit.AUDIT_CAP`). P2 has
no `Π₂`.

## Top statements

Each is proved as `Pous.SecurityProofs.<name>` in `Security/Proofs/Pous.lean` and stated as `Pous.Guarantees.<name>` in
`Security/Definitions/Pous/Guarantees.lean` unless its entry names other files, and `Security/lean-audit.json` locks
each statement. The schemes' statements hold in the ideal model each docstring names.

- `BandMultiMeetsFamily`: the band store, at every segment count from 1 to `2^64` and every `d ≥ 5`, meets the
  requirement at `D = 5`, `Q = 2^20`, `k = 111` and `ε = 2^-128` with the implementation's per-segment tags, with no
  hypothesis. It is `band.SECURE_FIRST`'s certificate.
- `BandPubMeetsFamily`: the same with an untrusted server encoding `W`, the verifier's public mode (`PubMeets`,
  `Security/Definitions/Pous/Accounting/PubMeets.lean`).
- `BandGradedMeetsFamily`: the band store's audit that accepts up to `t` late answers, at `(t, k)` of `(0, 111)`,
  `(1, 160)`, `(2, 203)` and `(3, 243)` (`GradedMeets`, `Security/Definitions/Pous/Accounting/GradedMeets.lean`).
- `BackgroundRebuildExcluded`: within the memory challenger's cap on an audit's duration, no audit re-encodes a whole
  band segment from `W` alone, in the ideal-permutation model. It does not exclude a prover that keeps a few stretches
  of consecutive labels and rebuilds the segment from them.
- `ChainDenseMeets64`: the dense scheme, 512 labels of 64 KiB on the complete DAG, meets the requirement at `D = 5`,
  `Q = 2^20`, `k = 111` and `ε = 2^-128`. It is `dense.POINT_64K`'s certificate.
- `P3ConcreteMeets64` and `P3ConcreteMeets14GB`: P3 with a concrete `H` and 64 KB labels meets the requirement at
  `D = 10`, `Q = 2^20`, `k = 106` and `ε = 2^-128`, for one 14.4 MB segment and jointly for 971 segments (14.005 GB).
  They are about Lean's chain, which starts from another `h_0` than the code's, so P3 has no certificate
  (`DISCREPANCIES.md`, item 16).
- `ContinuousAudit`: for any scheme with `TimedINC`, the continuous audit's memory challenger (modelled in
  `Security/Definitions/Pous/Model/Continuous.lean`; `continuous.py` runs it) accepts at tolerance `f` while the average
  freed fraction of memory over the challenges exceeds `ε_proof + g(f)` with probability at most `βe + n·(β + 1)·π`.
- `BandFamilyContinuousTimeAverage`: the continuous audit's time-averaged bound (`ContinuousTimeAverage`: arrivals the
  memory challenger samples itself, memory read at every idle slot) on the band store at every segment count, at
  `(D, k)` of `(0, 89)`, `(1, 93)`, `(2, 97)` and `(5, 111)`, with no hypothesis.
- `PousVerdict.verdictProg_accepts` (`Security/Proofs/Pous/Continuous/Verdict.lean`): `verdict_circuit`'s
  `PousVerdict_v1{K, M, LIMIT}` outputs 1 iff at most `M` of the `K` rounds miss.
- `StreamedWipe` (`Security/Definitions/Pous/Guarantees/StreamedWipe.lean`): in the wipe game, an accepted wipe leaves
  of the prior content at most about the labels the prover didn't hold, the egress and a counting term, except with
  `(g/N)^k`, a compression term and the labels' PRF advantage, with no named assumption.
- `StreamedWipeMeetsReads` (`Security/Definitions/Pous/Guarantees/StreamedWipeReads.lean`): at any `k` with
  `(1 − 1/r)^k ≤ 2^-129`, an accepted wipe leaves at most the egress plus `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130` bits
  and the memory outside the fill, at `2^-128`. It is the bound `wipe_period` publishes.
- `StreamedWipeMeetsServer` (`Security/Definitions/Pous/Guarantees/StreamedWipeServer.lean`): the same per server,
  against its assumed memory capacity over every tier, with the count of the missing labels in place of `n`
  (`StreamedWipeMeetsReadsTight`). At 5 GB a server (`StreamedWipeMeetsServer5GB`) it covers one H200 at its deployed
  `k = 5500`, and a whole 8×H200 server, HBM, 2 TB of RAM and 12 TB of disk, at `k = 1,268,143` with 3.9 GB outside
  the fill (`StreamedWipeMeetsServer8H200`).
- `StreamedWipeMeetsDisk` (`Security/Definitions/Pous/Guarantees/StreamedWipeDisk.lean`): the same with the node's
  disks in the run, counting what the fill doesn't overwrite (the used filesystem, its metadata, each drive's spare
  area and cache).
- The wipe at fixed points: `StreamedWipeMeets2600GB` (`Security/Definitions/Pous/Guarantees/StreamedWipe.lean`), 2.6
  TB at `k = 74,000`, the egress plus 1.075 GB at `2^-40`; `StreamedWipeMeetsFamily`
  (`Security/Definitions/Pous/Guarantees/StreamedWipeFamily.lean`), at `k = 2152` for every fill of a 32 MiB segment or
  more, the egress plus `⌈M/19⌉` bits at `2^-128`, with `StreamedWipeMeets96GiB` and `StreamedWipeMeets141GB` there;
  and `StreamedWipeMeets141GB5` (`Security/Definitions/Pous/Guarantees/StreamedWipeReads.lean`), an H200 at
  `k = 5231` within 5 GB with its driver reserve.
- `P2SlackFamilyUncond` (`Security/Definitions/Pous/Guarantees/P2.lean`): abstract M1 on `[0, p)` meets the `λ = 128`
  row at `k = 105` for every label count from `2^8` to `2^23`. It is P2's certificate, and it is about the abstract
  model, not the deployed P2.
- `ObligationWide320` (`Security/Definitions/Pous/Guarantees/Obligation.lean`, proved in
  `Security/Proofs/Pous/Obligation.lean`): at every label count `N` from `2^19` to `2^27`, a responder holding at most
  `⌊ρ·N·16448⌋` bits at each of 320 challenge times passes those 320 reads of P2's ring-model audit, at a snapshot of a
  mutable store, with probability at most `2^-20`, under the open conjecture `P2RingRoundSecureWide`.
- `Pous.Period.SecurityProofs.periodHeldAt` (`Security/Proofs/Pous/PeriodAt.lean`, stating `PeriodHeldAt` in
  `Security/Properties/Pous/PeriodAt.lean`): PoUS's per-period guarantee, at any point and label count up to `2^27`:
  the verifier accepts a period in which the store held fewer bit-ticks than its certificate with probability at most
  `δ + ε_max`, under the same open conjecture.

## Where to look

- `DISCREPANCIES.md`: where the Lean models, their prose and this code differ. Each module's docstring states its part
  of the lifecycle (the modules and the byte formats are under "API").
- `check/`: the package `pous_check`, P2's verifier in Lean (below); `lean/`: the grader and where PoUS's Lean is in
  `Security/` (its README).
- `verity/core/devices/memory_challenger`: the device that runs the timed rounds and the wipe, and its verdicts in Lean.
- `benchmarks/pous/`: P2's reference (`p2_v1`) and the wipe's runs on hardware.
- Tests: `uv run tools/verity/check/suites.py verity/core/protocols/pous`. P3's 64 KB-label segment vector runs with
  `VERITY_POUS_SLOW=1`. Regenerate the vectors with `python verity/core/protocols/pous/tests/pous_vectors.py`, and the
  Lean vectors with `tests/lean/generate.sh`.

## API

### Modules

| Module | What it holds |
|---|---|
| `params`, `labels`, `oracle`, `primitives`, `scheme` | P3: its parameters, the bit layout (Lean's `codeEquiv`), oracle programs and their rounds, the primitives `H` and `P_T` with their backends (`reference`, `ideal`, the unanalysed `heuristic_arx`, and the broken `XorSponge` and `MiyaguchiPreneel`, kept to replay their attacks), and the scheme |
| `band`, `dense` | the band and dense schemes over the overwrite chain: `SECURE_FIRST`, `POINT_64K`, `check_deployment` (the cliff) |
| `codec`, `layout` | `Codec`, the seam every module above the scheme uses, and `Geometry`, the sizes they read; tensors laid out in `W`, and per-tensor decode |
| `setup`, `verifier` | the commitment to `W`, the verifier's salt, and `vk` by the public encoder (the default) or the verifier's own; `VerifierKey`, its leaves and audit paths |
| `audit` | the challenge key, `UnsignedRounds` and `run_timed` (the sequential reveal, strict), `simulate` (oracle rounds) |
| `protocol`, `schemes` | the generic protocol (`Scheme`, `Certificate`, `TimedAudit`, `StorageProfile`, `LOCAL_CLAIMS`) and the registered schemes |
| `memory_challenger`, `challenger_party`, `period_reports` | the memory challenger, which signs its rounds; it as a party of its own; its runs under its party's per-period root |
| `verdict_circuit`, `answer_circuit`, `decode_circuit`, `circuit` | Definitions: the verdict as one proof block, an answer's label, P2's decode and its modular squaring |
| `run`, `period` | the composed check of a recorded run, and one period on one node |
| `continuous` | the continuous audit's memory challenger |
| `adversary` | holdings replayed against the real labels (Lean's `ColState` and `colRound`) |
| `streamed_wipe`, `wipe_party`, `wipe_parties`, `wipe_period`, `wipe_store`, `wipe_stream`, `wipe_monitoring` | the streamed-label wipe: the protocol, its memory challenger as a party, one node's wipe, the prover's store on a GPU, the native fill, and the wipe over a monitoring period |

### Byte formats

A second implementation (a GPU codec) matches these, and each changes only with a version bump in a string it carries.
`primitives.py`'s docstring has the reference backend's layout, `dense.py`'s the dense chain's, and `verifier.py`'s
`vk`'s.

| Name | Value |
|---|---|
| `params.SALT_BITS`, `params.TAG_BITS` | 192, 256: `tag_s = salt ‖ u64(s)` |
| `primitives.CAPACITY_BITS` | 512: `Π₂` is `2m + 512` bits |
| `primitives.HEADER` | `b"pous-p3-v1-H"`, the first 12 bytes of P3's `h_0` |
| `primitives.PI_DOMAIN`, `primitives.P_DOMAIN` | `0x48` (`Π₂`), `0x50` (`P_T`): the first byte of every round function's input |
| `primitives.EM_KEY_DOMAIN` | `0x4B`: the heuristic `P_T`'s key |
| `verifier.leaf_hash`, `verifier.node_hash` | prefixes `0x00`, `0x01` |
| `setup.COMMIT_TAG`, `setup.SALT_DOMAIN` | `"verity-pous/w-commitment/v2"`, `"verity/pous/salt/v1"` |
| `audit.CHALLENGE_DOMAIN` | `"verity/pous/audit-challenge/v1"` |

### The codecs and the lifecycle

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

### The memory challenger

On the deployed path the timed rounds run on the memory challenger (`memory_challenger`), an auditor-trusted device
on a machine the prover doesn't control at the prover's site (the auditor's own VM or box, under the auditor's account),
never a process on the prover's node. It commits its indices in a coin tree before the run, hashes each whole answer
after reading its receipt time (`block_digest`, its `sha512/row/v2` digest: the inner digest of C's leaf, which
`answer_circuit.py` opens), and signs the run's round records once, under a header bound to the setup's root of `C*`.
A run's id (`session_id`) is the auditor's for one audit, and its `SessionLog` refuses an id it has used. `verdict`
checks a signed run in the clear and publishes only the verdict. It accepts S1's LMS over SHA-512 at `lms.DEPLOYED`
(`memory_challenger.LMS`) and nothing else: a test-signed or unsigned record is refused. `verdict_circuit`'s
`PousVerdict_v1{K, M, LIMIT}` is the verdict over the rounds as one proof block, its one output bit the public verdict:
1 iff at most `M` of the `K` rounds miss, a round hitting when its record names it, is on time within `LIMIT` and holds
the expected digest of its label (`PousVerdict.verdictProg_accepts`). Its gates as Lean code, `PousCheck.Verdict.prog`
(`check/PousCheck/Verdict.lean`), are `verdictProg` gate for gate (`Security/Proofs/Pous/Continuous/VerdictCode.lean`),
and `check/tests/test_verdict_difftest.py` holds the Definition's compiled circuit to that code. An audit's public
outputs are the setup's (`run.Setup`: the commitment to `W`, the root of `C*`, the salt and the block count), the
verdict, and the coin tree's openings once the run has closed. Each round's times and digests are witness in the signed
record; `audit.AuditResult`, which keeps every answer and its latency in the clear, is the reference's, never a public
output.

As an auditor party with a device identity, the challenger reports its runs under its party's per-period root instead
(`period_reports`, on `verity.core.attest`): each run's one public output is its liveness record, and the period's root
is signed once at its close, for the prover only. The party starts its attachment on its key once
(`lms.PartyKey.start`), so a challenger restarted within the attachment signs no more of its runs.
`period_reports.PERIOD` is not in `memory_challenger.SCHEMES` until `PousCheck.RunProtocol` reads `Attest.accepts`.

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

### A recorded run

A recorded run is checked whole by `run.check(trusted, setup, digests, session, session_id)`: the setup's public
outputs, the memory challenger's table of `C*`'s row digests (the harness's `DIGESTS.bin`) and its signed run, joined
at P2's deployed point, the run under the id the auditor designated and signed under a key of `trusted`, the
verifier's own list of memory-challenger keys, which never comes from the run's record. `pous-run TRUSTED.json SETUP.json
DIGESTS.bin SESSION.json SESSION_ID` (`check/`) is the same check in Lean.

### The streamed-label wipe

**The streamed-label wipe** (`streamed_wipe`, experimental) is a proof of secure erasure for a memory wipe: the memory
challenger fills every memory tier of one server with labels it derives itself (SHA-512 under a key fresh for the server
and the wipe), then reveals `k` indices sampled with live coins once the fill is complete and checks each answer against
its own label, under a deadline per batch. Its guarantees are the wipe's top statements above, for the sequential
reveal, the only one the verdict accepts (batch size 1). `streamed_wipe.py`'s docstring has the messages and the
records, and `Pous.Guarantees.StreamedWipe`'s module docstring the bound, its physical premises and what it doesn't
cover. The verdict is decided by its Lean port, `MemoryChallenger.StreamedWipe` (`streamed-wipe-verdict`
in `verity/core/devices/memory_challenger/lean/`), through `wipe_verdict.judge` there; `streamed_wipe.verdict` is the
reference it is difftested against, and runs beside it under `both`.

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
`period.json`'s `challenger_host` (`wipe_party`, which streams the fill and times the reads over a socket), judged by
the wipe's own protocol, `verity/wipe/v0` (`MemoryChallenger.WipeProtocol`, `wipe-protocol` in
`verity/core/devices/memory_challenger/lean/`); onsite doesn't compose the wipe yet. An accepted wipe publishes `M`
(every declared tier), the reads `k` the memory challenger chose (`wipe_party --reads`, 2152 by default; the prover
never names them), `r = ratio(k)` and the foreign content it may leave, `StreamedWipeMeetsReads`'
`512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E` at `2^-128` with the egress counted (`R` the declared memory past the
fill), and `M` (vacuous) otherwise; an unsigned run is undecided. `wipe_period.leg(period_dir, node, out)` runs the
wipe in the all-protocols period runner.

`wipe_monitoring` reports one accepted wipe over monitoring periods of a week, a month and a year of `machines`
machines, each wiped every `cadence_s` seconds (a day by default), at δ = 1% and 10⁻⁶: the T wipes, at most M − 1 of
them failing their bound except with δ (`prCoin_count_le_choose`'s count bound, which PoUB's period shares), every one
meeting it except with T·`eps`, and what each certifies (`foreign_bits` of `memory_bits`, its tiers and devices).
`wipe_parties` is the same memory challenger and prover as parties of the party runtime (`verity.core.deploy`), the
reads the challenger's machine gives it: on the same coins, clock readings, key and reads it signs the bytes
`wipe_party.serve` signs.

On a GPU, `period.json`'s `wipe` puts the fill in device memory (`wipe_store`: the CUDA driver through ctypes, gated
before the fill, with host memory as its CPU stand-in) and has the challenger stream it natively (`wipe_stream.c`, built
and gated on the challenger's host by `wipe_stream`); `benchmarks/pous/streamed_wipe/demo.sh` runs it on one H200. Its
`foreign_bytes` makes the prover the cheat that keeps prior content in place of the fill's first labels (`fM`: one label
past the published figure, which must reject), its `reads` the reads the run expects, and its `placement` `simulated`
names a challenger on the prover's node the stand-in for one on its own host. The run's record names what the bound
doesn't cover (`uncertified`): the links leaving the node that no network certifier counts (the placement's `links`:
`wipe.links`, and the challenger's as the prover connected to it), the node's memory outside the declared tier
(`undeclared_tiers`: the device's past the declared tier, which a GPU's run declares whole, the host's DRAM, its disks),
a GPU's driver reserve when NVML gave no total (the run declares the larger of CUDA's and NVML's totals, and NVML's
counts the reserve: `wipe_store.Cuda.info`) and an egress signed `UNCOUNTED`. It also carries `t0` (the prover
connected) and `t1` (the run signed) on the challenger's clock, between which a network certifier counts the egress.

`wipe.disk` puts the node's disks in the same run (`wipe_store.Disk`, laid ahead of the card or host memory by
`wipe_store.Joint`; `demo.sh --disk PATH`): a fill file of the run's own in each target directory, or a raw block
device, written with `O_DIRECT` and synced before the prover acknowledges the fill. The whole disks under the targets
are declared, with `spare_bytes` for each drive's spare area and cache, so what the fill doesn't overwrite (the used
filesystem, its metadata, the reserve, the spare area) counts in `R`. That is `Pous.Guarantees.StreamedWipeMeetsDisk`.
A virtual disk's backing store, an undeclared spare area, buffered I/O and the disks outside the targets go in
`uncertified`: every block device in `/sys/block` but loop and RAM devices, mounted or not, each whole. `wipe.dram` puts
the host's DRAM in that run too, after the card (`demo.sh --dram`): the fill takes what `MemAvailable` says less a
reserve, and the whole DRAM is declared (`wipe_store.HostMemory(whole=True)`), so the rest counts in `R`.

### P2

P2 v4 (`p2/v4`, `schemes/p2.py`) encodes payload `m` into label `j` as `E_j(m) = ρ(σ_K(ρ((m + t_j) mod p)))` and
decodes it as `D_j(c) = (d(σ_K(d(c))) − t_j) mod p`: `d(z)` is `z² mod p`, negated when `z` is odd, `ρ` its inverse (one
root, its parity chosen), and `σ_K(r)` is `r ⊕ K` when that is below `p` and `r` otherwise, never reduced. Its point is
`w = 16,448` bits, `p = 2^16448 − 21065` (`p2-p16448-c21065`, `p ≡ 3 mod 4`), the payload profile `p2-2054B` (2,054
payload bytes in each 2,056-byte label), `B = 2^19` labels in 8 segments of `2^16`, `k = 105` challenges revealed one at
a time, `Δ = 0.5 ms` plus a measured RTT allowance of at most 354 µs, and SHA-512 with its whole output for every digest.
Every integer is fixed-width, unsigned and big-endian. A label's `K` and `t` are expanded where they are used:
counter-mode SHA-512 over a prefix that frames the setup salt, sampled after `W` is committed, zero-padded to whole
128-byte chunks; `K` is the stream's first `w/8` bytes and `t` its first `w/8`-byte word below `p`. The challenges are
`verity.ml.randomness` v2, bound to the epoch, the verification root and the salt. P2 is experimental: its
instantiation assumption, P2–M1-SGI (`P2RingReducesToM1p`), is not accepted.

- **Lean.** The package `pous_check` (`check/`) is P2's verifier: `PousCheck.P2`, `Keys` and `Wire.Codec` (the maps and
  keys), `Commit`, `Merkle` and `Challenge` (the commitment to `W`, the verification tree in both leaf forms, the
  challenges), `Wire.verifyAudit` and `Audit.check` (a setup and one timed audit, to its `FINAL` frame byte for byte),
  `Certificate.gate` and `assess` (the certificate gate, `deadlineRefusals` and the StorageProfile), and
  `RunProtocol.check` (a recorded run, whole). Its executables are `pous-setup`, `pous-timed-audit`, `pous-run` and
  `pous-run-protocol`, and the difftests `pous-setup-difftest`, `pous-audit-difftest` and `pous-verdict-difftest`;
  `check/interop.py --all` holds it to `wire.py` byte for byte on every vector.
- **Guarantees** (`Security/Definitions/Pous/Guarantees/P2.lean`): `P2SlackFamilyUncond` (the certificate, above),
  `P2SlackB19Uncond`, `P2ErrorSplitB19Uncond`, `P2MeetsM1pB19Uncond`, `P2MeetsM1pFamilyUncond`,
  `P2MeetsM1pW8192Uncond`, `P2SlackFamilyFreeBlocks`, `P2SlackFamilyWide`, `P2SlackFamilyFreeBlocksWide`, and the
  ring-model forms `P2RingAuditSecureOfReduces`, `P2RingAuditSecureOfRoundSecure` and `P2RingGradedOfRoundSecure`;
  `P2ContentAbsorbs` and `ObligationWide320` (`Security/Definitions/Pous/Guarantees/Obligation.lean`); above `2^23`
  labels, `Pous.Period.SecurityProofs.periodHeldAt`. The named assumptions
  (`Security/Definitions/Pous/Assumptions/P2.lean`): `PrimeP16448`, `P2RingAuditSecure`, `P2RingReducesToM1p`,
  `P2RingRoundSecure`, `P2RingRoundSecureWide` and `CoinCommitHides`. `schemes.p2.P2_CLAIMS` and `WIDE_CLAIMS` are the
  claims a certificate rests on.
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
the audit id and the round nonce are 32; `n = w/8`, and `h` is the label's RFC 9162 path length.

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
