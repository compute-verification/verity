# POUS (`verity.core.protocols.pous`)

A spec-level reference for POUS, proof of useful space for model weights, with three codecs behind one API:

- **the band scheme** (`verity.core.protocols.pous.band`, the secure-first default): one layer on the band DAG with in-degree 12, `H`
  an overwrite chain on a double-width permutation (Lean's `chainBand`). Deploy it only while the effective round
  budget stays at or below 12 calls (the cliff);
- **the dense scheme** (`verity.core.protocols.pous.dense`, selectable): the same on the complete DAG (Lean's `chainDense`);
- **P2** (`verity.core.protocols.pous.schemes.p2`, **experimental**): the squaring-based scheme, proved in abstract M1 under the
  pinned `P2SlackFamilyUncond` (k = 105), with its instantiation assumption not accepted; never the default.
  `p2-16448/v4` has the spec's (v4) counter-mode SHA-512 keys, `p2-16448/v2` ChaCha8 keys, which the spec does not allow;
- **P3** (`verity.core.protocols.pous.scheme`, the fast research codec): two layers over the band graph `B(220, 12)` with a transpose
  connector and a tweakable `P` (Lean's `p3Scheme`).

It matches the Lean models' definitions and is the oracle for GPU kernels. Pure Python, stdlib plus `verity`. It is
slow by design: a 32 MiB band segment decodes in about 25 s on one core, a dense one in about 8 minutes, and a
14.4 MB P3 segment in about 21 s.

`PROTOCOL.md` describes the lifecycle, the API and the byte formats. `DISCREPANCIES.md` lists where the Lean model,
the prose spec and this code differ.

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
never a process on the prover's node. It commits its indices in a coin tree it keeps until the session is signed, hashes
each whole answer after reading its receipt time (`block_digest`, its `sha512/row/v2` digest: the inner digest of C's
leaf, which `answer_circuit.py` opens), and signs the session's round records once, under a header bound to the setup's
root of `C*`; only then does it release the coin tree's root (`commitment`) and its openings.
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
(`StreamedWipeMeetsServer8H200`). `PROTOCOL.md` ("The
streamed-label wipe") has the messages, the bound, the measured fill and round trips, and the holes. The verdict is decided by its Lean port, `MemoryChallenger.StreamedWipe`
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
