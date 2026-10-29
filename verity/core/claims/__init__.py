"""The vocabulary of security claims, each with a stable id: what a configuration provides (guarantees: soundness, ZK,
non-interactive or designated-verifier, transparent setup), what that rests on (assumptions) and the setting it is proven
in (models).

Id format (shared with the docs site's assumptions registry): an assumption over a primitive is ``<property>/<instance>``,
composing a :data:`PROPERTIES` entry with an :data:`INSTANCES` entry -- ``cr/sha-512`` is collision resistance of SHA-512,
labelled ``CR · SHA-512``, and an algebraic instance makes it ``algebraic``; every other claim is a bare kebab-case id
(``logup``, ``live-verifier``) in :data:`ENTRIES`, and never contains ``/``.  This module is the source of the ids a render
cites.  :func:`get` resolves either.  Only what a render cites is here; a new claim is a new
entry, and an id, once cited, is never renamed.  Stdlib only; imports no other core module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["ENTRIES", "INSTANCES", "PROPERTIES", "Entry", "Instance", "Property", "assumption", "get"]

Kind = Literal["guarantee", "assumption", "model"]
# ``conjectured`` is for mathematics only; ``measured``: an assumption about a device, falsified by a measurement
Standing = Literal["standard", "algebraic", "proven", "conjectured", "measured"]


@dataclass(frozen=True, slots=True)
class Entry:
    """One claim: ``kind``, full ``name``, short ``label``, and for an assumption its ``standing``."""

    id: str
    kind: Kind
    name: str
    label: str
    standing: Standing | None = None


@dataclass(frozen=True, slots=True)
class Property:
    """A security property of a primitive (collision resistance, ...), defined once for any instance."""

    id: str
    name: str
    label: str


@dataclass(frozen=True, slots=True)
class Instance:
    """A primitive a property is assumed of; ``algebraic``: an algebraic hash (Poseidon2, Ajtai digests)."""

    id: str
    name: str
    algebraic: bool = False


PROPERTIES: dict[str, Property] = {p.id: p for p in (
    Property("cr", "Collision resistance", "CR"),
    # success linear in the finder's expected time (Chiesa-Dall'Agnol-Guan-Spooner-Yogev, ePrint 2024/1434, Thm 2); the constant
    # is the assumption's statement (A2 in Security/Proofs/Flock/Soundness/ASSUMPTIONS.md), not restated here
    Property("ecr", "Expected-time collision resistance", "ECR"),
    Property("xof", "Extendable-output-function security", "XOF"),
    Property("sis", "Ring-SIS binding", "SIS"),
    # A3 in Security/Proofs/Flock/Soundness/ASSUMPTIONS.md, for the unit sample's bytes
    Property("uniform", "Independent uniform output", "Uniform"),
    # A4 there, for verity.ml.randomness's keyed streams (prf/sha-512): SHA-512's compression function as a PRF keyed by
    # its chaining value (the cascade over prefix-free inputs, Bellare-Canetti-Krawczyk 1996) and as a dual PRF keyed by the
    # secret bytes in its first message block; its eta is stated there (at most 2^-128 for the chain's tests, for a 32-byte key)
    Property("prf", "Pseudorandom function", "PRF"),
    # of a signature scheme: no forgery on a new message after chosen-message queries, each one-time key used once
    Property("euf-cma", "Existential unforgeability under chosen-message attack", "EUF-CMA"),
)}

INSTANCES: dict[str, Instance] = {i.id: i for i in (
    Instance("sha-256", "SHA-256"),
    Instance("sha-512", "SHA-512"),
    Instance("blake3", "BLAKE3"),
    Instance("blake2b", "BLAKE2b"),
    Instance("shake-256", "SHAKE-256"),
    # PoUW's hiding tile leaf (row pouw/tile-h0/v1): TurboSHAKE128 at L = 32 bytes, domains 0x21 (chunks) and 0x22 (chunk digests)
    Instance("turboshake128", "TurboSHAKE128"),
    Instance("poseidon2-babybear", "Poseidon2-BabyBear", algebraic=True),
    Instance("poseidon2-koalabear", "Poseidon2-KoalaBear", algebraic=True),
    Instance("ajtai-n64", "Ajtai n = 64", algebraic=True),
    Instance("ajtai-n128", "Ajtai n = 128", algebraic=True),
    Instance("io-getrandombytes", "IO.getRandomBytes"),
    # A5: the PoUW window's secret, secrets.token_bytes(32), fresh per window (#423)
    Instance("python-secrets", "Python's secrets"),
    # A6: each round's coins from Flock's Rust coin server (/dev/urandom outside seed mode), sampled after the prover's message
    Instance("flock-coin-server", "Flock's coin server"),
    # verity.ml.crypto.lms: LMS (RFC 8554) over SHA-512, its whole 64-byte output, the network certifier's and the
    # memory challenger's signatures; its EUF-CMA rests on SHA-512 itself: multi-target second-preimage resistance under RFC
    # 8554's prefixes (its §9) and, for keys derived from a seed (its Appendix A), SHA-512 as a PRF keyed by the seed
    Instance("lms-sha512-n64", "LMS over SHA-512"),
)}

ENTRIES: dict[str, Entry] = {e.id: e for e in (
    Entry("statistical-soundness", "guarantee", "Statistical soundness", "Statistical soundness"),
    Entry("computational-soundness", "guarantee", "Computational soundness (argument)", "Computational soundness"),
    Entry("malicious-verifier-zk", "guarantee", "Zero knowledge against a malicious verifier", "Malicious-verifier ZK"),
    Entry("hvzk", "guarantee", "Honest-verifier zero knowledge", "HVZK"),
    Entry("no-zk", "guarantee", "No zero knowledge", "No ZK"),
    Entry("statistical-hiding", "guarantee", "Statistically hiding commitments, given uniform salts", "Statistical hiding"),
    Entry("non-interactive", "guarantee", "Non-interactive, transferable proof", "Non-interactive"),
    Entry("designated-verifier", "guarantee", "Designated-verifier proof (convinces only the live verifier)", "Designated verifier"),
    Entry("rs-proximity-udr", "assumption", "Reed–Solomon proximity gaps, unique-decoding regime", "RS proximity · UDR", "proven"),
    Entry("rs-proximity-johnson", "assumption", "Reed–Solomon proximity gaps up to the Johnson bound", "RS proximity · Johnson", "proven"),
    Entry("logup", "assumption", "LogUp fractional-sum soundness", "LogUp", "proven"),
    Entry("live-verifier", "model", "Live verifier: the verifier's coins arrive online during the proof", "Live verifier"),
    # `Flock.Assumptions.RegistrantCovers` (Security/Proofs/Flock/Soundness/Assumptions/Registrant.lean), a premise of the
    # recursive audit (Flock.Guarantees.RecursiveAuditC3At's _hRC) on the provers it covers; and the salt binder of
    # `TopOf` and `StackedTopAt` (Security/Proofs/Flock/Recursive/HoloExec.lean) and of Flock.Guarantees.RecursiveAuditC3At,
    # the salts the prover's workers committed R's level-0 top rows under, at which the guarantee is applied
    Entry("registrant-holds-openings", "model",
          "Registrant holds its openings: the prover holds a row and salt under every leaf it registered at a row the "
          "verifier's reads file names, and its own workers behind the firewall hold the salts they committed R's level-0 "
          "top rows under; a prover that registers a hash it can't open is outside the guarantee",
          "Registrant holds openings"),
    Entry("fiat-shamir", "model", "Fiat–Shamir (non-interactive) in the random-oracle model", "Fiat–Shamir"),
    Entry("common-reference-string", "model",
          "Common reference string: a public parameter treated as a uniform draw fixed before use, such as the pinned keys of "
          "hm96-sha256/v1 and hm96-sha512/v1 (their hiding bound holds except for a 2^-64 fraction of keys)", "CRS"),

    Entry("random-oracle", "assumption", "A hash behaves as a random oracle", "Random oracle", "standard"),
    # the bandwidth audit v0's device (Bandwidth.Guarantees.AuditRO, Security/Definitions/Accounting/Guarantees/Bandwidth.lean): the
    # guarantee quantifies over devices bounded by S_chip, cap_side * T and T / R_regen, and these say a deployed one is such;
    # a co-audited set is one such device over its members' stores (Bandwidth.Guarantees.AuditSet), and side sources that may
    # carry a peer's tags are Bandwidth.Guarantees.AuditRelay's
    Entry("bandwidth-device", "model",
          "Bandwidth device model: the bound is stated for a co-audited set of whole PCIe-only GPUs (no MIG slices, no NVLink; "
          "NVLink domains are to be certified as sets later) running simultaneous windows over pairwise-distinct stores, with "
          "every device that could hold a copy of a store in the set or isolated, so that the set's side sources are its "
          "external links. The set is a chip whose on-chip state, device memory and side sources are laid out before the "
          "verifier's fresh seed, which reads device memory only through counted ingress, computes every tag on chip, and "
          "commits to the tags before the sample is drawn. A device whose side sources may carry tags a peer computed is "
          "covered by AuditRelay instead, at the deficit c + m * b for m relayed blocks", "Bandwidth device"),
    # falsified by an on-chip working set above S_chip: a pointer-chase sweep on the stated device finding more bytes served
    # faster than device memory (registers, shared memory and L1, L2, any other SRAM), or a datasheet listing more
    Entry("bandwidth-s-chip", "assumption",
          "Bandwidth on-chip state: on the stated device, all on-chip storage (registers, shared memory and L1, L2 and any "
          "other SRAM) holds at most S_chip bits", "Bandwidth S_chip", "measured"),
    # falsified by a measured transfer into the device above cap_side during a window: host-to-device copies over PCIe, peer
    # copies over NVLink or PCIe, and any other path into the chip that bypasses device memory, run at their peak
    Entry("bandwidth-side-cap", "assumption",
          "Bandwidth side channels: during a window of length T, the stated device receives at most cap_side * T bits from "
          "outside its device memory and chip (PCIe from the host and its devices, peer links)", "Bandwidth side cap",
          "measured"),
    # falsified by a faster rebuild of one block of E from setup: for P2 blocks, the measurement behind pous-ring-speed (a
    # root's 16,447 sequential squarings, about 2 ms)
    Entry("bandwidth-regen-time", "assumption",
          "Bandwidth regeneration: rebuilding one block of E from setup takes at least R_regen on the stated device, so a "
          "window of length T rebuilds at most T / R_regen blocks per device that rebuilds", "Bandwidth regeneration time",
          "measured"),
    Entry("bandwidth-instantiation", "assumption",
          "The blocks of E, the PoUS encoding's, are no easier to hold information about than uniformly random blocks, "
          "except by rebuilding them from setup", "Bandwidth instantiation", "conjectured"),
    # the rate audit's device (Accounting.Guarantees.RateAudit, Security/Definitions/Accounting/Guarantees/RateAudit.lean): the
    # bandwidth device with a certified stream on its side link and rebuilds paid in compute
    Entry("rate-device", "model",
          "Rate device model: the bandwidth device model, with the trusted host streaming rho fresh bits to each device over "
          "its side link in every window, after the layout and before the deadline, and every tag absorbing all of them, so "
          "the link carries them and at most L - rho bits of anything else; its side sources are non-adaptive, each "
          "answering a read at an address the chip picks as a function of E fixed before the window, never of what the chip "
          "sent, the stream or the oracle; a word the device neither holds nor reads it rebuilds with the codec's inverse, "
          "whose ops are charged to accounts other than the compute account", "Rate device"),
    # falsified by a partial completion cheaper than x * c_enc / b: x bits of a word of E, rebuilt from setup, the weights
    # they encode and the rest of the word, in less than that of the compute account's units, by any schedule. At k = 2
    # the red team's 4-dimensional lattice completes a tail of up to b/10 bits (r20261012-163319-2804), and fplll, the
    # best of five reducers, costs 12.3x the claim at 512 bits (r20261012-164518-ccba); the 1.2x at the reduction's
    # operation-count floor is an estimate from operation counts, unmeasured (r20261012-170958-f49a). At k = 1 with the
    # payload known, F27 (r20261012-132841-565e) is such a completion, at a floor of half a GMP extended GCD
    # (r20261012-164518-ccba). In partial-completion form it is `Accounting.Assumptions.WorkHardness`
    # (Security/Definitions/Accounting/Assumptions/RateAudit.lean)
    Entry("rate-codec-work", "assumption",
          "Rate codec work: on the stated device, for a codec with k >= 2 rounds, rebuilding x bits of a word of E that it "
          "neither holds nor reads, from setup and the weights they encode, costs at least x * c_enc / b units of compute "
          "(b the word's bits), with no amortization across words; at k = 2 the best completion measured (keeping a "
          "word's top nine-tenths, a 4-dimensional lattice completes the rest) costs 12.3 times that at 512 bits, and the "
          "reduction's operation-count floor is estimated, not measured, at 1.2 times it on a GPU; at k = 1 it is false "
          "when the payload is known (F27: keeping a word's top two-thirds, a 2-dimensional lattice completes the rest at "
          "a floor of 0.025 of a root at 512 bits, down to 0.003 at 2,048, against a per-bit price of 0.32), and holds "
          "only for words whose payload is unknown, under Incompressible(W)",
          "Rate codec work", "measured"),
    # the sketch-then-hash receipt's model: falsified by a prover that passes while holding
    # some tier's share of the store as anything other than E's blocks. Accounting.Guarantees.RateSketchTable is why the
    # model is needed: a prover whose memory may hold any function of E (RateAudit's model) passes on every coin reading
    # only the digest's d bits per tile, from a table of |signs| * N * d bits, at any number of rows. What Lean proves is
    # the narrower verbatim class (Accounting.Guarantees.RateAuditSketch, RateAuditTensorSketchRank: on-chip state, device
    # memory, the side link and rebuilding each return whole sectors as they are); the restated form, where a device that
    # computes from blocks it read has those reads counted, is the remote audit's composition, not yet proved for the sketch
    Entry("rate-sketch-verbatim", "model",
          "Rate sketch store-as-blocks model: in every tier the prover holds the store only as its blocks (any number of "
          "copies, anywhere), and any device may compute from the blocks it reads, every such read counted; a layout from "
          "which sketches cost less to read than the blocks themselves, such as a table of precomputed sketches, is "
          "outside it", "Rate sketch store as blocks"),
    # what lets the stacking guarantees treat the blocks a pass reads as uniform when every appended block is keyed by
    # a live coin flipped after its root commits its digest, the writers' choices for each append fixed before its coin
    # and the pass's layout fixed apart from the encodings: `Accounting.Assumptions.KeyBinding`
    # (Security/Definitions/Accounting/Assumptions/Stacking.lean); the seed's key alone lets a writer choose a block's
    # encoding, and each coin keys a block apart (verity/core/protocols/remote_bandwidth/tests/test_key_binding.py)
    Entry("key-binding", "assumption",
          "Key binding (append-coin form): for the protocol's codec, whatever the writers choose for each append from "
          "the codec and the coins of earlier appends (the registration being append 0, whose coin is the seed; every "
          "appended block keyed by b_j(c_a) = H(DOMAIN || seed || j || d_j || c_a), its append's 32-byte coin c_a drawn "
          "once, after the append's root has committed every digest d_j), and for any layout of a pass's blocks fixed "
          "apart from the encodings, the blocks the pass reads are within delta of uniform on every event; in the "
          "ideal model a round is a random function on pairs {x, p - x} (rho(p - x) = p - rho(x), its output's "
          "parity its input's Legendre symbol), so delta is at most the writers' advantage at anticipating an "
          "append's key before its coin, the chance that two round inputs agree up to sign, and the words' values "
          "at or above p; one live registration per node; a layout chosen from the encodings after their coins is "
          "not covered",
          "Key binding", "conjectured"),
    Entry("hash-derived-key", "assumption",
          "A key derived as counter-mode hash output over a public label is not in the 2^-64 fraction of keys for which the "
          "pinned-key hiding bound (N * 2^-192) fails: hm96's pinned keys are H(label || u32be i), with H the instance's hash "
          "(SHA-256 for hm96-sha256/v1, SHA-512 for hm96-sha512/v1)", "Hash-derived key", "standard"),
    Entry("honest-verifier", "assumption", "The verifier follows the protocol", "Honest verifier", "standard"),
    # `Verity.Assumptions.TrustedHost` (Security/Definitions/Core/Assumptions.lean) states its part about one run; no
    # attestation hardware backs it
    Entry("trusted-host", "assumption",
          "The auditor's device runs its stated code, timing included, on a host the prover doesn't control, with its CPU "
          "isolated from other tenants and a monotonic clock; the host never restores its earlier state", "Trusted host",
          "standard"),
    # `Accounting.Covered` (Security/Definitions/Accounting/Physical.lean) is its Lean form: every boundary link of the
    # location is certified (uncertifiedCapacity = 0). The auditor's placement in the onsite protocol; nothing proves it
    Entry("certifier-coverage", "assumption",
          "Every uplink of the spatial unit passes through a network certifier the auditor trusts, with its parameters "
          "fixed before the run: every boundary link is certified, and nothing leaves the unit another way",
          "Certifier coverage", "standard"),
    # `Accounting.Placed.LatencyAtMost` (Security/Definitions/Accounting/Physical.lean) is its Lean form. Falsified by a
    # delivery slower than the ceiling, measured on the server's certified egress links
    Entry("latency-ceiling", "assumption",
          "Latency ceiling: every wire from the audited server to the network certifier on its egress delivers within "
          "the stated ceiling of its send, and nothing between them holds a value longer",
          "Latency ceiling", "measured"),
    Entry("transparent-setup", "guarantee", "Transparent setup (no trusted setup)", "Transparent setup"),
    # `Pous.EndToEnd.EndToEndRing`'s steps to the deployed system (Security/Properties/Pous/EndToEnd.lean), beside
    # prf/sha-512 for the memory challenger's index streams: the model, the numbers and the instantiation.  Its one
    # session per audit cites trusted-host.
    Entry("pous-ring-device", "model",
          "PoUS ring model of the server: the deployed server is a ring-model server whose state is bounded between "
          "setup, the session and its rounds, with one session per audit (the memory challenger refuses a used session "
          "id from its session log, and its host never restores an earlier copy of the state the device keeps there: "
          "trusted-host, pending network-accounting's widening) and the memory challenger's coins drawn online, "
          "after its state is fixed", "PoUS ring device"),
    # its falsification test is the measurement behind ROOT_CALL_NS (verity/core/protocols/pous/PROTOCOL.md, P2's SeqRoot):
    # the timing red team's modelled floor, note:20261004T2102Z-report-relay-p2-redteam-timing
    # (art:23a323f313d2193873c59ee8ab59a0f1197167a0b4c6d008346d771c3e72de57), and the 8-core cooperation run
    # r20260929-060834-0ec4 (art:b92b6d56fa12cc8e04b3f6843991c2e76f49366adc89140a96c33842f70b41ed); a faster squaring on the
    # stated device falsifies it
    Entry("pous-ring-speed", "assumption",
          "PoUS ring speed: on the stated device (up to 8 cores of one chiplet cooperating, each at least as slow as a "
          "Zen 5 core at 5.7 GHz with full-rate IFMA), within Δ + RTT (at most 854 µs) no server finishes a 16,447th "
          "sequential multiplication round (one root's depth, so D = 16,446), more than Q = 2^20 key queries or more than "
          "L = 2^64 ring operations per answer; its margin is a modelled floor cut by a measured speedup: the timing red "
          "team's modelled single-core floor of 361.55 ns per 16,448-bit squaring (that Zen 5 core, regate.py) over the "
          "measured 8-core cooperation speedup of 3.490× is about 103.6 ns per cooperating squaring, so 16,447 rounds take "
          "1.995× Δ + RTT, and 1.64× and 1.25× under the timing red team's sync model",
          "PoUS ring speed", "measured"),
    Entry("pous-ring-instantiation", "assumption",
          "P2 with the identity labeling, SHA-512-CTR keys and its published Merkle root of C* is no easier to answer "
          "than p2ring with a uniformly random odd labeling and uniformly random keys", "PoUS ring instantiation",
          "conjectured"),
    # `Pouw.PearlC.GγPerCoin (W1.w1Priced dv salt)` (Security/Definitions/Pouw/PearlC/W1Stop.lean): the one open hypothesis
    # of the compute audit's per-run bound at W1 priced (Accounting.Guarantees.ComputeOfPearlCStopPerRunPriced); it implies
    # the γ-gap property Gγ itself (Pouw.PearlC.W1Stop.gγ_of_withStop)
    Entry("pouw-gamma-per-coin", "assumption",
          "Pearl-C's γ-gap property with the budget per coin, at W1 priced (its arithmetic at W1's instruction prices and "
          "each word it first fetches at the fetch price): for every adversary within q oracle calls, the probability "
          "that its online cost is at most T and the honest work of its correct units exceeds T / (1 − γ) is at most "
          "η(q, N)", "PoUW γ-gap per coin", "conjectured"),
    # `Pouw.Assumptions.HonestClock` and its Pearl-C form `Pouw.PearlC.Assumptions.HonestClock`: the premise by which the
    # deadline guarantees (Pouw.Guarantees.WorkBetweenSaltAndDeadline, SpareCapacityAtDeadline, PearlCSampledByDeadline
    # and its sm_120 instances) read the verifier's clock in the cost model
    Entry("honest-clock", "assumption",
          "Honest clock: the cost behind a transcript the verifier receives by the deadline is spent after the salt's "
          "release and by the deadline, so the prover's computation stopped at the deadline outputs every transcript "
          "received by then at no more cost than the run spent in that window and no more oracle calls than its online "
          "phase. It fails if a transcript computed after the deadline can be stamped as received by it (a clock the "
          "prover can slow, a receipt it can backdate), or if the cost model charges that computation less than the run "
          "spent on it. The PoUW form is proved on the actual model's clock (Accounting.SecurityProofs.Actual."
          "honestClock); the Pearl-C form is not proved", "Honest clock", "standard"),
    # EPS1's tensor–scalar model (Security/Definitions/Accounting/Assumptions/TS.lean, note:20261017T0324Z-draft-eps1-spec):
    # the named assumptions Theorem D-ideal (I1)–(I8) and the compiler lemma rest on, and the compiler's per-stage form
    # (Security/Definitions/Accounting/Assumptions/Stages.lean)
    Entry("accounting-ts-facet-ceilings", "assumption",
          "Measured facets are ceilings for the stated device on the measured vector its statement names (sm_120 RTX "
          "PRO 6000: VF3s-16, the measured headline's (I1)): on every run the counts and cycles are nonnegative and the "
          "SM clock is in (0, f_max], each SM-clocked facet's load is at most the run's cycles, and each fixed-clock "
          "facet's load is at most f_max times its time", "TS facet ceilings", "measured"),
    Entry("accounting-ts-words-written", "assumption",
          "The owned right fixed words a writer kind heads are at most its ω times the instructions of its class the "
          "labelling gives them, ω the model's: an MMA form's m·n on a data sheet (128 for m16n8, 64 for m8n8k16), and "
          "on a measured vector the committed words an instruction of the form writes, m·n·k_dense/(32·split) rounded "
          "up (sm_120's VF3s-16: 128 for IMMA.16832, 32 for IMMA.8816); 1 for a uniform instruction (UNI) and 2 for a "
          "64-bit one (UNI64, on DS-120's model only)", "TS words per instruction", "measured"),
    Entry("accounting-ts-scalar-route", "assumption",
          "SCALAR-ROUTE at a per-instruction cap κ (every ALU instruction creates at most κ incidences): every right "
          "fresh gather-free vector-headed committed word costs at least 2.03 lane-instructions net of the fold and of "
          "operand delivery (for κ < 27.2 a slot), and one whose production reads a data-dependent load at least 1.22 "
          "lane-instructions (for κ ≤ 128) and 32(28 − 1.03κ)/(c_L − κ) bytes read from its memory level, 3.0 at κ = 8 "
          "(QSTAR's joint statement, from class P and the cap κ; no cap is proved, and ISA-CAP's 32/7 is false). "
          "sm_120's measured headline takes it at q = q_T = 9/8, ClassPSlots' slots once writers create at most 15 "
          "incidences, a cap nothing proves: its 0.591% rests on that. At A6's ½ floor, which Lean proves from A6 "
          "(SheetScalarRouteHalf), the same bound is 7.791% (10.24% on VF3r-16; r20261019-172339-a93b). The H200's "
          "twin headline on DC-90-ri (Sm90MeasuredI3Headline) takes it at q = q_T = 3/5, beside the proved ½ floor "
          "(Sm90MeasuredHeadline), not in its place: ClassPSlots counts a slot a word but takes a writer's slot as "
          "writing nothing else, which a 64-bit head writing two words a lane (IMAD.WIDE, DFMA) breaks. Both are "
          "over 1%. Cite it with its premise: DC-90-ri 1.086% under (I3) at 3/5; 1.412% at the proved ½ floor; "
          "credit against the built ♭, the cheater priced at every known fold, counted ones included",
          "SCALAR-ROUTE", "conjectured"),
    Entry("accounting-ts-side-units", "assumption",
          "A side unit's committed words cost at least its measured per-word vector", "TS side units", "measured"),
    Entry("accounting-ts-nosum", "assumption",
          "NOSUM: on the stated device no unit outside the route menu heads a fresh committed word", "NOSUM",
          "measured"),
    # What the declared capacity Π rests on, per node and per window (ASK-74 = A; note:20261019T0455Z-draft-rt-scope,
    # "Device identity and clock"): ε is relative to Π (decision 100), and ASK-65 = A made the floor the node's. Its
    # trust is the auditor's inventory of the node, as certifier-coverage's is the auditor's placement; nothing checks
    # it at run time but the prover's own driver, and NVIDIA's attestation states no clock and can't rule out an
    # undeclared GPU
    Entry("accounting-declared-device", "assumption",
          "The node's declared devices are its devices for the window: every GPU on the node is declared, each is the "
          "declared board and vBIOS, and no SM clock exceeds f_max, the top of that vBIOS's clock table (MIG and MPS "
          "need no clause: they can't raise the node's capacity). Trust: the auditor's inventory of the node, onsite "
          "or the host provider's instance record", "Declared device", "standard"),
    # The data-sheet axioms A1–A7 (Security/Definitions/Accounting/Assumptions/Sheet.lean, note:20261017T1818Z-draft-axioms):
    # vendor statements that build the data-sheet machine and give (I1)–(I5) at the ½ floor
    Entry("accounting-sheet-clock", "assumption",
          "A1 CLOCK: the SM clock never exceeds the device's f_max, from the device's own source. Assumed: sm_120's "
          "2430 MHz, like the B200's f_max, is the maximum SM clock its driver reports (nvidia-smi), not a data "
          "sheet's, the H200's 1980 MHz is EPS1 DC's measured f_max at default clocks (its sheet quotes TOPS at "
          "1830 MHz), and nothing checks a clock offset at run time; our nodes' driver allows none (NVML's offset range "
          "[0, 0], r20261019-121222-faa3)", "A1 clock (assumed)", "measured"),
    Entry("accounting-sheet-dispatch", "assumption",
          "A2 DISPATCH: each SMSP issues at most one warp instruction a cycle, whatever its pipe, uniform-datapath "
          "instructions included (A2u, which no vendor document states), and every warp instruction takes at least one "
          "issue slot whether or not its class is listed: one no other class names counts in the device's catch-all "
          "class at its issue slot (NOROW on sm_120 and on the H200's and B200's sheets, UNLISTED on the measured issue "
          "row on their measured readings, and on DC-90-ri the JI90 lines too, its instructions timed beside the "
          "IGMMA, r20261021-070616-d06c); a reading whose classes have none (sm_120's measured "
          "reading) leaves it free. A bulk or TMA instruction of size S counts S/99 KB in sm_120's UTMABIG, at most "
          "its one slot (A5b's count)",
          "A2 dispatch (whitepapers)", "measured"),
    Entry("accounting-sheet-tensor", "assumption",
          "A3 TENSOR: the tensor cores do at most the data sheet's dense MACs per SM clock for each operand type",
          "A3 tensor (data sheet)", "measured"),
    Entry("accounting-sheet-pipe", "assumption",
          "A4 PIPE: the instruction classes Nsight Compute puts on one pipe share that pipe's CUDA Programming Guide "
          "throughput", "A4 pipe (Programming Guide)", "measured"),
    Entry("accounting-sheet-smem", "assumption",
          "A5 SMEM: shared memory serves one request per SM clock, its distinct banks together (32 banks × 4 B = 128 "
          "B), loads and stores together, so each shared-memory instruction holds one request whatever its width; "
          "A5b, which no vendor document states: a bulk or TMA instruction holds its size S of bank bytes at that "
          "rate, what it reads or writes on one side of a block's shared memory (RT4-SHEET-1's fold on the bulk engine)",
          "A5 shared memory (Programming Guide)", "measured"),
    Entry("accounting-sheet-words", "assumption",
          "A6 WORDS: an instruction writes at most its destination registers' 32-bit words (its PTX fragment; two a "
          "lane for a lane instruction)", "A6 words (PTX ISA)", "measured"),
    Entry("accounting-sheet-semantics", "assumption",
          "A7 SEMANTICS: each instruction computes the function its ISA defines and writes nothing else, so no unit "
          "outside the route menu heads a fresh committed word and a side unit's words cost its per-word vector",
          "A7 semantics (PTX ISA)", "measured"),
    Entry("accounting-ts-fold-opt", "assumption",
          "FOLD-OPT: no program folds N committed words at a certificate's price below (1 − μ_f)·N·φ*(π). The TS "
          "statements ask it at every certificate π (FoldOpt), where a bit-exact fold falsifies it on DS-120 "
          "(RT2-LOGIC-1's IMAD rotate) and an ALU fold of ♭'s IMADs on the H200's and B200's readings (RT3-LOGIC-6); "
          "the headlines (sm_120's, the H200's and the B200's) ask it only at their own certificate π*, N the words "
          "the run folds (FoldOptFoldedAt), an open conjecture checked within a stated menu by test_lean_headline.py "
          "(the Lean menu, the bit-exact lane folds, the bulk-reduction floors and fold_bulk.cu's whole-block folds) "
          "and on bit-exact folds by test_lean_headline_dc.py; at DC-90-ri's π* the counted, unbuilt memory "
          "round-trip folds mixed with IMAD.WIDE (RT4-SHEET-13, -14) price at 1.0245 φ* or more, their classes on "
          "the JI90 lines the H200 measured beside the IGMMA (r20261021-070616-d06c, r20261021-074303-da7f)",
          "FOLD-OPT", "conjectured"),
    Entry("accounting-ts-fill-opt", "assumption",
          "FILL-OPT: no program runs the verified filler at a certificate's price below (1 − μ_x)·χ*(π), for every "
          "certificate π", "FILL-OPT", "conjectured"),
    Entry("accounting-ts-fan-in", "assumption",
          "An instruction merges at most m_k components, Σ(sources − 1) over its nodes, for every SASS form on sm_80, "
          "sm_89, sm_90, sm_100 and sm_120, counted by dependence: every register, uniform-register and predicate "
          "word it reads, its guard, a constant bank, the memory node, and every word and predicate it writes, which a "
          "guarded instruction keeps (a lane: ALU 9, a guarded IADD3.X with a cx[UR] operand on sm_80–sm_90, 8 on "
          "sm_100 and sm_120; VIMNMX 10, IMNMX.U64; NOROW 10, P2R; a shared load or store 7, a global store 13, a "
          "shared atomic 15, a global atomic 18, a texture 20; a warp instruction: SHFL 191, IMMA.16832 480, sm_100's "
          "LDTM.32x32b.x128 4098)", "TS fan-in", "measured"),
    Entry("accounting-ts-luck", "assumption",
          "(L) at p with freshness (F1)–(F2): given the past, a round with a deficient block passes with probability at "
          "most p", "TS luck at p", "proven"),
    Entry("accounting-ts-stage-luck", "assumption",
          "(L) at p under per-stage coins: in every round, at every stage, whatever the past, a unit of the stage that "
          "the cheater skips passes its proof with probability at most p over the stage's coin",
          "TS stage luck at p", "proven"),
    Entry("accounting-ts-receipt-luck", "assumption",
          "The receipt luck at pr: given the past, a deficient round whose deficient words are not all right keeps the "
          "honest committed value with probability at most pr (½ at LEDGER case (b), two or more wrong places a set), "
          "and a round at a unit whose fold the run skips passes with probability at most p₀ (½, B′'s value, per live "
          "coin; the whole-unit guess bound 2^-64 is not granted)",
          "TS receipt luck at pr", "conjectured"),
    Entry("accounting-ts-operand-h-mu", "assumption",
          "(H-μ) at the receipt's operands: given the past, a round's deficient words take any one value with "
          "probability at most μ given a view of the cheater that decides the round's share (¾ in the headlines)",
          "TS operand H-μ at μ", "conjectured"),
    Entry("accounting-ts-proofs-sound", "assumption",
          "Binding at δ_bind and proof soundness at δ_ZK, over a window of S rounds of n_π proofs whose leaves are each "
          "round's values committed before its σ. The TS statements take them in one premise with the protocol's "
          "order, pinning (accounting-ts-proofs-pin) and E1(c) (accounting-ts-right-fixed), ProofsSound: outside an "
          "event of probability at most S·(δ_bind + n_π·δ_ZK), an accepted window is accepted by the ideal game and its "
          "owned right fixed words per unit are at least (1 − Σ_w d_w/S)·N_w, rightness read on the path's coins. The "
          "headlines (sm_120's, the H200's and the B200's) take the two alone as named premises: binding in its fork "
          "form at δ_fork (BindingForkAt: at every round w, on a run and on its continuation from w on fresh coins, "
          "which share w's commitment, every proof of round w is accepted and some proof opens w's leaves differently, "
          "with probability at most δ_fork), read at δ_bind = √δ_fork through BindingOfFork; and each proof's "
          "soundness at "
          "δ_ZK (ProofSoundAt: a proof is accepted with an opening that fails its relation with probability at most "
          "δ_ZK), which Flock's EndToEndHiddenDrawn gives only for a cheater whose hash calls over the window are "
          "capped, and only against its committed transcript: that the relation's rows are the leaves' committed rows "
          "is not stated there", "TS proofs sound", "standard"),
    Entry("accounting-ts-proofs-pin", "assumption",
          "Pinning: the relations a window's proofs pass pin every committed value, so values that satisfy every "
          "proof's relation are right for their derivations on the path's coins (ProofsPin, ProofLayer.Pins). It is a "
          "property of our circuit Definitions, and no Lean checks it of the served relations: PouwSegCheck_v2 lacks it "
          "(it ties only the drawn segment's set to the one before, so its window's other absorbed prefix-word sets "
          "are free witness), PIN's checks (PouwSegCheck_v3, PouwCallCheck_v1, PouwWarpCheck_v1 and _v2, #2417) "
          "recompute every word they absorb, and nothing states that the served relations are PIN's", "TS proofs pin",
          "conjectured"),
    Entry("accounting-ts-accepted-live", "assumption",
          "A proof is credited only after flock-verify --zk --registered accepts its session. Every served path today "
          "verifies with plain flock-verify --zk and binds its served values through the public file (served_zk's Lean "
          "checks, served_commit's verdict, h1_prove); of the audit's runners only one_stage/a0.py passes --registered. "
          "So no served path meets it until the served trees are registered as ports, or a bridge for public-file "
          "binding is stated", "TS accepted live", "standard"),
    Entry("accounting-ts-window-law", "assumption",
          "The proof system's draws and audit coins are the window's live coins: at each round and proof, given what "
          "the path fixes before the session (its setting and its prover), the session's coins are the path's next "
          "coins, fresh and uniform. This rests on five facts: a refused session is never rerun on a fresh draw (no "
          "retry or redraw); the draw is revealed only after the session's registration and prover are fixed; the draw "
          "is no coin the setting reads (not σ_w, nothing derived from σ_w or from a prover's salt, not another "
          "session's draw); the path carries the draw (a later round's coin, or a draw beside σ_w in each round's "
          "coin, which the last round needs); and the session's audit coins (Flock's --zk coin tree) come from the "
          "auditor's verifier, on a machine the prover doesn't control, fresh for each session (trusted-host). "
          "Under full checking the draw is every unit, but the law still holds the audit coins fresh and uniform given "
          "the prover. served_zk's per-tile sessions today draw their coin tree from a serve --zk --operator loopback "
          "beside the prover, which doesn't meet the fifth fact", "TS window law", "standard"),
    Entry("accounting-ts-served-checks", "assumption",
          "A proof's relation (its served Definition's check) holds at its Flock session's extracted value layer "
          "wherever flock-verify accepts the session and every drawn unit is right at its own owned reads; under a "
          "sampled draw, only for a relation that reads the drawn units alone", "TS served checks", "conjectured"),
    Entry("accounting-ts-served-roots-linked", "assumption",
          "Flock's rows link covers the served roots: outside δ_s, where flock-verify accepts a session and every "
          "drawn unit is right at its owned reads, the session's extracted value layer is the served layer's opening "
          "(the round's served leaves, committed before σ_w)", "TS served roots linked", "conjectured"),
    Entry("accounting-ts-leaf-checks", "assumption",
          "Each of a leaf's 33 option-A statements (h1zy.PouwH1LaneZy_v1's 32 lanes and PouwH1Combine_v1, at their "
          "tile's K) computes its Definition at its decoded port rows, registered or not: on every wire assignment its "
          "circuit's evaluation agrees with, an ok word of 1 means the lane's partials row is the honest one and its "
          "32 cells have z = A·Bᵀ and y = dq(z), or for the combine that the rows' XOR is the receipt. That every port "
          "is registered and opens at its registered root is proved from flock-verify's complete check, not assumed",
          "TS leaf checks", "conjectured"),
    Entry("accounting-ts-ok-realizes", "assumption",
          "The checked unit's Flock circuit is its declared unit's: on any wire assignment the key program's "
          "evaluation agrees with at a unit, whose ok wires carry the ok row (the word 1), every check the declared "
          "circuit makes of that unit holds (the Σ check and the dequant check). Nothing in Lean derives a Flock "
          "circuit from a declared one or checks one against the other, and circuit-check binds the Definition to its "
          "Boolean lowering, not to the declared circuit", "TS ok realizes", "conjectured"),
    Entry("accounting-ts-tile-checks", "assumption",
          "A PouwH1TileZy_v1 session's circuit computes its Definition at its decoded registered ports: on any wire "
          "assignment the key program's evaluation agrees with at a unit, whose ok wires carry the ok row (the word "
          "1), every cell of the unit's decoded view has z = A·Bᵀ and y = dq(z). It holds of the circuit only if "
          "every input its ok reads is a registered port, which flock-verify's complete check establishes",
          "TS tile checks", "conjectured"),
    Entry("accounting-ts-lin-steer", "assumption",
          "LIN-STEER, false for a segmented consumer whose noise-active segments carry unequal counts: zeroing a "
          "fraction φ of a row's noise-active coordinates saves at most φ of the consumer's work",
          "LIN-STEER", "conjectured"),
    Entry("accounting-ts-right-fixed", "assumption",
          "Lemma E1(c) with the rounds' equal shares: on an accepted window whose committed values are right for their "
          "derivations on the path's coins, every word that isn't fixed is in its round's share", "TS right fixed",
          "proven"),
    Entry("accounting-ts-unit-fixed", "assumption",
          "Lemma E1(c) per unit with its fold clause, which doesn't read the draw: on an accepted window the fixed "
          "words per round are at least the good units' words over the rounds, a good unit being one the run computed "
          "(not skipped and guessed), with every unit writing it a value after the coin its noise reads, so a unit "
          "whose check passes by luck isn't credited; and the folded words per round are at least the words of the "
          "units not skipped whole over the rounds", "TS unit fixed", "proven"),
    # the two premises UnitFixed rests on under ASK-40's option A (Security/Definitions/Accounting/Assumptions/TS/Fixed.lean)
    Entry("accounting-ts-modes-run", "assumption",
          "The modes are the run's: a unit the run paid for has its words and the values it writes honest, from the "
          "values the layer committed, under its noise, and a unit not skipped whole has its receipt commit the fold of "
          "its words", "TS modes run", "standard"),
    Entry("accounting-ts-counts-fixed", "assumption",
          "The labelling counts fixed words as fixed: on an accepted window the owned right fixed words per round are at "
          "least the words of the units that are fixed under option A, computed, and with every unit steering them "
          "computed, over the rounds, and the folded words at least the words of the units whose receipts commit their "
          "folds", "TS counts fixed", "conjectured"),
    # `Claims.ZKServer.Loop.Src` (Security/Proofs/Flock/SessionLoop/Statement.lean) is its Lean form, a premise of
    # Flock.Guarantees.SessionLoopZKDeployed; its two halves are `Loop.Level0Src` and `Loop.RepSrc`
    Entry("flock-firewall-os-src", "assumption",
          "Every input the firewall's session loop takes from its host and its operating system is the session tape's: "
          "for each table's level 0, the witness's packed table at the tape's mask words and the tape's extra lanes, lane "
          "paddings and level-0 salts; for each rep, its pads, its pads tree's salts and its salts for levels 1 and up. "
          "The loop runs on the auditor-trusted host (trusted-host), whose operating system's random bytes are the "
          "uniform draws the tape models", "Firewall OS source", "standard"),
)}


def assumption(prop: str, instance: str) -> Entry:
    """The composed assumption ``<prop>/<instance>``; ``KeyError`` for an unknown property or instance."""
    p, i = PROPERTIES[prop], INSTANCES[instance]
    return Entry(f"{p.id}/{i.id}", "assumption", f"{p.name} of {i.name}", f"{p.label} · {i.name}", "algebraic" if i.algebraic else "standard")


def get(id: str) -> Entry:
    """The entry with this id, composed or not; ``KeyError`` when there is none."""
    if id in ENTRIES:
        return ENTRIES[id]
    prop, sep, instance = id.partition("/")
    if not sep:
        raise KeyError(id)
    return assumption(prop, instance)
