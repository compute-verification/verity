"""Drill-downs D1-D3 behind the canonical tables: non-canonical; layout fixed 2026-09-24 (campaign morning-tables BRIEF §1).

    python -m verity_numerical.bench.drilldown --root <store root> [--format md|json]

Read-only over the evidence store :mod:`verity_numerical.bench.tables` renders from, with its validity predicate
(:func:`tables.reject_codes`, imported, never restated):

* **D1 variants and security properties** -- one row per :data:`VARIANTS` entry: Variant | ZK | Operands hidden from verifier |
  Binding rests on | Soundness | In Table 2? (which column, or why not).  Definitional: the registry's words, taken from Table 1,
  the result envelopes' own ``security`` statements, the lane reports and the backend READMEs each entry cites; "unknown" where
  none states the fact.  The last column also names the Table 2 cells the variant holds in this store.
* **D2 fastest measured t.total per variant x target** (A100 BF16, H100 BF16, H100 FP8, 4090 FP8, 5090 NVFP4): cell
  ``seconds · overhead×`` of the variant's fastest result on that target, then ``✓`` when it is Table-2-valid, else the letter
  of its first reason (:data:`tables.REASON_LETTERS`); a footnote names the artifact and the reason.  Fastest = lowest
  ``R_native_spec / R_proved`` with ``R_proved = 2KB / t.total`` from the result's own B and K: the minimum ``t.total`` over the
  Table 2 workload (B = 4096, K = 1536), and a smaller off-workload run cannot hide it.
* **D3 communication and verifier cost, one row per Table 2 cell**: proof bytes per batch | prover -> verifier Gbit/s to keep
  pace (bytes x 8 / t.total) | verifier -> prover bytes (coins) | rounds | verifier CPU s per batch | verifier cores to keep pace
  | same-datacenter live tax (t.total_live / t.total), from the cell's result envelope only (:data:`D3_FIELDS`); ``…`` where the
  envelope lacks the field, and each row lists the fields it read.

A ``bench-result/v1`` artifact whose meta is a lane's summary but whose JSON payload is the result envelope counts in D2 with
that envelope and the letter of its own first reason, else ``X``: tables.py reads the meta only, so as registered it can never
be a Table 2 cell.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from verity.verification.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, NVFP4_SM120, TARGETS, Target
from verity_numerical.bench import contract
from verity_numerical.bench import tables as T

__all__ = ["D1_COLUMNS", "D2_COLUMNS", "D3_COLUMNS", "D3_FIELDS", "LAYOUT", "TARGET_COLUMNS", "VARIANTS", "Drilldowns", "Variant",
           "build", "classify", "features", "main", "render_json", "render_markdown"]

LAYOUT = "non-canonical; layout fixed 2026-09-24 (campaign morning-tables BRIEF §1)"
#: D2's columns, BRIEF §1's order: (label, target).
TARGET_COLUMNS: tuple[tuple[str, Target], ...] = (("A100 BF16", FIRST), ("H100 BF16", BF16_HOPPER), ("H100 FP8", FP8_HOPPER_WGMMA),
                                               ("4090 FP8", FP8_ADA_MMA), ("5090 NVFP4", NVFP4_SM120))
_TARGET_LABEL = {t.name: label for label, t in TARGET_COLUMNS}
D1_COLUMNS = ("Variant", "ZK", "Operands hidden from verifier", "Binding rests on", "Soundness", "In Table 2?")
D2_COLUMNS = ("Variant", *(label for label, _ in TARGET_COLUMNS))
D3_COLUMNS = ("Table 2 cell", "Proof bytes per batch", "Prover → verifier Gbit/s to keep pace", "Verifier → prover bytes (coins)",
              "Rounds", "Verifier CPU s per batch", "Verifier cores to keep pace", "Same-datacenter live tax")
#: The envelope fields D3 reads (``contract.UNITS`` names, plus the emitters' ``proof_bytes`` and the live evidence's session
#: count), in the order a row lists them.
D3_FIELDS = ("proof_bytes", "transcript.bytes", "net.bytes_in", "validation.evidence.live_verifier.sessions", "rounds.protocol",
             "rounds.sequential_depth", "verify.cpu_s", "verifier.seconds", contract.T_TOTAL_LIVE)
PASS = "✓"
PAYLOAD_ONLY = ("envelope only in the artifact's JSON payload (its meta is a lane summary): tables.py reads the meta, so as "
                "registered this result cannot be a Table 2 cell (register the result.json as --meta)")


# --- recognising a result's variant -----------------------------------------------------------------------------------------------

_RELATION = re.compile(r"^b-ligero chain, ([a-z0-9.-]+) relation")
_FAMILY = re.compile(r"-(v[23])(x4)?$")
_X4 = re.compile(r"\bx4\b|folded: ?4\b|4 (instruction )?steps per column")
_AJTAI_N = re.compile(r"ajtai\D*?(\d+)")


def _s(v: Any) -> str:
    return v.strip().lower() if isinstance(v, str) else ""


def _get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, Mapping):
            return None
        d = d.get(k)
    return d


def features(env: Mapping[str, Any], meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What :data:`VARIANTS` recognise a result by, from its envelope ``env`` (and the artifact's ``meta`` when that is a lane's
    summary beside a payload envelope):

    ``backend`` (``software.backend.name``, lower case); ``candidate`` (the Table 1 candidate its name selects, or ``Ligerito``);
    ``family`` (B-Ligero's relation: ``v1``, ``v2`` = ``pubsel``, ``v3`` = ``privsel``, from the name's ``<target>-v2`` / ``-v3``);
    ``fold`` (4 for the x4 relations: ``…x4 relation``, ``folded: 4 instruction steps``, ``relation``/``model`` "4 steps per
    column"; else 1); ``fused`` (the fused device hint kernel: meta ``hints = fused`` or ``software.backend.hints`` saying so);
    ``l`` (``security.rs_l``, columns per proof); ``mode`` (``interactive`` | ``fiat-shamir``, :func:`tables._mode_of`);
    ``proof_class``; ``authentication``; ``leaf`` (``poseidon2`` | ``blake3`` | ``ajtai-n<n>`` from ``hash`` / ``leaf.scheme`` /
    ``leaf.schema``); ``sharing`` (``sharing``, or ``row_sharing`` present); ``device`` (A-GKR's prover: ``gpu`` when the name,
    ``backend`` or toolchain says gpu / triton / torch / cuda, else ``cpu``); ``tcdot`` (the TC_DOT chip: ``tc-dot`` / ``tc_dot``
    / ``tcdot`` / ``modified sp1`` in the name)."""

    fp = env.get("workload_fingerprint") if isinstance(env.get("workload_fingerprint"), Mapping) else {}
    meta = meta or {}
    name = _s(_get(fp, "software", "backend", "name"))
    cand = T.candidate_for(name)
    f: dict[str, Any] = {"backend": name, "candidate": cand.id if cand else ("Ligerito" if name.startswith("ligerito") else None),
                         "family": None, "fold": 1, "fused": False, "l": _get(fp, "security", "rs_l"), "mode": T._mode_of(fp),
                         "proof_class": fp.get("proof_class"), "authentication": fp.get("authentication"), "leaf": None,
                         "sharing": fp.get("sharing") if isinstance(fp.get("sharing"), str) else ("row" if fp.get("row_sharing") else None),
                         "device": None, "tcdot": bool(re.search(r"tc[-_ ]?dot|modified sp1", name))}
    if f["candidate"] == "B-Ligero":
        m = _RELATION.match(name)
        fam = _FAMILY.search(m.group(1)) if m else None
        f["family"] = fam.group(1) if fam else "v1"
        if (fam and fam.group(2)) or _X4.search(" ".join((name, _s(fp.get("relation")), _s(fp.get("model"))))):
            f["fold"] = 4
        hints = " ".join((_s(meta.get("hints")), _s(_get(fp, "software", "backend", "hints"))))
        f["fused"] = bool(re.search(r"\bfused\b", hints))
    leaf = fp.get("leaf") if isinstance(fp.get("leaf"), Mapping) else {}
    named = " ".join(_s(x) for x in (fp.get("hash"), leaf.get("scheme"), leaf.get("schema")))
    if "ajtai" in named:
        n = _AJTAI_N.search(named)
        f["leaf"] = f"ajtai-n{n.group(1)}" if n else "ajtai"
    elif "blake3" in named:
        f["leaf"] = "blake3"
    elif "poseidon2" in named or "poseidon2" in name:
        f["leaf"] = "poseidon2"
    if f["candidate"] == "A-GKR":
        text = " ".join((name, _s(fp.get("backend")), _s(_get(fp, "software", "backend", "impl"))))
        tool = _get(fp, "software", "toolchain") if isinstance(_get(fp, "software", "toolchain"), Mapping) else {}
        f["device"] = "gpu" if re.search(r"gpu|triton|torch|cuda", text) or any(k in tool for k in ("torch", "triton")) else "cpu"
    return f


def _bare(f: Mapping[str, Any]) -> bool:
    """B-Ligero's bare relation, interactive: authentication excluded, not the Fiat–Shamir (HVZK) variant."""
    return (f["candidate"] == "B-Ligero" and f["authentication"] in (T.AUTHENTICATION, None) and f["mode"] != "fiat-shamir"
            and f["proof_class"] != "COMPLETE_HVZK_BACKEND")


def _hashed(f: Mapping[str, Any], leaf: str) -> bool:
    return f["candidate"] == "B-Ligero" and str(f["authentication"] or "").startswith("included-hash") and f["leaf"] == leaf


@dataclass(frozen=True)
class Variant:
    """One row of D1 and of D2.  ``recognise`` says in words what ``match`` tests on :func:`features`; the matchers are mutually
    exclusive (tested), so a result belongs to at most one variant.  ``sources`` names where the D1 facts come from."""

    id: str
    candidate: str | None
    recognise: str
    match: Callable[[Mapping[str, Any]], bool] = field(repr=False, compare=False)
    zk: str = "unknown"
    operands_hidden: str = "unknown"
    binding: str = "unknown"
    soundness: str = "unknown"
    in_table2: str = "unknown"
    sources: str = ""

    def d1_row(self) -> list[str]:
        return [self.id, self.zk, self.operands_hidden, self.binding, self.soundness, self.in_table2]

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "match"}


_B_ZK = ("Malicious-verifier ZK (interactive: coin commitment at step 0, simulator rewinds V*, CRH only); vacuous today, the "
         "statement publishes the operands (envelope security.zk_statement)")
_B_OPERANDS = "No: a, b and y16 are public in the chain statement (authentication excluded)"
_B_BINDING = ("collision resistance of Blake3 (Merkle) and Blake2b (coin commitments); SHAKE-256 challenge expander as a PRG of the "
              "verifier's coins (envelope security.hash_assumption)")
_B_SOUND = "2^-128 target and achieved, union bound over the sub-batch proofs; interactive: statistical, no Fiat–Shamir loss"
_B_COLUMN = "Yes: B-Ligero column (COMPLETE_ZK_BACKEND, authentication excluded)"
_B_SRC = "Table 1 (B-Ligero); result envelopes' security.{zk_statement, hash_assumption, bound_model}"
_H_ZK = "as the B-Ligero mode it runs in (Ligero masking): malicious-verifier ZK interactive, HVZK under Fiat–Shamir"
_H_SOUND = "2^-128 (Ligero union bound) plus the leaf's binding"
_KEEP_FROZEN = "column B-Ligero + in-proof hash is Poseidon2-BabyBear per row, no sharing (user decision keep_frozen, 2026-09-24)"

VARIANTS: tuple[Variant, ...] = (
    Variant("B-Ligero v1", "B-Ligero", "b-ligero chain, relation without -v2/-v3, one instruction step per column; bare, interactive",
            lambda f: _bare(f) and f["family"] == "v1" and f["fold"] == 1,
            _B_ZK, _B_OPERANDS, _B_BINDING, _B_SOUND + "; the WP2 checker (bits + one-hot selectors) proves decode, products, "
            "alignment over committed wires", _B_COLUMN, _B_SRC),
    Variant("B-Ligero v3 (private selection)", "B-Ligero", "b-ligero chain, <target>-v3 relation (privsel), no fold, hints not fused; bare",
            lambda f: _bare(f) and f["family"] == "v3" and f["fold"] == 1 and not f["fused"],
            _B_ZK, _B_OPERANDS + "; v3 keeps the operand words as committed wires bound by pin rows (relmin-private)", _B_BINDING,
            _B_SOUND + "; v1's statement in ~2.1x fewer rows (relmin-private)", _B_COLUMN, _B_SRC + "; relmin-private"),
    Variant("B-Ligero v3 fused hints", "B-Ligero", "as v3, with the fused device hint kernel (meta hints = fused / software.backend.hints)",
            lambda f: _bare(f) and f["family"] == "v3" and f["fold"] == 1 and f["fused"],
            _B_ZK, _B_OPERANDS, _B_BINDING, _B_SOUND + "; prover-side kernel only: statement and proof unchanged (hints-fused-2: "
            "fused == eager == reference 72/72)", _B_COLUMN + "; phase-sum rule applies (fused-phases fixes the accounting)",
            _B_SRC + "; hints-fused-2"),
    Variant("B-Ligero v3x4 fused (l=4096)", "B-Ligero", "as v3x4 with fused hints and security.rs_l = 4096 columns per proof",
            lambda f: _bare(f) and f["family"] == "v3" and f["fold"] == 4 and f["fused"] and f["l"] == 4096,
            _B_ZK, _B_OPERANDS, _B_BINDING, _B_SOUND + "; more, smaller proofs (fp8-ada: 13 of <= 341 VUs, 2^-128.33)",
            _B_COLUMN + "; phase-sum rule applies (fused-phases); instances as re-packed only with an accepted instance-equiv/v1",
            _B_SRC + "; hints-fused-2"),
    Variant("B-Ligero v1x4 (fold)", "B-Ligero", "v1 relation folded: 4 instruction steps per column (name 'folded: 4', model x4)",
            lambda f: _bare(f) and f["family"] == "v1" and f["fold"] == 4,
            _B_ZK, _B_OPERANDS, _B_BINDING, _B_SOUND + "; same statement as v1, 4 steps per column", _B_COLUMN, _B_SRC),
    Variant("B-Ligero v3x4 (fold)", "B-Ligero", "b-ligero chain, <target>-v3x4 relation, other than fused hints at l = 4096",
            lambda f: _bare(f) and f["family"] == "v3" and f["fold"] == 4 and not (f["fused"] and f["l"] == 4096),
            _B_ZK, _B_OPERANDS, _B_BINDING, _B_SOUND + "; same statement as v3, 4 steps per column", _B_COLUMN, _B_SRC + "; fold-private"),
    Variant("B-Ligero v2 (public selection)", "B-Ligero", "b-ligero chain, <target>-v2 relation (pubsel), no fold; bare",
            lambda f: _bare(f) and f["family"] == "v2" and f["fold"] == 1,
            _B_ZK, "No, by construction: the verifier recomputes products, zero flags, group maximum and truncated sums from the "
            "operands and pins them (relmin-lookup)", _B_BINDING,
            _B_SOUND + "; completeness gap in v2 (exact-zero sums rejected, relmin-private §4b): fix status unknown",
            "Yes: B-Ligero column; never the + in-proof hash column (needs the operands public)", _B_SRC + "; relmin-lookup; relmin-private"),
    Variant("B-Ligero v2x4 (public selection)", "B-Ligero", "b-ligero chain, <target>-v2x4 relation (pubsel, 4 steps per column)",
            lambda f: _bare(f) and f["family"] == "v2" and f["fold"] == 4,
            _B_ZK, "No, by construction (public selection, as v2)", _B_BINDING,
            _B_SOUND + "; completeness gap in v2 (relmin-private §4b): fix status unknown",
            "Yes: B-Ligero column; never the + in-proof hash column", _B_SRC + "; relmin-lookup; relmin-private"),
    Variant("B-Ligero Fiat–Shamir (HVZK)", "B-Ligero", "B-Ligero, authentication excluded, mode fiat-shamir or proof_class COMPLETE_HVZK_BACKEND",
            lambda f: f["candidate"] == "B-Ligero" and f["authentication"] in (T.AUTHENTICATION, None)
            and (f["mode"] == "fiat-shamir" or f["proof_class"] == "COMPLETE_HVZK_BACKEND"),
            "HVZK (Fiat–Shamir, transferable)", _B_OPERANDS, _B_BINDING + "; random oracle for Fiat–Shamir",
            "2^-128 including the Fiat–Shamir query term (security.fiat_shamir_queries_log2)",
            "No: B-Ligero's column is COMPLETE_ZK_BACKEND; HVZK is its drill-down class (listed under Table 2)", _B_SRC),
    Variant("B-Ligero + Poseidon2 per row", "B-Ligero", "B-Ligero, authentication included-hash, hash / leaf poseidon2, sharing none",
            lambda f: _hashed(f, "poseidon2") and f["authentication"] == "included-hash" and f["sharing"] in (T.IN_PROOF_SHARING, None),
            _H_ZK, "x, W private: committed rows bound in-proof by Poseidon2 digests checked against the operand Merkle roots; the "
            "digests are not hiding, privacy is Ligero's masking alone (red-team-leaf-3)",
            "Poseidon2 collision resistance (sponge chain over BabyBear); VU shape pinned (steps-pin closed H2)", _H_SOUND,
            "Yes: B-Ligero + in-proof hash column (included-hash, poseidon2-babybear-w24 per row, sharing none)",
            "red-team-leaf-3 (Table 1 assumption lines); steps-pin; Table 2 caption"),
    Variant("B-Ligero + Poseidon2 shared 64x64 tile", "B-Ligero", "B-Ligero, authentication included-hash-shared or sharing tile64x64, Poseidon2",
            lambda f: f["candidate"] == "B-Ligero" and f["leaf"] in ("poseidon2", None)
            and (f["authentication"] == T.SHARED_AUTHENTICATION or (str(f["authentication"] or "").startswith("included-hash")
                                                                    and f["sharing"] not in (T.IN_PROOF_SHARING, None))),
            "malicious-verifier ZK claimed (G and H both masked; shared-live-2), less the fingerprint leak",
            "x, W private except linear information about private rows disclosed by the fingerprint Proof.fp (F8, ZK_SHARED_NOTE)",
            "Poseidon2 collision resistance plus the fingerprint-collision bound (F7: n_op·(l // steps) slots, 3/2^32 coin bias); "
            "live G and H coins from the verifier (G3 fixed, shared-live-2 fe0c4f48)",
            "2^-128 union bound (2^-128.18 .. 2^-128.66 measured, share-logup-3 / shared-live-2)", "No: shared hashing is a drill-down; " + _KEEP_FROZEN,
            "share-logup-3; shared-live-2; red-team-leaf-3 (F7, F8, G3)"),
    Variant("B-Ligero + BLAKE3 in circuit", "B-Ligero", "B-Ligero, authentication included-hash*, hash / leaf blake3",
            lambda f: _hashed(f, "blake3"),
            _H_ZK, "x, W committed; digests deterministic, not hiding (red-team-leaf-3)",
            "BLAKE3 collision resistance with role-keyed, n_chunks-framed leaves, plus SHA-256 collision resistance for the tagged "
            "malformed-frame domain (red-team-leaf-3)", _H_SOUND, "No: " + _KEEP_FROZEN, "red-team-leaf-3; blake3-leaf-3"),
    Variant("B-Ligero + Ajtai n=64", "B-Ligero", "B-Ligero, authentication included-hash*, hash / leaf ajtai…n64",
            lambda f: _hashed(f, "ajtai-n64"),
            _H_ZK, "No hiding: binding-only, a low-entropy row can be brute-forced from its public digest (red-team-leaf-3)",
            "Ring-SIS over F_p[X]/(X^64 + 1) (BabyBear p), SHAKE-256-derived B; valid only for rows of <= 64 columns: H1 collision "
            "at steps > n (red-team-leaf-3), closed by steps-pin (verifier enforces steps <= n)", _H_SOUND,
            "No: " + _KEEP_FROZEN, "red-team-leaf-3; steps-pin; ajtai-leaf-3"),
    Variant("B-Ligero + Ajtai n=128", "B-Ligero", "B-Ligero, authentication included-hash*, hash / leaf ajtai…n128",
            lambda f: _hashed(f, "ajtai-n128"),
            _H_ZK, "No hiding: binding-only (red-team-leaf-3)",
            "Ring-SIS over F_p[X]/(X^128 + 1), SHAKE-256-derived B; rows of <= 128 columns only (H1, closed by steps-pin)", _H_SOUND,
            "No: " + _KEEP_FROZEN, "red-team-leaf-3; steps-pin"),
    Variant("B-Ligero native external authentication", "B-Ligero", "B-Ligero, authentication included (SHA-256 Merkle roots of x, W, y)",
            lambda f: f["candidate"] == "B-Ligero" and f["authentication"] == "included",
            "as the B-Ligero mode it runs in (the ZK claim is about the witness, unchanged)",
            "No: the verifier recomputes the leaves from the published words (auth-included)",
            "SHA-256 collision resistance: three committed Merkle roots over the words of x, W, y, recomputed and folded by the verifier",
            "the Ligero bound of its mode (2^-128.04 .. 2^-128.25 measured, auth-included)",
            "No: authentication 'included' is not a Table 2 column (columns: excluded | included-hash)", "auth-included"),
    Variant("Ligerito", None, "software.backend.name starting 'ligerito'",
            lambda f: f["candidate"] == "Ligerito",
            "No complete ZK: every row NON_ZK_PROOF_DIAGNOSTIC; --zk masks the sumcheck (LGSC0004) but no PCS ZK argument covers the "
            "zero / sparse g-row claims (red-team-ligerito-3 item 5, ligerito-relation-3)",
            "No: the a/b words are statement rows (red-team-ligerito-3 R3-3)",
            "collision-resistant hashing: Blake3 Merkle leaves, Blake2b coin commitments, SHAKE-256 challenges (ligerito-zk)",
            "2^-128 (PCS term 2^-128.017; sumcheck 184/|F| = 2^-177.9; live verify-session 2^-128.43)",
            "No: not a Table 1 candidate (no column)", "red-team-ligerito-3; ligerito-relation-3; ligerito-zk"),
    Variant("A-GKR CPU (Rust)", "A-GKR", "software.backend.name 'verity-gkr…' without a GPU marker (rustc toolchain)",
            lambda f: f["candidate"] == "A-GKR" and f["device"] == "cpu",
            "No (no ZK masking anywhere yet)", "No: committed without masking (only the output words are public endpoints)",
            "SHA-256 Merkle over the RS columns; Fiat–Shamir with SHA-256 (random oracle; the interactive object is not implemented); "
            "LogUp fractional-sum GKR (Table 1's text says BLAKE3 Merkle)",
            "2^-127.7 over Goldilocks^3 (IRS query term 2^-130.2; 0.3 bits short, the LogUp identity term); --features babybear: "
            "2^-130.2 by the accountant, label 2^-128", "Yes: A-GKR column (NON_ZK_PROOF_DIAGNOSTIC | NON_ZK_PROOF)",
            "backends/gkr/README.md; Table 1 (A-GKR)"),
    Variant("A-GKR GPU (torch/Triton)", "A-GKR", "software.backend.name 'verity-gkr…' with gpu / triton / torch / cuda in the name, "
            "fingerprint backend or toolchain",
            lambda f: f["candidate"] == "A-GKR" and f["device"] == "gpu",
            "No (no ZK masking)", "No: committed without masking",
            "SHA-256 Merkle (hash_gpu 'sha256' on CUDA) and SHA-256 Fiat–Shamir; LogUp over BabyBear^6 (Table 1's text says BLAKE3 "
            "Merkle)", "2^-127.7 (account_candidate_a, BabyBear^6, k = 4096, n = 16384, t = 192; dominated by the generic SHA-256 "
            "collision term)", "Yes: A-GKR column (NON_ZK_PROOF_DIAGNOSTIC | NON_ZK_PROOF); its verifier is Python (transcript replay), "
            "the Rust verifier is Goldilocks-only", "backends/gkr/gpu/README.md; backends/gkr/gpu/ligero.py; Table 1 (A-GKR)"),
    Variant("SP1 unmodified (relation-only / bare guest)", "SP1", "software.backend.name 'sp1…' without tc-dot",
            lambda f: f["candidate"] == "SP1" and not f["tcdot"],
            "unknown (SP1 6.4.0 core / compressed proofs; the README states no ZK property; Table 1: as configured)",
            "x, W are guest input; the 33 public bytes are h_O || h_L || 1 (sound guest) or sha256(K || x || W) || sha256(output "
            "word) || 0xEE (relation-only); hiding would need ZK: unknown",
            "SP1 / STARK hash assumptions (Table 1), sp1-sdk =6.4.0 unmodified; the sound guest authenticates Merkle openings of "
            "the boundary values, the relation-only guest verifies no statement", "unknown (Table 1: as configured)",
            "Yes: SP1 column in a sound class; the relation-only and skip-authentication guests commit 0xEE (DIAGNOSTIC, never "
            "accepted as sound)", "backends/sp1/README.md; Table 1 (SP1)"),
    Variant("SP1 + TC_DOT chip (modified SP1)", "SP1", "software.backend.name 'sp1 tc-dot…' (tc-dot / tc_dot / modified sp1 in the name)",
            lambda f: f["candidate"] == "SP1" and f["tcdot"],
            "unknown (as SP1)", "unknown", "SP1 / STARK hash assumptions plus the TC_DOT chip's constraints (a modified prover)", "unknown",
            "Yes: SP1 column = best valid SP1 variant (user decision 2026-09-24), footnoted 'modified SP1 (TC_DOT chip)'; the name must "
            "start with 'sp1' for the column to see it", "BRIEF §2; lane sp1-tcdot"),
)
_BY_ID = {v.id: v for v in VARIANTS}


def classify(f: Mapping[str, Any]) -> Variant | None:
    """The one :data:`VARIANTS` entry whose recogniser matches :func:`features` ``f``, or ``None``."""
    hits = [v for v in VARIANTS if v.match(f)]
    return hits[0] if hits else None


# --- build ----------------------------------------------------------------------------------------------------------------------------


@dataclass
class Measured:
    """One result as D2 sees it: its variant and target, ``t.total`` and overhead (own B, K), and the predicate's codes."""

    artifact: str
    variant: str
    target: str
    t_total: float
    overhead: float
    codes: list[tuple[str, str]]
    payload: bool = False

    @property
    def mark(self) -> str:
        return self.codes[0][0] if self.codes else PASS


@dataclass
class Drilldowns:
    tables: T.Tables
    measured: list[Measured] = field(default_factory=list)
    unrecognised: dict[str, list[str]] = field(default_factory=dict)   # candidate result that matches no variant: backend -> arts
    unplaced: dict[str, int] = field(default_factory=dict)             # variant id -> results with no D2 target or no t.total
    d3: list[dict[str, Any]] = field(default_factory=list)
    holds: dict[str, list[str]] = field(default_factory=dict)          # variant id -> Table 2 cells it holds


def _envelope(store: Any, art_id: str, doc: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, Mapping[str, Any], bool]:
    """(envelope, the artifact's meta, whether the envelope came from the payload)."""
    meta = doc.get("meta") or {}
    if isinstance(meta.get("workload_fingerprint"), Mapping):
        return meta, meta, False
    env = T._payload_json(store, doc)
    if isinstance(env, Mapping) and isinstance(env.get("workload_fingerprint"), Mapping):
        return {**env, **{k: meta[k] for k in T.PRODUCER_META_KEYS if k in meta and k not in env}}, meta, True
    return None, meta, False


def _overhead(fp: Mapping[str, Any], target: Target, t_total: float) -> float:
    b = fp.get("B") if T._num(fp.get("B")) and fp.get("B") > 0 else T.B
    k = fp.get("K") if T._num(fp.get("K")) and fp.get("K") > 0 else T.K
    return target.native_peak.flop_per_second / contract.proved_flop_per_second(t_total, b, k)


def build(store: Any) -> Drilldowns:
    D = Drilldowns(tables=T.build(store))
    equivs = T.instance_equivs(store)
    cell_variant: dict[str, str] = {}
    for row in store.index.select_artifacts(kind=T.RESULT_KIND):
        art_id = row["id"]
        doc = row.get("manifest") if isinstance(row.get("manifest"), Mapping) else store.get_manifest(art_id).to_doc()
        env, meta, payload = _envelope(store, art_id, doc)
        if env is None:
            continue
        fp = env["workload_fingerprint"]
        f = features(env, meta)
        v = classify(f)
        if v is None:
            if f["candidate"] is not None:
                D.unrecognised.setdefault(f["backend"] or "(no software.backend.name)", []).append(art_id)
            continue
        cell_variant[art_id] = v.id
        target = TARGETS.get(fp.get("profile")) if isinstance(fp.get("profile"), str) else None
        t_total = T._measurements(env).get(contract.T_TOTAL)
        if target is None or target.name not in _TARGET_LABEL or not T._num(t_total) or t_total <= 0:
            D.unplaced[v.id] = D.unplaced.get(v.id, 0) + 1
            continue
        cand = T.candidate_for(f["backend"])
        if cand is None:
            codes = [("X", f"software.backend.name {f['backend']!r} is no Table 1 candidate's: no Table 2 column")]
        else:
            _, attempt = T._attempt_of(store, art_id)
            codes = T.reject_codes(env, store.labels(art_id), attempt, target, cand, authentication=T.column_authentication(fp),
                                   equivs=equivs)
        if payload:
            codes = [*codes, ("X", PAYLOAD_ONLY)]
        D.measured.append(Measured(art_id, v.id, target.name, float(t_total), _overhead(fp, target, float(t_total)), codes, payload))
    for c in D.tables.cells:
        vid = cell_variant.get(c.artifact)
        label = f"{_TARGET_LABEL.get(c.target, c.target)} · {c.candidate}{T.AUTH_COLUMN_SUFFIX.get(c.authentication, '')}"
        if vid:
            D.holds.setdefault(vid, []).append(label)
        D.d3.append(d3_row(store.get_manifest(c.artifact).to_doc().get("meta") or {}, c, label, vid))
    return D


def fastest(D: Drilldowns, variant: str, target: str) -> tuple[Measured | None, Measured | None]:
    """(the variant's fastest result on the target, its fastest Table-2-valid one) by overhead, ties by artifact id."""
    pool = sorted((m for m in D.measured if m.variant == variant and m.target == target), key=lambda m: (m.overhead, m.artifact))
    ok = [m for m in pool if not m.codes]
    return (pool[0] if pool else None), (ok[0] if ok else None)


# --- D3 ------------------------------------------------------------------------------------------------------------------------------


def _field(meta: Mapping[str, Any], name: str) -> Any:
    if name.startswith("validation."):
        v = _get(meta, *name.split("."))
    else:
        v = T._measurements(meta).get(name)
    return v if T._num(v) else None


def d3_row(meta: Mapping[str, Any], cell: T.Cell, label: str, variant: str | None = None) -> dict[str, Any]:
    """One D3 row from the cell's result envelope ``meta``; ``None`` where a field is absent.  Proof bytes = ``proof_bytes``, else
    ``transcript.bytes``; coins = ``net.bytes_in`` / the live evidence's session count (one session proves the batch); verifier
    CPU s = ``verify.cpu_s`` (the live verifier's own clock), else ``verifier.seconds`` marked ``verifier_cpu_from`` (the
    emitter's verifier time per batch, CPU vs wall not stated)."""

    got = {k: _field(meta, k) for k in D3_FIELDS}
    t = cell.t_total
    nbytes = got["proof_bytes"] if got["proof_bytes"] is not None else got["transcript.bytes"]
    sessions = got["validation.evidence.live_verifier.sessions"]
    coins = got["net.bytes_in"] / sessions if got["net.bytes_in"] is not None and sessions else None
    cpu, cpu_from = (got["verify.cpu_s"], "verify.cpu_s") if got["verify.cpu_s"] is not None else (got["verifier.seconds"], "verifier.seconds")
    live = got[contract.T_TOTAL_LIVE]
    return {"cell": label, "artifact": cell.artifact, "variant": variant, "t_total": t, "proof_bytes": nbytes,
            "proof_bytes_from": ("proof_bytes" if got["proof_bytes"] is not None else "transcript.bytes") if nbytes is not None else None,
            "gbit_per_s": nbytes * 8 / t / 1e9 if nbytes is not None else None, "coins_bytes": coins, "rounds": got["rounds.protocol"],
            "sequential_depth": got["rounds.sequential_depth"], "verifier_cpu_s": cpu, "verifier_cpu_from": cpu_from if cpu is not None else None,
            "verifier_cores": cpu / t if cpu is not None else None, "live_tax": live / t if live is not None else None,
            "fields_present": [k for k in D3_FIELDS if got[k] is not None], "fields_absent": [k for k in D3_FIELDS if got[k] is None]}


# --- rendering ------------------------------------------------------------------------------------------------------------------------


def _bytes(n: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.3g} {unit}"
    return f"{n:.0f} B"


def _num_or(v: Any, fmt: Callable[[Any], str]) -> str:
    return fmt(v) if v is not None else T.UNMEASURED


def render_markdown(D: Drilldowns) -> str:
    out = [f"## D1 — Variants and security properties ({LAYOUT})", "",
           "Definitional: from Table 1, the result envelopes' own security statements, the lane reports and READMEs each entry "
           "cites; `unknown` where none states it. The last column adds the Table 2 cells the variant holds in this store.", ""]
    rows = []
    for v in VARIANTS:
        row = v.d1_row()
        held = D.holds.get(v.id)
        row[-1] += f" — holds: {', '.join(held)}" if held else ""
        rows.append(row)
    out += [T._md_table(D1_COLUMNS, rows), "", "Recognised by (software.backend.name + fingerprint fields; sources):"]
    out += [f"- {v.id}: {v.recognise} ({v.sources})" for v in VARIANTS]

    out += ["", "## D2 — Fastest measured t.total per variant × target", "",
            f"Cell `seconds · overhead×` of the variant's fastest result on the target (lowest R_native_spec / R_proved, R_proved = "
            f"2KB / t.total at the result's own B and K), then `{PASS}` = Table-2-valid (tables.py's predicate) or the first reason: "
            + ", ".join(f"{k} {v}" for k, v in T.REASON_LETTERS.items()) + f". `{T.EMPTY}` = no result. Footnotes: artifact, reason.", ""]
    notes: list[str] = []
    rows2 = []
    for v in VARIANTS:
        line = [v.id]
        for label, target in TARGET_COLUMNS:
            best, ok = fastest(D, v.id, target.name)
            if best is None:
                line.append(T.EMPTY)
                continue
            why = "" if not best.codes else f" — {best.codes[0][1]}"
            also = f"; fastest valid: {ok.artifact} {T.fmt_seconds(ok.t_total)} · {T.fmt_overhead(ok.overhead)}" if ok and ok is not best else ""
            notes.append(f"{best.artifact}{' (payload envelope)' if best.payload else ''}{why}{also}")
            line.append(f"{T.fmt_seconds(best.t_total)} · {T.fmt_overhead(best.overhead)} {best.mark} [{len(notes)}]")
        rows2.append(line)
    out += [T._md_table(D2_COLUMNS, rows2), ""]
    if notes:
        out += ["Footnotes (D2):", *[f"[{i}] {n}" for i, n in enumerate(notes, 1)], ""]
    if D.unplaced:
        out += ["Recognised results with no D2 cell (no profile among the five targets, or no t.total): "
                + ", ".join(f"{vid} ×{n}" for vid, n in D.unplaced.items()), ""]
    if D.unrecognised:
        n = sum(len(a) for a in D.unrecognised.values())
        out += [f"Candidate results no variant recognises ({n}; by software.backend.name):",
                *[f"- {name} ×{len(arts)}" for name, arts in sorted(D.unrecognised.items(), key=lambda kv: (-len(kv[1]), kv[0]))], ""]

    out += ["## D3 — Communication and verifier cost, one row per Table 2 cell", "",
            "From the cell's result envelope only; `…` = the envelope has no such field. Proof bytes = `proof_bytes` (else "
            "`transcript.bytes`); coins = `net.bytes_in` per live session; rounds = `rounds.protocol`; verifier CPU s = "
            "`verify.cpu_s` (the live verifier's clock), † = `verifier.seconds` instead (the emitter's verifier time, CPU vs wall not "
            "stated); live tax = `t.total_live / t.total`.", ""]
    rows3 = []
    for r in D.d3:
        dag = "†" if r["verifier_cpu_from"] == "verifier.seconds" else ""
        rows3.append([r["cell"], _num_or(r["proof_bytes"], _bytes), _num_or(r["gbit_per_s"], lambda x: f"{x:.3g}"),
                      _num_or(r["coins_bytes"], _bytes), _num_or(r["rounds"], lambda x: f"{x:.0f}"),
                      _num_or(r["verifier_cpu_s"], lambda x: f"{T.fmt_seconds(x)}{dag}"),
                      _num_or(r["verifier_cores"], lambda x: f"{x:.3g}{dag}"), _num_or(r["live_tax"], lambda x: f"{x:.3g}×")])
    out += [T._md_table(D3_COLUMNS, rows3) if rows3 else f"(no populated Table 2 cell {T.EMPTY} no row)", ""]
    if D.d3:
        out.append("Fields read per row (present; absent):")
        out += [f"- {r['cell']} ({r['artifact']}): {', '.join(r['fields_present']) or 'none'}; absent: {', '.join(r['fields_absent']) or 'none'}"
                for r in D.d3]
    return "\n".join(out) + "\n"


def render_json(D: Drilldowns) -> dict[str, Any]:
    d2 = []
    for v in VARIANTS:
        cells = {}
        for label, target in TARGET_COLUMNS:
            best, ok = fastest(D, v.id, target.name)
            cells[label] = None if best is None else {**asdict(best), "mark": best.mark, "fastest_valid": ok.artifact if ok else None}
        d2.append({"variant": v.id, "cells": cells})
    return {"schema": "verity/drilldowns/v1", "layout": LAYOUT, "canonical": False,
            "d1": {"columns": list(D1_COLUMNS), "variants": [{**v.to_json(), "holds": D.holds.get(v.id, [])} for v in VARIANTS]},
            "d2": {"columns": list(D2_COLUMNS), "reason_letters": T.REASON_LETTERS, "rows": d2, "unrecognised": D.unrecognised,
                   "unplaced": D.unplaced},
            "d3": {"columns": list(D3_COLUMNS), "fields": list(D3_FIELDS), "rows": D.d3}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, metavar="ROOT", help="store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--format", choices=("md", "json"), default="md")
    ns = ap.parse_args(argv)
    from research.store.local import LocalStore  # noqa: PLC0415  (only the CLI needs the store package)

    D = build(LocalStore(ns.root))
    if ns.format == "json":
        print(json.dumps(render_json(D), indent=2, sort_keys=True, default=str))
    else:
        print(render_markdown(D), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
