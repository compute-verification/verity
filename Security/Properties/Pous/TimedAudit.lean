import PousCheck.TimedAudit
import Properties.Pous

/-!
# The timed audit's property, stated against the Lean code

What `PousCheck.TimedAudit.verdict` (package `pous_check`, `memory_challenger.verdict` driving `protocol.TimedAudit`)
accepts, stated over its three inputs, `READY.json`, `LEAVES.bin` and the audit document, and over the store `C*` whose
leaves (`leaves`, as setup publishes them: `PublishesRoot`) have the document's root.

* `TimedAuditAccepts`: an accepted record reads a P2 v4 store and salt from `READY.json`, one 64-byte leaf digest a
  block from `LEAVES.bin` whose `Merkle.root` is the document's root, and `Δ`, the RTT allowance and `k` from the
  document; it has an audit, and in each audit each round `j` answers the challenged index `i` (round `j`'s
  `uniform(B, j)` under `derive(source, CHALLENGE_DOMAIN, {root, salt})`, a block of the store) with a block of the
  width whose leaf (`Merkle.packageLeaf`) is leaf `i` of `LEAVES.bin` and of every `C*` of `B` blocks under the root,
  within `Δ + RTT`. A strict record is covered: the store is the certificate's point, `k`, `Δ` and the allowance are
  within it, and it reports no gap exactly when `W` has the certificate's 8 segments (a strict record of another segment
  count is accepted with that gap: `Certificate.gate` refuses only `k`, `Δ` and the allowance).

It takes the binding form: the conclusion holds, or the run hashed two distinct strings to one SHA-512 digest
(`runStrings`: the inner nodes of the tree over `LEAVES.bin` the audit computes, and of the tree over `C*`'s leaves setup
computes). Proved (`Proofs.Properties.Pous.TimedAudit`).
-/

namespace PousCheck.Properties

open Lean (Json)
open Flock

/-- The document's field `k` holds the hex string of `b`. -/
def HexAt (j : Json) (k : String) (b : ByteArray) : Prop := ∃ t, j.getObjVal? k = .ok (.str t) ∧ unhex t = .ok b

/-- The document's field `k` holds the natural number `n`. -/
def NatAt (j : Json) (k : String) (n : Nat) : Prop := ∃ v, j.getObjVal? k = .ok v ∧ v.getNat? = .ok n

/-- `LEAVES.bin` read as `blocks` 64-byte leaf digests. -/
def fileLeaves (blocks : Nat) (leafBytes : ByteArray) : Array ByteArray :=
  (Array.range blocks).map fun i => leafBytes.extract (Merkle.DIGEST * i) (Merkle.DIGEST * (i + 1))

/-- The audit's `k`: the document's, else the certificate's at the store, else `audit.CHALLENGES`. -/
def challengesOf (s : Certificate.Store) (doc : Json) (k : Nat) : Prop :=
  match doc.getObjVal? "k" with
  | .ok j => j.getNat? = .ok k
  | .error _ => k = ((Certificate.p2 s.w s.c0 s.blocksPerSegment).map (·.k)).getD TimedAudit.CHALLENGES

/-- Round `j`'s challenged index in an audit keyed by `source`. -/
def challenged (s : Certificate.Store) (salt root source : ByteArray) (j : Nat) : Nat :=
  (Challenge.index (Randomness.derive source TimedAudit.CHALLENGE_DOMAIN [("root", .bytes root), ("salt", .bytes salt)])
    s.blocks j).getD s.blocks

/-- The inner nodes' inputs of `Merkle.mth leaves lo n`: what SHA-512 hashes for the tree beyond the leaves. -/
def nodeInputs (leaves : Array ByteArray) (lo n : Nat) : List ByteArray :=
  if h : n ≤ 1 then []
  else
    have := Merkle.split_lt (n := n) (by omega)
    have := Merkle.split_pos n
    (ByteArray.mk #[1] ++ Merkle.mth leaves lo (Merkle.split n) ++
        Merkle.mth leaves (lo + Merkle.split n) (n - Merkle.split n)) ::
      (nodeInputs leaves lo (Merkle.split n) ++ nodeInputs leaves (lo + Merkle.split n) (n - Merkle.split n))
termination_by n

/-- What the run hashes for its two trees: over `LEAVES.bin` (the audit's `Merkle.root`) and over `C*`'s leaves (setup's). -/
def runStrings (ready : Json) (leafBytes : ByteArray) (C : Array (Array ByteArray)) : List ByteArray :=
  match TimedAudit.store ready with
  | .ok (s, salt) =>
    nodeInputs (fileLeaves s.blocks leafBytes) 0 s.blocks ++ nodeInputs (leaves salt C) 0 (leaves salt C).size
  | .error _ => []

/-- The verdict accepts: its `accepted` is `true` (`pous-timed-audit` exits 0). -/
def TimedAuditAccepted (ready : Json) (leafBytes : ByteArray) (doc : Json) : Prop :=
  (TimedAudit.verdict ready leafBytes doc).getObjValD "accepted" = .bool true

/-- What an accepted record establishes, for the store `C`. -/
def TimedAuditAnswers (ready : Json) (leafBytes : ByteArray) (doc : Json) (C : Array (Array ByteArray)) : Prop :=
  ∃ (s : Certificate.Store) (salt root : ByteArray) (delta rtt k : Nat) (strict : Bool) (records : Array Json),
    TimedAudit.store ready = .ok (s, salt) ∧ leafBytes.size = s.blocks * Merkle.DIGEST ∧
    HexAt doc "root" root ∧ NatAt doc "deadline_ns" delta ∧ NatAt doc "rtt_allowance_ns" rtt ∧
    NatAt doc "limit_ns" (delta + rtt) ∧ doc.getObjVal? "strict" = .ok (.bool strict) ∧ challengesOf s doc k ∧
    doc.getObjVal? "records" = .ok (.arr records) ∧ 0 < records.size ∧
    Merkle.root (fileLeaves s.blocks leafBytes) = some root ∧
    (strict = true → (s.w, s.c0, s.blocksPerSegment) = Certificate.POINT ∧ Certificate.CERT_K ≤ k ∧
      delta = Certificate.DELTA_NS ∧ rtt ≤ Certificate.RTT_ALLOWANCE_CAP_NS ∧
      ((Certificate.gate s k delta rtt).2.1 ++ (Certificate.gate s k delta rtt).2.2 = [] ↔ s.segments = 8)) ∧
    ∀ rec ∈ records, ∃ (source : ByteArray) (answers : Array Json),
      HexAt rec "source" source ∧ 0 < source.size ∧ rec.getObjVal? "answers" = .ok (.arr answers) ∧ answers.size = k ∧
      ∀ j (hj : j < answers.size), ∃ (block : ByteArray) (elapsed : Nat),
        let i := challenged s salt root source j
        NatAt answers[j] "index" i ∧ NatAt answers[j] "elapsed_ns" elapsed ∧ HexAt answers[j] "block" block ∧
        i < s.blocks ∧ block.size = s.w / 8 ∧ elapsed ≤ delta + rtt ∧
        Merkle.packageLeaf salt (i / s.blocksPerSegment) (i % s.blocksPerSegment) block =
          (fileLeaves s.blocks leafBytes)[i]! ∧
        ((leaves salt C).size = s.blocks → Merkle.root (leaves salt C) = some root →
          Merkle.packageLeaf salt (i / s.blocksPerSegment) (i % s.blocksPerSegment) block = (leaves salt C)[i]!)

def TimedAuditAccepts : Prop :=
  ∀ (ready : Json) (leafBytes : ByteArray) (doc : Json) (C : Array (Array ByteArray)),
    TimedAuditAccepted ready leafBytes doc →
    TimedAuditAnswers ready leafBytes doc C ∨
      ∃ x ∈ runStrings ready leafBytes C, ∃ y ∈ runStrings ready leafBytes C, x ≠ y ∧ Sha512.hash x = Sha512.hash y

end PousCheck.Properties
