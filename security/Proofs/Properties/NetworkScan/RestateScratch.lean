module

public import Properties.NetworkScan
public import Properties.Spot.Members
public import Proofs.Flock.Claims.Sampling.Rewire

/-! Scratch: two candidate restatements of `ScanScopeAcc` over the deployed parse's positions. Not for the tree. -/

@[expose] public section

namespace Properties.NetworkScan.Restate

open NetworkCheck.Period

/-! ## Positions in SPOT[Flock]'s statement (`bitsProgs ms`) -/

/-- Unit `u`'s returned leaf `leaf` of member `m`: word `leaf / 16`, bit `leaf % 16` (each return field on words of its
own, the u16 on word 0), template instance 0, at the member's base. -/
def outAt (ms : Spot.Bits.Members) (m u leaf : ℕ) : ℕ :=
  let ci := ms.getD m default
  Spot.Bits.base ms m + (Spot.Bits.outBit ci.1 ci.2.size u 0 (leaf / 16) (leaf % 16)).getD 0

/-- Unit `u`'s input bit `j` of member `m`'s first parameter (the ingress row, on ports `0, 1, …` of 16 bits), template
instance 0, at the member's base. -/
def inAt (ms : Spot.Bits.Members) (m u j : ℕ) : ℕ :=
  let ci := ms.getD m default
  Spot.Bits.base ms m + (Spot.Bits.inBit ci.1 ci.2.size u 0 (j / 16) (j % 16)).getD 0

/-- **The scan's layout at the members** `ms` (start, ingress, egress: members 0, 1, 2): step `t`'s slot `σ` is the
egress member's unit `t`'s leaf 0 and its frame's leaves `16 + 8σB …`; step `t`'s ingress is the ingress member's unit
`t`'s row bits. -/
def membersLayout (ms : Spot.Bits.Members) (s : Schedule) : NetTiming.BoundaryGates 1 s.allocation :=
  let bits := 8 * s.cellBytes
  { egress := fun t _ σ => (outAt ms 2 t 0, (List.range bits).map fun b => outAt ms 2 t (outBits + σ.val * bits + b))
    ingress := fun t => (List.range (s.allocation * bits)).map fun j => inAt ms 1 t j }

/-- The template key of root node `templateNodes P`'s `m`-th, as `Spot.Bits.check` names it. -/
def templateKey (P : Flock.Program.Prog) (m : ℕ) : String :=
  (P.defs[((P.defs[P.root]!).body.nodes[(templateNodes P).getD m 0]!).fn]!).key

/-- **The parse's members**: each statement `D` parses has a descriptor Flock decodes and is SPOT[Flock]'s statement
over `msOf` of it (`bitsProgs`), three members, each laid out as its template of the Program (`Spot.Bits.check`). -/
def StmtMembers {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin} (D : Claims.SamplingGeneric.Compiled Coin A)
    (msOf : Claims.Sampling.AuditStatement → Spot.Bits.Members) : Prop :=
  ∀ x s, D.parse x = some s → ∃ P, Flock.Program.decode ⟨s.desc.toArray⟩ = .ok P ∧
    s.prog = Properties.Spot.bitsProgs (msOf s) ∧ (msOf s).length = 3 ∧
    ∀ m < 3, Spot.Bits.check ((msOf s).getD m default).1 P (templateKey P m) = none

/-! ## Candidate A: the fit at the parsed statement's own gates -/

/-- **Candidate A.** An accepted scan's statements are in `ScopeAcc` at the members' layout: `ScanScopeAcc` with
`StmtMembers` for `StmtFlattens` and `membersLayout` for `layOf`. -/
def ScanScopeAccA : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String),
    scanOf j program s = .ok sc → Flock.canon j = .ok c →
    ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Claims.SamplingGeneric.Arg Coin}
      (D : Claims.SamplingGeneric.Compiled Coin A) (msOf : Claims.Sampling.AuditStatement → Spot.Bits.Members)
      (x : Claims.Msg),
      StmtMembers D msOf → D.Scope x →
      (∀ st, D.parse x = some st → (⟨st.desc.toArray⟩ : ByteArray) = c.toUTF8) →
      NetTiming.ScopeAcc D (fun st => membersLayout (msOf st) s) (fun _ => s.window) x

/-! ## Candidate B: the fit at the connected program -/

/-- **The fit at the connected program**: the layout fits the parsed statement's gates read through the wiring `ω`,
returning those the boundary keeps (`Prog.wired`), as SPOT[Flock]'s held code connects its members
(`SpotFlock.wiredProg`). -/
def ScopeAccWired {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Claims.SamplingGeneric.Arg Coin}
    {L S : ℕ} (D : Claims.SamplingGeneric.Compiled Coin A)
    (lay : Claims.Sampling.AuditStatement → NetTiming.BoundaryGates L S) (k : Claims.Sampling.AuditStatement → ℕ)
    (ω : ℕ → ℕ) (keep : ℕ → Bool) (x : Claims.Msg) : Prop :=
  D.Scope x ∧ ∃ s, D.parse x = some s ∧ (lay s).Fits (s.prog.wired ω keep) (k s)

/-- **Candidate B.** As A, at the connected program: the code's wiring `ω` and boundary `keep` connect the members. -/
def ScanScopeAccB : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String),
    scanOf j program s = .ok sc → Flock.canon j = .ok c →
    ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Claims.SamplingGeneric.Arg Coin}
      (D : Claims.SamplingGeneric.Compiled Coin A) (msOf : Claims.Sampling.AuditStatement → Spot.Bits.Members)
      (ω : ℕ → ℕ) (keep : ℕ → Bool) (x : Claims.Msg),
      StmtMembers D msOf → D.Scope x →
      (∀ st, D.parse x = some st → (⟨st.desc.toArray⟩ : ByteArray) = c.toUTF8) →
      ScopeAccWired D (fun st => membersLayout (msOf st) s) (fun _ => s.window) ω keep x

/-! ## Why A alone is weak: no gate of a member's unit reads another unit's -/

/-- At SPOT[Flock]'s statement a gate's reads stay in its own unit (`Spot.Bits.keyProg`'s `wire`), so the egress and
ingress members' gates are never related by `Depends` across units: the fit's causality at `bitsProgs` holds of every
layout over its positions. Stated here only as the shape a proof of A's part (3) would take. -/
def NoCrossUnitReads (C : Claims.Sampling.Prog) (unit : ℕ → Option ℕ) : Prop :=
  ∀ g x, x ∈ (C.gates[g]?.map Claims.Sampling.BGate.reads).getD [] →
    unit x = none ∨ unit x = unit g

end Properties.NetworkScan.Restate
