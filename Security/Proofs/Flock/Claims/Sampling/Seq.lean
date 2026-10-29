import Proofs.Flock.Claims.ZK.Layout
import Proofs.Flock.Claims.Sampling.Deployed

/-!
# Sampling's zero knowledge by sequential composition

The masked sampled run is a first part whose messages the framing fixes (the registration and the draw), then `M`
sessions back to back, each a complete run of one session protocol (Flock's masked record: its own `Hello`, coin tree
and coins) at the statement the router puts it under, on a tape of its own. Its zero knowledge is the sessions' by
sequential composition (`zkmalAt_layout` at `c = 0`): the first part costs nothing, since the router fixes its
messages from `(x, hist)` (`zkmalAt_public`), and `M` sessions cost `Σ_{j < M} eB j` and `M·g`.

* `StepZ`, `Framing`: the router, by slot (`StepZ.session j xS hS`), and the code's reads and witness encoding, with the
  parts' rounds (`k₀`, `n`). `Framing.prover` is the masked sampled reference prover: `Reference.refP` with each
  session the given session prover on slot `j`'s tape, so two sessions under one statement have independent tapes.
* `Framing.Routed`: the router puts each round in its part, and each session under the statement of its first
  history. `Framing.prover_flat`: then the prover sends the composed prover's messages and makes at least its calls
  (`Framing.honestSeq`, `HonestSeq`).
* `Framing.SessionWit`: with no faulty unit, the openings a session reveals make a witness in the session's relation,
  at every history that starts it. `Seq.RelOnAt` is then a theorem (`Framing.relOnAt`), from `Rhon` and the reveal's
  facts (`Reference.found_opens`, `found_val`, `found_covers`).
* `Framing.Shaped`: at every history that starts session `j`, the session protocol's rounds are the slot's, its
  simulator's coins are `CB j x` up to a bijection, and its errors are at most `eB j` and `g`.
* **`Framing.zkAt`**: `ZKmalAt` at one statement, on the nested tape read flat; `Framing.zkmalR` at a constant number
  of sessions `M` (the deployed code's `sessions = 1` is `M = 1`), and `Framing.zkmalRx` at `M x` sessions on
  statement-indexed tapes.
* **`SamplingDeploy.zk_seq`**: `Claims.Sampling.ZK` at the masked record `deployZ` (the deployed record with its
  reference prover `Framing.prover` at the session prover `DF.prover` of a masked Flock record `DF`), from
  `Claims.Flock.ZK H DF εF γF`, at `ε = Σ_{j < M} eB j` and `γ = M·g`.

The simulator is given the published leaf layer and doesn't read it: what the prover sends outside the sessions is a
function of `(x, hist)`.
-/

namespace Claims.Sampling

open Claims Claims.ZKMal

/-- **Where a masked run is, for the reference prover**, at a history: slot `j`'s session under the statement `xS` at
its own history `hS`, or a message the run's framing fixes outside the sessions. -/
inductive StepZ where
  | session (j : ℕ) (xS : Msg) (hS : List (Msg × Msg))
  | say (m : Msg)

/-- The session's statement; `[]` outside the sessions. -/
def StepZ.stmt : StepZ → Msg
  | .session _ xS _ => xS
  | .say _ => []

/-- What the framing says; `[]` in a session. -/
def StepZ.said : StepZ → Msg
  | .session .. => []
  | .say m => m

/-- **The masked sampled prover's framing**: the code's parser, partition evaluation, trace commitment, reads and
witness encoding (`Reference.refP`'s); the router by slot (`step`); and the parts' rounds: `k₀ x` before the sessions,
`n x j` in session `j`. -/
structure Framing where
  parse : Msg → Option AuditStatement
  partition : AuditStatement → Option Partition
  commit : Commitment Bool
  reads : AuditStatement → Msg → List ℕ
  wit : AuditStatement → Msg → List (Opening Bool) → Claims.Flock.Wit
  step : Msg → List (Msg × Msg) → StepZ
  k₀ : Msg → ℕ
  n : Msg → ℕ → ℕ

/-- The first part's tape: none. -/
abbrev unitTape : SeqTape := ⟨Unit, Unit⟩

namespace Framing

variable (F : Framing)

/-- **The masked sampled reference prover** at the witness `w`, with the session prover `PS`: what the framing fixes
outside the sessions; in slot `j`'s session under `xS`, `PS` at the session's history, on the Flock witness of what
it reveals and slot `j`'s tape. -/
def prover {TS : Type} (PS : Claims.Flock.Wit → Prover TS) (w : Wit) : Prover (ℕ → TS) :=
  ⟨fun x hist ω =>
    match F.step x hist with
    | .say m => .ret m
    | .session j xS hS =>
      match F.parse x with
      | some s => Claims.Sampling.ocBind (Reference.revealOC F.commit s.roots w (F.reads s xS)) fun os =>
          (PS (F.wit s xS os)).next xS hS (ω j)
      | none => .ret []⟩

/-- **The framing's parts**: the first part, then session `j` at the statement the router puts its first history
under; no witness. -/
def parts : Layout Wit Claims.Flock.Wit (Unit × Unit) where
  k₀ := F.k₀
  n := F.n
  xs x _ h := (F.step x h).stmt
  ws _ _ _ _ _ := ⟨[]⟩

/-- **A session's witness**: the Flock witness of what it reveals of `w` at its statement's reads, against `H`. -/
def wsOf (H : Msg → Msg) (w : Wit) (x : Msg) (h : List (Msg × Msg)) : Claims.Flock.Wit :=
  match F.parse x with
  | some s => F.wit s (F.step x h).stmt (Reference.found H F.commit s.roots w (F.reads s (F.step x h).stmt))
  | none => ⟨[]⟩

/-- The parts with each session's witness. -/
def lay (H : Msg → Msg) : Layout Wit Claims.Flock.Wit (Unit × Unit) :=
  { F.parts with ws := fun _ w _ x h => F.wsOf H w x h }

theorem lay_start (H : Msg → Msg) : (F.lay H).start = F.parts.start := Layout.start_congr rfl rfl

/-- The first part's prover: what the framing says. -/
def sayP : Wit → Prover (Unit × Unit) := fun _ => ⟨fun x hist _ => .ret (F.step x hist).said⟩

/-- The first part's simulator: it runs the framing. -/
def sim₀ : Simulator where
  C _ := Unit
  prog x _ := spRun (fun h => .ret (F.step x h).said) (F.k₀ x) []

/-- **The composed simulator** of `M` sessions after the first part: the framing, then each session's simulator
`SF` at the statement its simulated history routes to, on coins `CB j x`. -/
noncomputable def sim (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (SF : Simulator)
    (CB : ℕ → Msg → Type) [∀ j x, Fintype (CB j x)] [∀ j x, Nonempty (CB j x)] (M : ℕ) : Simulator :=
  iterSimG unitTape 0 NB ΩB (Layout.seq unitTape 0 NB ΩB F.parts) F.sim₀ CB
    (fun j x u => progOf SF (CB j x) (F.step x u).stmt) (fun j x => goesOn (F.parts.start x j)) M

/-- **The router puts each round in its part** at `x`, for `M` sessions: within the first part it says what the
framing fixes, and within session `j < M` it is in slot `j`'s session under the statement of the session's first
history, at the session's own history. -/
def Routed (M : ℕ) (x : Msg) : Prop :=
  (∀ hist : List (Msg × Msg), hist.length < F.k₀ x → ∃ m, F.step x hist = .say m) ∧
    ∀ j < M, ∀ hist : List (Msg × Msg), F.parts.start x j ≤ hist.length → hist.length < F.parts.start x (j + 1) →
      F.step x hist = .session j (F.step x (hist.take (F.parts.start x j))).stmt (hist.drop (F.parts.start x j))

/-- **An honest unit's openings make a witness in its session's relation** `RF` at `x`: with no faulty unit, at every
history that starts session `j < M`, openings that pass the commitment's check under the roots, are the committed
values `Wit.val` reads, and cover every position the session reads under its statement that has a committed value,
encode (`wit`) a witness in the statement's relation. `SessionWit`'s form at the masked router, at every history
rather than the runs of the public-coin verifier, and on the claim's relation `Rhon`, which reads no scope. -/
def SessionWit (H : Msg → Msg) (RF : Rel Claims.Flock.Wit) (M : ℕ) (x : Msg) : Prop :=
  ∀ s Pt w, F.parse x = some s → F.partition s = some Pt → faulty H F.commit s Pt w = ∅ →
    ∀ j < M, ∀ u : List (Msg × Msg), u.length = F.parts.start x j →
      ∀ os : List (Opening Bool),
        (∀ o ∈ os, F.commit.Opens H s.roots o = true ∧ w.val H F.commit s o.pos = some o.val) →
        (∀ g ∈ F.reads s (F.step x u).stmt, ∀ b, w.val H F.commit s g = some b → ∃ o ∈ os, o.pos = g) →
          RF (F.step x u).stmt (F.wit s (F.step x u).stmt os)

/-- **The sessions at `x` are of one shape**: at every history that starts session `j < M`, the session protocol's
rounds (`kF`) are the slot's, its simulator's coins are `CB j x` up to a bijection, and its errors are at most `eB j`
and `g`. -/
def Shaped (kF : Msg → ℕ) (SF : Simulator) (εF γF : Msg → ℝ) (CB : ℕ → Msg → Type) (eB : ℕ → ℝ) (g : ℝ) (M : ℕ)
    (x : Msg) : Prop :=
  ∀ j < M, ∀ u : List (Msg × Msg), u.length = F.parts.start x j →
    kF (F.step x u).stmt = F.n x j ∧ Nonempty (CB j x ≃ SF.C (F.step x u).stmt) ∧ εF (F.step x u).stmt ≤ eB j ∧
      γF (F.step x u).stmt ≤ g

/-- **The prover sends the composed prover's messages**, and makes at least its calls, at every history within `M`
sessions, on the flat tape: the first part's prover, then session `j` at the history before it. -/
theorem prover_flat (H : Msg → Msg) {TS : Type} (PS : Claims.Flock.Wit → Prover TS) (M : ℕ) {x : Msg} {s : AuditStatement}
    (hs : F.parse x = some s) (hr : F.Routed M x) (w : Wit) (hist : List (Msg × Msg))
    (ω : (Unit × Unit) × (ℕ → TS)) (hh : hist.length < F.parts.start x M) :
    (((F.prover PS w).next x hist ω.2).eval H).1 = ((((F.lay H).prover F.sayP PS M w).next x hist ω).eval H).1 ∧
      ((((F.lay H).prover F.sayP PS M w).next x hist ω).eval H).2 ⊆
        (((F.prover PS w).next x hist ω.2).eval H).2 := by
  by_cases h0 : hist.length < F.k₀ x
  · obtain ⟨m, hm⟩ := hr.1 hist h0
    rw [Layout.prover_pre (F.lay H) F.sayP PS M w x hist ω h0]
    simp [prover, sayP, hm, StepZ.said, OC.eval]
  · rw [← F.lay_start H] at hh
    obtain ⟨j, hj, h1, h2⟩ := (F.lay H).exists_slot x hist.length (Nat.le_of_not_lt h0) M hh
    rw [Layout.prover_session (F.lay H) F.sayP PS M j hj w x hist ω h1 h2, F.lay_start H]
    rw [F.lay_start H] at h1 h2
    have hst := hr.2 j hj hist h1 h2
    simp only [prover, hst, hs, Claims.Sampling.ocBind_eval, Reference.revealOC_eval, lay, wsOf, parts, true_and]
    exact List.subset_append_right _ _

/-- **`HonestSeq`**: the prover on the flat tape `flatten` reads sends the nested composed prover's messages
(`iterProver` at the parts, the composition `zkmalAt_iterG` reads), and makes at least its calls. -/
theorem honestSeq (H : Msg → Msg) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (PS : Claims.Flock.Wit → Prover (NB × ΩB)) (M : ℕ) {x : Msg} {s : AuditStatement} (hs : F.parse x = some s)
    (hr : F.Routed M x) (w : Wit) (hist : List (Msg × Msg)) (t : unitTape.at 0 NB ΩB M)
    (hh : hist.length < F.parts.start x M) :
    (((F.prover PS w).next x hist (Layout.flatten unitTape 0 NB ΩB M t).2).eval H).1 =
        (((iterProver unitTape 0 NB ΩB (Layout.seq unitTape 0 NB ΩB (F.lay H)) F.sayP (fun _ => PS) M w).next x
          hist t).eval H).1 ∧
      (((iterProver unitTape 0 NB ΩB (Layout.seq unitTape 0 NB ΩB (F.lay H)) F.sayP (fun _ => PS) M w).next x
          hist t).eval H).2 ⊆ (((F.prover PS w).next x hist (Layout.flatten unitTape 0 NB ΩB M t).2).eval H).2 := by
  rw [Layout.iterProver_flat unitTape 0 NB ΩB (F.lay H) F.sayP PS M w x hist t]
  exact F.prover_flat H PS M hs hr w hist _ hh

/-- **`Seq.RelOnAt` from `Rhon`**: where no unit is faulty, each session's witness is in its relation at every history
that starts it, so at every unit an honest run selects. -/
theorem relOnAt (H : Msg → Msg) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (RF : Rel Claims.Flock.Wit) (M : ℕ) (x : Msg) (RA : Rel Wit)
    (hRA : ∀ w, RA x w → Rhon H F.commit F.parse F.partition x w) (hw : F.SessionWit H RF M x) {j : ℕ} (hj : j < M)
    (P : Wit → Prover (unitTape.at 0 NB ΩB j)) :
    (Layout.seq unitTape 0 NB ΩB (F.lay H) j).RelOnAt H P RA RF x := by
  refine Layout.relOnAt_of unitTape 0 NB ΩB (F.lay H) H P RA RF x fun w hwA t₀ u hu => ?_
  rw [F.lay_start H] at hu
  obtain ⟨s, Pt, hs, hPt, hf⟩ := hRA w hwA
  show RF (F.step x u).stmt (F.wsOf H w x u)
  simp only [wsOf, hs]
  exact hw s Pt w hs hPt hf j hj u hu _ (fun o ho => ⟨Reference.found_opens ho, Reference.found_val ho⟩)
    fun g hg b hb => Reference.found_covers hg hb

/-- **Sampling's zero knowledge at one statement, by sequential composition**: at `x`, where the router puts each
round in its part (`Routed`), honest openings make session witnesses in `RF` (`SessionWit`), the session protocol
`PS` is `ZKmalR` at `SF` with `εF` and `γF` on `RF`, and the sessions are of one shape (`Shaped`), the masked prover
on the flat tape is `ZKmalAt` at the composed simulator, with `Σ_{j < M} eB j` and `M·g`, on any relation within
`Rhon`, at `k`, the parts' rounds. -/
theorem zkAt (H : Msg → Msg) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (PS : Claims.Flock.Wit → Prover (NB × ΩB)) (RF : Rel Claims.Flock.Wit) (SF : Simulator) (kF : Msg → ℕ)
    (εF γF : Msg → ℝ) (hF : ZKmalR H kF PS RF SF εF γF) (M : ℕ) (x : Msg) (k : ℕ)
    (hk : k = F.parts.start x M) (RA : Rel Wit) (hRA : ∀ w, RA x w → Rhon H F.commit F.parse F.partition x w)
    (hr : (∃ w, RA x w) → F.Routed M x) (hw : F.SessionWit H RF M x) (CB : ℕ → Msg → Type)
    [∀ j x, Fintype (CB j x)] [∀ j x, Nonempty (CB j x)] (eB : ℕ → ℝ) (g : ℝ)
    (hsh : F.Shaped kF SF εF γF CB eB g M x) (he0 : ∀ j < M, 0 ≤ eB j) (hg : 0 ≤ g) :
    ZKmalAt H k (fun w => (F.prover PS w).comap fun t => (Layout.flatten unitTape 0 NB ΩB M t).2) (RA x) x
      ((F.sim NB ΩB SF CB M).C x) ((F.sim NB ΩB SF CB M).prog x) (∑ j ∈ Finset.range M, eB j) (M * g) := by
  subst hk
  have h₀ : ZKmalAt H ((F.lay H).k₀ x) F.sayP (RA x) x (F.sim₀.C x) (F.sim₀.prog x) 0 0 :=
    zkmalAt_public H (F.k₀ x) F.sayP (RA x) x (fun h => .ret (F.step x h).said) fun _ _ _ _ _ => rfl
  have hB : ∀ j < M, ∀ u, (Layout.seq unitTape 0 NB ΩB (F.lay H) j).Reach H
      (iterProver unitTape 0 NB ΩB (Layout.seq unitTape 0 NB ΩB (F.lay H)) F.sayP (fun _ => PS) j) RA x u →
      ZKmalAt H ((F.lay H).n x j) PS (RF ((F.lay H).xs x j u)) ((F.lay H).xs x j u) (CB j x)
        (progOf SF (CB j x) (F.step x u).stmt) (eB j) g := by
    intro j hj u hu
    have hl := Layout.reach_length unitTape 0 NB ΩB (F.lay H) H _ RA x hu
    rw [F.lay_start H] at hl
    obtain ⟨hkF, hC, hε, hγ⟩ := hsh j hj u hl
    have e := hF.at_equiv (F.step x u).stmt hC.some
    rw [hkF] at e
    rw [progOf_of hC]
    exact e.mono hε hγ
  have h := zkmalAt_layout unitTape 0 (F.lay H) H x M F.sayP PS RA RF F.sim₀ CB
    (fun j x u => progOf SF (CB j x) (F.step x u).stmt) (fun j x => goesOn (F.parts.start x j)) 0 0 g eB h₀ hB
    (fun j hj => F.relOnAt H RF M x RA hRA hw hj _) (fun _ _ _ _ _ _ _ _ => by rw [F.lay_start H]) (by simp) he0 hg
  rw [Layout.iterSimG_seq_congr unitTape 0 NB ΩB (F.lay H) F.parts rfl rfl rfl, F.lay_start H] at h
  simp only [Nat.cast_zero, zero_add] at h
  refine h.congr (mul_nonneg (Nat.cast_nonneg _) hg) (fun w hwA hist t hh => ?_) fun w hwA hist t hh => ?_
  · obtain ⟨s, -, hs, -⟩ := hRA w hwA
    exact (F.prover_flat H PS M hs (hr ⟨w, hwA⟩) w hist _ hh).1.symm
  · obtain ⟨s, -, hs, -⟩ := hRA w hwA
    exact (F.prover_flat H PS M hs (hr ⟨w, hwA⟩) w hist _ hh).2

/-- **`Framing.zkAt` at every statement**, for a constant number of sessions `M`: `ZKmalR` on one tape type. -/
theorem zkmalR (H : Msg → Msg) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (PS : Claims.Flock.Wit → Prover (NB × ΩB)) (RF : Rel Claims.Flock.Wit) (SF : Simulator) (kF : Msg → ℕ)
    (εF γF : Msg → ℝ) (hF : ZKmalR H kF PS RF SF εF γF) (M : ℕ) (RA : Rel Wit)
    (hRA : ∀ x w, RA x w → Rhon H F.commit F.parse F.partition x w) (hr : ∀ x, (∃ w, RA x w) → F.Routed M x)
    (hw : ∀ x, F.SessionWit H RF M x) (CB : ℕ → Msg → Type) [∀ j x, Fintype (CB j x)] [∀ j x, Nonempty (CB j x)]
    (eB : ℕ → Msg → ℝ) (g : Msg → ℝ) (hsh : ∀ x, F.Shaped kF SF εF γF CB (fun j => eB j x) (g x) M x)
    (he0 : ∀ j x, 0 ≤ eB j x) (hg : ∀ x, 0 ≤ g x) :
    ZKmalR H (fun x => F.parts.start x M)
      (fun w => (F.prover PS w).comap fun t => (Layout.flatten unitTape 0 NB ΩB M t).2) RA (F.sim NB ΩB SF CB M)
      (fun x => ∑ j ∈ Finset.range M, eB j x) (fun x => M * g x) :=
  (zkmalR_iff_at _ _ _ _ _ _ _).2 fun x =>
    F.zkAt H PS RF SF kF εF γF hF M x _ rfl RA (hRA x) (hr x) (hw x) CB (fun j => eB j x) (g x) (hsh x)
      (fun j _ => he0 j x) (hg x)

/-- The composed simulator of `M x` sessions at each `x`. -/
noncomputable def simX (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (SF : Simulator)
    (CB : ℕ → Msg → Type) [∀ j x, Fintype (CB j x)] [∀ j x, Nonempty (CB j x)] (M : Msg → ℕ) : Simulator where
  C x := (F.sim NB ΩB SF CB (M x)).C x
  prog x := (F.sim NB ΩB SF CB (M x)).prog x

/-- **`Framing.zkAt` on statement-indexed tapes**: `M x` sessions at each `x`, each statement's prover on the tape of
its own sessions (`ZKmalRx`). -/
theorem zkmalRx (H : Msg → Msg) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (PS : Claims.Flock.Wit → Prover (NB × ΩB)) (RF : Rel Claims.Flock.Wit) (SF : Simulator) (kF : Msg → ℕ)
    (εF γF : Msg → ℝ) (hF : ZKmalR H kF PS RF SF εF γF) (M : Msg → ℕ) (RA : Rel Wit)
    (hRA : ∀ x w, RA x w → Rhon H F.commit F.parse F.partition x w) (hr : ∀ x, (∃ w, RA x w) → F.Routed (M x) x)
    (hw : ∀ x, F.SessionWit H RF (M x) x) (CB : ℕ → Msg → Type) [∀ j x, Fintype (CB j x)]
    [∀ j x, Nonempty (CB j x)] (eB : ℕ → Msg → ℝ) (g : Msg → ℝ)
    (hsh : ∀ x, F.Shaped kF SF εF γF CB (fun j => eB j x) (g x) (M x) x) (he0 : ∀ j x, 0 ≤ eB j x)
    (hg : ∀ x, 0 ≤ g x) :
    ZKmalRx (fun x => (unitTape.iter 0 NB ΩB (M x)).N) (fun x => (unitTape.iter 0 NB ΩB (M x)).Ω) H
      (fun x => F.parts.start x (M x))
      (fun w x => (F.prover PS w).comap fun t => (Layout.flatten unitTape 0 NB ΩB (M x) t).2) RA
      (F.simX NB ΩB SF CB M) (fun x => ∑ j ∈ Finset.range (M x), eB j x) (fun x => M x * g x) := fun x =>
  F.zkAt H PS RF SF kF εF γF hF (M x) x _ rfl RA (hRA x) (hr x) (hw x) CB (fun j => eB j x) (g x) (hsh x)
    (fun j _ => he0 j x) (hg x)

end Framing

end Claims.Sampling

namespace SamplingDeploy

open Claims Claims.Sampling Claims.ZKMal

/-- **The code's masked framing** at a router by slot `step` and the parts' rounds `k₀`, `n`: the code's parser,
partition evaluation, trace commitment, reads and witness encoding. -/
def framing (c : Code) (step : Msg → List (Msg × Msg) → StepZ) (k₀ : Msg → ℕ) (n : Msg → ℕ → ℕ) : Framing :=
  ⟨c.parse, c.partition, commit c, c.reads, c.wit, step, k₀, n⟩

/-- **The masked deployed record**: the deployed record with its reference prover the masked one (`Framing.prover`),
each session the masked Flock record `DF`'s reference prover (`DF.prover`, its `Hello` first) on slot `j`'s tape. The
verifier is the code's, so its rounds must count each session's `Hello` (`zk_seq`'s `hV`). The other fields are
`deploy rel c`'s; completeness at this record is not claimed here. -/
noncomputable def deployZ (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)
    (step : Msg → List (Msg × Msg) → StepZ) (k₀ : Msg → ℕ) (n : Msg → ℕ → ℕ) {CoinF ColF TF : Type}
    (DF : Claims.Flock.ZkDeploy CoinF ColF TF) :
    Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace :=
  { deploy rel c with
    Tape := ℕ → DF.Nonce × DF.Rest
    tapeNonempty := inferInstance
    P := (framing c step k₀ n).prover DF.prover }

/-- **The masked record's tape** at `M` sessions: the nested tape of `zkmalAt_iterG`, read as one tape per slot. -/
noncomputable def tapeZ (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (M : ℕ) :
    unitTape.at 0 NB ΩB M → ℕ → NB × ΩB :=
  fun t => (Layout.flatten unitTape 0 NB ΩB M t).2

/-- **The masked record's simulator** at `M` sessions, given the leaf layer, which it doesn't read. -/
noncomputable def simLeakZ (F : Framing) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (SF : Simulator) (CB : ℕ → Msg → Type) [∀ j x, Fintype (CB j x)] [∀ j x, Nonempty (CB j x)] (M : ℕ) :
    SimLeak where
  C x := (F.sim NB ΩB SF CB M).C x
  prog x _ c := (F.sim NB ΩB SF CB M).prog x c

/-- **`HonestSeq` at the masked deployed record**: at a statement that parses where the router puts each round in its
part, the record's reference prover on the masked tape sends the composed prover's messages and makes at least its
calls, at every history within its rounds. -/
theorem deployZ_honestSeq (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (c : Code) (step : Msg → List (Msg × Msg) → StepZ) (k₀ : Msg → ℕ) (n : Msg → ℕ → ℕ) (M : ℕ)
    {CoinF ColF TF : Type} (DF : Claims.Flock.ZkDeploy CoinF ColF TF) {x : Msg} {s : AuditStatement}
    (hs : c.parse x = some s) (hV : c.V.k x = (framing c step k₀ n).parts.start x M)
    (hr : (framing c step k₀ n).Routed M x) (w : Wit) (hist : List (Msg × Msg))
    (t : unitTape.at 0 DF.Nonce DF.Rest M) (hh : hist.length < (deployZ rel c step k₀ n DF).V.k x) :
    ((((deployZ rel c step k₀ n DF).P w).next x hist (tapeZ DF.Nonce DF.Rest M t)).eval H).1 =
        (((iterProver unitTape 0 DF.Nonce DF.Rest (Layout.seq unitTape 0 DF.Nonce DF.Rest ((framing c step k₀ n).lay H))
          (framing c step k₀ n).sayP (fun _ => DF.prover) M w).next x hist t).eval H).1 ∧
      (((iterProver unitTape 0 DF.Nonce DF.Rest (Layout.seq unitTape 0 DF.Nonce DF.Rest ((framing c step k₀ n).lay H))
          (framing c step k₀ n).sayP (fun _ => DF.prover) M w).next x hist t).eval H).2 ⊆
        ((((deployZ rel c step k₀ n DF).P w).next x hist (tapeZ DF.Nonce DF.Rest M t)).eval H).2 :=
  (framing c step k₀ n).honestSeq H DF.prover M hs hr w hist t (hV ▸ hh)

/-- **Sampling's zero knowledge by sequential composition**, at the masked deployed record: `Claims.Sampling.ZK` of
`deployZ`'s reference prover on `tapeZ`'s tape at the composed simulator, with `ε = Σ_{j < M} eB j` and `γ = M·g`,
from Flock's zero knowledge at the masked record `DF` its sessions run (`hF`). What remains, at each statement that
parses: the verifier's rounds are the parts' (`hV`), the router puts each round in its part (`hr`), honest openings
make witnesses in Flock's relation (`hw`), and the sessions are of one shape (`hsh`). -/
theorem zk_seq (H : Msg → Msg) (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)
    (step : Msg → List (Msg × Msg) → StepZ) (k₀ : Msg → ℕ) (n : Msg → ℕ → ℕ) (M : ℕ) {CoinF ColF TF : Type}
    (DF : Claims.Flock.ZkDeploy CoinF ColF TF) (εF γF : Msg → ℝ) (hF : Claims.Flock.ZK H DF εF γF)
    (CB : ℕ → Msg → Type) [∀ j x, Fintype (CB j x)] [∀ j x, Nonempty (CB j x)] (eB : ℕ → Msg → ℝ) (g : Msg → ℝ)
    (hV : ∀ x, (∃ s, c.parse x = some s) → c.V.k x = (framing c step k₀ n).parts.start x M)
    (hr : ∀ x, (∃ s, c.parse x = some s) → (framing c step k₀ n).Routed M x)
    (hw : ∀ x, (framing c step k₀ n).SessionWit H (Claims.Flock.R DF.parse) M x)
    (hsh : ∀ x, (∃ s, c.parse x = some s) →
      (framing c step k₀ n).Shaped (fun xS => DF.V.k xS + 1) DF.S εF γF CB (fun j => eB j x) (g x) M x)
    (he0 : ∀ j x, 0 ≤ eB j x) (hg : ∀ x, 0 ≤ g x) :
    Claims.Sampling.ZK H ((deployZ rel c step k₀ n DF).mask (unitTape.iter 0 DF.Nonce DF.Rest M).N
        (unitTape.iter 0 DF.Nonce DF.Rest M).Ω (tapeZ DF.Nonce DF.Rest M)
        (simLeakZ (framing c step k₀ n) DF.Nonce DF.Rest DF.S CB M))
      (fun x => ∑ j ∈ Finset.range M, eB j x) (fun x => M * g x) := by
  refine ⟨rfl, fun l => (zkmalRx_iff_at _ _ _ _ _ _ _ _ _).2 fun x => ?_⟩
  by_cases hx : ∃ s, c.parse x = some s
  · exact (framing c step k₀ n).zkAt H DF.prover (Claims.Flock.R DF.parse) DF.S (fun xS => DF.V.k xS + 1) εF γF hF.2
      M x _ (hV x hx) (fun x w => Rhon H (commit c) c.parse c.partition x w ∧ leafLayer H (commit c) c.parse x w = l x)
      (fun w hw => hw.1) (fun _ => hr x hx) (hw x) CB (fun j => eB j x) (g x) (hsh x hx) (fun j _ => he0 j x) (hg x)
  · intro w hw
    obtain ⟨s, -, hs, -⟩ := hw.1
    exact absurd ⟨s, hs⟩ hx

end SamplingDeploy
