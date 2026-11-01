import Proofs.Flock.EndToEnd.Blocks.Statement
import Proofs.Flock.Soundness.LinkCarry

/-!
# The integrity audit over a monitoring period of blocks, with carried state (trusted)

A block's carried state (the KV cache) crosses a wipe as committed values: each value is one hiding leaf, registered in
the public file of every session that reads it, on both sides of every wipe it crosses (`CarriedLeaf`). `ServerBlocks`'
statements read each session's values at its own plurality, so they say nothing of whether a session after a wipe reads
the value the block before committed. Here a carried leaf's value is one value read in every session that shares the
leaf: the plurality of its finder over all of them (`CarriedBlocks.Xm`), each session's wrong units counted at that value
there (`CarriedBlocks.wrongSet`).

* `IntegrityHorizonCarried`: `IntegrityHorizonBlocks` with carried leaves. The probability that every session is
  accepted and the wrong units over the monitoring period, each carried leaf read at its one value, total at least `ln(1/δ)/q` is
  at most `δ + Σ_b Σ_{t ∈ b} (ε_ks,t + δ_link,t) + Σ_ℓ δ_carry,ℓ`, with each carried leaf's term
  $$\delta_{carry,\ell} = \Big(\sum_{t \in \ell} \frac{\pi_{\ell,t}}{1 - \rho_t}\Big)
    \Big(\frac{2 \sum_{t \in \ell} (\sum_{j < t} t'_j + t'_t (1 + k_t))}{2^{256}} + \frac{m_\ell}{e M_\ell}\Big),$$
  `π_{ℓ,t}` the probability that session `t`'s draw reads the leaf and `m_ℓ` the number of sessions that share it: `m_ℓ`
  times the binding error of the leaf's finder (`LinkCarry`).

The leaf's finder (`CarriedBlocks.mix`) picks one of the leaf's sessions uniformly, plays the monitoring period up to it with the
prover's strategy, then that session's trial at the leaf, and so on `M` more times; each session's values embed in one
type through their bytes (`CarriedLeaf.e`), and two readings of different bytes, from any two sessions, give a SHA-512
collision (`hm96`). Its cost is the SHA-512 evaluations of the runs it plays: `t'_j` for each session `j` before the
one it picked, and that session's trial's (`CarriedLeaf.costMix`). The named assumptions are `IntegrityHorizonBlocks`',
and `ecr/sha-512` (`Assumptions.SHA512CRExpected`) at each carried leaf's finder, with that cost. The proof is
`Flock.SecurityProofs.IntegrityHorizonCarried`.
-/

namespace FlockSoundness.Game

/-- **`g h` if `c` holds, else `a` at once.** -/
def optG {γ : Type} (c : Prop) (g : c → Game γ) (a : γ) : Decidable c → Game γ
  | .isTrue h => g h
  | .isFalse _ => .ret a

/-- The strategy `optG` is played by: `s h` if `c` holds. -/
def optS {γ : Type} (c : Prop) (g : c → Game γ) (a : γ) (s : (h : c) → Strategy (g h)) :
    (d : Decidable c) → Strategy (optG c g a d)
  | .isTrue h => s h
  | .isFalse _ => PUnit.unit

end FlockSoundness.Game

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkBind
  FlockSoundness.Binding FlockSoundness.LinkFinder
open scoped ENNReal

namespace SamplingBlock

variable (w : SamplingBlock)

/-- The public file's layout of committed values. -/
noncomputable abbrev lay :=
  layZHJ w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU (pub := w.pub) (Reg := w.Reg)

/-- The public file's rows: each commit string's bytes, and whether it is an `hm96` row. -/
noncomputable abbrev rs :=
  rsZHJ w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg)

/-- An opening's salt. -/
def salt (o : w.vb.Opening) : List UInt8 := (o : w.lay.Pos × w.lay.Val × List UInt8).2.2

open Classical in
/-- The probability that the session's draw reads commit string `c`. -/
noncomputable def readP (c : w.vb.Pos) : ℝ :=
  (Fintype.card (ΩcZ execArith w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb w.L c) : ℝ) / Fintype.card w.L.Ω

end SamplingBlock

/-- **A carried leaf**: one hiding leaf `leaf`, registered in the public file of each session in `sessions`, at commit
string `pos t` of session `t`, an `hm96` row there whose bytes determine the value. `M` is its finder's further
trials. -/
structure CarriedLeaf (W : ℕ → SamplingBlock) where
  sessions : Finset ℕ
  nonempty : sessions.Nonempty
  pos : (t : ℕ) → (W t).vb.Pos
  leaf : Dig
  row : ∀ t ∈ sessions, (W t).rs.isRow (pos t)
  inj : ∀ t ∈ sessions, Function.Injective ((W t).rs.enc (pos t))
  registered : ∀ t ∈ sessions, ∀ R, (W t).lay.regLeaf R (pos t) = leaf
  M : ℕ
  hM : 1 ≤ M

/-- A carried leaf's trial in some session: the session, the strategy that played it, and the trial's outcome. -/
abbrev TrialAt (W : ℕ → SamplingBlock) : Type :=
  Option (Σ t : ℕ, Strategy (W t).game × TrialOutZC FlockLevel3.GF128 (W t).L Dig (List UInt8))

/-- An opening from some session. -/
abbrev OpenAt (W : ℕ → SamplingBlock) : Type := Option (Σ t : ℕ, (W t).vb.Opening)

/-- A trial's cost: its session's trial's (`cTZC`). -/
noncomputable def trialCost {W : ℕ → SamplingBlock} : TrialAt W → ℝ
  | none => 0
  | some ⟨t, τ, o⟩ => cTZC (planZHJ (W t).dj (W t).y₀) (reg ((W t).sim τ)) (W t).t' o

namespace CarriedLeaf

variable {W : ℕ → SamplingBlock} (x : CarriedLeaf W)

/-- The leaf's sessions, as a type. -/
abbrev Ses : Type := {t // t ∈ x.sessions}

instance instNonemptySes : Nonempty x.Ses := x.nonempty.coe_sort

/-- The bytes the leaf's sessions encode its values as. -/
noncomputable def encs : Finset (List UInt8) :=
  x.sessions.biUnion fun t => Finset.univ.image ((W t).rs.enc (x.pos t))

/-- The leaf's common values: its sessions' bytes. -/
abbrev V : Type := Option {l // l ∈ x.encs}

/-- Session `t`'s value as a common value: its bytes. -/
noncomputable def e (t : ℕ) (v : (W t).vb.Val) : x.V :=
  if h : (W t).rs.enc (x.pos t) v ∈ x.encs then some ⟨_, h⟩ else none

/-- A draw of session `t` reads the leaf. -/
abbrev Read (t : ℕ) : Prop :=
  Nonempty (ΩcZ execArith (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb (W t).L (x.pos t))

/-- `trialGZC` at the leaf in session `t`, at the strategy `τ` that plays it. -/
noncomputable def trialG (t : ℕ) (τ : Strategy (W t).game) (h : x.Read t) : Game (TrialAt W) :=
  haveI := h
  (trialGZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb
    (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) (W t).k (W t).Rw).map fun o => (some ⟨t, τ, o⟩ : TrialAt W)

/-- **The leaf's trial in session `t`**, at the strategy `τ` that plays it: `trialGZC` at the leaf, when a draw reads
it. -/
noncomputable def trial (t : ℕ) (τ : Strategy (W t).game) : Game (TrialAt W) :=
  optG (x.Read t) (x.trialG t τ) none (Classical.dec _)

/-- The prover's session strategy in the trial. -/
noncomputable def trialS (t : ℕ) (τ : Strategy (W t).game) : Strategy (x.trial t τ) :=
  optS (x.Read t) (x.trialG t τ) none (fun h => haveI := h
    Strategy.map _ _ (trialStratZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀)
      (W t).vb (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) (W t).k (W t).Rw)) (Classical.dec _)

/-- **What a trial reads at the leaf**, as common values. -/
noncomputable def rd : TrialAt W → Finset x.V
  | none => ∅
  | some ⟨t, τ, o⟩ => if t ∈ x.sessions then
      (rdTZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb
        (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) o).image (x.e t)
    else ∅

/-- **The finder's cost per trial**: `t'_j` for each session `j` played before the one it picked, and that session's
trial's. -/
noncomputable def costMix (y : Σ _ : x.Ses, TrialAt W) : ℝ := (∑ j ∈ Finset.range y.1.1, (W j).t') + trialCost y.2

/-- An opening a trial found for a common value: its session's opening of the value's preimage. -/
noncomputable def op : (Σ _ : x.Ses, TrialAt W) → x.V → OpenAt W
  | ⟨_, none⟩, _ => none
  | ⟨_, some ⟨t, τ, o⟩⟩, v => some ⟨t, opOfVZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀)
      (tabZHJ (W t).dj (W t).y₀) (W t).vb (reg ((W t).sim τ)) (cont ((W t).sim τ)) (x.pos t) o (pull (x.e t) v)⟩

/-- **The collision from two openings of the leaf**, from any two sessions: `hm96`'s extractor on their bytes and
salts. -/
noncomputable def col : OpenAt W → OpenAt W → List UInt8 × List UInt8
  | some ⟨t, o⟩, some ⟨t', o'⟩ => hm96Pair ((W t).rs.enc (x.pos t) ((W t).vb.val o)) ((W t).salt o)
      ((W t').rs.enc (x.pos t') ((W t').vb.val o')) ((W t').salt o')
  | _, _ => ([], [])

/-- **The carried leaf's term `δ_carry`**: the probability that a session's draw reads the leaf over `1 − ρ`, summed
over its sessions, times its finder's bound: `2·Σ_{t ∈ ℓ} (Σ_{j < t} t'_j + t'_t(1 + k_t))/2^256 + m/(eM)`. -/
noncomputable def carry : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ t ∈ x.sessions, (W t).readP (x.pos t) / (1 - (W t).ρ)) *
    (2 * (∑ t ∈ x.sessions, ((∑ j ∈ Finset.range t, (W j).t') + (W t).t' * (1 + (W t).k))) / 2 ^ (256 : ℝ) +
      x.sessions.card / (Real.exp 1 * x.M)))

end CarriedLeaf

/-- **A monitoring period of server-time blocks with carried leaves**: `ServerBlocks` and `nL` leaves, each read in sessions of
the monitoring period. -/
structure CarriedBlocks extends ServerBlocks where
  nL : ℕ
  leaf : Fin nL → CarriedLeaf W
  hin : ∀ x, ∀ t ∈ (leaf x).sessions, t < start B

namespace CarriedBlocks

variable (S : CarriedBlocks) (s : Strategy S.game)

/-- **The leaf's trial in session `i`, in the monitoring period**: the monitoring period's sessions before `i` played by `s`, then the leaf's
trial in session `i` at the strategy `s` leaves for it. -/
noncomputable def comp (x : Fin S.nL) (i : (S.leaf x).Ses) : Game (TrialAt S.W) :=
  compTo (fun t => (S.W t).game) (S.leaf x).trial i.1 0 S.T (S.hin x i.1 i.2) s

/-- `s`, then the prover's session strategy in the trial. -/
noncomputable def compS (x : Fin S.nL) (i : (S.leaf x).Ses) : Strategy (S.comp s x i) :=
  compStrat (fun t => (S.W t).game) (S.leaf x).trial (S.leaf x).trialS i.1 0 S.T (S.hin x i.1 i.2) s

/-- **The leaf's trial**: one of its sessions, uniformly, then its trial there. -/
noncomputable def mix (x : Fin S.nL) : Game (Σ _ : (S.leaf x).Ses, TrialAt S.W) := mixG (S.comp s x)

/-- The prover's strategy in the leaf's trial. -/
noncomputable def mixS (x : Fin S.nL) : Strategy (S.mix s x) := mixStrat (S.comp s x) (S.compS s x)

/-- **The leaf's value**: the common value its trial reads alone most often. -/
noncomputable def Xm (x : Fin S.nL) : (S.leaf x).V :=
  Classical.choose (Finset.exists_max_image Finset.univ
    (pRd (S.mix s x) (S.mixS s x) (rdSig fun _ => (S.leaf x).rd)) Finset.univ_nonempty)

open Classical in
/-- **Session `t`'s values**: at a carried leaf's commit string, the leaf's value; elsewhere the session's plurality
value (`XvalZC`). -/
noncomputable def Xcar (t : ℕ) (R : (S.W t).Reg)
    (τ : Cont (S.W t).L (S.W t).Reg (batchedSessionZC execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀)) R)
    (c : (S.W t).vb.Pos) : (S.W t).vb.Val :=
  if h : ∃ x, t ∈ (S.leaf x).sessions ∧ (S.leaf x).pos t = c then pull ((S.leaf h.choose).e t) (S.Xm s h.choose)
  else XvalZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb R τ
    (S.W t).k (S.W t).Rw c

/-- **Session `t`'s wrong units, each carried leaf read at its value**: `wrongRegZ` at the value layer of `Xcar`. -/
noncomputable def wrongSet (t : ℕ) (τ : Strategy (S.W t).game) : Finset (Fin (S.W t).n) :=
  wrongRegZ execArith (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
    (reg ((S.W t).sim τ))
    (XlayerZC execArith (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
      (S.Xcar s t (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ))))

/-- Session `t`'s record, its wrong units those of `wrongSet`. -/
noncomputable def record (t : ℕ) (τ : Strategy (S.W t).game) (o : (S.W t).Reg × Finset (Fin (S.W t).n) × Discharge.Exec.Transcript) :
    WindowRec :=
  ⟨t, LiveAcceptsZK (S.W t).I o, (S.wrongSet s t τ).card, ∃ u ∈ o.2.1, u ∈ S.wrongSet s t τ⟩

/-- The monitoring period's run at `s`, with these records. -/
noncomputable def run : Game (List WindowRec) := periodRun (fun t => (S.W t).game) (fun t => S.record s t) 0 S.T s

end CarriedBlocks

/-- **The integrity audit over a monitoring period of server-time blocks, with carried state.** In a monitoring period as in
`IntegrityHorizonBlocks` with carried leaves (`CarriedBlocks`), whose draws each cover at least a fraction `q > 0` of
their session's units, the probability that every session is accepted and the wrong units over the monitoring period, each
carried leaf read at its one value in every session that shares it, total at least `ln(1/δ)/q` is at most
$$\delta + \sum_{b < B} \sum_{t \in b} (\varepsilon_{ks,t} + \delta_{link,t}) + \sum_\ell \delta_{carry,\ell}.$$
Its named assumptions are `IntegrityHorizonBlocks`', and `ecr/sha-512` (`Assumptions.SHA512CRExpected`) at each carried
leaf's finder, whose cost counts `t'_j` for each session `j` it plays before the one it picked. -/
def IntegrityHorizonCarried : Prop :=
  ∀ (S : CarriedBlocks) (q δ : ℝ), 0 < q → 0 < δ →
    (∀ t < S.T, q ≤ ((S.W t).kd : ℝ) / (S.W t).n) →
    (∀ t < S.T, Assumptions.UniformRandomBytes (S.W t).Lb (S.W t).src) →
    ∀ s : Strategy S.game,
    (∀ t (τ : Strategy (S.W t).game), Period.Plays S.W S.T s t τ → ∀ ω j,
      TableCRZ execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀ ((S.W t).L.draw ω) (reg ((S.W t).sim τ))) j
        (zcStrat execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀) (cont ((S.W t).sim τ) ω)) (S.W t).qF (S.W t).qS) →
    (∀ t (τ : Strategy (S.W t).game), Period.Plays S.W S.T s t τ →
      LinkCRZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
        (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).t') →
    (∀ x, Assumptions.SHA512CRExpected H512 (finderG (S.mix s x) (S.leaf x).M)
      (finderStrat (S.mix s x) (S.mixS s x) (S.leaf x).M) (outOf (rdSig fun _ => (S.leaf x).rd) (S.leaf x).op (S.leaf x).col)
      (finderCost (rdSig fun _ => (S.leaf x).rd) (S.leaf x).costMix)) →
    value (fun l => (∀ r ∈ l, r.accepted) ∧ Real.log (1 / δ) / q ≤ ((l.map WindowRec.wrong).sum : ℝ)) (S.run s) ≤
      ENNReal.ofReal δ + ∑ b ∈ Finset.range S.B, S.err b + ∑ x, (S.leaf x).carry

end Flock.Guarantees
