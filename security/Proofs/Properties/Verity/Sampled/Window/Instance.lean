module

public import Proofs.Properties.Verity.Sampled.Window.Read
public import Proofs.Properties.Verity.Sampled.Window.UnboundLaid
public import Proofs.Properties.Verity.Sampled.Instance

/-!
# SPOT's vocabulary as deployed over windows of several members

The vocabulary the window's leg reads (`WindowDeployed`, `WindowCovers`) built from the other subprotocols' names over
the deployed device (`WindowInstance.base`), as `Sampled/Instance.lean` builds the one statement's, so
`windowSpotSound_sessions` applies to it with every field of `WindowDeployed` and `WindowCovers` either proved or one
of the named `Prop`s below.

* **The base** (`WindowInstance`): a vocabulary whose signer is the deployed device's and whose declared calls cover
  the deployed device's reading of what the prover sent at its history (`CostsRead`), the window the deployment runs at
  each parameters (`window`), and the committed files the verifier sets that window's sessions up on (`SetsUp`), at
  which the window reads `MAX_COINS` frames' bytes (`lb`) and its members each have the window's one table with
  hm96-sha512 rows (`members`, `Window.table`).
* **The windows** (`WindowInstance.windows`): the deployment's window wherever the verifier sets it up on the
  committed files.
* **The sampled names** (`WindowInstance.vocab`, at constructions `K` over the base, whose game and declared calls
  are the vocabulary's): v2's log, the binding's read finder and forger at `2t` calls, the further finders (the file
  finders under `ecr/sha-512`, and at a window `K`'s two-continuation finders under `cr/sha-512` and each member's
  roots finder under `ecr/sha-512`, `rootsSpecs`), the window's
  premises at the projected strategy (`JointFinderCR`), no loss, and the window's δ (`Window.delta`) at the
  deployment's window, `⊤` where it runs none.
* **The claim's transfer** (`ClaimTransfers`, `claimTransfers`): the units the extracted transcript doesn't bind are
  at most as many as the window's incorrect proof blocks (`Window.unboundWrong`,
  `JointAuditV5.UnboundWrong.card_le`), wherever the claim reads the window (`ClaimReads`): the window's members are
  laid out in order (`Window.Laid`, which holds at every cost and widening, `Window.Laid.joint`,
  `Window.Laid.extend`), the published share and the committed files' units are the window's, and, where every
  member's roots fix their domains (`Window.RootsFix`, which the window's δ pays for at each member's roots term,
  `Window.rootsErr`), the committed files are read at the claim, each member at its own gates
  (`JointAuditV5.ReadsAt`).
* **`WindowDeployed`** (`deployed`): the finder, the forger, the loss, the per-finder premises and the published δ by
  construction (`Window.delta_extend`: the window's δ doesn't read the widening), and the claim's transfer from
  `ClaimReads`.
* **`WindowCovers`** (`covers`): where the deployment runs no window the published δ is `⊤`; where it runs one the
  verifier doesn't set up on the committed files, no play is accepted (`RegisteredFiles`, `WindowAt`).
* **The leg and remote's soundness** (`spotSound`, `remote_soundness_window`): `windowSpotSound_sessions` at the
  constructions `constructionsAt` assembles at the deployed device's history, the sessions' readings and decoders at
  the window's table, the family the parameters name (`labelAt`) and the window's bytes as `MAX_COINS` frames' coins.
* **What stays open**: what the claim reads of the window (`ClaimReads`), the verifier's window at the registered
  files (`WindowAt`), and what the verifier's acceptance gives of a play (`WindowReads`, `WindowRead`), each of the
  windows the deployment runs.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The window's table's name**: its first member's. Every member's, at a window whose members have one table
(`MembersAt`). -/
def Window.table (W : Window) : String := (W.ms.head?.map fun m => m.I.tags.table).getD ""

/-- **The window's δ doesn't read the widening** (`Window.sessionDelta_extend`). -/
theorem Window.delta_extend (W : Window) (S : Widening) (t : ℕ) (cr : ℝ) : (W.extend S).delta t cr = W.delta t cr := by
  unfold Window.delta
  rw [W.sessionDelta_extend S t]
  rfl

/-- **A list's mapped entries, by place**: what holds of `f` at each entry of `l` with its place holds of each entry
of `l.map f` with its place. -/
theorem get_map_of {α β : Type*} (f : α → β) (P : ℕ → β → Prop) :
    ∀ (l : List α), (∀ j : Fin l.length, P j.1 (f (l.get j))) → ∀ k : Fin (l.map f).length, P k.1 ((l.map f).get k)
  | [], _, k => absurd k.2 (Nat.not_lt_zero _)
  | _ :: _, h, ⟨0, _⟩ => h ⟨0, Nat.succ_pos _⟩
  | _ :: l, h, ⟨k + 1, hk⟩ =>
    get_map_of f (fun j => P (j + 1)) l (fun j => h ⟨j.1 + 1, Nat.succ_lt_succ j.2⟩) ⟨k, Nat.lt_of_succ_lt_succ hk⟩

/-- Flock's members' counts are the members', at any budget. -/
theorem Member.map_n_toV5 {nJ : ℕ} (t : ℕ) : ∀ l : List (Member nJ),
    ((l.map fun m => m.toV5 t).map fun m' => m'.n) = l.map fun m => m.n
  | [] => rfl
  | m :: l => congrArg (m.n :: ·) (Member.map_n_toV5 t l)

/-- **A list's entries, by place, from its mapped entries'**: `get_map_of`'s converse. -/
theorem get_of_map {α β : Type*} (f : α → β) (P : ℕ → β → Prop) :
    ∀ (l : List α), (∀ k : Fin (l.map f).length, P k.1 ((l.map f).get k)) → ∀ j : Fin l.length, P j.1 (f (l.get j))
  | [], _, j => absurd j.2 (Nat.not_lt_zero _)
  | _ :: _, h, ⟨0, _⟩ => h ⟨0, Nat.succ_pos _⟩
  | _ :: l, h, ⟨j + 1, hj⟩ =>
    get_of_map f (fun j => P (j + 1)) l (fun k => h ⟨k.1 + 1, Nat.succ_lt_succ k.2⟩) ⟨j, Nat.lt_of_succ_lt_succ hj⟩

/-- Flock's window's counts are the members', at any budget. -/
theorem Window.counts_joint (W : Window) (t : ℕ) : (W.joint t).counts = W.ms.map fun m => m.n :=
  Member.map_n_toV5 t W.ms

/-- **A window laid out in order (`Window.Laid`) is Flock's window laid out in order at every budget**
(`JointAuditV5.Laid`): the layout reads the members' counts and places alone, which the registration fixes, and no
budget, coin or sample. -/
theorem Window.Laid.joint {W : Window} (h : W.Laid) (t : ℕ) : (W.joint t).Laid where
  total := by
    rw [W.counts_joint t, ← W.counts_joint 0]
    exact h.total
  emb k v := by
    have hb : ∀ (s : ℕ) (j : Fin (W.joint s).ms.length),
        (W.joint s).base j = MemberLayout.base (W.ms.map fun m => m.n) j.1 := fun s j =>
      congrArg (fun cnt => MemberLayout.base cnt j.1) (W.counts_joint s)
    have h0 : ∀ (j : Fin W.ms.length) (u : Fin (W.ms.get j).n),
        ((W.ms.get j).emb u).val = MemberLayout.base (W.ms.map fun m => m.n) j.1 + u.val :=
      get_of_map (fun m : Member W.nJ => m.toV5 0)
        (fun j m' => ∀ u : Fin m'.n, (m'.emb u).val = MemberLayout.base (W.ms.map fun m => m.n) j + u.val) W.ms
        fun j u => (h.emb j u).trans (congrArg (· + u.val) (hb 0 j))
    rw [hb t k]
    exact get_map_of (fun m : Member W.nJ => m.toV5 t)
      (fun j m' => ∀ u : Fin m'.n, (m'.emb u).val = MemberLayout.base (W.ms.map fun m => m.n) j + u.val)
      W.ms h0 k v

/-- **A window laid out in order stays so with the sample's coins widened**: the members are the window's. -/
theorem Window.Laid.extend {W : Window} (h : W.Laid) (S : Widening) : (W.extend S).Laid :=
  ⟨h.total, h.emb⟩

/-- **The deployment over windows, before SPOT's names**: the other subprotocols' vocabulary over the deployed device
(`base`), whose signer is the deployed device's (`deployment`, `signer`) and whose declared calls cover the deployed
device's reading at its history (`costs`, `Device.dev`); the window the deployment runs at each parameters
(`window`), whose bound's parameters `θ` fixes; and the committed files the verifier sets that window's sessions up
on (`SetsUp`), at which the window reads `MAX_COINS` frames' bytes (`lb`) and its members each have the window's one
table with hm96-sha512 rows (`members`). A deployed instance fills `signer`, `lb` and `members` at its construction;
no theorem takes any of them as a hypothesis of its own. -/
structure WindowInstance (d : Accounting.Device ι) (α : Type) [Fintype α] where
  base : Vocab d α
  deployment : Device.Deployment
  signer : ∀ θ, base.signer θ = deployment.signer base.families θ
  costs : CostsRead base (Device.dev (deployment.setupOf base.families))
  window : ByteArray → Option Window
  SetsUp : ByteArray → Array ByteArray → Window → Prop
  lb : ∀ θ F W, SetsUp θ F W → W.Lb 0 = _root_.Spot.Challenger.MAX_COINS * 64
  members : ∀ θ F W, SetsUp θ F W → MembersAt W W.table

namespace WindowInstance

variable (I : WindowInstance d α)

/-- **The deployed device's history at each play** (`Device.dev`), at the base's layouts. -/
noncomputable abbrev dev : Exchange → Properties.Challenger.History :=
  Device.dev (I.deployment.setupOf I.base.families)

/-- **SPOT's further finders as deployed**: the circuit file's and the public file's (`Vocab.fileFinder`,
`Vocab.publicFinder`) at two calls each. -/
noncomputable def fileSpecs (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c)) :=
  [{ finder := fun _ => I.base.fileFinder θ F, budget := fun _ => 2, calls := fun _ _ => Nat.le_refl 2 },
   { finder := fun _ => I.base.publicFinder θ F, budget := fun _ => 2, calls := fun _ _ => Nat.le_refl 2 }]

noncomputable def coinsReadOf : Subprotocol → Sent → Option (List Release)
  | .sampled => Verity.Sampled.coinsRead
  | a => I.base.coinsRead a

noncomputable def finderOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (I.base.game θ F c) → Claims.Finder
  | .sampled => Binding.readFinder I.base I.dev
  | a => I.base.finder a

noncomputable def forgerOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (I.base.game θ F c) → Hash.Assumptions.LmsForger
  | .sampled => Binding.forger I.base I.dev
  | a => I.base.forger a

def callsMulOf (θ : ByteArray) : Subprotocol → ℕ
  | .sampled => 2
  | a => I.base.callsMul θ a

def callsAddOf (θ : ByteArray) : Subprotocol → ℕ
  | .sampled => 0
  | a => I.base.callsAdd θ a

open Classical in
/-- **The deployment's windows**: at the parameters `θ` and the committed files `F`, the deployment's window where
the verifier sets it up on them (`SetsUp`). -/
noncomputable def windows (θ : ByteArray) (F : Array ByteArray) : Option Window :=
  (I.window θ).bind fun W => if I.SetsUp θ F W then some W else none

theorem windows_some {θ : ByteArray} {F : Array ByteArray} {W : Window} (h : I.windows θ F = some W) :
    I.window θ = some W ∧ I.SetsUp θ F W := by
  unfold windows at h
  cases hW : I.window θ with
  | none => rw [hW] at h; cases h
  | some B =>
    rw [hW] at h
    dsimp only [Option.bind] at h
    split_ifs at h with hs
    cases h
    exact ⟨rfl, hs⟩

theorem windows_of {θ : ByteArray} {F : Array ByteArray} {W : Window} (hW : I.window θ = some W)
    (hs : I.SetsUp θ F W) : I.windows θ F = some W := by
  unfold windows
  rw [hW]
  dsimp only [Option.bind]
  rw [if_pos hs]

variable (K : WindowConstructions I.base)

/-- **A two-continuation finder as a further finder**: `K`'s at round `j` of the window `W`, asked under `cr/sha-512`
at its `2t + 2h` calls (`WindowConstructions.msg_calls`, `Window.qM`). -/
noncomputable def msgSpec (W : Window) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (j : Fin W.rounds) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder σ := K.msgFinder W θ F c σ j
  budget t := 2 * t + 2 * W.h
  calls σ x := by
    have h := K.msg_calls W θ F c σ j x
    unfold Window.qM at h
    exact_mod_cast h
  claim := .cr

/-- **The record's two-continuation finder as a further finder**: `K`'s at the window `W`, asked under `cr/sha-512`
at its `2t + 2h` calls (`WindowConstructions.record_calls`). -/
noncomputable def recordSpec (W : Window) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder σ := K.recordFinder W θ F c σ
  budget t := 2 * t + 2 * W.h
  calls σ x := by
    have h := K.record_calls W θ F c σ x
    unfold Window.qM at h
    exact_mod_cast h
  claim := .cr

/-- **The bound digests' two-continuation finder as a further finder**: `K`'s at the window `W`, asked under
`cr/sha-512` at its `2t + 2h` calls (`WindowConstructions.bind_calls`). -/
noncomputable def bindSpec (W : Window) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder σ := K.bindFinder W θ F c σ
  budget t := 2 * t + 2 * W.h
  calls σ x := by
    have h := K.bind_calls W θ F c σ x
    unfold Window.qM at h
    exact_mod_cast h
  claim := .cr

/-- **The window's two-continuation finders as further finders**: at the deployment's window, one per round
(`msgSpec`), the record's (`recordSpec`) and the bound digests' (`bindSpec`); none elsewhere. -/
noncomputable def strictSpecs (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c)) :=
  match I.windows θ F with
  | some W => (List.finRange W.rounds).map (I.msgSpec K W θ F c) ++ [I.recordSpec K W θ F c, I.bindSpec K W θ F c]
  | none => []

/-- **A member's roots finder as a further finder**: the finder that recomputes the member's roots
(`Member.rootsFinder`), asked under `ecr/sha-512` at its calls (`Member.rootsCalls`), which no strategy changes. Its
premise is `Member.RootsCR`. -/
noncomputable def rootsSpec {nJ : ℕ} (m : Member nJ) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    FinderSpec (I.base.game θ F c) (I.base.cost θ F c) where
  finder _ := m.rootsFinder
  budget _ := m.rootsCalls
  calls _ _ := Nat.le_refl _

/-- **The members' roots finders as further finders**: at the deployment's window, each member's (`rootsSpec`), of
the members with a proof block in the sample and of those with none; none elsewhere. -/
noncomputable def rootsSpecs (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c)) :=
  match I.windows θ F with
  | some W => W.ms.map fun m => I.rootsSpec m θ F c
  | none => []

noncomputable def findersOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    List (FinderSpec (I.base.game θ F c) (I.base.cost θ F c))
  | .sampled => fun θ F c => I.fileSpecs θ F c ++ I.strictSpecs K θ F c ++ I.rootsSpecs θ F c
  | a => I.base.finders a

/-- **SPOT's window premises as deployed**: at the deployment's window, `JointShareV5`'s per-finder premises at the
strategy projected there (`JointFinderCR`), asked of this strategy alone; `True` where there is none. -/
def premisesOf : Subprotocol → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (I.base.game θ F c) → Prop
  | .sampled => fun θ F c σ =>
    ∀ W, I.windows θ F = some W →
      JointFinderCR ((K.live W θ F c σ).joint (I.base.cost θ F c σ)) (K.project W θ F c σ)
  | a => I.base.premises a

/-- The read finder's published collision bound at `t` calls (`Vocab.crBound` at `callsMulOf`, `callsAddOf`). -/
noncomputable def sampledBound (θ : ByteArray) (t : ℕ) : ℝ :=
  ((I.callsMulOf θ .sampled * t + I.callsAddOf θ .sampled : ℕ) : ℝ) / 2 ^ (256 : ℝ)

/-- **SPOT's δ at the parameters `θ`**: the deployment's window's (`Window.delta`) at the read finder's published
bound; `⊤` where the deployment runs no window, which bounds nothing there. -/
noncomputable def deltaOf (θ : ByteArray) : Subprotocol → ℕ → ℝ≥0∞
  | .sampled, t => ((I.window θ).map fun W => W.delta t (I.sampledBound θ t)).getD ⊤
  | a, t => I.base.delta θ a t

theorem deltaOf_some {θ : ByteArray} {W : Window} (h : I.window θ = some W) (t : ℕ) :
    I.deltaOf θ .sampled t = W.delta t (I.sampledBound θ t) := by
  simp [deltaOf, h]

theorem deltaOf_none {θ : ByteArray} (h : I.window θ = none) (t : ℕ) : I.deltaOf θ .sampled t = ⊤ := by
  simp [deltaOf, h]

noncomputable def lossOf : Subprotocol → ByteArray → ℝ≥0∞ → ℝ≥0∞
  | .sampled => fun _ _ => 0
  | a => I.base.loss a

theorem finderOf_calls : ∀ a θ F c σ (x : (I.finderOf a θ F c σ).S),
    (I.finderOf a θ F c σ).cost x ≤ I.callsMulOf θ a * I.base.cost θ F c σ + I.callsAddOf θ a
  | .sampled, θ, F, c, σ, x =>
    calc (I.finderOf .sampled θ F c σ).cost x
        = I.base.cost θ F c σ + (Binding.breakCalls (I.dev (Verity.Game.Tape.out _ σ x))
            (Verity.Game.Tape.out _ σ x).sent + (Verity.Game.Tape.out _ σ x).sent.blobs.size) := Nat.add_assoc _ _ _
      _ ≤ I.base.cost θ F c σ + I.base.cost θ F c σ := Nat.add_le_add_left (I.costs θ F c σ x) _
      _ = 2 * I.base.cost θ F c σ + 0 := by omega
  | .pouw, θ, F, c, σ, x => I.base.finder_calls .pouw θ F c σ x
  | .inferenceOnly, θ, F, c, σ, x => I.base.finder_calls .inferenceOnly θ F c σ x
  | .pous, θ, F, c, σ, x => I.base.finder_calls .pous θ F c σ x
  | .poub, θ, F, c, σ, x => I.base.finder_calls .poub θ F c σ x
  | .network, θ, F, c, σ, x => I.base.finder_calls .network θ F c σ x
  | .wipe, θ, F, c, σ, x => I.base.finder_calls .wipe θ F c σ x

theorem forgerOf_calls : ∀ a θ F c σ (x : (I.forgerOf a θ F c σ).S),
    (I.forgerOf a θ F c σ).cost x ≤ I.callsMulOf θ a * I.base.cost θ F c σ + I.callsAddOf θ a
  | .sampled, θ, F, c, σ, x =>
    calc (I.forgerOf .sampled θ F c σ).cost x
        = I.base.cost θ F c σ + Binding.breakCalls (I.dev (Verity.Game.Tape.out _ σ x))
            (Verity.Game.Tape.out _ σ x).sent := rfl
      _ ≤ I.base.cost θ F c σ + I.base.cost θ F c σ :=
          Nat.add_le_add_left ((Nat.le_add_right _ _).trans (I.costs θ F c σ x)) _
      _ = 2 * I.base.cost θ F c σ + 0 := by omega
  | .pouw, θ, F, c, σ, x => I.base.forger_calls .pouw θ F c σ x
  | .inferenceOnly, θ, F, c, σ, x => I.base.forger_calls .inferenceOnly θ F c σ x
  | .pous, θ, F, c, σ, x => I.base.forger_calls .pous θ F c σ x
  | .poub, θ, F, c, σ, x => I.base.forger_calls .poub θ F c σ x
  | .network, θ, F, c, σ, x => I.base.forger_calls .network θ F c σ x
  | .wipe, θ, F, c, σ, x => I.base.forger_calls .wipe θ F c σ x

theorem forgerOf_signed : ∀ a θ F c σ (x : (I.forgerOf a θ F c σ).S),
    ((I.forgerOf a θ F c σ).signed x).length ≤ I.base.signatures θ
  | .sampled, θ, F, c, σ, x => Binding.forger_signed_le I.base I.dev θ F c σ x
  | .pouw, θ, F, c, σ, x => I.base.forger_signed .pouw θ F c σ x
  | .inferenceOnly, θ, F, c, σ, x => I.base.forger_signed .inferenceOnly θ F c σ x
  | .pous, θ, F, c, σ, x => I.base.forger_signed .pous θ F c σ x
  | .poub, θ, F, c, σ, x => I.base.forger_signed .poub θ F c σ x
  | .network, θ, F, c, σ, x => I.base.forger_signed .network θ F c σ x
  | .wipe, θ, F, c, σ, x => I.base.forger_signed .wipe θ F c σ x

/-- **The deployed vocabulary**: the base with SPOT's names (v2's log, the binding's read finder and forger, the
further finders and the window's premises at `K`'s finders and projection, the window's δ and no loss). -/
noncomputable def vocab : Vocab d α :=
  { I.base with
    coinsRead := I.coinsReadOf
    finder := I.finderOf
    callsMul := I.callsMulOf
    callsAdd := I.callsAddOf
    finder_calls := I.finderOf_calls
    finders := I.findersOf K
    premises := I.premisesOf K
    forger := I.forgerOf
    forger_calls := I.forgerOf_calls
    forger_signed := I.forgerOf_signed
    delta := I.deltaOf
    loss := I.lossOf }

/-- **Constructions over the base at the deployed vocabulary**: `K'`'s, whose strategies are the same. -/
noncomputable def liftOf (K' : WindowConstructions I.base) : WindowConstructions (I.vocab K) where
  coins := K'.coins
  project := K'.project
  readFinder := K'.readFinder
  forger := K'.forger
  msgFinder := K'.msgFinder
  recordFinder := K'.recordFinder
  bindFinder := K'.bindFinder
  msg_calls := K'.msg_calls
  record_calls := K'.record_calls
  bind_calls := K'.bind_calls

/-! ## The fields of `WindowDeployed` -/

theorem fileCR {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {σ : Verity.Game.Strategy ((I.vocab K).game θ F c)} (h : (I.vocab K).finderCR .sampled θ F c σ) :
    (I.base.fileFinder θ F).Pr Verity.sha512Msg < 1 ∧ (I.base.publicFinder θ F).Pr Verity.sha512Msg < 1 := by
  have hall := List.forall_iff_forall_mem.1
    (show finderCROf (I.fileSpecs θ F c ++ I.strictSpecs K θ F c ++ I.rootsSpecs θ F c) σ from h)
  have hl : finderCROf (I.fileSpecs θ F c) σ :=
    List.forall_iff_forall_mem.2 fun f hf => hall f (List.mem_append_left _ (List.mem_append_left _ hf))
  obtain ⟨h1, h2⟩ := hl
  have lt2 : ((2 : ℕ) : ℝ) / 2 ^ (256 : ℝ) < 1 := budget_lt (by norm_num)
  exact ⟨lt_of_le_of_lt h1 lt2, lt_of_le_of_lt h2 lt2⟩

/-- **The two-continuation finders' premises**: at the deployment's window, where the further finders meet theirs,
each two-continuation finder `K` builds from the strategy meets the strict birthday bound (`strictSpecs`). -/
theorem strictCR {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {W : Window}
    {σ : Verity.Game.Strategy ((I.vocab K).game θ F c)} (hW : I.windows θ F = some W)
    (h : (I.vocab K).finderCR .sampled θ F c σ) :
    (∀ j, Claims.StrictBirthday Verity.sha512Msg (K.msgFinder W θ F c σ j)) ∧
      Claims.StrictBirthday Verity.sha512Msg (K.recordFinder W θ F c σ) ∧
      Claims.StrictBirthday Verity.sha512Msg (K.bindFinder W θ F c σ) := by
  have hall := List.forall_iff_forall_mem.1
    (show finderCROf (I.fileSpecs θ F c ++ I.strictSpecs K θ F c ++ I.rootsSpecs θ F c) σ from h)
  have hs : I.strictSpecs K θ F c =
      (List.finRange W.rounds).map (I.msgSpec K W θ F c) ++ [I.recordSpec K W θ F c, I.bindSpec K W θ F c] := by
    unfold strictSpecs
    rw [hW]
  refine ⟨fun j => hall (I.msgSpec K W θ F c j) (List.mem_append_left _ (List.mem_append_right _ ?_)),
    hall (I.recordSpec K W θ F c) (List.mem_append_left _ (List.mem_append_right _ ?_)),
    hall (I.bindSpec K W θ F c) (List.mem_append_left _ (List.mem_append_right _ ?_))⟩
  · rw [hs]
    exact List.mem_append_left _ (List.mem_map.2 ⟨j, List.mem_finRange j, rfl⟩)
  · rw [hs]
    exact List.mem_append_right _ (by simp)
  · rw [hs]
    exact List.mem_append_right _ (by simp)

/-- **The members' roots finders' premises**: at the deployment's window, where the further finders meet theirs,
each member's roots finder outputs a SHA-512 collision with probability at most its calls over `2^256`
(`rootsSpecs`, `Window.RootsCR`). -/
theorem rootsCR {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {W : Window}
    {σ : Verity.Game.Strategy ((I.vocab K).game θ F c)} (hW : I.windows θ F = some W)
    (h : (I.vocab K).finderCR .sampled θ F c σ) : W.RootsCR := by
  have hall := List.forall_iff_forall_mem.1
    (show finderCROf (I.fileSpecs θ F c ++ I.strictSpecs K θ F c ++ I.rootsSpecs θ F c) σ from h)
  have hs : I.rootsSpecs θ F c = W.ms.map fun m => I.rootsSpec m θ F c := by
    unfold rootsSpecs
    rw [hW]
  intro m hm
  exact hall (I.rootsSpec m θ F c) (List.mem_append_right _ (by rw [hs]; exact List.mem_map.2 ⟨m, hm, rfl⟩))

/-- **The claim's transfer at the deployment's windows** (`WindowDeployed.claim`, at the constructions `K`): where
every member's roots fix their domains (`Window.RootsFix`), at a strategy whose extracted transcript fails the claim
on the committed files at the published share, more than the window's `ε` share of its population's proof blocks is
incorrect at the projected strategy (`JointAuditV5.wrongSet`). -/
def ClaimTransfers : Prop :=
  ∀ θ F W, I.windows θ F = some W → W.RootsFix → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (I.base.game θ F c)),
    ¬ Accounting.IntegrityHolds (I.base.partsOf θ F) (I.base.Opens θ F c) (I.base.extract θ F c σ)
        (I.base.certOf θ).ε →
      (K.live W θ F c σ).ε * (K.live W θ F c σ).nJ <
        ((((K.live W θ F c σ).joint (I.base.cost θ F c σ)).wrongSet (K.project W θ F c σ)).card : ℝ)

/-- **What the claim reads at the deployment's windows**, at the constructions `K`. At each window the verifier
sets up on the committed files: the window's members are laid out in order (`laid`, `Window.Laid`); the published
share is the window's (`share`) and the committed files' units are the window's population (`units`); and, where
every member's roots fix their domains (`Window.RootsFix`: the files check's one clause that isn't a decision of
the public files alone, which the window's δ pays for), at every strategy the committed files are read at the claim
at the projected strategy (`reads`, `JointAuditV5.ReadsAt`): the vocabulary's read is the members' files in order,
each member's files pass the verifier's files check (`MemberV5.Checked`), the commitment's openings hold wherever a
member's deployed commitment opens and at its files' constants, and the extracted transcript is, at each member's
gates, that member's transcript at the trace its session fixes at the joint registration
(`JointAuditV5.memberX`). It is asked of every member, with a proof block in the sample or not. -/
structure ClaimReads : Prop where
  laid : ∀ θ F W, I.windows θ F = some W → W.Laid
  share : ∀ θ F W, I.windows θ F = some W → (I.base.certOf θ).ε = W.ε
  units : ∀ θ F W, I.windows θ F = some W → I.base.units θ F = W.nJ
  reads : ∀ θ F W, I.windows θ F = some W → W.RootsFix →
    ∀ (c : ByteArray) (σ : Verity.Game.Strategy (I.base.game θ F c)),
    ((K.live W θ F c σ).joint (I.base.cost θ F c σ)).ReadsAt (I.base.read θ F) (I.base.Opens θ F c)
      (I.base.extract θ F c σ) (K.project W θ F c σ)

/-- **The claim's transfer holds wherever the claim reads the window** (`ClaimTransfers`, from `ClaimReads`): the
units the extracted transcript doesn't bind are at most as many as the window's incorrect proof blocks at the
projected strategy (`Window.unboundWrong`, `JointAuditV5.UnboundWrong.card_le`, at the window widened by the
projection's coins, laid out as the window is, `Window.Laid.extend`, `Window.Laid.joint`), and a claim that fails at
the published share has more than that share of the committed files' units unbound, which are the window's share and
population. -/
theorem claimTransfers (h : I.ClaimReads K) : I.ClaimTransfers K := by
  intro θ F W hW hR c σ hI
  have hcard : (Accounting.unbound (I.base.partsOf θ F) (I.base.Opens θ F c) (I.base.extract θ F c σ)).card ≤
      (((K.live W θ F c σ).joint (I.base.cost θ F c σ)).wrongSet (K.project W θ F c σ)).card :=
    Flock.Guarantees.JointAuditV5.UnboundWrong.card_le
      (Window.unboundWrong (K.live W θ F c σ) (I.base.cost θ F c σ))
      (((h.laid θ F W hW).extend (K.coins W θ F c σ)).joint (I.base.cost θ F c σ))
      (I.base.read θ F) (I.base.Opens θ F c) (I.base.extract θ F c σ) (K.project W θ F c σ) (h.reads θ F W hW hR c σ)
  have hlt : (I.base.certOf θ).ε * ((I.base.units θ F : ℕ) : ℝ) <
      ((Accounting.unbound (I.base.partsOf θ F) (I.base.Opens θ F c) (I.base.extract θ F c σ)).card : ℝ) :=
    not_le.1 hI
  have he : (K.live W θ F c σ).ε * ((K.live W θ F c σ).nJ : ℝ) =
      (I.base.certOf θ).ε * ((I.base.units θ F : ℕ) : ℝ) :=
    congrArg₂ (fun (e : ℝ) (n : ℕ) => e * (n : ℝ)) (h.share θ F W hW).symm (h.units θ F W hW).symm
  exact he.trans_lt (lt_of_lt_of_le hlt (Nat.cast_le.2 hcard))

/-- **`WindowDeployed` at the deployed vocabulary**, from what the claim reads of the window (`ClaimReads`,
`claimTransfers`), at constructions whose read finder and forger are the binding's
(`WindowConstructions.withBinding`): the leg's per-finder premises are `error`'s condition at this strategy
(`strictSpecs`, `rootsSpecs`, `premisesOf`), and the published δ is the window's, which doesn't read the widening
(`Window.delta_extend`). -/
theorem deployed (hKr : ∀ θ F c σ, K.readFinder θ F c σ = Binding.readFinder I.base I.dev θ F c σ)
    (hKf : ∀ θ F c σ, K.forger θ F c σ = Binding.forger I.base I.dev θ F c σ) (hC : I.ClaimReads K) :
    WindowDeployed (I.vocab K) (I.liftOf K K) I.windows where
  finder θ F c σ := (hKr θ F c σ).symm
  forger θ F c σ := (hKf θ F c σ).symm
  loss _ _ := rfl
  claim θ F W h hR c σ := I.claimTransfers K hC θ F W h hR c σ
  finderCR θ F W h c σ hcr hpr := by
    obtain ⟨hm, hr, hb⟩ := I.strictCR K h hcr
    exact ⟨hpr W h, hm, hr, hb, I.rootsCR K h hcr⟩
  delta θ F W h c σ :=
    le_of_eq ((Window.delta_extend W _ _ _).trans (I.deltaOf_some (I.windows_some h).1 _).symm)

/-! ## `WindowCovers` -/

/-- **The registered files pass the file check wherever some files do**, at parameters naming the one-stage form
(`Instance.RegisteredFiles`, at this base). -/
def RegisteredFiles : Prop :=
  ∀ θ F, (∃ p, ParamsAt θ p) → I.base.Registered θ F → I.base.Registered θ (I.base.registered θ)

/-- **The verifier's window at the registered files' files**: at parameters naming the one-stage form where the
deployment runs a window, committed files that name the registered circuit and public files, and that some play's
sampled verdict accepts, are files the verifier sets that window up on. -/
def WindowAt (k : Verity.Registration.Composite) : Prop :=
  ∀ θ F W, I.window θ = some W → (∃ p, ParamsAt θ p) →
    I.base.circuitFile θ F = I.base.circuitFile θ (I.base.registered θ) →
    I.base.publicFile θ F = I.base.publicFile θ (I.base.registered θ) →
    (∃ c o, Verity.Game.Reaches (I.base.game θ F c) o ∧ I.base.Verdict k .sampled o) → I.SetsUp θ F W

/-- **`WindowCovers` at the deployed vocabulary**: where the deployment runs no window, the published δ is `⊤`
(`deltaOf`); where it runs one that the verifier doesn't set up on the committed files, an accepting play's files
pass the file check, so under the further finders' premises they name the registered files (`namedFile_eq`), where
the verifier sets the window up (`WindowAt`). -/
theorem covers (k : Verity.Registration.Composite) (hRg : I.RegisteredFiles) (hWa : I.WindowAt k) :
    WindowCovers (I.vocab K) k I.windows := by
  intro θ F hS hP c σ hcr
  rcases hA : I.window θ with _ | W
  · exact le_top.trans_eq (I.deltaOf_none hA _).symm
  refine le_of_eq_of_le ?_ zero_le
  by_contra hne
  obtain ⟨t, hv⟩ := Instance.exists_tape_of_prob_ne hne
  have hreach : Verity.Game.Reaches (I.base.game θ F c) (Verity.Game.Tape.out _ σ t) :=
    Verity.Game.Tape.reaches_out _ σ t
  have hv' : I.base.Verdict k .sampled (Verity.Game.Tape.out _ σ t) := hv
  obtain ⟨hθ, hF, -⟩ := Binding.play_reaches hreach
  have hreg : I.base.Registered θ F := by
    have h1 := hv'.1
    rw [hθ, hF] at h1
    exact h1
  have hR := hRg θ F hP hreg
  obtain ⟨h1, h2⟩ := I.fileCR K hcr
  have hs := hWa θ F W hA hP (namedFile_eq hreg.1 hR.1 h1) (namedFile_eq hreg.2 hR.2 h2) ⟨c, _, hreach, hv'⟩
  rw [I.windows_of hA hs] at hS
  cases hS

/-! ## The leg at the deployment -/

/-- **The constructions at the deployment**: `K₀`'s with the binding's finders at the deployed device's history
(`dev`) and composite `k`, the window's projection at the sessions' readings and decoders at the window's table
(`sessionReadings`, `sessionDecoders`, `Window.table`), and the sample-coupled coin maps at the family the
parameters name (`labelAt`), the digest `afterAt` and the window's bytes as `MAX_COINS` fresh frames, as
`windowSpotSound_sessions` reads them. -/
noncomputable def constructionsAt (K₀ : WindowConstructions I.base) (k : Verity.Registration.Composite)
    (reg : Option Claims.Msg → ByteArray) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray)
    (record₀ : Window → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Window → ByteArray → Array ByteArray → ByteArray → Claims.Msg) : WindowConstructions I.base :=
  (K₀.withBinding I.dev k).withProject I.base (sessionReadings Window.table ev record₀ recordDigest)
    (fun W _ _ _ _ _ => sessionDecoders reg W.table filesDec)
    (windowSampleMaps fieldEnc I.base (sessionReadings Window.table ev record₀ recordDigest)
      (fun θ _ _ _ => labelAt θ) (afterAt I.base k) fun W => W.srcCoinsAt _root_.Spot.Challenger.MAX_COINS)

/-- **What the verifier's acceptance gives of each play, at the deployment's windows** (`WindowRead`), at composite
`k`: at every play on files the verifier sets one of the deployment's windows up on, which `k`'s sampled verdict
accepts, whose log v2 read is the device's, whose blobs sent hold each pinned table at its registered file and whose
sample answers the strategy's `afterAt`, every release of the family the parameters name precedes every opening, and
the play has the sessions of the window's sample at the coins that family releases. -/
def WindowReads (k : Verity.Registration.Composite) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray) : Prop :=
  ∀ θ F W, I.windows θ F = some W →
  ∀ (c : ByteArray) (σ : Strategy (I.base.game θ F c)) (z : CoinTape (I.base.game θ F c) σ),
    I.base.Verdict k .sampled (CoinTape.run (I.base.game θ F c) σ z) →
    Binding.Logged Binding.readsV2 (CoinTape.run (I.base.game θ F c) σ z) →
    TablesSent (CoinTape.run (I.base.game θ F c) σ z).sent →
    SampledDigest k (CoinTape.run (I.base.game θ F c) σ z).sent (afterAt I.base k θ F c σ) →
    WindowRead W W.table ev filesDec (labelAt θ) (afterAt I.base k θ F c σ) _root_.Spot.Challenger.MAX_COINS
      (CoinTape.run (I.base.game θ F c) σ z)

/-- **SPOT's leg at the deployment's windows** (`windowSpotSound_sessions`, at the deployed vocabulary and the
constructions `constructionsAt` assembles): `WindowDeployed` and `WindowCovers` are `deployed`'s and `covers`'s, the
deployed device printed each play's log and signed what its signatures are on (`Device.printsLog_dev`,
`Device.signsPlay_dev`, at the base's signer, `WindowInstance.signer`), and each window the verifier sets up reads
`MAX_COINS` frames' bytes and has members of its one table (`lb`, `members`). What is left is what the claim reads of
the window (`ClaimReads`), the registered files' file check (`RegisteredFiles`), the verifier's window at the
registered files (`WindowAt`) and what the verifier's acceptance gives of each play (`WindowReads`). -/
theorem spotSound (K₀ : WindowConstructions I.base) (k : Verity.Registration.Composite)
    (reg : Option Claims.Msg → ByteArray) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray)
    (record₀ : Window → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Window → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hC : I.ClaimReads (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest))
    (hRg : I.RegisteredFiles) (hWa : I.WindowAt k) (hWR : I.WindowReads k ev filesDec) :
    (I.vocab (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest)).SpotSound k :=
  windowSpotSound_sessions (I.vocab (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest)) k
    (I.liftOf (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest) K₀) I.dev reg Window.table ev filesDec
    record₀ recordDigest (fun θ _ _ _ => labelAt θ) _root_.Spot.Challenger.MAX_COINS
    (I.deployed (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest) (fun _ _ _ _ => rfl)
      (fun _ _ _ _ => rfl) hC)
    (I.covers (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest) k hRg hWa)
    (fun _ _ _ _ o _ => Device.printsLog_dev _ o)
    (fun θ F _ _ c => Device.signsPlay_dev (I.vocab (I.constructionsAt K₀ k reg ev filesDec record₀ recordDigest))
      I.deployment θ F c (I.signer θ))
    (fun θ F W h => I.lb θ F W (I.windows_some h).2)
    (fun θ F W h => I.members θ F W (I.windows_some h).2)
    (fun θ F W h c σ z hv hl ht hd => hWR θ F W h c σ z hv hl ht hd)

end WindowInstance

/-- **Soundness at remote's verifier at the deployed vocabulary over windows of several members**: remote's compiled
verifier accepts files that passed the commit phase's check while one of its three claims fails with probability at
most their errors summed, at the vocabulary `WindowInstance.vocab` builds from the deployment and the constructions
`WindowInstance.constructionsAt` assembles, wherever what the claim reads of the window (`ClaimReads`), the
registered files' file check (`RegisteredFiles`), the verifier's window at the registered files (`WindowAt`), what
the verifier's acceptance gives of each play (`WindowReads`), and the base's PoUW and inference-only targets hold.
The sampled leg's error at a window is `Window.delta`: the sample's own miss and the members' knowledge and binding
terms summed (`JointShareV5`), each member's registration's roots term (`Window.rootsErr`), a two-continuation term
per round opened after a digest in every member's session, its closing round among them, one for the record and one
for the bound digests, the coin trees' hiding term, and the read finder's bound. The premises at the finders the
constructions build from a strategy are `error`'s condition at that strategy. -/
theorem remote_soundness_window (I : WindowInstance d α) (K₀ : WindowConstructions I.base)
    (reg : Option Claims.Msg → ByteArray) (ev : ℕ → Exchange → Option Claims.Msg)
    (filesDec : Claims.Msg → Array ByteArray)
    (record₀ : Window → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Window → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hC : I.ClaimReads (I.constructionsAt K₀ .remote reg ev filesDec record₀ recordDigest))
    (hRg : I.RegisteredFiles) (hWa : I.WindowAt .remote) (hWR : I.WindowReads .remote ev filesDec)
    (hW : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (I.constructionsAt K₀ .remote reg ev filesDec record₀ recordDigest)).SoundnessAt .remote :=
  Verity.remote_soundness_of_subprotocols ι α d _
    (I.spotSound K₀ .remote reg ev filesDec record₀ recordDigest hC hRg hWa hWR) hW hI

end Verity.Sampled
