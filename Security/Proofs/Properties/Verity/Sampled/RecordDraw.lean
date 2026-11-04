import Proofs.Properties.Verity.Sampled.Assembled
import Proofs.Flock.Claims.JsonNorm
import Proofs.Properties.Challenger.CanonRead
import Proofs.Properties.Challenger.Coins

/-!
# The record's draw is the session's draw file

Clause (i) of `MovesOf`: the record v2 reads in an accepted play decodes (`Exec.recordDraw`) to the session's draw file
(`Live.dj`) at the proof blocks the play's sample selects.

* **v2's refusal** (`DrawHeld`): wherever v2 drew `units` under the law `L` over `n` units (`Drew`), the units are
  ascending below `n` (`unitsAscending`) and the record's `unit_draw` prints as `UnitDraw.toJson ⟨L, n, units, #[]⟩`
  does (`recordDrawn`). It is a `Prop` about v2's code, taken as a hypothesis.
* **The record half** (`recordDraw_of_drew`): where `DrawHeld` holds, v2 drew `units`, and the session's law and
  population are `L` and `n`, the record decodes to the session's draw file at the proof blocks `units` names, at a
  session whose draw files `norm` fixes. Both are rebuilt values (`Flock.norm`) that `canonUtf8` prints alike, so they
  are equal (`eq_of_canonUtf8`): a rebuilt value is well formed (`FlockDeploy.wf_norm`), so its objects hold their keys
  in order (`sortedKeys_of_wf`), and sorted values that print alike are equivalent
  (`Properties.Challenger.CanonRead.canon_inj`). The draw file at the units' proof blocks is
  `UnitDraw.toJson ⟨L, n, units, #[]⟩` (`drawOf_filter`).
* **The draw files `norm` fixes** (`norm_printedDraw`): the printed draw file at every law but the work law. A work
  stratum's object (`WorkStratum.toJson`) lists its fields out of key order, so its tree isn't `norm`'s.
* **v2's coins fit `MAX_COINS` frames** (`coins_size_le`): v2's evidence holds its segment's openings within `LIMITS`
  (`Properties.Challenger.Limits.evidenceOf_within`), and the coins v2 reads are one frame per opening
  (`challengerStream_ok`).
* **What the sample's fill reads of the play**: the walk and the fill a node at a time (`walk_send_of`, `fill_coin_of`),
  an output reached from the walk's stop is reached from the node (`playReaches_walk`), readings that pick only openings
  (`PicksOpenings`), and the family and digest v2 reads the sample's coins at (`SampleNamed`).
* **Where the walk from the sample's fill stops** (`WalkFill`, `walkFill`): from the play's start, at readings that pick
  only openings, the walk from the tape the fill writes stops where every output's log is releases `L`, then nothing or
  an opening (`StopTail`), and `L`'s coins of the sample's family agree with the written coins. Only a fresh frame of
  the family answering the digest takes a written coin: a node is `FreshAt` the play's evidence only at the release its
  outputs' logs show there (`FreshAt.release`, `LogsAt.unique`).
* **The record's draw on the joint law's support** (`recordDraw_reachesOff`, `MovesOf`'s clause (i)): at the
  sample-coupled coin map, an output off the held-off events is reached from the first stop, so its log is the fill's
  releases, then nothing or an opening. Where every release of the sample's family precedes every opening
  (`SampleLinkFirst`), the play's coins of the family are those releases' (`freshCoins_of_stopTail`), and
  `recordDraw_of_coins` gives the record's draw.
* **The record's draw from the sample's coins** (`recordDraw_of_coins`): in an accepted play whose log v2 read is the
  device's, wherever the play's coins of the sample's family and the sample's coins agree (`CoinsAgree`: one list
  starts the other), the record decodes to the session's draw file at the sample. v2's coins start the play's
  (`logged_draw_v2`) and fit the sample's (`coins_size_le`), so they start the sample's bytes, and the sample is v2's
  units (`draw_of_drew`).
-/

namespace Verity.Game

variable {α V ι : Type}

/-! ## The walk and the fill, a node at a time -/

/-- **The walk only descends**: an output a play reaches from the walk's stop, it reaches from the node. -/
theorem playReaches_walk {g : Game α} {σ : Strategy g} :
    ∀ (g' : Game α) (s' : Strategy g') (Ks : ι → DigestCut V g') (x : CoinTape g' s')
      (back : CoinTape g' s' → CoinTape g σ) (o : α),
      PlayReaches (Pos.walk g' s' Ks x back).g' (Pos.walk g' s' Ks x back).s' o → PlayReaches g' s' o
  | .ret _, _, _, _, _, _, h => h
  | .send _ k, s', Ks, x, back, o, h => by
    by_cases hK : ∃ i, (Ks i).isHere
    · rw [Pos.walk, ite_eq_left_of_eq_true _ _ (eq_true hK)] at h
      exact h
    · rw [Pos.walk, ite_eq_right_of_eq_false _ _ (eq_false hK)] at h
      exact playReaches_walk (k s'.1) s'.2 _ x back o h
  | @Game.coin _ _ _ _ k, s', Ks, x, back, o, h => by
    by_cases hK : ∃ i, (Ks i).isHere
    · rw [Pos.walk, ite_eq_left_of_eq_true _ _ (eq_true hK)] at h
      exact h
    · rw [Pos.walk, ite_eq_right_of_eq_false _ _ (eq_false hK)] at h
      exact ⟨x.1, playReaches_walk (k x.1) (s' x.1) _ (x.2 x.1) _ o h⟩

theorem walk_send_of {g : Game α} {σ : Strategy g} {M : Type} {k : M → Game α} {s' : Strategy (.send M k)}
    {Ks : ι → DigestCut V (.send M k)} (hK : ∀ i, (Ks i).isHere = false) (x : CoinTape (.send M k) s')
    (back : CoinTape (.send M k) s' → CoinTape g σ) :
    Pos.walk (.send M k) s' Ks x back = Pos.walk (k s'.1) s'.2 (fun i => (Ks i).atSend s'.1) x back := by
  rw [Pos.walk]
  split_ifs with h
  · exact absurd h (by simp [hK])
  · rfl

open Classical in
theorem walk_coin_of {g : Game α} {σ : Strategy g} {C : Type} {iF : Fintype C} {iN : Nonempty C}
    {k : C → Game α} {s' : Strategy (@Game.coin _ C iF iN k)} {Ks : ι → DigestCut V (@Game.coin _ C iF iN k)}
    (hK : ∀ i, (Ks i).isHere = false) (x : CoinTape (@Game.coin _ C iF iN k) s')
    (back : CoinTape (@Game.coin _ C iF iN k) s' → CoinTape g σ) :
    Pos.walk (@Game.coin _ C iF iN k) s' Ks x back = Pos.walk (k x.1) (s' x.1) (fun i => (Ks i).atCoin x.1) (x.2 x.1)
      (fun y => back (x.1, Function.update x.2 x.1 y)) := by
  rw [Pos.walk]
  split_ifs with h
  · exact absurd h (by simp [hK])
  · rfl

theorem fill_send_of {Q : Type} (sel : FillSel α Q) {M : Type} {k : M → Game α} {s' : Strategy (.send M k)}
    {Ks : ι → DigestCut V (.send M k)} (hK : ∀ i, (Ks i).isHere = false) (m : ℕ) (κ : Fin m → Q)
    (z : CoinTape (.send M k) s') :
    fill sel (.send M k) s' Ks m κ z = fill sel (k s'.1) s'.2 (fun i => (Ks i).atSend s'.1) m κ z := by
  rw [fill]
  exact ite_eq_right (by simp [hK])

theorem fill_coin_of {Q : Type} (sel : FillSel α Q) {C : Type} {iF : Fintype C} {iN : Nonempty C}
    {k : C → Game α} {s' : Strategy (@Game.coin _ C iF iN k)} {Ks : ι → DigestCut V (@Game.coin _ C iF iN k)}
    (hK : ∀ i, (Ks i).isHere = false) (m : ℕ) (κ : Fin m → Q) (z : CoinTape (@Game.coin _ C iF iN k) s') :
    fill sel (@Game.coin _ C iF iN k) s' Ks m κ z =
      fillAt (T := fun c => CoinTape (k c) (s' c))
        (fun c m κ y => fill sel (k c) (s' c) (fun i => (Ks i).atCoin c) m κ y) (@sel C iF iN k) m κ z := by
  rw [fill]
  exact ite_eq_right (by simp [hK])

end Verity.Game

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open Properties.Challenger (bitsOf Draws)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The sample's family and the readings -/

/-- **The readings pick only openings**: each round's cut is fixed only at a frame that opens a coin tree's round. -/
def PicksOpenings {R : ℕ} (rd : Reading R) : Prop :=
  ∀ j pre f, rd.round j pre f → ∃ i s n, f = .opening i s n

/-- **v2 reads the sample's coins of the family `label` answering `after`** in what the prover sent composite `k`:
wherever v2's case accepts at coins it read under the challenger `dr`, `dr` names that family and digest. -/
def SampleNamed (k : Verity.Registration.Composite) (s : Verity.Sent) (label : String) (after : ByteArray) : Prop :=
  ∀ a b e p c ev ξ dr, V2Case k s a b e → V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr →
    dr.label = label ∧ dr.after = after

/-- **The log after a stop**: nothing, or an opening first. -/
def StopTail (log : List Release) : Prop :=
  log = [] ∨ ∃ r rest i s n, log = r :: rest ∧ r.frame = .opening i s n

/-- The coins `κ` as lists of bits, in order. -/
abbrev fillCoins {m : ℕ} (κ : Fin m → Fin 512 → Bool) : List (List Bool) := List.ofFn fun j => List.ofFn (κ j)

/-- **One list of coins starts the other.** -/
def CoinsAgree (a b : List (List Bool)) : Prop := a <+: b ∨ b <+: a

theorem CoinsAgree.cons {a b : List (List Bool)} (x : List Bool) (h : CoinsAgree a b) :
    CoinsAgree (x :: a) (x :: b) :=
  h.imp (fun h => List.cons_prefix_cons.2 ⟨rfl, h⟩) (fun h => List.cons_prefix_cons.2 ⟨rfl, h⟩)

theorem fillCoins_succ {m : ℕ} (κ : Fin (m + 1) → Fin 512 → Bool) :
    fillCoins κ = List.ofFn (κ 0) :: fillCoins (Fin.tail κ) :=
  List.ofFn_succ

theorem freshCoins_cons_of {label : String} {after : ByteArray} {r : Release} (L : List Release)
    (h : ¬ (r.frame = .fresh label ∧ r.answers = after)) :
    freshCoins label after (r :: L) = freshCoins label after L := by
  rcases r with ⟨fr, a, cs⟩
  cases fr with
  | fresh l =>
    have : ¬ (l = label ∧ a = after) := fun ⟨h1, h2⟩ => h ⟨by rw [h1], h2⟩
    simp [freshCoins, this]
  | hello _ => simp [freshCoins]
  | opening _ _ _ => simp [freshCoins]

theorem freshCoins_cons_fresh (label : String) (after : ByteArray) (cs : List Bool) (L : List Release) :
    freshCoins label after (⟨.fresh label, after, cs⟩ :: L) = cs :: freshCoins label after L := by
  simp [freshCoins]

theorem not_picks_fresh {R : ℕ} {rd : Reading R} (h : PicksOpenings rd) (j : Fin R) (pre : List Release)
    (l : String) : ¬ rd.round j pre (.fresh l) := fun hp => by
  obtain ⟨_, _, _, he⟩ := h j pre _ hp
  cases he

theorem not_picks_hello {R : ℕ} {rd : Reading R} (h : PicksOpenings rd) (j : Fin R) (pre : List Release)
    (spec : Flock.CoinTree.Spec) : ¬ rd.round j pre (.hello spec) := fun hp => by
  obtain ⟨_, _, _, he⟩ := h j pre _ hp
  cases he

/-! ## The releases' outputs -/

/-- **An output the releases reach is one `f` reaches** at some log. -/
theorem reaches_relK {β : Type} : ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (f : List Release → Game β) (o : β), (∃ s, PlayReaches (relK k fs ts f) s o) →
      ∃ log, ∃ s, PlayReaches (f log) s o
  | 0, _, _, _, _, h => ⟨[], h⟩
  | k + 1, fs, ts, f, o, ⟨⟨msg, s⟩, h⟩ => by
    rcases msg with _ | ⟨l | spec | ⟨i, sn, n⟩, b⟩
    · exact ⟨[], s, h⟩
    · have h' : ∃ s, PlayReaches (optGame (takeCoin l fs) (f []) fun fs' =>
          .coin (Fin 512 → Bool) fun x => relK k fs' ts fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)) s o :=
        ⟨s, h⟩
      clear h s
      generalize takeCoin l fs = tc at h'
      cases tc with
      | none => exact ⟨[], h'⟩
      | some fs' =>
        obtain ⟨s, x, h⟩ := h'
        obtain ⟨log, h⟩ := reaches_relK k fs' ts _ o ⟨s x, h⟩
        exact ⟨_, h⟩
    · obtain ⟨t, h⟩ := h
      obtain ⟨log, h⟩ := reaches_relK k _ _ _ o ⟨s t, h⟩
      exact ⟨_, h⟩
    · have h' : ∃ s, PlayReaches (optGame ts[i]? (f []) fun T => optGame (T.next sn n) (f []) fun p =>
          haveI : Nonempty (TreeOpenable T.nonce T.spec T.sample p.1 p.2) := ⟨⟨T.sample.2 p.1 p.2, rfl⟩⟩
          .coin (TreeOpenable T.nonce T.spec T.sample p.1 p.2) fun d =>
            relK k fs (ts.set i (T.bump p.1)) fun log =>
              f (⟨.opening i sn n, b, TreeDraw.opened d.1 n⟩ :: log)) s o :=
        ⟨s, h⟩
      clear h s
      generalize ts[i]? = oT at h'
      cases oT with
      | none => exact ⟨[], h'⟩
      | some T =>
        change ∃ s, PlayReaches (optGame (T.next sn n) (f []) _) s o at h'
        generalize T.next sn n = op at h'
        cases op with
        | none => exact ⟨[], h'⟩
        | some p =>
          obtain ⟨s, d, h⟩ := h'
          obtain ⟨log, h⟩ := reaches_relK k _ _ _ o ⟨s d, h⟩
          exact ⟨_, h⟩

/-- **The releases reach what `f` reaches at the empty log**: the prover closes at once. -/
theorem reaches_relK_nil {β : Type} : ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (f : List Release → Game β) (o : β), (∃ s, PlayReaches (f []) s o) → ∃ s, PlayReaches (relK k fs ts f) s o
  | 0, _, _, _, _, h => h
  | _ + 1, _, _, _, _, ⟨s, h⟩ => ⟨⟨none, s⟩, h⟩

theorem reaches_evidenceG {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {L : List Release} {o : Exchange}
    (h : ∃ s, PlayReaches (evidenceG θ F c L) s o) : o.log = L := by
  obtain ⟨s, h⟩ := h
  have h' : o = ⟨θ, F, c, L, s.1.1, s.1.2⟩ := h
  rw [h']

theorem exists_reaches_evidenceG (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (L : List Release) :
    ∃ o, (∃ s, PlayReaches (evidenceG θ F c L) s o) ∧ o.log = L :=
  ⟨⟨θ, F, c, L, .empty, #[]⟩, ⟨⟨(.empty, #[]), ⟨⟩⟩, rfl⟩, rfl⟩

/-- **A node whose outputs' logs are `pre`, the release `r`, then any**, one of them `pre` and `r` alone. -/
def LogsAt (g : Game Exchange) (pre : List Release) (r : Release) : Prop :=
  (∀ o, (∃ s, PlayReaches g s o) → ∃ log, o.log = pre ++ r :: log) ∧
    ∃ o, (∃ s, PlayReaches g s o) ∧ o.log = pre ++ [r]

theorem logsAt_relK (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (k : ℕ) (fs : List (String × ℕ))
    (ts : List OpenTree) (pre : List Release) (r : Release) :
    LogsAt (relK k fs ts fun log => evidenceG θ F c (pre ++ r :: log)) pre r := by
  refine ⟨fun o h => ?_, ?_⟩
  · obtain ⟨log, h⟩ := reaches_relK k fs ts _ o h
    exact ⟨log, reaches_evidenceG h⟩
  · obtain ⟨o, ho, hl⟩ := exists_reaches_evidenceG θ F c (pre ++ [r])
    exact ⟨o, reaches_relK_nil k fs ts _ o ho, hl⟩

theorem LogsAt.unique {g : Game Exchange} {pre pre' : List Release} {r r' : Release} (h : LogsAt g pre r)
    (h' : LogsAt g pre' r') : pre = pre' ∧ r = r' := by
  obtain ⟨o, ho, hl⟩ := h.2
  obtain ⟨o', ho', hl'⟩ := h'.2
  obtain ⟨log, hlog⟩ := h'.1 o ho
  obtain ⟨log', hlog'⟩ := h.1 o' ho'
  rw [hl] at hlog
  rw [hl'] at hlog'
  have h1 := congrArg List.length hlog
  have h2 := congrArg List.length hlog'
  simp only [List.length_append, List.length_cons, List.length_nil] at h1 h2
  have hnil : log = [] := List.eq_nil_of_length_eq_zero (by omega)
  subst hnil
  obtain ⟨hp, hr⟩ := List.append_inj' hlog rfl
  exact ⟨hp, (List.cons.inj hr).1⟩

/-- **Only a fresh frame of the family answering the digest is `FreshAt`**: a coin node whose every coin's
continuation logs `pre`, then a release (`LogsAt`), is `FreshAt` only where that release is a fresh frame of `label`
answering `after`. -/
theorem FreshAt.release {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {label : String} {after : ByteArray}
    {C : Type} {iF : Fintype C} {iN : Nonempty C} {k' : C → Game Exchange} {pre : List Release}
    {r : C → Release} (hk : ∀ x, LogsAt (k' x) pre (r x))
    (h : FreshAt (evidenceG θ F c) label after (@Game.coin _ C iF iN k')) (x : C) :
    (r x).frame = .fresh label ∧ (r x).answers = after := by
  obtain ⟨k, fs, ts, pre', iF', iN', heq⟩ := h
  injection heq with hC _ hk'
  subst hC
  have hk'' := eq_of_heq hk'
  have h2 : LogsAt (k' x) pre' ⟨.fresh label, after, List.ofFn x⟩ := by
    rw [hk'']
    exact logsAt_relK θ F c k fs ts pre' _
  obtain ⟨-, hr⟩ := (hk x).unique h2
  rw [hr]
  exact ⟨rfl, rfl⟩

theorem freshSel_none {E : List Release → Game Exchange} {label : String} {after : ByteArray} {C : Type}
    {iF : Fintype C} {iN : Nonempty C} {k' : C → Game Exchange}
    (h : FreshAt E label after (@Game.coin _ C iF iN k') → False) : @freshSel E label after C iF iN k' = none := by
  unfold freshSel
  exact dite_eq_right (fun ⟨x⟩ => h x)

theorem freshSel_refl {E : List Release → Game Exchange} {label : String} {after : ByteArray}
    {iF : Fintype (Fin 512 → Bool)} {iN : Nonempty (Fin 512 → Bool)} {k' : (Fin 512 → Bool) → Game Exchange}
    (h : FreshAt E label after (@Game.coin _ _ iF iN k')) :
    @freshSel E label after _ iF iN k' = some (Equiv.refl _) := by
  unfold freshSel
  rw [dite_eq_left ⟨h⟩]
  rfl

/-! ## The releases and the cuts, one frame at a time -/

/-- **The releases' node after the prover's frame `m`**, at `k + 1` frames (`relK`). -/
def relKNext {β : Type} (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β) :
    Option (Frame × ByteArray) → Game β
  | none => f []
  | some (.fresh l, b) => optGame (takeCoin l fs) (f []) fun fs' =>
      .coin (Fin 512 → Bool) fun x => relK k fs' ts fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)
  | some (.hello spec, ν) => .coin (TreeSample spec) fun t =>
      relK k fs (ts ++ [⟨ν, spec, t, fun _ => 0⟩]) fun log =>
        f (⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩ :: log)
  | some (.opening i s n, b) => optGame ts[i]? (f []) fun T => optGame (T.next s n) (f []) fun p =>
      haveI : Nonempty (TreeOpenable T.nonce T.spec T.sample p.1 p.2) := ⟨⟨T.sample.2 p.1 p.2, rfl⟩⟩
      .coin (TreeOpenable T.nonce T.spec T.sample p.1 p.2) fun d =>
        relK k fs (ts.set i (T.bump p.1)) fun log => f (⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log)

theorem relK_succ {β : Type} (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (f : List Release → Game β) :
    relK (k + 1) fs ts f = .send _ (relKNext k fs ts f) := rfl

open Classical in
/-- **A frame's cut after the prover's frame `m`**, at `k + 1` frames (`frameCut`). -/
noncomputable def frameNext {β : Type} (P : List Release → Frame → Prop) (rv : β → Option Claims.Msg) (k : ℕ)
    (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β) (pre : List Release) :
    (m : Option (Frame × ByteArray)) → DigestCut.AgreeCut sha512Msg (relKNext k fs ts f m)
  | none => .skip
  | some (.fresh l, b) =>
    if P pre (.fresh l) then cutAt rv b.data.toList else
      optCut (takeCoin l fs) .skip fun fs' => .coin fun x =>
        frameCut P rv k fs' ts _ (pre ++ [⟨.fresh l, b, List.ofFn x⟩])
  | some (.hello spec, ν) =>
    if P pre (.hello spec) then cutAt rv ν.data.toList else
      .coin fun t => frameCut P rv k fs (ts ++ [⟨ν, spec, t, fun _ => 0⟩]) _
        (pre ++ [⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩])
  | some (.opening i s n, b) =>
    if P pre (.opening i s n) then cutAt rv b.data.toList else
      optCut ts[i]? .skip fun T => optCut (T.next s n) .skip fun p => .coin fun d =>
        frameCut P rv k fs (ts.set i (T.bump p.1)) _ (pre ++ [⟨.opening i s n, b, TreeDraw.opened d.1 n⟩])

theorem frameCut_succ {β : Type} (P : List Release → Frame → Prop) (rv : β → Option Claims.Msg) (k : ℕ)
    (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β) (pre : List Release) :
    HEq (frameCut P rv (k + 1) fs ts f pre).cut
      (DigestCut.send (V := Claims.Msg) fun m => (frameNext P rv k fs ts f pre m).cut) := HEq.rfl

/-! ## Where the walk from the sample's fill stops -/

section WalkFill

variable {g : Game Exchange} {σ : Strategy g} (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (label : String)
  (after : ByteArray) {R : ℕ} (rd : Reading R)

/-- **What the sample's fill and the walk keep, at `k` frames more**: from a node of the device's releases after the
log `pre`, at cuts each past or round `j`'s at `rd`, the walk from the tape the fill writes stops at an output whose log
is `pre`, then releases `L`, then nothing or an opening (`StopTail`); and `L`'s coins of the family `label` answering
`after` and the written coins `κ` agree. -/
def WalkFill (k : ℕ) : Prop :=
  ∀ (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Release) (f : List Release → Game Exchange),
    (∀ log, f log = evidenceG θ F c (pre ++ log)) → ∀ (g' : Game Exchange), g' = relK k fs ts f →
    ∀ (s' : Strategy g') (Ks : Fin R → DigestCut Claims.Msg g'),
    (∀ j, Ks j = .skip ∨ HEq (Ks j) (frameCut (rd.round j) (rd.reveal j) k fs ts f pre).cut) →
    ∀ (m : ℕ) (κ : Fin m → Fin 512 → Bool) (z : CoinTape g' s') (back : CoinTape g' s' → CoinTape g σ),
    ∃ L, CoinsAgree (freshCoins label after L) (fillCoins κ) ∧
      ∀ o, PlayReaches (Pos.walk g' s' Ks (fill (freshSel (evidenceG θ F c) label after) g' s' Ks m κ z) back).g'
        (Pos.walk g' s' Ks (fill (freshSel (evidenceG θ F c) label after) g' s' Ks m κ z) back).s' o →
        ∃ log', o.log = pre ++ L ++ log' ∧ StopTail log'

variable {θ F c label after rd}

theorem walkFill_nil {pre : List Release} {f : List Release → Game Exchange}
    (hf : ∀ log, f log = evidenceG θ F c (pre ++ log)) {g'' : Game Exchange} (hn : g'' = f []) (s'' : Strategy g'')
    (Ks : Fin R → DigestCut Claims.Msg g'') (x : CoinTape g'' s'') (back : CoinTape g'' s'' → CoinTape g σ)
    (o : Exchange) (h : PlayReaches (Pos.walk g'' s'' Ks x back).g' (Pos.walk g'' s'' Ks x back).s' o) :
    ∃ log', o.log = pre ++ [] ++ log' ∧ StopTail log' := by
  have h1 : ∃ s, PlayReaches g'' s o := ⟨s'', playReaches_walk g'' s'' Ks x back o h⟩
  rw [hn, hf] at h1
  exact ⟨[], by simp [reaches_evidenceG h1], .inl rfl⟩

theorem walkFill_opening {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {pre : List Release}
    {f : List Release → Game Exchange} (hf : ∀ log, f log = evidenceG θ F c (pre ++ log)) {i s n : ℕ}
    {b : ByteArray} {g'' : Game Exchange} (hn : g'' = relKNext k fs ts f (some (.opening i s n, b)))
    (s'' : Strategy g'') (Ks : Fin R → DigestCut Claims.Msg g'') (x : CoinTape g'' s'')
    (back : CoinTape g'' s'' → CoinTape g σ) (o : Exchange)
    (h : PlayReaches (Pos.walk g'' s'' Ks x back).g' (Pos.walk g'' s'' Ks x back).s' o) :
    ∃ log', o.log = pre ++ [] ++ log' ∧ StopTail log' := by
  have h1 : ∃ s, PlayReaches g'' s o := ⟨s'', playReaches_walk g'' s'' Ks x back o h⟩
  rw [hn] at h1
  change ∃ t₀, PlayReaches (optGame ts[i]? (f []) _) t₀ o at h1
  generalize ts[i]? = oT at h1
  cases oT with
  | none =>
    rw [optGame, hf] at h1
    exact ⟨[], by simp [reaches_evidenceG h1], .inl rfl⟩
  | some T =>
    change ∃ t₀, PlayReaches (optGame (T.next s n) (f []) _) t₀ o at h1
    generalize T.next s n = op at h1
    cases op with
    | none =>
      rw [optGame, hf] at h1
      exact ⟨[], by simp [reaches_evidenceG h1], .inl rfl⟩
    | some p =>
      obtain ⟨s₁, d, h1⟩ := h1
      obtain ⟨log, h1⟩ := reaches_relK k _ _ _ o ⟨s₁ d, h1⟩
      rw [hf] at h1
      refine ⟨⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log, ?_, .inr ⟨_, log, i, s, n, rfl, rfl⟩⟩
      simp [reaches_evidenceG h1]

open Classical in
/-- **One coin of the releases**: at a coin node of the releases, whose coin `x` releases `r x`, where the fill writes
the coin `c₀` and fills on with `κ'`, the walk goes on to `c₀`'s continuation. -/
theorem walkFill_coin {k : ℕ} (IH : WalkFill (g := g) (σ := σ) θ F c label after rd k) {C : Type} {iF : Fintype C}
    {iN : Nonempty C} (fs' : C → List (String × ℕ)) (ts' : C → List OpenTree) (pre : List Release)
    (f : List Release → Game Exchange) (hf : ∀ log, f log = evidenceG θ F c (pre ++ log)) (r : C → Release)
    (s'' : Strategy (@Game.coin _ C iF iN fun x => relK k (fs' x) (ts' x) fun log => f (r x :: log)))
    (Ks : Fin R → DigestCut Claims.Msg
      (@Game.coin _ C iF iN fun x => relK k (fs' x) (ts' x) fun log => f (r x :: log)))
    (hK : ∀ j, Ks j = .skip ∨ Ks j = DigestCut.coin fun x =>
      (frameCut (rd.round j) (rd.reveal j) k (fs' x) (ts' x) (fun log => f (r x :: log)) (pre ++ [r x])).cut)
    (m : ℕ) (κ : Fin m → Fin 512 → Bool) (z : CoinTape (@Game.coin _ C iF iN fun x =>
      relK k (fs' x) (ts' x) fun log => f (r x :: log)) s'')
    (back : CoinTape (@Game.coin _ C iF iN fun x => relK k (fs' x) (ts' x) fun log => f (r x :: log)) s'' →
      CoinTape g σ)
    (c₀ : C) (m' : ℕ) (κ' : Fin m' → Fin 512 → Bool)
    (hfill : fill (freshSel (evidenceG θ F c) label after) _ s'' Ks m κ z =
      (c₀, Function.update z.2 c₀ (fill (freshSel (evidenceG θ F c) label after) _ (s'' c₀)
        (fun j => (Ks j).atCoin c₀) m' κ' (z.2 c₀)))) :
    ∃ L, CoinsAgree (freshCoins label after L) (fillCoins κ') ∧
      ∀ o, PlayReaches (Pos.walk _ s'' Ks (fill (freshSel (evidenceG θ F c) label after) _ s'' Ks m κ z) back).g'
        (Pos.walk _ s'' Ks (fill (freshSel (evidenceG θ F c) label after) _ s'' Ks m κ z) back).s' o →
        ∃ log', o.log = pre ++ (r c₀ :: L) ++ log' ∧ StopTail log' := by
  have hH : ∀ j, (Ks j).isHere = false := fun j => by
    rcases hK j with h | h <;> rw [h] <;> rfl
  rw [walk_coin_of hH]
  generalize hX : fill (freshSel (evidenceG θ F c) label after)
    (@Game.coin _ C iF iN fun x => relK k (fs' x) (ts' x) fun log => f (r x :: log)) s'' Ks m κ z = X
  rw [hfill] at hX
  subst hX
  dsimp only
  rw [Function.update_self]
  obtain ⟨L, hA, hO⟩ := IH (fs' c₀) (ts' c₀) (pre ++ [r c₀]) (fun log => f (r c₀ :: log))
    (fun log => by rw [hf, List.append_assoc, List.singleton_append]) _ rfl (s'' c₀)
    (fun j => (Ks j).atCoin c₀)
    (fun j => by
      rcases hK j with h | h
      · exact .inl (by rw [h]; rfl)
      · exact .inr (by rw [h]; rfl))
    m' κ' (z.2 c₀) _
  exact ⟨L, hA, fun o ho => by
    obtain ⟨log', h1, h2⟩ := hO o ho
    exact ⟨log', by rw [h1]; simp, h2⟩⟩

theorem frameNext_fresh {β : Type} (hrd : PicksOpenings rd) (j : Fin R) (rv : β → Option Claims.Msg) (k : ℕ)
    (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β) (pre : List Release) (l : String)
    (b : ByteArray) :
    frameNext (rd.round j) rv k fs ts f pre (some (.fresh l, b)) =
      optCut (takeCoin l fs) .skip fun fs' => DigestCut.AgreeCut.coin fun x =>
        frameCut (rd.round j) rv k fs' ts (fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log))
          (pre ++ [⟨.fresh l, b, List.ofFn x⟩]) := by
  rw [frameNext]
  exact ite_eq_right (not_picks_fresh hrd j pre l)

theorem frameNext_hello {β : Type} (hrd : PicksOpenings rd) (j : Fin R) (rv : β → Option Claims.Msg) (k : ℕ)
    (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β) (pre : List Release)
    (spec : Flock.CoinTree.Spec)
    (ν : ByteArray) :
    frameNext (rd.round j) rv k fs ts f pre (some (.hello spec, ν)) =
      DigestCut.AgreeCut.coin fun t : TreeSample spec =>
        frameCut (rd.round j) rv k fs (ts ++ [(⟨ν, spec, t, fun _ => 0⟩ : OpenTree)])
          (fun log => f (⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩ :: log))
          (pre ++ [⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩]) := by
  rw [frameNext]
  exact ite_eq_right (not_picks_hello hrd j pre spec)

/-- **After the prover's frame**: `WalkFill` at `k` frames gives it at the node after any frame the prover sends. -/
theorem walkFill_next (hrd : PicksOpenings rd) {k : ℕ} (IH : WalkFill (g := g) (σ := σ) θ F c label after rd k)
    (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Release) (f : List Release → Game Exchange)
    (hf : ∀ log, f log = evidenceG θ F c (pre ++ log)) (msg : Option (Frame × ByteArray)) (g'' : Game Exchange)
    (hn : g'' = relKNext k fs ts f msg) (s'' : Strategy g'') (Ks : Fin R → DigestCut Claims.Msg g'')
    (hK : ∀ j, Ks j = .skip ∨ HEq (Ks j) (frameNext (rd.round j) (rd.reveal j) k fs ts f pre msg).cut)
    (m : ℕ) (κ : Fin m → Fin 512 → Bool) (z : CoinTape g'' s'') (back : CoinTape g'' s'' → CoinTape g σ) :
    ∃ L, CoinsAgree (freshCoins label after L) (fillCoins κ) ∧
      ∀ o, PlayReaches
        (Pos.walk g'' s'' Ks (fill (freshSel (evidenceG θ F c) label after) g'' s'' Ks m κ z) back).g'
        (Pos.walk g'' s'' Ks (fill (freshSel (evidenceG θ F c) label after) g'' s'' Ks m κ z) back).s' o →
        ∃ log', o.log = pre ++ L ++ log' ∧ StopTail log' := by
  rcases msg with _ | ⟨l | spec | ⟨i, s, n⟩, b⟩
  · exact ⟨[], .inl List.nil_prefix, walkFill_nil hf hn s'' Ks _ back⟩
  · have hn1 : g'' = optGame (takeCoin l fs) (f []) fun fs' =>
        .coin (Fin 512 → Bool) fun x => relK k fs' ts fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log) := hn
    have hK1 : ∀ j, Ks j = .skip ∨
        HEq (Ks j) (optCut (takeCoin l fs) .skip fun fs' => DigestCut.AgreeCut.coin fun x =>
          frameCut (rd.round j) (rd.reveal j) k fs' ts (fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log))
            (pre ++ [⟨.fresh l, b, List.ofFn x⟩])).cut :=
      fun j => (hK j).imp_right fun h =>
        h.trans (heq_of_eq (congrArg DigestCut.AgreeCut.cut
          (frameNext_fresh hrd j (rd.reveal j) k fs ts f pre l b)))
    clear hK hn
    generalize takeCoin l fs = tc at hn1 hK1
    cases tc with
    | none => exact ⟨[], .inl List.nil_prefix, walkFill_nil hf hn1 s'' Ks _ back⟩
    | some fs' =>
      change g'' = Game.coin (Fin 512 → Bool) _ at hn1
      subst hn1
      have hK2 : ∀ j, Ks j = .skip ∨ Ks j = DigestCut.coin fun x =>
          (frameCut (rd.round j) (rd.reveal j) k ((fun _ => fs') x) ((fun _ => ts) x)
            (fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)) (pre ++ [⟨.fresh l, b, List.ofFn x⟩])).cut :=
        fun j => (hK1 j).imp_right eq_of_heq
      have hH : ∀ j, (Ks j).isHere = false := fun j => by
        rcases hK2 j with h | h <;> rw [h] <;> rfl
      by_cases hs : l = label ∧ b = after
      · obtain ⟨rfl, rfl⟩ := hs
        have hF : FreshAt (evidenceG θ F c) l b (Game.coin (Fin 512 → Bool) fun x =>
            relK k fs' ts fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)) :=
          ⟨k, fs', ts, pre, inferInstance, inferInstance, by simp only [hf]⟩
        have hsel := freshSel_refl hF
        cases m with
        | zero =>
          obtain ⟨L, -, hO⟩ := walkFill_coin IH (fun _ => fs') (fun _ => ts) pre f hf
            (fun x => ⟨.fresh l, b, List.ofFn x⟩) s'' Ks hK2 0 κ z back z.1 0 κ
            (by rw [fill_coin_of _ hH]; rfl)
          exact ⟨_, .inr (by rw [show fillCoins κ = [] from List.ofFn_zero]; exact List.nil_prefix), hO⟩
        | succ m₀ =>
          obtain ⟨L, hA, hO⟩ := walkFill_coin IH (fun _ => fs') (fun _ => ts) pre f hf
            (fun x => ⟨.fresh l, b, List.ofFn x⟩) s'' Ks hK2 (m₀ + 1) κ z back (κ 0) m₀ (Fin.tail κ)
            (by rw [fill_coin_of _ hH, hsel]; rfl)
          refine ⟨_, ?_, hO⟩
          rw [freshCoins_cons_fresh, fillCoins_succ]
          exact hA.cons _
      · have hns : ¬ ((⟨.fresh l, b, List.ofFn z.1⟩ : Release).frame = .fresh label ∧
            (⟨.fresh l, b, List.ofFn z.1⟩ : Release).answers = after) := fun ⟨h1, h2⟩ =>
          hs ⟨Frame.fresh.inj h1, h2⟩
        have hnone := freshSel_none (E := evidenceG θ F c) (label := label) (after := after)
          (k' := fun x : Fin 512 → Bool => relK k fs' ts fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log))
          (iF := inferInstance) (iN := inferInstance) fun h => by
            obtain ⟨h1, h2⟩ := FreshAt.release (r := fun x => (⟨.fresh l, b, List.ofFn x⟩ : Release))
              (pre := pre)
              (fun x => by
                show LogsAt (relK k fs' ts fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)) pre _
                simp only [hf]
                exact logsAt_relK θ F c k fs' ts pre _) h z.1
            exact hns ⟨h1, h2⟩
        obtain ⟨L, hA, hO⟩ := walkFill_coin IH (fun _ => fs') (fun _ => ts) pre f hf
          (fun x => ⟨.fresh l, b, List.ofFn x⟩) s'' Ks hK2 m κ z back z.1 m κ
          (by rw [fill_coin_of _ hH, hnone]; cases m <;> rfl)
        refine ⟨_, ?_, hO⟩
        rw [freshCoins_cons_of _ hns]
        exact hA
  · have hn1 : g'' = Game.coin (TreeSample spec) fun t =>
        relK k fs (ts ++ [⟨b, spec, t, fun _ => 0⟩]) fun log =>
          f (⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩ :: log) :=
      hn
    subst hn1
    have hK2 : ∀ j, Ks j = .skip ∨ Ks j = DigestCut.coin fun t =>
        (frameCut (rd.round j) (rd.reveal j) k ((fun _ => fs) t) ((fun t => ts ++ [⟨b, spec, t, fun _ => 0⟩]) t)
          (fun log => f (⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩ :: log))
          (pre ++ [⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩])).cut :=
      fun j => (hK j).imp_right fun h =>
        eq_of_heq (h.trans (heq_of_eq (congrArg DigestCut.AgreeCut.cut
          (frameNext_hello hrd j (rd.reveal j) k fs ts f pre spec b))))
    have hH : ∀ j, (Ks j).isHere = false := fun j => by
      rcases hK2 j with h | h <;> rw [h] <;> rfl
    have hns : ∀ t : TreeSample spec,
        ¬ ((⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩ : Release).frame = .fresh label ∧
          (⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩ : Release).answers = after) :=
      fun _ ⟨h1, _⟩ => Frame.noConfusion h1
    have hnone := freshSel_none (E := evidenceG θ F c) (label := label) (after := after)
      (k' := fun t : TreeSample spec => relK k fs (ts ++ [⟨b, spec, t, fun _ => 0⟩]) fun log =>
        f (⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩ :: log))
      (iF := inferInstance) (iN := inferInstance) fun h => by
        obtain ⟨h1, h2⟩ := FreshAt.release
          (r := fun t => (⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩ : Release)) (pre := pre)
          (fun t => by
            show LogsAt (relK k fs (ts ++ [⟨b, spec, t, fun _ => 0⟩]) fun log =>
              f (⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩ :: log)) pre _
            simp only [hf]
            exact logsAt_relK θ F c k _ _ pre _) h z.1
        exact hns z.1 ⟨h1, h2⟩
    obtain ⟨L, hA, hO⟩ := walkFill_coin IH (fun _ => fs) (fun t => ts ++ [⟨b, spec, t, fun _ => 0⟩]) pre f hf
      (fun t => ⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩) s'' Ks hK2 m κ z back z.1 m κ
      (by rw [fill_coin_of _ hH, hnone]; cases m <;> rfl)
    refine ⟨_, ?_, hO⟩
    rw [freshCoins_cons_of _ (hns z.1)]
    exact hA
  · exact ⟨[], .inl List.nil_prefix, walkFill_opening hf hn s'' Ks _ back⟩

/-- **`WalkFill` at every number of frames**, at readings that pick only openings. -/
theorem walkFill (hrd : PicksOpenings rd) : ∀ k, WalkFill (g := g) (σ := σ) θ F c label after rd k
  | 0 => fun _ _ _ _ hf _ hg s' Ks _ _ _ _ back => ⟨[], .inl List.nil_prefix, walkFill_nil hf hg s' Ks _ back⟩
  | k + 1 => fun fs ts pre f hf g' hg s' Ks hK m κ z back => by
    rw [relK_succ] at hg
    subst hg
    have hK' : ∀ j, Ks j = .skip ∨
        Ks j = DigestCut.send fun msg => (frameNext (rd.round j) (rd.reveal j) k fs ts f pre msg).cut :=
      fun j => (hK j).imp_right fun h => eq_of_heq (h.trans (frameCut_succ _ _ k fs ts f pre))
    have hH : ∀ j, (Ks j).isHere = false := fun j => by
      rcases hK' j with h | h <;> rw [h] <;> rfl
    rw [walk_send_of hH, fill_send_of _ hH]
    exact walkFill_next hrd (walkFill hrd k) fs ts pre f hf s'.1 _ rfl s'.2 _ (fun j => by
      rcases hK' j with h | h
      · exact .inl (by rw [h]; rfl)
      · exact .inr (by rw [h]; rfl)) m κ z back

end WalkFill

theorem freshCoins_nil_of {label : String} {after : ByteArray} :
    ∀ {L : List Release}, (∀ q ∈ L, ¬ (q.frame = .fresh label ∧ q.answers = after)) →
      freshCoins label after L = []
  | [], _ => rfl
  | q :: L, h => by
    rw [freshCoins_cons_of _ (h q List.mem_cons_self),
      freshCoins_nil_of fun q' hq' => h q' (List.mem_cons_of_mem _ hq')]

/-- **Off the sample-first order, the play's coins of the sample's family are those before the stop**: where the log is
`L`, then nothing or an opening (`StopTail`), and every release of the family `label` answering `after` precedes every
opening (`SampleLinkFirst`), the log's coins of that family are `L`'s. -/
theorem freshCoins_of_stopTail {label : String} {after : ByteArray} {L log' : List Release}
    (hP : SampleLinkFirst label after (L ++ log')) (ht : StopTail log') :
    freshCoins label after (L ++ log') = freshCoins label after L := by
  rcases ht with rfl | ⟨r, rest, i, s, n, rfl, hr⟩
  · rw [List.append_nil]
  · have hq := (hP L r rest rfl i s n hr).1
    have happ : ∀ A B : List Release, freshCoins label after (A ++ B) =
        freshCoins label after A ++ freshCoins label after B := fun _ _ => List.filterMap_append
    rw [happ, freshCoins_cons_of _ (fun ⟨h, _⟩ => by rw [hr] at h; cases h), freshCoins_nil_of hq, List.append_nil]

/-! ## The record's draw -/

/-- **The units are ascending below `n`**: `UnitDraw.ofJson`'s check of a draw's units. -/
def unitsAscending (n : ℕ) (units : Array ℕ) : Bool :=
  (List.range units.size).all fun i => units[i]! < n && (i == 0 || units[i - 1]! < units[i]!)

/-- **The record's draw is the one v2 drew**: the units are ascending below `n`, and the record's `unit_draw`
(`Flock.Witness.recordDraw`) prints as the draw of `units` under the law `L` over `n` units does. -/
def recordDrawn (L : Flock.Draw.Law) (n : ℕ) (units : Array ℕ) (record : String) : Bool :=
  unitsAscending n units && match Flock.Witness.recordDraw record with
    | some r => match Flock.canonUtf8 r, Flock.canonUtf8 (Flock.Draw.UnitDraw.toJson ⟨L, n, units, #[]⟩) with
      | .ok a, .ok b => a == b
      | _, _ => false
    | none => false

/-- **v2 holds the record's draw to its own** in what composite `k` accepts: wherever v2 drew `units` under the law `L`
over `n` units in what the prover sent (`Drew`), the record v2 reads there is drawn so (`recordDrawn`). -/
def DrawHeld (k : Verity.Registration.Composite) : Prop :=
  ∀ s dr L n ξ units record, Drew k s dr L n ξ units → sentRecord s = some record →
    recordDrawn L n units record = true

theorem exec_recordDraw (record : String) :
    FlockSoundness.Discharge.Exec.recordDraw record = Flock.Witness.recordDraw record :=
  rfl

/-- **The units of an ascending draw** are below `n` and strictly ascending. -/
theorem unitsAscending_spec {n : ℕ} {units : Array ℕ} (h : unitsAscending n units = true) :
    (∀ x ∈ units.toList, x < n) ∧ units.toList.Pairwise (· < ·) := by
  have hi : ∀ i (hi : i < units.size), units[i] < n ∧ (i = 0 ∨ units[i - 1]! < units[i]) := fun i hi => by
    have := List.all_eq_true.1 h i (List.mem_range.2 hi)
    simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true, beq_iff_eq, getElem!_pos units i hi] at this
    exact this
  have step : ∀ i (h₁ : i + 1 < units.size), units[i] < units[i + 1] := fun i h₁ => by
    rcases (hi (i + 1) h₁).2 with h0 | h0
    · omega
    · simpa [getElem!_pos units i (by omega)] using h0
  have far : ∀ d i (h₁ : i + d + 1 < units.size), units[i] < units[i + d + 1] := by
    intro d
    induction d with
    | zero => exact fun i h₁ => step i h₁
    | succ d ih => exact fun i h₁ => (ih i (by omega)).trans (step (i + d + 1) h₁)
  refine ⟨fun x hx => ?_, ?_⟩
  · obtain ⟨i, hi', rfl⟩ := List.getElem_of_mem hx
    exact (hi i (by simpa using hi')).1
  · rw [List.pairwise_iff_getElem]
    intro i j h₁ h₂ hij
    obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_lt hij
    simpa using far d i (by simpa using h₂)

/-- **The verifier's own draw of the proof blocks an ascending draw names** is that draw. -/
theorem drawOf_filter {n : ℕ} {units : Array ℕ} (h : unitsAscending n units = true) (L : Flock.Draw.Law) :
    FlockSoundness.Discharge.ZkLink.drawOf L n (Finset.univ.filter fun u : Fin n => u.val ∈ units) =
      ⟨L, n, units, #[]⟩ := by
  obtain ⟨hlt, hs⟩ := unitsAscending_spec h
  have himg : (Finset.univ.filter fun u : Fin n => u.val ∈ units).image Fin.val = units.toList.toFinset := by
    ext x
    simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and, List.mem_toFinset,
      Array.mem_toList_iff]
    constructor
    · rintro ⟨u, hu, rfl⟩
      exact hu
    · intro hx
      exact ⟨⟨x, hlt x (Array.mem_toList_iff.2 hx)⟩, hx, rfl⟩
  have hsort : units.toList.toFinset.sort (· ≤ ·) = units.toList :=
    (List.toFinset_sort (· ≤ ·) (hs.imp ne_of_lt)).2 (hs.imp le_of_lt)
  simp only [FlockSoundness.Discharge.ZkLink.drawOf, himg, hsort, Array.toArray_toList]

open Properties.Challenger.CanonRead (sortedKeys_arr sortedKeys_obj increasing_iff) in
/-- **A well-formed value's objects hold their keys in order.** -/
theorem sortedKeys_of_wf : ∀ j : Json, Claims.Json.Wf j → _root_.SampledProofs.sortedKeys j = true := by
  refine FlockDeploy.json_ind (fun _ => by simp [_root_.SampledProofs.sortedKeys])
    (fun _ _ => by simp [_root_.SampledProofs.sortedKeys]) (fun _ _ => by simp [_root_.SampledProofs.sortedKeys])
    (fun _ _ => by simp [_root_.SampledProofs.sortedKeys]) ?_ ?_
  · intro a ih hw
    cases hw with
    | arr h => exact sortedKeys_arr.2 fun x hx => ih x hx (h x hx)
  · intro m ih hw
    cases hw with
    | obj hm h =>
      refine sortedKeys_obj.2 ⟨?_, fun p hp => ?_⟩
      · rw [Std.TreeMap.Raw.toList_toArray, increasing_iff, List.pairwise_map]
        exact (Std.TreeMap.Raw.ordered_keys_toList hm).imp fun hlt => compareOfLessAndEq_eq_lt.1 hlt
      · have hp' : p ∈ m.toList := by
          rw [← Std.TreeMap.Raw.toList_toArray]
          exact Array.mem_toList_iff.2 hp
        refine ih p hp' (h p.1 p.2 ?_)
        rw [Std.TreeMap.Raw.get?_eq_getElem?]
        exact (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp'

/-- **Two values `norm` fixes that `canonUtf8` prints alike are equal.** -/
theorem eq_of_canonUtf8 {a b : Json} {s : String} (ha : Flock.norm a = a) (hb : Flock.norm b = b)
    (hsa : Flock.canonUtf8 a = .ok s) (hsb : Flock.canonUtf8 b = .ok s) : a = b := by
  have wa : Claims.Json.Wf a := by
    rw [← ha]
    exact FlockDeploy.wf_norm a
  have wb : Claims.Json.Wf b := by
    rw [← hb]
    exact FlockDeploy.wf_norm b
  exact FlockDeploy.eq_of_jsonEquiv_norm ha hb
    (Properties.Challenger.CanonRead.canon_inj (sortedKeys_of_wf a wa) (sortedKeys_of_wf b wb) hsa hsb)

/-- **The record's draw is the session's draw file at v2's units**: where v2 holds the record's draw to its own
(`DrawHeld`), v2 drew `units` under the law `L` over `n` units in what the prover sent, and the session's law and
population are `L` and `n`, the record v2 reads decodes to the session's draw file at the proof blocks `units` names,
at a session whose draw files `norm` fixes. -/
theorem recordDraw_of_drew {k : Verity.Registration.Composite} (hH : DrawHeld k) {s : Verity.Sent} {dr : Draws}
    {L : Flock.Draw.Law} {n : ℕ} {ξ : ByteArray} {units : Array ℕ} (hd : Drew k s dr L n ξ units) (A : Live)
    (hL : A.law = L) (hn : A.n = n) (hfix : ∀ S, Flock.norm (A.dj S) = A.dj S) {record : String}
    (hr : sentRecord s = some record) :
    FlockSoundness.Discharge.Exec.recordDraw record =
      some (A.dj (Finset.univ.filter fun u : Fin A.n => u.val ∈ units)) := by
  have h := hH s dr L n ξ units record hd hr
  subst hL hn
  unfold recordDrawn at h
  rw [Bool.and_eq_true] at h
  obtain ⟨hu, h⟩ := h
  have ht : Flock.Draw.UnitDraw.toJson ⟨A.law, A.n, units, #[]⟩ =
      A.dj (Finset.univ.filter fun u : Fin A.n => u.val ∈ units) := by
    show _ = (FlockSoundness.Discharge.ZkLink.drawOf A.law A.n _).toJson
    rw [drawOf_filter hu]
  rw [ht] at h
  rw [exec_recordDraw]
  cases hw : Flock.Witness.recordDraw record with
  | none => simp [hw] at h
  | some r =>
    have hnr : Flock.norm r = r := Claims.Bridge.Custody.recordDraw_norm (record := record) (by
      rw [exec_recordDraw]
      exact hw)
    cases ha : Flock.canonUtf8 r with
    | error _ => simp [hw, ha] at h
    | ok a =>
      cases hb : Flock.canonUtf8 (A.dj (Finset.univ.filter fun u : Fin A.n => u.val ∈ units)) with
      | error _ => simp [hw, ha, hb] at h
      | ok b =>
        simp only [hw, ha, hb, beq_iff_eq] at h
        subst h
        exact congrArg some (eq_of_canonUtf8 hnr (hfix _) ha hb)

/-! ## The draw files `norm` fixes -/

theorem norm_arr_of {a : Array Json} (h : ∀ x ∈ a, Flock.norm x = x) : Flock.norm (.arr a) = .arr a := by
  rw [FlockDeploy.norm_arr]
  congr 1
  exact (Array.map_congr_left h).trans (Array.map_id' a)

theorem norm_nats (a : Array ℕ) :
    Flock.norm (.arr (a.map fun x => Lean.toJson x)) = .arr (a.map fun x => Lean.toJson x) :=
  norm_arr_of fun x hx => by
    obtain ⟨y, -, rfl⟩ := Array.mem_map.1 hx
    rfl

theorem norm_stratum (s : Flock.Draw.Stratum) : Flock.norm s.toJson = s.toJson := by
  have h : ∀ a b c : Json, Flock.norm (Json.mkObj [("template", a), ("units", b), ("k", c)]) =
      Json.mkObj [("template", Flock.norm a), ("units", Flock.norm b), ("k", Flock.norm c)] :=
    fun _ _ _ => rfl
  show Flock.norm (Json.mkObj [("template", Json.str s.template),
    ("units", Json.arr (s.units.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])),
    ("k", Lean.toJson s.k)]) = _
  rw [h]
  congr 4
  refine norm_arr_of fun x hx => ?_
  obtain ⟨⟨lo, hi⟩, -, rfl⟩ := Array.mem_map.1 hx
  exact norm_arr_of fun y hy => by
    simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hy
    rcases hy with rfl | rfl <;> rfl

/-- **`norm` fixes the printed draw file at every law but the work law.** -/
theorem norm_printedDraw {n : ℕ} {law : Flock.Draw.Law} (hw : ∀ K ws, law ≠ .work K ws) (S : Finset (Fin n)) :
    Flock.norm (printedDraw law S) = printedDraw law S := by
  have hu := norm_nats ((((S.image Fin.val).sort (· ≤ ·)).toArray))
  cases law with
  | subset k =>
    have h : ∀ a b c d : Json, Flock.norm (Json.mkObj [("law", a), ("population", b), ("k", c), ("units", d)]) =
        Json.mkObj [("law", Flock.norm a), ("population", Flock.norm b), ("k", Flock.norm c),
          ("units", Flock.norm d)] :=
      fun _ _ _ _ => rfl
    show Flock.norm (Json.mkObj [("law", "subset"), ("population", Lean.toJson n), ("k", Lean.toJson k),
      ("units", .arr (((S.image Fin.val).sort (· ≤ ·)).toArray.map fun x => Lean.toJson x))]) = _
    rw [h, hu]
    rfl
  | bernoulli a b =>
    have h : ∀ w x y z : Json, Flock.norm (Json.mkObj [("law", w), ("population", x), ("p", y), ("units", z)]) =
        Json.mkObj [("law", Flock.norm w), ("population", Flock.norm x), ("p", Flock.norm y),
          ("units", Flock.norm z)] :=
      fun _ _ _ _ => rfl
    show Flock.norm (Json.mkObj [("law", "bernoulli"), ("population", Lean.toJson n), ("p", Json.str s!"{a}/{b}"),
      ("units", .arr (((S.image Fin.val).sort (· ≤ ·)).toArray.map fun x => Lean.toJson x))]) = _
    rw [h, hu]
    rfl
  | stratified K St =>
    have h : ∀ a b c d e : Json,
        Flock.norm (Json.mkObj [("law", a), ("population", b), ("k", c), ("strata", d), ("units", e)]) =
          Json.mkObj [("law", Flock.norm a), ("population", Flock.norm b), ("k", Flock.norm c),
            ("strata", Flock.norm d), ("units", Flock.norm e)] :=
      fun _ _ _ _ _ => rfl
    show Flock.norm (Json.mkObj [("law", "stratified"), ("population", Lean.toJson n), ("k", Lean.toJson K),
      ("strata", Json.arr (St.map (·.toJson))),
      ("units", .arr (((S.image Fin.val).sort (· ≤ ·)).toArray.map fun x => Lean.toJson x))]) = _
    rw [h, hu, norm_arr_of fun x hx => by
      obtain ⟨s, -, rfl⟩ := Array.mem_map.1 hx
      exact norm_stratum s]
    rfl
  | work K ws => exact absurd rfl (hw K ws)

/-- **`norm` fixes the session's draw files** at a session whose law isn't the work law. -/
theorem Live.norm_dj (A : Live) (hw : ∀ K ws, A.law ≠ .work K ws) : ∀ S, Flock.norm (A.dj S) = A.dj S :=
  norm_printedDraw hw

/-! ## v2's coins fit `MAX_COINS` frames -/

/-- **The coins v2 reads at an accepted case fit `MAX_COINS` frames**: v2's evidence holds its segment's openings
within `LIMITS`, and v2 reads one frame of `COIN_BYTES` per opening. -/
theorem coins_size_le {bs : Core.Blobs} {a b e : ByteArray} {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {ev : _root_.SampledProofs.OneStageZk.Evidence} {ξ : ByteArray}
    {dr : Draws} (h : V2Accepts bs a b e p c ev ξ dr) :
    ξ.size ≤ _root_.SampledProofs.Challenger.COIN_BYTES * _root_.SampledProofs.Challenger.MAX_COINS := by
  obtain ⟨-, -, hev, hξ, hdr, -⟩ := h
  obtain ⟨b', hb'⟩ := Binding.dispatch_evidence hev
  obtain ⟨-, -, -, -, hops, -⟩ :=
    Properties.Challenger.Limits.withinAt_spec (Properties.Challenger.Limits.evidenceOf_within hb')
  obtain ⟨raw, hraw, rfl⟩ := Properties.Challenger.oneStageZk_reads p c dr hdr ev ξ hξ
  obtain ⟨cm, roots, -, hstream⟩ := drawCoins_ok hraw
  obtain ⟨l, start, openings, os, -, -, hop, -, -, hr⟩ := Properties.Challenger.Coins.stream_ok hstream
  rw [challengerStream_ok hstream, ← zipIdx_filter_length, hr, List.length_range']
  rw [hop] at hops
  exact Nat.mul_le_mul (le_refl _) hops

/-! ## The record's draw from the sample's coins -/

theorem length_bitsOf (b : ByteArray) : (bitsOf b).length = 8 * b.size := by
  have key : ∀ l : List UInt8, (l.flatMap byteBits).length = 8 * l.length := by
    intro l
    induction l with
    | nil => rfl
    | cons x l ih =>
      have h8 : (byteBits x).length = 8 := by simp [byteBits]
      rw [List.flatMap_cons, List.length_append, ih, h8, List.length_cons]
      omega
  rw [bitsOf_eq, key, Array.length_toList]
  rfl

theorem length_fillCoins : ∀ {m : ℕ} (κ : Fin m → Fin 512 → Bool), (fillCoins κ).flatten.length = m * 512
  | 0, _ => by simp
  | m + 1, κ => by
    rw [fillCoins_succ, List.flatten_cons, List.length_append, List.length_ofFn, length_fillCoins (Fin.tail κ),
      Nat.add_mul, Nat.one_mul]
    omega

theorem prefix_flatten {a b : List (List Bool)} (h : a <+: b) : a.flatten <+: b.flatten := by
  obtain ⟨t, rfl⟩ := h
  exact ⟨t.flatten, by rw [List.flatten_append]⟩

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The record's draw is the session's draw file at the sample**, wherever the play's coins of the sample's family
agree with the sample's coins (`CoinsAgree`): in a play at the parameters `θ` of a session v2 sets up and tied to v2's
verifier, whose sampled verdict accepts, whose log v2 read is the device's, and where v2 reads the coins of the family
`label` answering `after`, the record decodes to the session's draw file at the sample of the device's coins `ω`, whose
bytes are the coins `κ`. Where the play holds more of the family's coins than `κ`, v2's still fit `κ`'s
(`coins_size_le`, `hm`). -/
theorem recordDraw_of_coins {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live} {θ : ByteArray}
    {F : Array ByteArray} (hA : SetsUp V θ F A) (ht : PopTied θ A) (hR : DrawReadAll k coinsRead) (hH : DrawHeld k)
    (hfix : ∀ S, Flock.norm (A.dj S) = A.dj S) {m : ℕ} (hm : _root_.SampledProofs.Challenger.MAX_COINS ≤ m)
    {κ : Fin m → Fin 512 → Bool} {ω : A.Ω}
    (hsrc : _root_.Properties.SampledProofs.bytesOf (A.src ω 0) =
      _root_.Properties.SampledProofs.bytesOf (coinBytes m 64 κ 0))
    {o : Verity.Exchange} (hθ : o.params = θ) (hv : V.Verdict k .sampled o) (hl : Binding.Logged Binding.readsV2 o)
    {label : String} {after : ByteArray} (hN : SampleNamed k o.sent label after)
    (hc : CoinsAgree (freshCoins label after o.log) (fillCoins κ)) {record : String}
    (hr : sentRecord o.sent = some record) :
    FlockSoundness.Discharge.Exec.recordDraw record = some (A.dj (A.L.draw ω)) := by
  obtain ⟨p₀, hp₀, -⟩ := hA.params
  obtain ⟨dr, L, n, ξ, units, hd, hs⟩ := logged_draw_v2 hR (by rw [hθ]; exact hp₀) hv hl
  have hn := n_of_popTied ht hv.2.1 hθ hd
  have hL := law_of_setsUp hA hv.2.1 hθ hd hn
  obtain ⟨a, b, e, p, c, ev, hcase, hacc, -⟩ := id hd
  obtain ⟨hlab, haft⟩ := hN a b e p c ev ξ dr hcase hacc
  have hξ := coins_size_le hacc
  unfold ShowsDraw at hs
  rw [hlab, haft] at hs
  have e₀ : bitsOf (_root_.Properties.SampledProofs.bytesOf (coinBytes m 64 κ 0)) = (fillCoins κ).flatten :=
    bitsOf_streamOf_coinBytes (B := 64) κ
  have hpre : bitsOf ξ <+: bitsOf (_root_.Properties.SampledProofs.bytesOf (A.src ω 0)) := by
    rw [hsrc, e₀]
    rcases hc with hc | hc
    · exact hs.trans (prefix_flatten hc)
    · refine List.prefix_of_prefix_length_le hs (prefix_flatten hc) ?_
      have h64 : _root_.SampledProofs.Challenger.COIN_BYTES = 64 := rfl
      rw [h64] at hξ
      rw [length_bitsOf, length_fillCoins]
      omega
  rw [draw_of_drew hd A ω hL hn hpre]
  exact recordDraw_of_drew hH hd A hL hn hfix hr

/-! ## The record's draw on the joint law's support -/

/-- **Round `j`'s cut is its frame's cut on the releases**, the period's game read as the releases then the evidence
(`game_eq_relK`). -/
theorem roundCuts_heq (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) {R : ℕ} (rd : Reading R)
    (j : Fin R) :
    HEq (roundCuts V θ F c rd j)
      (frameCut (rd.round j) (rd.reveal j) (frames (V.families θ)) (V.families θ) [] (evidenceG θ F c) []).cut := by
  have key : ∀ {g g' : Game Exchange} (h : g = g') (K : DigestCut.AgreeCut sha512Msg g),
      HEq (K.cast h).cut K.cut := by
    intro g g' h K
    subst h
    rfl
  exact key (game_eq_relK V θ F c).symm _

/-- **The record's draw is the session's draw file at the outcome's sample** (`MovesOf`'s clause (i)): on the joint
law's support off the held-off events at the sample-coupled coin map, wherever the play's sampled verdict accepts, the
log v2 read is the device's, v2 reads the sample's coins of the family `label` answering `after`, and every release of
that family precedes every opening (`SampleLinkFirst`), the record decodes to the session's draw file at the outcome's
sample. The play's output is reached from the first stop, where the walk from the sample's fill stops at a log of the
fill's releases, then nothing or an opening (`walkFill`). The order leaves the play's coins of the family those
releases' (`freshCoins_of_stopTail`), which agree with the sample's coins, and `recordDraw_of_coins` reads the rest. -/
theorem recordDraw_reachesOff (e : 𝔽 ≃ (Fin 128 → Bool)) {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {k : Verity.Registration.Composite} {A : Live} {rd : Reading A.rounds}
    {σ : Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg} {BL : LinkHold A.mP} {B : RoundHold}
    {label : String} {after : ByteArray} {m : ℕ} {κ : A.Ω → Fin m → Fin 512 → Bool}
    (hA : SetsUp V θ F A) (ht : PopTied θ A) (hR : DrawReadAll k coinsRead) (hH : DrawHeld k)
    (hrd : PicksOpenings rd) (hm : _root_.SampledProofs.Challenger.MAX_COINS ≤ m)
    (hsrc : ∀ ω, _root_.Properties.SampledProofs.bytesOf (A.src ω 0) =
      _root_.Properties.SampledProofs.bytesOf (coinBytes m 64 (κ ω) 0))
    (hfix : ∀ S, Flock.norm (A.dj S) = A.dj S) {o : A.Out × CoinTape (V.game θ F c) σ}
    (ho : ReachesOff V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ) BL B o)
    (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2))
    (hl : Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ o.2))
    (hN : SampleNamed k (CoinTape.run (V.game θ F c) σ o.2).sent label after)
    (hP : SampleLinkFirst label after (CoinTape.run (V.game θ F c) σ o.2).log) :
    ∀ record, sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some record →
      FlockSoundness.Discharge.Exec.recordDraw record = some (A.dj o.1.2.1) := by
  intro record hrec
  obtain ⟨ω, pts, -, hr⟩ := ho
  obtain ⟨_, h1, -⟩ := hr.out
  have hS : o.1.2.1 = A.L.draw ω.1 :=
    (congrArg (fun x : A.Out => x.2.1) h1).trans (sample_extend A _ ω)
  have hinv :
      (firstStop V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) ω).Inv (roundCuts V θ F c rd) :=
    (Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c rd) _)).2.2
  obtain ⟨q, hq, -, hoq⟩ := hr.through (Pos.inv_consume (Pos.inv_rewrite hinv _))
  obtain ⟨h, hh⟩ := ((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans hq
  have hreach : PlayReaches (firstStop V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) ω).g'
      (firstStop V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) ω).s'
      (CoinTape.run (V.game θ F c) σ o.2) := by
    rw [hoq]
    show PlayReaches _ _ (CoinTape.run (V.game θ F c) σ (q.back q.x))
    rw [hh, hinv.run]
    exact run_reaches _ _ _
  rw [firstStop_sampleMap e V θ F c A rd σ label after κ ω] at hreach
  obtain ⟨L, hc, hO⟩ := walkFill (label := label) (after := after) hrd (frames (V.families θ)) (V.families θ) [] []
    (evidenceG θ F c) (fun _ => rfl) (V.game θ F c) (game_eq_relK V θ F c) σ (roundCuts V θ F c rd)
    (fun j => .inr (roundCuts_heq V θ F c rd j)) m (κ ω.1) ω.2 id
  obtain ⟨log', hlog, htail⟩ := hO _ hreach
  rw [List.nil_append] at hlog
  have hc' : CoinsAgree (freshCoins label after (CoinTape.run (V.game θ F c) σ o.2).log) (fillCoins (κ ω.1)) := by
    rw [hlog] at hP ⊢
    rw [freshCoins_of_stopTail hP htail]
    exact hc
  rw [hS]
  exact recordDraw_of_coins hA ht hR hH hfix hm (hsrc ω.1) (play_ends V θ F c σ o.2).1 hv hl hN hc' hrec

end Verity.Sampled
