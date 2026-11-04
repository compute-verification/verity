import Proofs.Properties.Verity.Sampled.Messages
import Proofs.Core.Game.InterleaveBridge

/-!
# Sampled proofs' coupling: where the walk stops

The projection (`Project`) walks a play of the period's game to the next node where one of the rounds' cuts is fixed
(`Game.Pos.walk`). On the period's game as the device's releases and then the evidence (`relK`), at the cuts fixed at
the first frame each round's reading picks (`frameCut`), the walk stops only at the play's end or right after an opening
frame a live round picks, when each reading picks only openings (`walk_relK`):

* **The walk, a step at a time** (`Pos.walk_here`, `Pos.walk_send`, `Pos.walk_coin`, `Pos.walk_skip`): at a node where a
  cut is fixed the walk stays; elsewhere it follows the prover's message or the tape's coin; where every cut is past,
  it runs to the play's end.
* **The live rounds' cuts** (`DigestCut.LiveCuts`): each live round's cut, every other round's past. After a frame the
  prover sends, a live round's cut is the one after the frame (`freshCut`, `helloCut`, `openCut`); after an opening
  frame, it is fixed there where the round's reading picks the frame, and follows the opening's coin otherwise
  (`optOpenCut`).
* **Where the walk stops** (`StopOf`, `walk_relK`, `walk_cast`): after releases no live round picks (`NoPick`), the
  play's end, every cut past there; or the node after an opening frame a live round picks after the device's log before
  the walk and those releases, with the live rounds' cuts there. Consuming that stop leaves live the rounds that didn't
  pick it (`liveCuts_consume`), and the walk from there stops the same way (`stopOf_next`).
-/

namespace Verity.Game

variable {α V ι : Type}

namespace DigestCut

/-- **`Ks` are the cuts `K` at the rounds `a` holds of**, every other round's cut past. -/
def LiveCuts {g : Game α} (a : ι → Prop) (K Ks : ι → DigestCut V g) : Prop :=
  ∀ i, a i ∧ Ks i = K i ∨ ¬ a i ∧ Ks i = .skip

namespace LiveCuts

variable {a : ι → Prop}

theorem eq_of {g : Game α} {K Ks : ι → DigestCut V g} (h : LiveCuts a K Ks) {i : ι} (ha : a i) : Ks i = K i :=
  ((h i).resolve_right fun h' => h'.1 ha).2

theorem skip_of {g : Game α} {K Ks : ι → DigestCut V g} (h : LiveCuts a K Ks) {i : ι} (ha : ¬ a i) :
    Ks i = .skip :=
  ((h i).resolve_left fun h' => ha h'.1).2

theorem isHere {g : Game α} {K Ks : ι → DigestCut V g} (h : LiveCuts a K Ks) (hK : ∀ i, a i → (K i).isHere = false)
    (i : ι) : (Ks i).isHere = false := by
  by_cases ha : a i
  · rw [h.eq_of ha]
    exact hK i ha
  · rw [h.skip_of ha]
    rfl

theorem eq_skip {g : Game α} {K Ks : ι → DigestCut V g} (h : LiveCuts a K Ks) (hK : ∀ i, a i → K i = .skip)
    (i : ι) : Ks i = .skip := by
  by_cases ha : a i
  · rw [h.eq_of ha]
    exact hK i ha
  · exact h.skip_of ha

theorem congr {g : Game α} {K K' Ks : ι → DigestCut V g} (h : LiveCuts a K Ks) (hK : ∀ i, a i → K i = K' i) :
    LiveCuts a K' Ks := fun i =>
  (h i).imp (fun ⟨ha, e⟩ => ⟨ha, e.trans (hK i ha)⟩) id

theorem map {g g' : Game α} {K Ks : ι → DigestCut V g} (h : LiveCuts a K Ks) (φ : DigestCut V g → DigestCut V g')
    (hφ : φ .skip = .skip) : LiveCuts a (fun i => φ (K i)) (fun i => φ (Ks i)) := fun i =>
  (h i).imp (fun ⟨ha, e⟩ => ⟨ha, congrArg φ e⟩) fun ⟨ha, e⟩ => ⟨ha, (congrArg φ e).trans hφ⟩

theorem atSend {M : Type} {k : M → Game α} {K Ks : ι → DigestCut V (.send M k)} (h : LiveCuts a K Ks) (m : M) :
    LiveCuts a (fun i => (K i).atSend m) (fun i => (Ks i).atSend m) :=
  h.map (fun K => K.atSend m) rfl

theorem atCoin {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    {K Ks : ι → DigestCut V (@Game.coin _ C iF iN k)} (h : LiveCuts a K Ks) (c : C) :
    LiveCuts a (fun i => (K i).atCoin c) (fun i => (Ks i).atCoin c) :=
  h.map (fun K => K.atCoin c) rfl

end LiveCuts

end DigestCut

namespace Pos

variable {g : Game α} {σ : Strategy g}

/-- **At a node where a cut is fixed, the walk stays.** -/
theorem walk_here : ∀ (g' : Game α) (s' : Strategy g') (Ks : ι → DigestCut V g') (x : CoinTape g' s')
    (back : CoinTape g' s' → CoinTape g σ), (∃ i, (Ks i).isHere) → walk g' s' Ks x back = ⟨g', s', Ks, x, back⟩
  | .ret _, _, _, _, _, _ => rfl
  | .send _ _, _, _, _, _, h => by rw [walk, ite_eq_left_of_eq_true _ _ (eq_true h)]
  | @Game.coin _ _ _ _ _, _, _, _, _, h => by rw [walk, ite_eq_left_of_eq_true _ _ (eq_true h)]

/-- **Where no cut is fixed at the prover's message, the walk follows it.** -/
theorem walk_send {M : Type} {k : M → Game α} (s' : Strategy (.send M k)) (Ks : ι → DigestCut V (.send M k))
    (x : CoinTape (.send M k) s') (back : CoinTape (.send M k) s' → CoinTape g σ) (h : ∀ i, (Ks i).isHere = false) :
    walk (.send M k) s' Ks x back = walk (k s'.1) s'.2 (fun i => (Ks i).atSend s'.1) x back := by
  have hf : ¬ ∃ i, (Ks i).isHere := fun ⟨i, hi⟩ => by
    rw [h i] at hi
    cases hi
  rw [walk, ite_eq_right_of_eq_false _ _ (eq_false hf)]

open Classical in
/-- **Where no cut is fixed at a coin, the walk follows the tape's coin.** -/
theorem walk_coin {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (s' : Strategy (@Game.coin _ C iF iN k)) (Ks : ι → DigestCut V (@Game.coin _ C iF iN k))
    (x : CoinTape (@Game.coin _ C iF iN k) s') (back : CoinTape (@Game.coin _ C iF iN k) s' → CoinTape g σ)
    (h : ∀ i, (Ks i).isHere = false) :
    walk (@Game.coin _ C iF iN k) s' Ks x back = walk (k x.1) (s' x.1) (fun i => (Ks i).atCoin x.1) (x.2 x.1)
      fun y => back (x.1, Function.update x.2 x.1 y) := by
  have hf : ¬ ∃ i, (Ks i).isHere := fun ⟨i, hi⟩ => by
    rw [h i] at hi
    cases hi
  rw [walk, ite_eq_right_of_eq_false _ _ (eq_false hf)]

/-- **Where every cut is past, the walk runs to the play's end**, every cut past there. -/
theorem walk_skip : ∀ (g' : Game α) (s' : Strategy g') (Ks : ι → DigestCut V g') (x : CoinTape g' s')
    (back : CoinTape g' s' → CoinTape g σ), (∀ i, Ks i = .skip) →
      (∀ i, (walk g' s' Ks x back).Ks i = .skip) ∧ (walk g' s' Ks x back).g' = .ret (CoinTape.run g' s' x)
  | .ret _, _, _, _, _, h => ⟨h, rfl⟩
  | .send _ k, s', Ks, x, back, h => by
    rw [walk_send s' Ks x back (fun i => show (Ks i).isHere = false by rw [h i]; rfl)]
    exact walk_skip (k s'.1) s'.2 _ x back fun i => show (Ks i).atSend s'.1 = .skip by rw [h i]; rfl
  | @Game.coin _ _ _ _ k, s', Ks, x, back, h => by
    rw [walk_coin s' Ks x back (fun i => show (Ks i).isHere = false by rw [h i]; rfl)]
    exact walk_skip (k x.1) (s' x.1) _ (x.2 x.1) _ fun i => show (Ks i).atCoin x.1 = .skip by rw [h i]; rfl

end Pos

end Verity.Game

namespace Verity

open Game

variable {β ι : Type}

/-! ## The releases after a frame, and each round's cut there -/

/-- **The releases after a fresh frame of the family `l`**, at the families the device's coin leaves (`o`, `takeCoin`'s;
`none` where it refuses the coin). -/
def freshK (k : ℕ) (ts : List OpenTree) (f : List Release → Game β) (l : String) (b : ByteArray)
    (o : Option (List (String × ℕ))) : Game β :=
  optGame o (f []) fun fs' => .coin (Fin 512 → Bool) fun x =>
    relK k fs' ts fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)

/-- **The releases after a `Hello` of the shape `spec`** under the nonce `ν`. -/
def helloK (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β)
    (spec : Flock.CoinTree.Spec) (ν : ByteArray) : Game β :=
  .coin (TreeSample spec) fun t => relK k fs (ts ++ [⟨ν, spec, t, fun _ => 0⟩]) fun log =>
    f (⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩ :: log)

/-- **The releases after an opening of the tree `T`'s stream `s`**, at the round it opens (`q`, `OpenTree.next`'s;
`none` where the device refuses the opening). -/
def treeOpenK (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β) (i s n : ℕ)
    (b : ByteArray) (T : OpenTree) (q : Option (Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog))) : Game β :=
  optGame q (f []) fun p =>
    haveI : Nonempty (TreeOpenable T.nonce T.spec T.sample p.1 p.2) := ⟨⟨T.sample.2 p.1 p.2, rfl⟩⟩
    .coin (TreeOpenable T.nonce T.spec T.sample p.1 p.2) fun d =>
      relK k fs (ts.set i (T.bump p.1)) fun log => f (⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log)

/-- **The releases after an opening of the `i`-th tree's stream `s`**, at the tree (`o`, `ts[i]?`). -/
def optOpenK (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β) (i s n : ℕ)
    (b : ByteArray) (o : Option OpenTree) : Game β :=
  optGame o (f []) fun T => treeOpenK k fs ts f i s n b T (T.next s n)

section Cuts

variable (P : List Release → Frame → Prop) (rv : β → Option Claims.Msg)

/-- A round's cut after a fresh frame it doesn't pick. -/
noncomputable def freshCut (k : ℕ) (ts : List OpenTree) (f : List Release → Game β) (pre : List Release) (l : String)
    (b : ByteArray) (o : Option (List (String × ℕ))) : DigestCut.AgreeCut sha512Msg (freshK k ts f l b o) :=
  optCut o .skip fun fs' => .coin fun x =>
    frameCut P rv k fs' ts (fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)) (pre ++ [⟨.fresh l, b, List.ofFn x⟩])

/-- A round's cut after a `Hello` it doesn't pick. -/
noncomputable def helloCut (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β)
    (pre : List Release) (spec : Flock.CoinTree.Spec) (ν : ByteArray) :
    DigestCut.AgreeCut sha512Msg (helloK k fs ts f spec ν) :=
  .coin fun t => frameCut P rv k fs (ts ++ [⟨ν, spec, t, fun _ => 0⟩])
    (fun log => f (⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩ :: log))
    (pre ++ [⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec t ++ treeKey t.1)⟩])

/-- A round's cut after an opening of the tree `T` it doesn't pick, at the round the opening opens. -/
noncomputable def treeOpenCut (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β)
    (pre : List Release) (i s n : ℕ) (b : ByteArray) (T : OpenTree)
    (q : Option (Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog))) :
    DigestCut.AgreeCut sha512Msg (treeOpenK k fs ts f i s n b T q) :=
  optCut q .skip fun p => .coin fun d =>
    frameCut P rv k fs (ts.set i (T.bump p.1)) (fun log => f (⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log))
      (pre ++ [⟨.opening i s n, b, TreeDraw.opened d.1 n⟩])

/-- A round's cut after an opening of the `i`-th tree it doesn't pick, at the tree. -/
noncomputable def optOpenCut (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β)
    (pre : List Release) (i s n : ℕ) (b : ByteArray) (o : Option OpenTree) :
    DigestCut.AgreeCut sha512Msg (optOpenK k fs ts f i s n b o) :=
  optCut o .skip fun T => treeOpenCut P rv k fs ts f pre i s n b T (T.next s n)

open Classical in
/-- **A round's cut after an opening frame**: fixed there, at the bytes the frame answers, where the round's reading
picks the frame after the log `pre`; otherwise the cut after the opening's coin. -/
noncomputable def openCut (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β)
    (pre : List Release) (i s n : ℕ) (b : ByteArray) (o : Option OpenTree) :
    DigestCut Claims.Msg (optOpenK k fs ts f i s n b o) :=
  (if P pre (.opening i s n) then cutAt rv b.data.toList else optOpenCut P rv k fs ts f pre i s n b o).cut

variable {P rv}

open Classical in
theorem frameCut_fresh {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {pre : List Release} {l : String} {b : ByteArray} (hn : ¬ P pre (.fresh l)) :
    ((frameCut P rv (k + 1) fs ts f pre).cut).atSend (some (.fresh l, b)) =
      (freshCut P rv k ts f pre l b (takeCoin l fs)).cut := by
  show (if P pre (.fresh l) then cutAt rv b.data.toList else freshCut P rv k ts f pre l b (takeCoin l fs)).cut = _
  rw [ite_eq_right hn]

open Classical in
theorem frameCut_hello {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {pre : List Release} {spec : Flock.CoinTree.Spec} {ν : ByteArray} (hn : ¬ P pre (.hello spec)) :
    ((frameCut P rv (k + 1) fs ts f pre).cut).atSend (some (.hello spec, ν)) =
      (helloCut P rv k fs ts f pre spec ν).cut := by
  show (if P pre (.hello spec) then cutAt rv ν.data.toList else helloCut P rv k fs ts f pre spec ν).cut = _
  rw [ite_eq_right hn]

theorem frameCut_opening {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {pre : List Release} {i s n : ℕ} {b : ByteArray} :
    ((frameCut P rv (k + 1) fs ts f pre).cut).atSend (some (.opening i s n, b)) =
      openCut P rv k fs ts f pre i s n b ts[i]? :=
  rfl

section OpenCut

variable {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β} {pre : List Release}
  {i s n : ℕ} {b : ByteArray}

open Classical in
theorem isHere_openCut (h : P pre (.opening i s n)) (o : Option OpenTree) :
    (openCut P rv k fs ts f pre i s n b o).isHere = true := by
  unfold openCut
  rw [ite_eq_left h]
  rfl

open Classical in
theorem openCut_neg (h : ¬ P pre (.opening i s n)) (o : Option OpenTree) :
    openCut P rv k fs ts f pre i s n b o = (optOpenCut P rv k fs ts f pre i s n b o).cut := by
  unfold openCut
  rw [ite_eq_right h]

theorem isHere_optOpenCut : ∀ o : Option OpenTree, (optOpenCut P rv k fs ts f pre i s n b o).cut.isHere = false
  | none => rfl
  | some T => by
    show (treeOpenCut P rv k fs ts f pre i s n b T (T.next s n)).cut.isHere = false
    cases T.next s n <;> rfl

end OpenCut

end Cuts

/-! ## Where the walk stops -/

/-- **No release of `L` answers a frame a live round picks**: after the log `pre` and the releases of `L` before it, the
reading of no round `a` holds of picks the release's frame. -/
def NoPick (P : ι → List Release → Frame → Prop) (a : ι → Prop) (pre L : List Release) : Prop :=
  ∀ j, a j → ∀ L₁ r L₂, L = L₁ ++ r :: L₂ → ¬ P j (pre ++ L₁) r.frame

theorem noPick_nil (P : ι → List Release → Frame → Prop) (a : ι → Prop) (pre : List Release) :
    NoPick P a pre [] := fun _ _ L₁ _ _ h => by
  cases L₁ <;> simp at h

theorem noPick_cons {P : ι → List Release → Frame → Prop} {a : ι → Prop} {pre : List Release} {r : Release}
    {L : List Release} (hr : ∀ j, a j → ¬ P j pre r.frame) (hL : NoPick P a (pre ++ [r]) L) :
    NoPick P a pre (r :: L) := by
  intro j hj L₁ r' L₂ h
  cases L₁ with
  | nil =>
    obtain ⟨rfl, -⟩ := List.cons.inj h
    simpa using hr j hj
  | cons x L₁ =>
    obtain ⟨rfl, rfl⟩ := List.cons.inj h
    simpa [List.append_assoc] using hL j hj L₁ r' L₂ rfl

variable {g : Game β} {σ : Strategy g}

/-- **Where the walk stops**, from the device's log `pre`, the continuation `f` and the live rounds `a`: after releases
`L` no live round picks (`NoPick`), either at the play's end, every cut past there, the end of a play of `f L`; or at
the node after an opening frame of the `i`-th tree's stream `s` that a live round picks after the log `pre ++ L`, the
device's frames left `k`, families `fs` and trees `ts`, the continuation `f` after `L`, with the live rounds' cuts
there (`openCut`). -/
def StopOf (P : ι → List Release → Frame → Prop) (rv : ι → β → Option Claims.Msg) (pre : List Release)
    (f : List Release → Game β) (a : ι → Prop) (q : Pos g σ Claims.Msg ι) : Prop :=
  ∃ L, NoPick P a pre L ∧
    (((∀ j, q.Ks j = .skip) ∧ ∃ y, q.g' = .ret y ∧ ∃ s₀ x₀, CoinTape.run (f L) s₀ x₀ = y) ∨
      ∃ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (f' : List Release → Game β) (i s n : ℕ)
        (b : ByteArray), (∀ log, f' log = f (L ++ log)) ∧ (∃ j, a j ∧ P j (pre ++ L) (.opening i s n)) ∧
        ∃ (s'' : Strategy (optOpenK k fs ts f' i s n b ts[i]?))
          (Ks : ι → DigestCut Claims.Msg (optOpenK k fs ts f' i s n b ts[i]?))
          (x'' : CoinTape (optOpenK k fs ts f' i s n b ts[i]?) s'')
          (back'' : CoinTape (optOpenK k fs ts f' i s n b ts[i]?) s'' → CoinTape g σ),
          q = ⟨optOpenK k fs ts f' i s n b ts[i]?, s'', Ks, x'', back''⟩ ∧
            DigestCut.LiveCuts a (fun j => openCut (P j) (rv j) k fs ts f' (pre ++ L) i s n b ts[i]?) Ks)

/-- **The walk from the device's releases with `k` frames left stops as `StopOf` says**, at every families, trees,
continuation, log and live rounds, the live rounds' cuts those at the first frame their readings pick. -/
def RelStops (P : ι → List Release → Frame → Prop) (rv : ι → β → Option Claims.Msg) (k : ℕ) : Prop :=
  ∀ (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game β) (pre : List Release) (a : ι → Prop)
    (s' : Strategy (relK k fs ts f)) (Ks : ι → DigestCut Claims.Msg (relK k fs ts f))
    (x : CoinTape (relK k fs ts f) s') (back : CoinTape (relK k fs ts f) s' → CoinTape g σ),
    DigestCut.LiveCuts a (fun j => (frameCut (P j) (rv j) k fs ts f pre).cut) Ks →
      StopOf P rv pre f a (Pos.walk (relK k fs ts f) s' Ks x back)

variable {P : ι → List Release → Frame → Prop} {rv : ι → β → Option Claims.Msg}

theorem stopOf_cons {pre : List Release} {f : List Release → Game β} {a : ι → Prop} {q : Pos g σ Claims.Msg ι}
    (r : Release) (hr : ∀ j, a j → ¬ P j pre r.frame) (h : StopOf P rv (pre ++ [r]) (fun log => f (r :: log)) a q) :
    StopOf P rv pre f a q := by
  obtain ⟨L, hL, h⟩ := h
  refine ⟨r :: L, noPick_cons hr hL, ?_⟩
  rcases h with ⟨hK, y, hy, s₀, x₀, hrun⟩ |
    ⟨k, fs, ts, f', i, s, n, b, hf, ⟨j, hj, hp⟩, s'', Ks, x'', back'', hq, hKs⟩
  · exact Or.inl ⟨hK, y, hy, s₀, x₀, hrun⟩
  · rw [List.append_assoc] at hp hKs
    exact Or.inr ⟨k, fs, ts, f', i, s, n, b, hf, ⟨j, hj, hp⟩, s'', Ks, x'', back'', hq, hKs⟩

theorem stopOf_end {pre : List Release} {f : List Release → Game β} {a : ι → Prop} (s' : Strategy (f []))
    (Ks : ι → DigestCut Claims.Msg (f [])) (x : CoinTape (f []) s') (back : CoinTape (f []) s' → CoinTape g σ)
    (h : ∀ j, Ks j = .skip) : StopOf P rv pre f a (Pos.walk (f []) s' Ks x back) := by
  obtain ⟨hK, hg⟩ := Pos.walk_skip (f []) s' Ks x back h
  exact ⟨[], noPick_nil P a pre, Or.inl ⟨hK, _, hg, s', x, rfl⟩⟩

theorem stopOf_open {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {pre : List Release} {i s n : ℕ} {b : ByteArray} {a : ι → Prop}
    (s' : Strategy (optOpenK k fs ts f i s n b ts[i]?)) (Ks : ι → DigestCut Claims.Msg (optOpenK k fs ts f i s n b ts[i]?))
    (x : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s')
    (back : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s' → CoinTape g σ)
    (hKs : DigestCut.LiveCuts a (fun j => openCut (P j) (rv j) k fs ts f pre i s n b ts[i]?) Ks)
    (hp : ∃ j, a j ∧ P j pre (.opening i s n)) :
    StopOf P rv pre f a (Pos.walk (optOpenK k fs ts f i s n b ts[i]?) s' Ks x back) := by
  obtain ⟨j₀, ha₀, hp₀⟩ := hp
  have hh : ∃ j, (Ks j).isHere := ⟨j₀, by rw [hKs.eq_of ha₀]; exact isHere_openCut hp₀ _⟩
  rw [Pos.walk_here _ s' Ks x back hh]
  refine ⟨[], noPick_nil P a pre, Or.inr ⟨k, fs, ts, f, i, s, n, b, fun _ => rfl, ⟨j₀, ha₀, ?_⟩, s', Ks, x, back,
    rfl, ?_⟩⟩
  · rwa [List.append_nil]
  · rw [List.append_nil]
    exact hKs

section Steps

variable {k : ℕ} (IH : RelStops (g := g) (σ := σ) P rv k)
include IH

theorem walk_freshK {ts : List OpenTree} {f : List Release → Game β} {pre : List Release} {l : String}
    {b : ByteArray} {a : ι → Prop} (hn : ∀ j, a j → ¬ P j pre (.fresh l)) :
    ∀ (o : Option (List (String × ℕ))) (s' : Strategy (freshK k ts f l b o))
      (Ks : ι → DigestCut Claims.Msg (freshK k ts f l b o)) (x : CoinTape (freshK k ts f l b o) s')
      (back : CoinTape (freshK k ts f l b o) s' → CoinTape g σ),
      DigestCut.LiveCuts a (fun j => (freshCut (P j) (rv j) k ts f pre l b o).cut) Ks →
        StopOf P rv pre f a (Pos.walk (freshK k ts f l b o) s' Ks x back)
  | none, s', Ks, x, back, hKs => stopOf_end s' Ks x back (hKs.eq_skip fun _ _ => rfl)
  | some fs', s', Ks, x, back, hKs => by
    rw [show Pos.walk (freshK k ts f l b (some fs')) s' Ks x back = _ from
      Pos.walk_coin s' Ks x back (hKs.isHere fun _ _ => rfl)]
    exact stopOf_cons ⟨.fresh l, b, List.ofFn x.1⟩ hn
      (IH fs' ts (fun log => f (⟨.fresh l, b, List.ofFn x.1⟩ :: log))
        (pre ++ [(⟨.fresh l, b, List.ofFn x.1⟩ : Release)]) a (s' x.1) _ (x.2 x.1) _ (hKs.atCoin x.1))

theorem walk_helloK {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β} {pre : List Release}
    {spec : Flock.CoinTree.Spec} {ν : ByteArray} {a : ι → Prop} (hn : ∀ j, a j → ¬ P j pre (.hello spec))
    (s' : Strategy (helloK k fs ts f spec ν)) (Ks : ι → DigestCut Claims.Msg (helloK k fs ts f spec ν))
    (x : CoinTape (helloK k fs ts f spec ν) s') (back : CoinTape (helloK k fs ts f spec ν) s' → CoinTape g σ)
    (hKs : DigestCut.LiveCuts a (fun j => (helloCut (P j) (rv j) k fs ts f pre spec ν).cut) Ks) :
    StopOf P rv pre f a (Pos.walk (helloK k fs ts f spec ν) s' Ks x back) := by
  rw [show Pos.walk (helloK k fs ts f spec ν) s' Ks x back = _ from
    Pos.walk_coin s' Ks x back (hKs.isHere fun _ _ => rfl)]
  exact stopOf_cons ⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec x.1 ++ treeKey x.1.1)⟩ hn
    (IH fs (ts ++ [(⟨ν, spec, x.1, fun _ => 0⟩ : OpenTree)])
      (fun log => f (⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec x.1 ++ treeKey x.1.1)⟩ :: log))
      (pre ++ [(⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec x.1 ++ treeKey x.1.1)⟩ : Release)]) a (s' x.1) _
      (x.2 x.1) _ (hKs.atCoin x.1))

theorem walk_treeOpenK {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {pre : List Release} {i s n : ℕ} {b : ByteArray} {a : ι → Prop} (hn : ∀ j, a j → ¬ P j pre (.opening i s n))
    (T : OpenTree) :
    ∀ (q : Option (Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)))
      (s' : Strategy (treeOpenK k fs ts f i s n b T q)) (Ks : ι → DigestCut Claims.Msg (treeOpenK k fs ts f i s n b T q))
      (x : CoinTape (treeOpenK k fs ts f i s n b T q) s')
      (back : CoinTape (treeOpenK k fs ts f i s n b T q) s' → CoinTape g σ),
      DigestCut.LiveCuts a (fun j => (treeOpenCut (P j) (rv j) k fs ts f pre i s n b T q).cut) Ks →
        StopOf P rv pre f a (Pos.walk (treeOpenK k fs ts f i s n b T q) s' Ks x back)
  | none, s', Ks, x, back, hKs => stopOf_end s' Ks x back (hKs.eq_skip fun _ _ => rfl)
  | some p, s', Ks, x, back, hKs => by
    rw [show Pos.walk (treeOpenK k fs ts f i s n b T (some p)) s' Ks x back = _ from
      Pos.walk_coin s' Ks x back (hKs.isHere fun _ _ => rfl)]
    exact stopOf_cons ⟨.opening i s n, b, TreeDraw.opened x.1.1 n⟩ hn
      (IH fs (ts.set i (T.bump p.1)) (fun log => f (⟨.opening i s n, b, TreeDraw.opened x.1.1 n⟩ :: log))
        (pre ++ [(⟨.opening i s n, b, TreeDraw.opened x.1.1 n⟩ : Release)]) a (s' x.1) _ (x.2 x.1) _
        (hKs.atCoin x.1))

theorem walk_optOpenK {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {pre : List Release} {i s n : ℕ} {b : ByteArray} {a : ι → Prop} (hn : ∀ j, a j → ¬ P j pre (.opening i s n)) :
    ∀ (o : Option OpenTree) (s' : Strategy (optOpenK k fs ts f i s n b o))
      (Ks : ι → DigestCut Claims.Msg (optOpenK k fs ts f i s n b o)) (x : CoinTape (optOpenK k fs ts f i s n b o) s')
      (back : CoinTape (optOpenK k fs ts f i s n b o) s' → CoinTape g σ),
      DigestCut.LiveCuts a (fun j => (optOpenCut (P j) (rv j) k fs ts f pre i s n b o).cut) Ks →
        StopOf P rv pre f a (Pos.walk (optOpenK k fs ts f i s n b o) s' Ks x back)
  | none, s', Ks, x, back, hKs => stopOf_end s' Ks x back (hKs.eq_skip fun _ _ => rfl)
  | some T, s', Ks, x, back, hKs => walk_treeOpenK IH hn T (T.next s n) s' Ks x back hKs

end Steps

/-- **The walk from the device's releases stops at the play's end or right after an opening frame a live round
picks**, where each round's reading picks only openings. -/
theorem walk_relK (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n) :
    ∀ k, RelStops (g := g) (σ := σ) P rv k
  | 0 => fun _ _ _ _ _ s' Ks x back hKs => stopOf_end s' Ks x back (hKs.eq_skip fun _ _ => rfl)
  | k + 1 => fun fs ts f pre a s' Ks x back hKs => by
    rw [show Pos.walk (relK (k + 1) fs ts f) s' Ks x back = _ from
      Pos.walk_send s' Ks x back (hKs.isHere fun _ _ => rfl)]
    have h1 := hKs.atSend s'.1
    obtain ⟨m, s₁⟩ := s'
    rcases m with _ | ⟨l | spec | ⟨i, s, n⟩, b⟩
    · exact stopOf_end s₁ _ x back (h1.eq_skip fun _ _ => rfl)
    · have hn : ∀ j, a j → ¬ P j pre (.fresh l) := fun j _ hp => by
        obtain ⟨_, _, _, h⟩ := hP j pre _ hp
        cases h
      exact walk_freshK (walk_relK hP k) hn (takeCoin l fs) s₁ _ x back
        (h1.congr (K' := fun j => (freshCut (P j) (rv j) k ts f pre l b (takeCoin l fs)).cut) fun j hj =>
          frameCut_fresh (hn j hj))
    · have hn : ∀ j, a j → ¬ P j pre (.hello spec) := fun j _ hp => by
        obtain ⟨_, _, _, h⟩ := hP j pre _ hp
        cases h
      exact walk_helloK (walk_relK hP k) hn s₁ _ x back
        (h1.congr (K' := fun j => (helloCut (P j) (rv j) k fs ts f pre spec b).cut) fun j hj =>
          frameCut_hello (hn j hj))
    · have h2 := h1.congr (K' := fun j => openCut (P j) (rv j) k fs ts f pre i s n b ts[i]?) fun _ _ => rfl
      by_cases hpk : ∃ j, a j ∧ P j pre (.opening i s n)
      · exact stopOf_open s₁ _ x back h2 hpk
      · have hn : ∀ j, a j → ¬ P j pre (.opening i s n) := fun j hj hp => hpk ⟨j, hj, hp⟩
        exact walk_optOpenK (walk_relK hP k) hn ts[i]? s₁ _ x back
          (h2.congr fun j hj => openCut_neg (hn j hj) _)

/-- **The walk from any node the device's releases are equal to stops as `StopOf` says** (`walk_relK`), at the cuts
cast onto it. -/
theorem walk_cast (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n) {k : ℕ} {fs : List (String × ℕ)}
    {ts : List OpenTree} {f : List Release → Game β} {pre : List Release} {a : ι → Prop} {g' : Game β}
    (h : relK k fs ts f = g') (s' : Strategy g') (Ks : ι → DigestCut Claims.Msg g') (x : CoinTape g' s')
    (back : CoinTape g' s' → CoinTape g σ)
    (hKs : DigestCut.LiveCuts a (fun j => ((frameCut (P j) (rv j) k fs ts f pre).cast h).cut) Ks) :
    StopOf P rv pre f a (Pos.walk g' s' Ks x back) := by
  subst h
  exact walk_relK hP k fs ts f pre a s' Ks x back hKs

/-- **Consuming a stop after an opening frame leaves live the rounds that didn't pick it**, each at its cut after the
opening's coin. -/
theorem liveCuts_consume {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {pre : List Release} {i s n : ℕ} {b : ByteArray} {o : Option OpenTree} {a : ι → Prop}
    {Ks : ι → DigestCut Claims.Msg (optOpenK k fs ts f i s n b o)}
    (h : DigestCut.LiveCuts a (fun j => openCut (P j) (rv j) k fs ts f pre i s n b o) Ks) :
    DigestCut.LiveCuts (fun j => a j ∧ ¬ P j pre (.opening i s n))
      (fun j => (optOpenCut (P j) (rv j) k fs ts f pre i s n b o).cut)
      (fun j => if (Ks j).isHere then .skip else Ks j) := fun j => by
  by_cases ha : a j
  · by_cases hp : P j pre (.opening i s n)
    · exact Or.inr ⟨fun h' => h'.2 hp, ite_eq_left (by rw [h.eq_of ha]; exact isHere_openCut hp o)⟩
    · refine Or.inl ⟨⟨ha, hp⟩, (ite_eq_right fun e => ?_).trans ?_⟩
      · rw [h.eq_of ha, openCut_neg hp, isHere_optOpenCut] at e
        cases e
      · rw [h.eq_of ha, openCut_neg hp]
  · refine Or.inr ⟨fun h' => ha h'.1, ?_⟩
    by_cases hc : (Ks j).isHere = true
    · exact ite_eq_left hc
    · exact (ite_eq_right hc).trans (h.skip_of ha)

/-- **The walk from a stop after an opening frame, consumed, stops as `StopOf` says**, at the rounds that didn't pick
the frame and on any rewriting of the tape there. -/
theorem stopOf_next (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n) {k : ℕ} {fs : List (String × ℕ)}
    {ts : List OpenTree} {f : List Release → Game β} {pre : List Release} {i s n : ℕ} {b : ByteArray}
    {a : ι → Prop} {q : Pos g σ Claims.Msg ι} (s'' : Strategy (optOpenK k fs ts f i s n b ts[i]?))
    (Ks : ι → DigestCut Claims.Msg (optOpenK k fs ts f i s n b ts[i]?))
    (x'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s'')
    (back'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s'' → CoinTape g σ)
    (hq : q = ⟨optOpenK k fs ts f i s n b ts[i]?, s'', Ks, x'', back''⟩)
    (hKs : DigestCut.LiveCuts a (fun j => openCut (P j) (rv j) k fs ts f pre i s n b ts[i]?) Ks)
    (φ : CoinTape q.g' q.s' → CoinTape q.g' q.s') :
    StopOf P rv pre f (fun j => a j ∧ ¬ P j pre (.opening i s n)) (q.rewrite φ).consume.toStop := by
  subst hq
  exact walk_optOpenK (walk_relK hP k) (fun _ h => h.2) ts[i]? s'' _ (φ x'') back'' (liveCuts_consume hKs)

end Verity
