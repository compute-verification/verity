import Proofs.Flock.Claims.Recursion.PartAccepts
import Proofs.Flock.Claims.Recursion.BoundRun

/-!
# The composition: V\*'s parts' checks together make `IVStar` accept (`ivstar_of_parts`)

`DecompB` at what each part's circuit accepts (`PartAccepts`) is one premise (`exists_chk_iff`), and it mixes two
facts: what each V\* part's circuit computes, and that what the parts compute together is `IVStar`'s acceptance. This
file proves the second, at every bound record, layout and deployment, for per-part predicates of the code side's
choice, so that whichever way each part's share is discharged (a named per-part lowering fact, or a proof of the part's
lowering) plugs in.

**`IVStar`'s checks** (`IVCheck`, at the inner statement, `ivChecks`). `IVStar` accepts the messages `ms` at the coins
`c` exactly when its last message reads (`final`) as rows and salts with the schedule's counts, each earlier round
opens to them, and the element-level verifier accepts the opened messages (`Bound.IVStar`). So its checks are:

* `shape`: there are rounds, and the last message reads as `os.length + 1 = k` rounds' rows and salts with the
  schedule's counts (`Bound.nAt`);
* `len j`, for each round `j` before the last: its message is 128 bytes a row;
* `salt j i` and `row j i`, for each of round `j`'s `Bound.nAt` commit strings: the salt's hash is the string's last 64
  bytes, and the row's digest is its first 64 unmasked at the salt (`Bound.opens`);
* `elem q`, for each check of a split of the element-level verifier's acceptance (`ElemSplit`), at the opened messages.

Every check reads `ms` and `c` alone; which checks there are is fixed by the inner statement. None of them is
`IVStar`'s acceptance: `shape`, `len`, `salt` and `row` are single comparisons, and an `elem` check is whatever the
split names. A split is a structure whose field `sound` is the proof that its checks give the element-level verifier's
acceptance, so a split's checks are only as coarse as its author makes them. The one split proved here at every bound
record is by step (`stepSplit`); at the deployment the steps before the last are proved from two checks no part need
compute (`deploySplit`: the statement sets up, and an opened round's tag), and the last step, the verdict, is the
split's parameter (`LastSplit`), which a split of the verdict into the parts' element-level checks fills.

**The composition** (`ivstar_of_parts`). An inventory `own` gives each check to a part (`some s`) or to the compiled
statement (`none`). Take any per-part predicates `PC` such that each check given to part `s` is a part of the statement
(`s < nS`) and follows from `PC s` (`OwnedBy`), and each check given to the statement holds at the readers' messages of
every witnesses whose calls are true (`StmtFixes`). Then `DecompB` holds at `PC`: wherever every part's `PC` holds at
the readers' messages, `IVStar` accepts them. With per-part lowering facts, `PartAccepts → PC` at each part, that is
`DecompB` at `PartAccepts` (`decompB_partAccepts_of_parts`), and the dense corollaries take the facts in its place
(`RecDeploy.ksLin2_deployed_parts`, `RecDeploy.bitsLin2_deployed_parts`).

`PartComputes`, part `s`'s share of the checks, is the least such `PC` (`ownedBy_partComputes`), and the default
inventory (`ownHolders`) gives each commit string's salt and row checks to the round's first holder, the part whose
message the readers take, and the rest to the statement or to the code side's `ownE`. Which checks the statement
fixes is a code fact about the layout and `final`; no part computes `final`'s reading or an opened round's tag.

The composition loses nothing: where every check holds, `IVStar` accepts (`ivstar_acc_of_checks`), and that is the
only fact about `IVStar` it reads.
-/

namespace Claims.Recursion.VStarB

open Claims

/-! ## Oracle programs -/

/-- Programs in order return `true` where each does. -/
theorem allOC_of_forall (H : Msg → Msg) : ∀ ps : List (OC Bool), (∀ p ∈ ps, (p.eval H).1 = true) →
    ((allOC ps).eval H).1 = true
  | [], _ => rfl
  | p :: ps, h => by
    show (OC.eval H (bindOC p fun b => if b then allOC ps else .ret false)).1 = true
    rw [eval_bindOC, h p (by simp)]
    exact allOC_of_forall H ps fun q hq => h q (by simp [hq])

theorem eval_bindOC_of {α β : Type} (H : Msg → Msg) (p : OC α) (f : α → OC β) {a : α} (hp : (p.eval H).1 = a) :
    (OC.eval H (bindOC p f)).1 = (OC.eval H (f a)).1 := by
  rw [eval_bindOC, hp]

section Checks

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-! ## Splits of the element-level verifier's acceptance -/

/-- **A split of the element-level verifier's acceptance into checks**: at each inner statement `x`, the checks
(`checks x`, fixed by the statement), each holding or not at the opened messages and the coins (`holds`), and the proof
that where every check holds at messages and coins of `DE`'s rounds, the element-level verifier accepts them
(`Bound.agrees`, `sound`). -/
structure ElemSplit (H : Msg → Msg) (Bd : Bound Coin Col T) (Q : Type) where
  checks : Msg → List Q
  holds : Q → Msg → List Msg → List Coin → Prop
  sound : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ms.length = Bd.DE.V.k x → cs.length = ms.length →
    (∀ q ∈ checks x, holds q x ms cs) → ((Bd.agrees x ms cs).eval H).1 = true

/-- **Step `j` of the element-level verifier gives what `agrees` asks**: on the first `j + 1` messages and coins, the
round's coin's answer before the last round, acceptance at the last. -/
def StepOk (H : Msg → Msg) (Bd : Bound Coin Col T) (x : Msg) (ms : List Msg) (cs : List Coin) (j : ℕ) : Prop :=
  ((Bd.DE.V.next x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
    if j + 1 < ms.length then .msg (Bd.ans (cs.getD j default)) else .acc

/-- **The split by step**: one check a round of `DE`, its step (`StepOk`). -/
def stepSplit (H : Msg → Msg) (Bd : Bound Coin Col T) : ElemSplit H Bd ℕ where
  checks x := List.range (Bd.DE.V.k x)
  holds j x ms cs := StepOk H Bd x ms cs j
  sound x ms cs hk _ h := by
    unfold Bound.agrees
    refine allOC_of_forall H _ fun p hp => ?_
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hp
    rw [eval_bindOC_of H _ _ (h j (by rw [List.mem_range] at hj ⊢; omega))]
    simp [OC.eval]

/-- **A split refined at its checks**: each check `q` of `S` replaced by checks of its own (`checksR q x`), which
together give it (`soundR`). -/
def ElemSplit.refine {H : Msg → Msg} {Bd : Bound Coin Col T} {Q : Type} (S : ElemSplit H Bd Q) {R : Type}
    (checksR : Q → Msg → List R) (holdsR : Q → R → Msg → List Msg → List Coin → Prop)
    (soundR : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ms.length = Bd.DE.V.k x → cs.length = ms.length →
      ∀ q ∈ S.checks x, (∀ r ∈ checksR q x, holdsR q r x ms cs) → S.holds q x ms cs) :
    ElemSplit H Bd (Q × R) where
  checks x := (S.checks x).flatMap fun q => (checksR q x).map fun r => (q, r)
  holds p x ms cs := holdsR p.1 p.2 x ms cs
  sound x ms cs hk hc h := S.sound x ms cs hk hc fun q hq => soundR x ms cs hk hc q hq fun r hr =>
    h (q, r) (List.mem_flatMap.2 ⟨q, hq, List.mem_map.2 ⟨r, hr, rfl⟩⟩)

/-! ## `IVStar`'s checks -/

/-- **A check of `IVStar`'s decision**: the last message's reading (`shape`), round `j`'s length (`len j`), the salt's
hash (`salt j i`) and the row's digest (`row j i`) at round `j`'s `i`-th commit string, or the element-level split's
check `q` (`elem q`). -/
inductive IVCheck (Q : Type) where
  | shape
  | len (j : ℕ)
  | salt (j i : ℕ)
  | row (j i : ℕ)
  | elem (q : Q)

/-- Round message `a`'s `i`-th commit string, as `Bound.opensRound` cuts it. -/
def cstr (a : Msg) (i : ℕ) : Msg := (a.drop (128 * i)).take 128

/-- **What each check says** at the inner statement `x`, the coins `c` and the messages `ms` (`IVStar`'s `k` of them,
read with `getD`, as its run reads them). -/
def IVCheck.Holds {Q : Type} (H : Msg → Msg) (Bd : Bound Coin Col T) (S : ElemSplit H Bd Q) (x : Msg) (c : ℕ → Coin)
    (ms : List Msg) : IVCheck Q → Prop
  | .shape => 0 < Bd.DE.V.k x ∧ ∃ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) ∧
      os.length + 1 = Bd.DE.V.k x ∧ ∀ j < os.length, (os.getD j []).length = Bd.nAt x j
  | .len j => ∀ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) →
      (ms.getD j []).length = 128 * (os.getD j []).length
  | .salt j i => ∀ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) → ∀ o,
      (os.getD j [])[i]? = some o → H (Bd.saltPrefix ++ List.ofFn o.2) = (cstr (ms.getD j []) i).drop 64
  | .row j i => ∀ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) → ∀ o,
      (os.getD j [])[i]? = some o → H (Bd.rowIn o.1) = xorMask ((cstr (ms.getD j []) i).take 64) (Bd.mask o.2)
  | .elem q => ∀ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) →
      S.holds q x (Bd.opened os last) ((List.range (Bd.DE.V.k x)).map c)

/-- **`IVStar`'s checks at the inner statement `x`**: the last message's reading; at each round before the last, its
length and each of its `nAt` commit strings' salt and row; then the split's checks. -/
def ivChecks {Q : Type} {H : Msg → Msg} (Bd : Bound Coin Col T) (S : ElemSplit H Bd Q) (x : Msg) : List (IVCheck Q) :=
  .shape :: ((List.range (Bd.DE.V.k x - 1)).flatMap fun j =>
    .len j :: (List.range (Bd.nAt x j)).flatMap fun i => [.salt j i, .row j i]) ++ (S.checks x).map .elem

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
/-- **`IVStar` accepts where its opening layer and the element-level verifier do**: at messages of its `k` rounds
whose last reads as rows and salts with the schedule's counts, each earlier round opening to them, and the element-level
verifier accepting the opened messages. -/
theorem ivstar_next_acc (H : Msg → Msg) (Bd : Bound Coin Col T) (x : Msg) (as : List Msg) (cs : List Coin)
    (hlen : as.length = Bd.DE.V.k x) (os : List (List (Msg × (Fin Bd.saltLen → UInt8)))) (last : Msg)
    (hfin : Bd.final (as.getLastD []) = some (os, last)) (hos : os.length + 1 = as.length)
    (hcnt : ∀ j < os.length, (os.getD j []).length = Bd.nAt x j)
    (hopen : ∀ p ∈ as.dropLast.zip os, ((Bd.opensRound p.1 p.2).eval H).1 = true)
    (hagr : ((Bd.agrees x (Bd.opened os last) cs).eval H).1 = true) :
    ((Bd.IVStar.next x as cs).eval H).1 = .acc := by
  have h1 : ¬ as.length < Bd.DE.V.k x := by omega
  simp only [Bound.IVStar, h1, ↓reduceIte, hfin]
  rw [ite_eq_left ⟨hos, hcnt⟩, eval_bindOC_of H _ _ (allOC_of_forall H _ fun p hp => by
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hp
    exact hopen q hq)]
  rw [ite_eq_left rfl, eval_bindOC_of H _ _ hagr]
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **A round opens where its length and each commit string's salt and row check.** -/
theorem opensRound_true (H : Msg → Msg) (Bd : Bound Coin Col T) (a : Msg)
    (os : List (Msg × (Fin Bd.saltLen → UInt8))) (hl : a.length = 128 * os.length)
    (hs : ∀ i o, os[i]? = some o → H (Bd.saltPrefix ++ List.ofFn o.2) = (cstr a i).drop 64)
    (hr : ∀ i o, os[i]? = some o → H (Bd.rowIn o.1) = xorMask ((cstr a i).take 64) (Bd.mask o.2)) :
    ((Bd.opensRound a os).eval H).1 = true := by
  unfold Bound.opensRound
  rw [ite_eq_left hl]
  refine allOC_of_forall H _ fun p hp => ?_
  obtain ⟨i, hi, rfl⟩ := List.mem_mapIdx.1 hp
  have ho : os[i]? = some os[i] := List.getElem?_eq_getElem hi
  simp only [Bound.opens, OC.eval]
  exact decide_eq_true ⟨hs i _ ho, hr i _ ho⟩

/-- **Where every check holds, `IVStar` accepts**: its run on the replayed messages at the coins `c` accepts. -/
theorem ivstar_acc_of_checks {Q : Type} (H : Msg → Msg) (Bd : Bound Coin Col T) (S : ElemSplit H Bd Q) (x : Msg)
    (c : ℕ → Coin) (ms : List Msg) (h : ∀ q ∈ ivChecks Bd S x, q.Holds H Bd S x c ms) :
    (run H (replay ms) Bd.IVStar x () fun i => c i.val).out = .acc := by
  obtain ⟨hk, os, last, hfin, hos, hcnt⟩ := h .shape (List.mem_cons_self ..)
  have hmem : ∀ q, q ∈ ((List.range (Bd.DE.V.k x - 1)).flatMap fun j =>
      IVCheck.len j :: (List.range (Bd.nAt x j)).flatMap fun i => [IVCheck.salt j i, IVCheck.row j i]) →
      q.Holds H Bd S x c ms := fun q hq =>
    h q (List.mem_cons_of_mem _ (List.mem_append_left _ hq))
  rw [Bd.run_ivstar H x ms _ hk]
  show (Bd.decAt H x ms _).1 = .acc
  unfold Bound.decAt
  obtain ⟨n, hn⟩ : ∃ n, Bd.DE.V.k x = n + 1 := ⟨Bd.DE.V.k x - 1, by omega⟩
  have hcs : (List.range (Bd.DE.V.k x)).map (coinSeq (k := Bd.IVStar.k x) fun i => c i.val) =
      (List.range (Bd.DE.V.k x)).map c :=
    List.map_congr_left fun i hi => by
      rw [List.mem_range] at hi
      simp [coinSeq, show i < Bd.IVStar.k x from hi]
  rw [hcs]
  refine ivstar_next_acc H Bd x _ _ (by simp) os last ?_ (by simp [hos]) hcnt ?_ ?_
  · rw [hn, List.range_succ, List.map_append, List.map_singleton, List.getLastD_concat]
    rw [hn, Nat.add_sub_cancel] at hfin
    exact hfin
  · intro p hp
    obtain ⟨j, hj, rfl⟩ := List.mem_iff_getElem.1 hp
    simp only [List.length_zip, List.length_dropLast, List.length_map, List.length_range] at hj
    have hjo : j < os.length := by omega
    have hjk : j < Bd.DE.V.k x - 1 := by omega
    have e1 : ((List.range (Bd.DE.V.k x)).map fun i => ms.getD i []).dropLast[j]'(by simp; omega) = ms.getD j [] := by
      rw [List.getElem_dropLast]
      simp
    have e2 : os[j] = os.getD j [] := by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjo]
    rw [List.getElem_zip, e1, e2]
    have hjm : ∀ q, q ∈ IVCheck.len j :: (List.range (Bd.nAt x j)).flatMap (fun i => [IVCheck.salt j i,
        IVCheck.row j i]) → q.Holds H Bd S x c ms := fun q hq =>
      hmem q (List.mem_flatMap.2 ⟨j, List.mem_range.2 hjk, hq⟩)
    have hnj : (os.getD j []).length = Bd.nAt x j := hcnt j hjo
    refine opensRound_true H Bd _ _ (hjm _ (List.mem_cons_self ..) os last hfin) ?_ ?_
    · intro i o hi
      have hil : i < Bd.nAt x j := hnj ▸ (List.getElem?_eq_some_iff.1 hi).1
      exact hjm (.salt j i) (List.mem_cons_of_mem _ (List.mem_flatMap.2 ⟨i, List.mem_range.2 hil, by simp⟩))
        os last hfin o hi
    · intro i o hi
      have hil : i < Bd.nAt x j := hnj ▸ (List.getElem?_eq_some_iff.1 hi).1
      exact hjm (.row j i) (List.mem_cons_of_mem _ (List.mem_flatMap.2 ⟨i, List.mem_range.2 hil, by simp⟩))
        os last hfin o hi
  · refine S.sound x _ _ (by simp [Bound.opened, hos]) (by simp [Bound.opened, hos]) fun q hq => ?_
    exact h (.elem q) (List.mem_cons_of_mem _ (List.mem_append_right _ (List.mem_map.2 ⟨q, hq, rfl⟩))) os last hfin

end Checks

/-! ## The composition -/

section Compose

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Each check the inventory gives a part is that part's**, at the shape: at a compiled statement `x`, a check of
`IVStar` at the inner statement that `own` gives part `s` has `s` among the statement's parts, and follows from part
`s`'s predicate `PC s` wherever it holds. -/
def OwnedBy {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T) (shape : Form → Prop)
    (S : ElemSplit H Bd Q) (own : Msg → IVCheck Q → Option ℕ) (PC : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) :
    Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ q ∈ ivChecks Bd S (xI D x), ∀ s,
    own x q = some s → s < D.nS x ∧ ∀ c ms, PC s x c ms → q.Holds H Bd S (xI D x) c ms

/-- **Each check the inventory gives the statement holds off any witnesses** (a code fact about the layout, `final`
and the split): at a compiled statement `x` of the shape, a check `own` gives no part holds at the coins and at the
readers' messages of any witnesses, one a part, whose recorded calls are true evaluations. -/
def StmtFixes {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (S : ElemSplit H Bd Q) (own : Msg → IVCheck Q → Option ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ q ∈ ivChecks Bd S (xI D x), own x q = none →
    ∀ (c : ℕ → Coin) (ws : List Flock.Wit), ws.length = D.nS x → (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
      q.Holds H Bd S (xI D x) c (L.msgs x ws)

/-- **The composition**: where the inventory gives each of `IVStar`'s checks to a part whose predicate implies it
(`OwnedBy`) or to the statement, which fixes it (`StmtFixes`), the parts' predicates together are `IVStar`'s acceptance
at the readers' messages: `DecompB` holds at them. Its escape is never taken. -/
theorem ivstar_of_parts {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (S : ElemSplit H Bd Q) (own : Msg → IVCheck Q → Option ℕ)
    (PC : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (hown : OwnedBy H D Bd shape S own PC)
    (hfix : StmtFixes H D Bd L shape S own) : DecompB H D Bd.IVStar L shape PC := by
  intro x st hx hf c ws hl hT hC
  refine Or.inl (ivstar_acc_of_checks H Bd S (xI D x) c _ fun q hq => ?_)
  cases ho : own x q with
  | none => exact hfix x st hx hf q hq ho c ws hl hT
  | some s =>
    obtain ⟨hs, hq'⟩ := hown x st hx hf q hq s ho
    exact hq' c _ (hC s hs)

/-- **Part `s`'s share of `IVStar`'s checks**: at a compiled statement `x`, every check at the inner statement that the
inventory `own` gives part `s` holds at the coins and messages. -/
def PartComputes {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T) (S : ElemSplit H Bd Q)
    (own : Msg → IVCheck Q → Option ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin) (ms : List Msg) : Prop :=
  ∀ q ∈ ivChecks Bd S (xI D x), own x q = some s → q.Holds H Bd S (xI D x) c ms

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
/-- **`PartComputes` is owned**, wherever the inventory gives checks only to the statement's parts. -/
theorem ownedBy_partComputes {Q : Type} (H : Msg → Msg) (D : Deploy Coin) (Bd : Bound Coin Col T)
    (shape : Form → Prop) (S : ElemSplit H Bd Q) (own : Msg → IVCheck Q → Option ℕ)
    (hcov : ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ q ∈ ivChecks Bd S (xI D x), ∀ s,
      own x q = some s → s < D.nS x) :
    OwnedBy H D Bd shape S own (PartComputes H D Bd S own) :=
  fun x st hx hf q hq s ho => ⟨hcov x st hx hf q hq s ho, fun _ _ h => h q hq ho⟩

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
/-- **`PartComputes` is the least owned predicate**: any `PC` the inventory owns implies it. -/
theorem partComputes_of_ownedBy {Q : Type} {H : Msg → Msg} {D : Deploy Coin} {Bd : Bound Coin Col T}
    {shape : Form → Prop} {S : ElemSplit H Bd Q} {own : Msg → IVCheck Q → Option ℕ}
    {PC : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} (hown : OwnedBy H D Bd shape S own PC) {s : ℕ} {x : Msg}
    {st : Stmt} (hx : D.parse x = some st) (hf : shape st.form) {c : ℕ → Coin} {ms : List Msg}
    (h : PC s x c ms) : PartComputes H D Bd S own s x c ms :=
  fun q hq ho => (hown x st hx hf q hq s ho).2 c ms h

/-- **`DecompB` at what each part's circuit accepts, from the parts' lowering facts and the composition**: wherever
part `s`'s circuit accepting at a witness that agrees with the messages implies its predicate `PC s`, at every part,
and the inventory covers `IVStar`'s checks (`OwnedBy`, `StmtFixes`), `DecompB` holds at `PartAccepts`. -/
theorem decompB_partAccepts_of_parts {Q Col' T' : Type} (H : Msg → Msg) (D : Deploy Coin)
    (DO : Flock.Deploy Coin Col' T') (Bd : Bound Coin Col T) (L : LayoutB) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (S : ElemSplit H Bd Q) (own : Msg → IVCheck Q → Option ℕ)
    (PC : ℕ → Msg → (ℕ → Coin) → List Msg → Prop)
    (hlow : ∀ s x c ms, PartAccepts H D DO L shape shows s x c ms → PC s x c ms)
    (hown : OwnedBy H D Bd shape S own PC) (hfix : StmtFixes H D Bd L shape S own) :
    DecompB H D Bd.IVStar L shape (PartAccepts H D DO L shape shows) :=
  decompB_mono hlow (ivstar_of_parts H D Bd L shape S own PC hown hfix)

/-- The first of the statement's parts holding round `j` whole, as the readers take it (`LayoutB.msg`). -/
def firstHolder (D : Deploy Coin) (L : LayoutB) (x : Msg) (j : ℕ) : Option ℕ :=
  (List.range (D.nS x)).find? fun s => (L.msgAt x s j).isSome

/-- **The default inventory**: each commit string's salt and row checks to the first part holding its round whole, the
part whose commit string the readers take; the element-level checks as the code side gives them (`ownE`); the last
message's reading and the rounds' lengths to the statement. -/
def ownHolders {Q : Type} (D : Deploy Coin) (L : LayoutB) (ownE : Msg → Q → Option ℕ) (x : Msg) :
    IVCheck Q → Option ℕ
  | .shape => none
  | .len _ => none
  | .salt j _ => firstHolder D L x j
  | .row j _ => firstHolder D L x j
  | .elem q => ownE x q

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem firstHolder_lt {D : Deploy Coin} {L : LayoutB} {x : Msg} {j s : ℕ} (h : firstHolder D L x j = some s) :
    s < D.nS x :=
  List.mem_range.1 (List.mem_of_find?_eq_some h)

end Compose

end Claims.Recursion.VStarB
