module

public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Flock.SpotFlock.Sessions
public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.Sampled.Moves
public import Proofs.Properties.Verity.Sampled.MemberForm.Order

/-! The second stage's sessions' coins: wherever the check read one, it is the device's release
(`coinsRead_sessions`), and the statements' round reads no coin (`stmtReadsNone`). -/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Verity.Sampled (helloSpecs repStreamName)
open Verity.Sampled.Binding (Read)

def repsOf (K : ℕ) : ℕ := (K - 2) / 2

/-- Round `j` of a session at `R` rounds a rep: the link's opening at `j = 0`, then rep 0's rounds, then rep
1's, as Flock's deployed verifier takes them; not one-stage v2's `--zk` layout, which alternates the two reps. -/
def slotStream (tab : String) (R j : ℕ) : String × ℕ :=
  if j = 0 then (Spot.Challenger.TREE_SESSION, 0)
  else if j - 1 < R then (repStreamName tab 0, j - 1) else (repStreamName tab 1, j - 1 - R)

def firstsOf : List ℕ → List ℕ → List ℕ
  | acc, [] => acc
  | acc, i :: is => firstsOf (if i ∈ acc then acc else acc ++ [i]) is

def openTree : Verity.Frame → Option ℕ
  | .opening i _ _ => some i
  | _ => none

def treesOpened (log : List Release) : List ℕ :=
  firstsOf [] (log.filterMap fun r => openTree r.frame)

def roundOf (tab : String) (R : ℕ) (name : Option String) (k : ℕ) : Option ℕ :=
  if name = some Spot.Challenger.TREE_SESSION ∧ k = 0 then some 0
  else if name = some (repStreamName tab 0) ∧ k < R then some (k + 1)
  else if name = some (repStreamName tab 1) ∧ k < R then some (R + 1 + k)
  else none

def sessLoc (off K : ℕ → ℕ) (M : ℕ) (tab : ℕ → String) (pre : List Release) : Verity.Frame → Option ℕ
  | .opening i s _ =>
    let t := (treesOpened pre).idxOf i
    (roundOf (tab t) (repsOf (K t)) ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?)
        (openedOf pre i s)).bind fun j =>
      if t < M ∧ j < K t then some (off t + j) else none
  | _ => none

def locOf (A : Live₂) (tab : ℕ → String) : List Release → Verity.Frame → Option ℕ :=
  sessLoc (SpotFlock.off A.h) (SpotFlock.k₀ A.h) A.h.ms.length tab

def LocatesSessions (A : Live₂) (tab : ℕ → String) (L : List Release → Verity.Frame → ByteArray → Option ℕ) : Prop :=
  ∀ pre fr b n, groupOf A n = .sessions → (L pre fr b = some n ↔ locOf A tab pre fr = some n)

def located (loc : List Release → Verity.Frame → Option ℕ) (n : ℕ) :
    List Release → List Release → List (Verity.Frame × List Bool)
  | _, [] => []
  | pre, r :: log => (if loc pre r.frame = some n then [(r.frame, r.coins)] else []) ++ located loc n (pre ++ [r]) log

def DevLocal (A : Live₂) (loc : List Release → Verity.Frame → Option ℕ)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) : Prop :=
  ∀ n, groupOf A n = .sessions → ∀ log log', located loc n [] log = located loc n [] log' → dev log n = dev log' n

/-- The coins an opening of `n` coins carries: `n` elements of GF(2^128). -/
def widthOf : Verity.Frame → ℕ
  | .opening _ _ n => n
  | _ => 0

def interiorRead (C : CaseOf) (o : Exchange) : Option (List Release) := do
  let k ← C o
  let p ← (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson).toOption
  let e ← (VStar.Recursive.SampledTwo.evidenceOf k.evidence).toOption
  let s ← (Flock.canon e.record).toOption
  let c₂ ← (Spot.OneStage.commitmentOf s.toUTF8).toOption
  let d ← Properties.Challenger.oneStageZkSource p.audit c₂
  let ξ₂ ← (Spot.OneStageZk.coinsOf p.audit c₂ e.interior).toOption
  let recs ← (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn k.blobs)
  Verity.Sampled.readLog d.label d.after recs ξ₂ (← (Verity.Sampled.linesOf e.interior).toOption)

def InteriorLogged (C : CaseOf) (Good : Exchange → Prop) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ l, interiorRead C o = some l →
    ∃ q l', o.log = q ++ l' ∧ (∀ r ∈ q, openTree r.frame = none ∧ ∀ sp, r.frame ≠ .hello sp) ∧
      List.Forall₂ Read.Of (l.map Read.ofV2) l'

def InteriorReads (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∃ l, interiorRead C o = some l

def withCoins (A : Live₂) (C : CaseOf) (tab : ℕ → String) (dev : List Release → ℕ → Option FlockDeploy.Coin)
    (Ms : SessionReading) : SessionReading :=
  { Ms with
    coin := fun o t j => (interiorRead C o).bind fun l =>
      if (located (locOf A tab) (SpotFlock.off A.h t + j) [] l).all (fun x => !x.2.isEmpty) then
        dev l (SpotFlock.off A.h t + j)
      else none
    agree := fun o t j c c' => ∃ l, interiorRead C o = some l ∧
      ∀ e : Fin 65536, e.val < ((located (locOf A tab) (SpotFlock.off A.h t + j) [] l).map fun x => widthOf x.1).sum →
        c e = c' e }

def CoinsReadSome (A : Live₂) (Rd : Reading A) (Acc Good : Exchange → Prop) (g : Group) : Prop :=
  ∀ o : Exchange, Acc o → Good o → ∀ i, Rd.reads i → groupOf A i = g → ∀ c, Rd.coin o i = some c →
    Rd.dev o.log i = some c

theorem stmtReadsNone (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) : StmtReadsNone A (formReading A C S dev) := by
  intro i hi hg
  have hne : i ≠ A.h.stage + 1 := (show readsAt A.h i from hi).1
  unfold groupOf at hg
  split_ifs at hg

def CoinsReadSessions : Prop :=
  ∀ (A : Live₂) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (Good : Exchange → Prop),
    DevLocal A (locOf A tab) dev → InteriorLogged C Good →
      CoinsReadSome A (formReading A C (withCoins A C tab dev Ms) dev) (FormAccepts C) Good .sessions

theorem firstsOf_append : ∀ (acc xs ys : List ℕ), firstsOf acc (xs ++ ys) = firstsOf (firstsOf acc xs) ys
  | _, [], _ => rfl
  | acc, x :: xs, ys => by
    simp only [List.cons_append, firstsOf]
    exact firstsOf_append _ xs ys

theorem treesOpened_append (a b : List Release) :
    treesOpened (a ++ b) = firstsOf (treesOpened a) (b.filterMap fun r => openTree r.frame) := by
  unfold treesOpened
  rw [List.filterMap_append, firstsOf_append]

def Same (pre pre' : List Release) : Prop :=
  treesOpened pre = treesOpened pre' ∧ helloSpecs pre = helloSpecs pre' ∧
    ∀ i s, openedOf pre i s = openedOf pre' i s

theorem sessLoc_congr {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String} {pre pre' : List Release} (h : Same pre pre')
    (fr : Verity.Frame) : sessLoc off K M tab pre fr = sessLoc off K M tab pre' fr := by
  obtain ⟨h1, h2, h3⟩ := h
  cases fr with
  | fresh _ => rfl
  | hello _ => rfl
  | opening i s n => simp only [sessLoc, h1, h2, h3]

theorem Same.snoc {pre pre' : List Release} (h : Same pre pre') {r r' : Release} (hf : r.frame = r'.frame) :
    Same (pre ++ [r]) (pre' ++ [r']) := by
  obtain ⟨h1, h2, h3⟩ := h
  refine ⟨?_, ?_, fun i s => ?_⟩
  · rw [treesOpened_append, treesOpened_append, h1]
    simp only [List.filterMap_cons, List.filterMap_nil, hf]
  · unfold Verity.Sampled.helloSpecs at h2 ⊢
    rw [List.filterMap_append, List.filterMap_append, h2]
    simp only [List.filterMap_cons, List.filterMap_nil, hf]
  · unfold openedOf at h3 ⊢
    rw [List.countP_append, List.countP_append, h3 i s]
    simp only [List.countP_cons, List.countP_nil, hf]

theorem same_append {pre q : List Release} (hq : ∀ r ∈ q, openTree r.frame = none ∧ ∀ sp, r.frame ≠ .hello sp) :
    Same pre (pre ++ q) := by
  refine ⟨?_, ?_, fun i s => ?_⟩
  · have hnil : (q.filterMap fun r => openTree r.frame) = [] :=
      List.filterMap_eq_nil_iff.2 fun r hr => (hq r hr).1
    rw [treesOpened_append, hnil]
    rfl
  · have hnil : helloSpecs q = [] := by
      unfold Verity.Sampled.helloSpecs
      apply List.filterMap_eq_nil_iff.2
      intro r hr
      obtain ⟨-, h2⟩ := hq r hr
      cases hf : r.frame with
      | fresh _ => rfl
      | hello sp => exact absurd hf (h2 sp)
      | opening _ _ _ => rfl
    unfold Verity.Sampled.helloSpecs at hnil ⊢
    rw [List.filterMap_append, hnil, List.append_nil]
  · have hz : openedOf q i s = 0 := by
      unfold openedOf
      apply List.countP_eq_zero.2
      intro r hr
      obtain ⟨h1, -⟩ := hq r hr
      cases hf : r.frame with
      | fresh _ => simp [Verity.Frame.opensAt]
      | hello _ => simp [Verity.Frame.opensAt]
      | opening _ _ _ => rw [hf] at h1; cases h1
    unfold openedOf at hz ⊢
    rw [List.countP_append, hz, Nat.add_zero]

theorem located_append (loc : List Release → Verity.Frame → Option ℕ) (n : ℕ) :
    ∀ (pre a b : List Release), located loc n pre (a ++ b) = located loc n pre a ++ located loc n (pre ++ a) b
  | pre, [], b => by simp [located]
  | pre, r :: a, b => by
    simp only [List.cons_append, located, List.append_assoc]
    rw [located_append loc n (pre ++ [r]) a b]
    simp [List.append_assoc]

theorem located_none {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String} {n : ℕ} (q : List Release)
    (hq : ∀ r ∈ q, openTree r.frame = none) : ∀ pre, located (sessLoc off K M tab) n pre q = [] := by
  induction q with
  | nil => intro _; rfl
  | cons r q ih =>
    intro pre
    have h1 : ¬ sessLoc off K M tab pre r.frame = some n := by
      have hr := hq r (List.mem_cons.2 (Or.inl rfl))
      cases hf : r.frame with
      | fresh _ => simp [sessLoc]
      | hello _ => simp [sessLoc]
      | opening _ _ _ => rw [hf] at hr; cases hr
    show (if sessLoc off K M tab pre r.frame = some n then [(r.frame, r.coins)] else []) ++
      located (sessLoc off K M tab) n (pre ++ [r]) q = []
    rw [if_neg h1, List.nil_append]
    exact ih (fun x hx => hq x (List.mem_cons_of_mem _ hx)) _

theorem located_same {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String} {n : ℕ} {l l' : List Release}
    (hl : List.Forall₂ (fun r r' => r.frame = r'.frame ∧ (r.coins ≠ [] → r.coins = r'.coins)) l l') :
    ∀ {pre pre' : List Release}, Same pre pre' →
      (located (sessLoc off K M tab) n pre l).all (fun x => !x.2.isEmpty) = true →
      located (sessLoc off K M tab) n pre' l' = located (sessLoc off K M tab) n pre l := by
  induction hl with
  | nil => intro _ _ _ _; rfl
  | @cons r r' l l' hr _ ih =>
    intro pre pre' hS hall
    obtain ⟨hf, hc⟩ := hr
    simp only [located, List.all_append, Bool.and_eq_true] at hall ⊢
    rw [ih (hS.snoc hf) hall.2, ← sessLoc_congr hS r'.frame, ← hf]
    by_cases hloc : sessLoc off K M tab pre r.frame = some n
    · have hne : r.coins ≠ [] := by
        intro he
        simp [hloc, he] at hall
      rw [if_pos hloc, if_pos hloc, hc hne]
    · rw [if_neg hloc, if_neg hloc]

theorem of_ofV2 {r r' : Release} (h : (Read.ofV2 r).Of r') :
    r.frame = r'.frame ∧ (r.coins ≠ [] → r.coins = r'.coins) := by
  obtain ⟨hf, -, hc⟩ := h
  refine ⟨hf, fun hne => hc r.coins ?_⟩
  cases hcs : r.coins with
  | nil => exact absurd hcs hne
  | cons a as => simp [Read.ofV2, hcs]

theorem slotOf_some {h : SpotFlock.Held} {n t : ℕ} (hs : SpotFlock.slotOf h n = some t) :
    SpotFlock.off h t < n ∧ n ≤ SpotFlock.off h t + SpotFlock.k₀ h t := by
  unfold SpotFlock.slotOf at hs
  simpa using List.find?_some hs

theorem coinsRead_sessions : CoinsReadSessions := by
  intro A C Ms tab dev Good hdev hlog o hacc hgood i _ hg c hc
  have h0 : i ≠ 0 := by
    intro e
    (simp only [groupOf, if_pos e] at hg) <;> cases hg
  have h1 : ¬ i ≤ A.h.stage := by
    intro e
    (simp only [groupOf, if_neg h0, if_pos e] at hg) <;> cases hg
  have h2 : i ≠ A.h.stage + 1 := by
    intro e
    (simp only [groupOf, if_neg h0, if_neg h1, if_pos e] at hg) <;> cases hg
  have hc' : (formReading A C (withCoins A C tab dev Ms) dev).coin o i = some c := hc
  simp only [formReading, if_neg h0, dif_neg (show ¬ (i - 1 < A.h.stage) by omega), if_neg h2] at hc'
  cases hs : SpotFlock.slotOf A.h (i + 1) with
  | none => simp [hs] at hc'
  | some t =>
    simp only [hs] at hc'
    obtain ⟨hlt, -⟩ := slotOf_some hs
    have hn : SpotFlock.off A.h t + (i - SpotFlock.off A.h t) = i := by omega
    simp only [withCoins, hn] at hc'
    cases hl : interiorRead C o with
    | none => simp [hl] at hc'
    | some l =>
      simp only [hl, Option.bind] at hc'
      by_cases hall : (located (locOf A tab) i [] l).all (fun x => !x.2.isEmpty) = true
      · rw [if_pos hall] at hc'
        obtain ⟨q, l', hlog', hq, hR⟩ := hlog o hacc hgood l hl
        have hR' : List.Forall₂ (fun r r' => r.frame = r'.frame ∧ (r.coins ≠ [] → r.coins = r'.coins)) l l' :=
          (List.forall₂_map_left_iff.1 hR).imp fun _ _ h => of_ofV2 h
        show dev o.log i = some c
        rw [hdev i hg o.log l ?_, hc']
        unfold locOf at hall ⊢
        rw [hlog', located_append, located_none q (fun r hr => (hq r hr).1) []]
        simp only [List.nil_append]
        have hS : Same [] q := by simpa using same_append (pre := []) hq
        exact located_same hR' hS hall
      · rw [if_neg hall] at hc'
        cases hc'

def sessionRecords (C : CaseOf) (o : Exchange) : Option (List Json) := do
  let k ← C o
  let e ← (VStar.Recursive.SampledTwo.evidenceOf k.evidence).toOption
  (e.statements.toList.flatMap (·.sessions.toList)).mapM (Verity.Sampled.recordOn k.blobs)

def LocatedOwn (A : Live₂) (C : CaseOf) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l → ∀ t j, t < A.h.ms.length → j < SpotFlock.k₀ A.h t →
    ∀ x ∈ located (locOf A tab) (SpotFlock.off A.h t + j) [] l, ∃ recs rec,
      sessionRecords C o = some recs ∧ recs[t]? = some rec ∧
      (∃ i s n, x.1 = .opening i s n) ∧
      x.2 = ((Verity.Sampled.recordRound rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).1
        (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).2).map Verity.bitsOfBytes).getD []

def MembersAccepted (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l →
    ∃ (label : String) (after : ByteArray) (ξ : ByteArray) (lines : List String)
      (trees : Array Spot.Challenger.LoggedTree) (records : Array (String × Array (String × Option ℕ) × Json × String)),
      Spot.Challenger.coinTreesOf lines = .ok trees ∧
      Verity.Sampled.readLog label after (records.toList.map (·.2.2.1)) ξ lines = some l ∧
      Spot.Challenger.membersTreeReason trees lines records = none ∧
      sessionRecords C o = some (records.toList.map (·.2.2.1))

theorem off_mono {off K : ℕ → ℕ} (hs : ∀ t, off (t + 1) = off t + K t) {a b : ℕ} (h : a ≤ b) : off a ≤ off b := by
  induction b, h using Nat.le_induction with
  | base => exact le_rfl
  | succ b _ ih => rw [hs]; omega

theorem off_sep {off K : ℕ → ℕ} (hs : ∀ t, off (t + 1) = off t + K t) {t t' j j' : ℕ} (hj : j < K t)
    (hj' : j' < K t') (h : off t + j = off t' + j') : t = t' ∧ j = j' := by
  have lt : ∀ {a b ja jb : ℕ}, a < b → ja < K a → off a + ja < off b + jb := by
    intro a b ja jb hab hja
    have := off_mono hs (show a + 1 ≤ b from hab)
    rw [hs] at this
    omega
  rcases Nat.lt_trichotomy t t' with h1 | rfl | h1
  · exact absurd h (Nat.ne_of_lt (lt h1 hj))
  · exact ⟨rfl, by omega⟩
  · exact absurd h.symm (Nat.ne_of_lt (lt h1 hj'))

theorem roundOf_slot {tab : String} {R : ℕ} {name : Option String} {k j : ℕ} (h : roundOf tab R name k = some j) :
    name = some (slotStream tab R j).1 ∧ k = (slotStream tab R j).2 := by
  unfold roundOf at h
  split_ifs at h with h1 h2 h3
  · obtain ⟨rfl, rfl⟩ := h1
    cases h
    simp [slotStream]
  · obtain ⟨rfl, hk⟩ := h2
    cases h
    have hs : slotStream tab R (k + 1) = (repStreamName tab 0, k) := by
      unfold slotStream
      rw [if_neg (Nat.succ_ne_zero k), if_pos (by omega)]
      rfl
    rw [hs]
    exact ⟨rfl, rfl⟩
  · cases h
    have hs : slotStream tab R (R + 1 + k) = (repStreamName tab 1, k) := by
      unfold slotStream
      rw [if_neg (by omega), if_neg (by omega)]
      congr 1
      omega
    rw [hs]
    exact ⟨h3.1, rfl⟩

theorem roundOf_ne_last (tab : String) {K : ℕ} (hK : 2 ≤ K) (name : Option String) (k : ℕ) :
    roundOf tab (repsOf K) name k ≠ some (K - 1) := by
  intro h
  unfold roundOf at h
  by_cases h1 : name = some Spot.Challenger.TREE_SESSION ∧ k = 0
  · rw [ite_eq_left h1] at h
    have hj := Option.some.inj h
    omega
  · rw [ite_eq_right h1] at h
    by_cases h2 : name = some (repStreamName tab 0) ∧ k < repsOf K
    · rw [ite_eq_left h2] at h
      have hk := h2.2
      have hj := Option.some.inj h
      unfold repsOf at hk
      omega
    · rw [ite_eq_right h2] at h
      by_cases h3 : name = some (repStreamName tab 1) ∧ k < repsOf K
      · rw [ite_eq_left h3] at h
        have hk := h3.2
        have hj := Option.some.inj h
        unfold repsOf at hk hj
        omega
      · rw [ite_eq_right h3] at h
        cases h

theorem locOf_ne_last (A : Live₂) (tab : ℕ → String) (pre : List Release) (fr : Verity.Frame) {t : ℕ}
    (hK : 2 ≤ SpotFlock.k₀ A.h t) :
    locOf A tab pre fr ≠ some (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) := by
  intro h
  cases fr with
  | fresh _ => simp [locOf, sessLoc] at h
  | hello _ => simp [locOf, sessLoc] at h
  | opening i s c =>
    simp only [locOf, sessLoc] at h
    obtain ⟨j, hr, hif⟩ := Option.bind_eq_some_iff.1 h
    have hc : (treesOpened pre).idxOf i < A.h.ms.length ∧ j < SpotFlock.k₀ A.h ((treesOpened pre).idxOf i) := by
      by_contra hc
      rw [ite_eq_right hc] at hif
      cases hif
    rw [ite_eq_left hc] at hif
    have hlast : SpotFlock.k₀ A.h t - 1 < SpotFlock.k₀ A.h t := by omega
    obtain ⟨rfl, rfl⟩ := off_sep (SpotFlock.off_succ A.h) hc.2 hlast (Option.some.inj hif)
    exact roundOf_ne_last _ hK _ _ hr

theorem mem_firstsOf {x : ℕ} : ∀ (acc xs : List ℕ), x ∈ acc ∨ x ∈ xs → x ∈ firstsOf acc xs
  | acc, [], h => by simpa [firstsOf] using h
  | acc, i :: is, h => by
    simp only [firstsOf]
    apply mem_firstsOf
    rcases h with h | h
    · left
      split_ifs <;> simp [h]
    · rcases List.mem_cons.1 h with rfl | h
      · left
        split_ifs with hi <;> simp [hi]
      · exact Or.inr h

theorem mem_treesOpened {pre : List Release} {q : Release} (hq : q ∈ pre) {i s n : ℕ}
    (hf : q.frame = .opening i s n) : i ∈ treesOpened pre := by
  unfold treesOpened
  exact mem_firstsOf [] _ (Or.inr (List.mem_filterMap.2 ⟨q, hq, by simp [hf, openTree]⟩))

theorem treesOpened_snoc_open {pre : List Release} {r : Release} {i s n : ℕ} (hr : r.frame = .opening i s n) :
    treesOpened (pre ++ [r]) = if i ∈ treesOpened pre then treesOpened pre else treesOpened pre ++ [i] := by
  rw [treesOpened_append]
  simp only [List.filterMap_cons, List.filterMap_nil, hr, openTree]
  rfl

theorem treesOpened_snoc_none {pre : List Release} {r : Release} (hr : openTree r.frame = none) :
    treesOpened (pre ++ [r]) = treesOpened pre := by
  rw [treesOpened_append]
  simp only [List.filterMap_cons, List.filterMap_nil, hr]
  rfl

theorem idxOf_map_range {f : ℕ → ℕ} {N : ℕ} (hf : ∀ a b, a < N → b < N → f a = f b → a = b) :
    ∀ {m : ℕ}, m ≤ N → ∀ {j : ℕ}, j < m → ((List.range m).map f).idxOf (f j) = j := by
  intro m
  induction m with
  | zero => intro _ j hj; omega
  | succ m ih =>
    intro hm j hj
    rw [show List.range (m + 1) = List.range m ++ [m] from List.range_succ, List.map_append, List.idxOf_append]
    by_cases hjm : j < m
    · rw [if_pos (List.mem_map.2 ⟨j, List.mem_range.2 hjm, rfl⟩)]
      exact ih (by omega) hjm
    · have hj' : j = m := by omega
      rw [hj']
      have hn : f m ∉ (List.range m).map f := by
        intro hmem
        obtain ⟨a, ha, hfa⟩ := List.mem_map.1 hmem
        have ha' := List.mem_range.1 ha
        have := hf a m (by omega) (by omega) hfa
        omega
      rw [if_neg hn, List.map_singleton, List.idxOf_cons_self, List.length_map, List.length_range, Nat.zero_add]

theorem find?_of_first {α : Type} {p : α → Bool} : ∀ {l : List α} {t : ℕ} {a : α}, l[t]? = some a → p a = true →
    (∀ u b, u < t → l[u]? = some b → p b = false) → l.find? p = some a
  | [], _, _, h, _, _ => by simp at h
  | x :: l, 0, a, h, hp, _ => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp [hp]
  | x :: l, t + 1, a, h, hp, hb => by
    have hx : p x = false := hb 0 x (Nat.succ_pos _) (by simp)
    rw [List.find?_cons, hx]
    exact find?_of_first (by simpa using h) hp fun u b hu hub => hb (u + 1) b (by omega) (by simpa using hub)

theorem mem_located {loc : List Release → Verity.Frame → Option ℕ} {n : ℕ} :
    ∀ {pre l : List Release} {x : Verity.Frame × List Bool}, x ∈ located loc n pre l →
      ∃ a r b, l = a ++ r :: b ∧ loc (pre ++ a) r.frame = some n ∧ x = (r.frame, r.coins)
  | _, [], _, h => by simp [located] at h
  | pre, r :: l, x, h => by
    simp only [located, List.mem_append] at h
    rcases h with h | h
    · split_ifs at h with hl
      · exact ⟨[], r, l, rfl, by simpa using hl, by simpa using h⟩
      · simp at h
    · obtain ⟨a, r', b, rfl, hl, rfl⟩ := mem_located h
      exact ⟨r :: a, r', b, rfl, by simpa using hl, rfl⟩

theorem located_two {loc : List Release → Verity.Frame → Option ℕ} {n : ℕ} :
    ∀ {pre l : List Release}, 2 ≤ (located loc n pre l).length →
      ∃ a r b r' c, l = a ++ r :: (b ++ r' :: c) ∧ loc (pre ++ a) r.frame = some n ∧
        loc (pre ++ a ++ r :: b) r'.frame = some n
  | _, [], h => by simp [located] at h
  | pre, r :: l, h => by
    simp only [located, List.length_append] at h
    by_cases hr : loc pre r.frame = some n
    · rw [if_pos hr] at h
      obtain ⟨y, hy⟩ : ∃ y, y ∈ located loc n (pre ++ [r]) l := by
        cases hl : located loc n (pre ++ [r]) l with
        | nil => rw [hl] at h; simp at h
        | cons y _ => exact ⟨y, by simp⟩
      obtain ⟨b, r', c, rfl, hl, -⟩ := mem_located hy
      exact ⟨[], r, b, r', c, rfl, by simpa using hr, by simpa using hl⟩
    · rw [if_neg hr] at h
      obtain ⟨a, r₁, b, r₂, c, rfl, h1, h2⟩ := located_two (by simpa using h)
      exact ⟨r :: a, r₁, b, r₂, c, rfl, by simpa using h1, by simpa using h2⟩

structure MembersHeld (lines : List String) (trees : Array Spot.Challenger.LoggedTree) (log : List Release)
    (records : Array (String × Array (String × Option ℕ) × Json × String)) (tree : ℕ → ℕ) : Prop where
  nodup : (Verity.Sampled.MemberForm.usedOf records).toList.Nodup
  order : Spot.Challenger.membersOrder lines = some (Verity.Sampled.MemberForm.usedOf records)
  idx : ∀ j ρ, (Verity.Sampled.MemberForm.usedOf records)[j]? = some ρ →
    (Verity.Sampled.heldTrees trees).findIdx? (·.1 == ρ) = some (tree j)
  roots : ∀ (j : ℕ) (x : String × Array (String × Option ℕ) × Json × String), records[j]? = some x →
    ∃ ρ, Spot.getD (Spot.getD x.2.2.1 "coin_tree") "root" = Json.str ρ ∧
      (Verity.Sampled.MemberForm.usedOf records)[j]? = some ρ
  closes : ∀ j, j < records.size → ∃ Lc c L₂ s n, log = Lc ++ c :: L₂ ∧ c.frame = .opening (tree j) s n
  before : ∀ pre r post, log = pre ++ r :: post → ∀ j, j < records.size → ∀ s n, r.frame = .opening (tree j) s n →
    ∀ q ∈ pre, ∀ j', j' < records.size → ∀ s' n', q.frame = .opening (tree j') s' n' → j' ≤ j

section Held

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Release} {trees : Array Spot.Challenger.LoggedTree}
  {records : Array (String × Array (String × Option ℕ) × Json × String)} {tree : ℕ → ℕ}

theorem membersHeld_of (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log)
    (hM : Spot.Challenger.membersTreeReason trees lines records = none) :
    ∃ tree, MembersHeld lines trees log records tree := by
  obtain ⟨-, hnod, hO⟩ := Verity.Sampled.MemberForm.membersTreeReason_none hM
  obtain ⟨tree, h1, h2, -⟩ := Verity.Sampled.MemberForm.members_read hT hR hM
  refine ⟨tree, hnod, hO, fun j ρ hρ => ?_, fun j x hx => ?_, fun j hj => ?_, h2⟩
  · have hj : j < records.size := by
      have := (Array.getElem?_eq_some_iff.1 hρ).1
      rwa [Verity.Sampled.MemberForm.usedOf_size] at this
    obtain ⟨t, -, hroot, hfi, -⟩ := h1 j _ (Array.getElem?_eq_getElem hj)
    have hu := Verity.Sampled.MemberForm.usedOf_getElem? (Array.getElem?_eq_getElem hj)
    rw [hroot, Verity.Sampled.MemberForm.strOf_str, hρ] at hu
    rw [Option.some.inj hu]
    exact hfi
  · obtain ⟨t, -, hroot, -⟩ := h1 j x hx
    refine ⟨t.root, hroot, ?_⟩
    rw [Verity.Sampled.MemberForm.usedOf_getElem? hx, hroot, Verity.Sampled.MemberForm.strOf_str]
  · obtain ⟨t, -, -, -, -, Lc, c, L₂, s, n, he, hc, -⟩ := h1 j _ (Array.getElem?_eq_getElem hj)
    exact ⟨Lc, c, L₂, s, n, he, hc⟩

theorem MembersHeld.inj (H : MembersHeld lines trees log records tree) {a b : ℕ} (ha : a < records.size)
    (hb : b < records.size) (h : tree a = tree b) : a = b :=
  Verity.Sampled.MemberForm.tree_inj H.nodup H.idx (by rw [Verity.Sampled.MemberForm.usedOf_size]; exact ha)
    (by rw [Verity.Sampled.MemberForm.usedOf_size]; exact hb) h

theorem MembersHeld.opening (H : MembersHeld lines trees log records tree)
    (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log) {pre : List Release} {r : Release}
    {post : List Release} (he : log = pre ++ r :: post) {i s n : ℕ} (hr : r.frame = .opening i s n) :
    ∃ j, j < records.size ∧ i = tree j := by
  obtain ⟨l, ρ, name, k, hl, hro, hkey, -, -⟩ := Verity.Sampled.readLog_round hT hR he hr
  obtain ⟨a, b, hls⟩ := List.append_of_mem hl
  obtain ⟨bo, hbo⟩ := Verity.Sampled.MemberForm.memberLine_opensLine hro.1.opensLine
  obtain ⟨A, B, hused, -⟩ := Verity.Sampled.MemberForm.membersOrder_order H.order hls hbo
  have hmem : ρ ∈ (Verity.Sampled.MemberForm.usedOf records).toList := by
    rw [hused]
    simp
  obtain ⟨j, hj, hjρ⟩ := List.getElem_of_mem hmem
  have hρ : (Verity.Sampled.MemberForm.usedOf records)[j]? = some ρ := by
    rw [← Array.getElem?_toList, List.getElem?_eq_getElem hj, hjρ]
  have hj' : j < records.size := by
    rw [Array.length_toList, Verity.Sampled.MemberForm.usedOf_size] at hj
    exact hj
  exact ⟨j, hj', Option.some.inj ((Verity.Sampled.MemberForm.keyAt_idx hkey).symm.trans (H.idx j ρ hρ))⟩

theorem MembersHeld.earlier (H : MembersHeld lines trees log records tree) {pre : List Release} {r : Release}
    {post : List Release} (he : log = pre ++ r :: post) {j s n : ℕ} (hj : j < records.size)
    (hr : r.frame = .opening (tree j) s n) {j' : ℕ} (hlt : j' < j) :
    ∃ q ∈ pre, ∃ s' n', q.frame = .opening (tree j') s' n' := by
  have hj' : j' < records.size := by omega
  obtain ⟨Lc, c, L₂, sc, nc, hec, hcf⟩ := H.closes j' hj'
  have hne : c ≠ r := by
    intro e
    rw [e, hr] at hcf
    have := H.inj hj hj' (Verity.Frame.opening.inj hcf).1
    omega
  rcases List.append_eq_append_iff.1 (hec.symm.trans he) with ⟨a', hpre, ha⟩ | ⟨c', hLc, hc⟩
  · cases a' with
    | nil => exact absurd (List.cons.inj ha).1 hne
    | cons y a'' =>
      rw [List.cons_append] at ha
      obtain ⟨rfl, -⟩ := List.cons.inj ha
      exact ⟨_, by rw [hpre]; simp, sc, nc, hcf⟩
  · cases c' with
    | nil => exact absurd (List.cons.inj hc).1.symm hne
    | cons y c'' =>
      rw [List.cons_append] at hc
      obtain ⟨rfl, -⟩ := List.cons.inj hc
      have := H.before Lc c L₂ hec j' hj' sc nc hcf _ (by rw [hLc]; simp) j hj s n hr
      omega

theorem MembersHeld.le (H : MembersHeld lines trees log records tree) {pre : List Release} {r : Release}
    {post : List Release} (he : log = pre ++ r :: post) {j s n m : ℕ} (hj : j < records.size)
    (hr : r.frame = .opening (tree j) s n) (hm : treesOpened pre = (List.range m).map tree) : j ≤ m := by
  by_contra hlt
  obtain ⟨q, hq, s', n', hqf⟩ := H.earlier he hj hr (j' := m) (by omega)
  have hmem := mem_treesOpened hq hqf
  rw [hm] at hmem
  obtain ⟨a, ha, hfa⟩ := List.mem_map.1 hmem
  have ha' := List.mem_range.1 ha
  have := H.inj (by omega) (by omega) hfa
  omega

theorem MembersHeld.prefix (H : MembersHeld lines trees log records tree)
    (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log) :
    ∀ pre post, log = pre ++ post → ∃ m, m ≤ records.size ∧ treesOpened pre = (List.range m).map tree := by
  intro pre
  induction pre using List.reverseRecOn with
  | nil => intro _ _; exact ⟨0, Nat.zero_le _, rfl⟩
  | append_singleton pre r ih =>
    intro post he
    have he' : log = pre ++ r :: post := by simp [he]
    obtain ⟨m, hm, hpre⟩ := ih (r :: post) he'
    cases hf : openTree r.frame with
    | none => exact ⟨m, hm, by rw [treesOpened_snoc_none hf, hpre]⟩
    | some i =>
      obtain ⟨s, n, hr⟩ : ∃ s n, r.frame = .opening i s n := by
        cases hfr : r.frame with
        | opening i' s n =>
          rw [hfr] at hf
          simp only [openTree, Option.some.injEq] at hf
          subst hf
          exact ⟨s, n, rfl⟩
        | fresh _ => rw [hfr] at hf; simp [openTree] at hf
        | hello _ => rw [hfr] at hf; simp [openTree] at hf
      obtain ⟨j, hj, rfl⟩ := H.opening hT hR he' hr
      rw [treesOpened_snoc_open hr, hpre]
      by_cases hin : tree j ∈ (List.range m).map tree
      · exact ⟨m, hm, by rw [if_pos hin]⟩
      · have hjm : j = m := by
          have h1 := H.le he' hj hr hpre
          rcases Nat.lt_or_ge j m with h2 | h2
          · exact absurd (List.mem_map.2 ⟨j, List.mem_range.2 h2, rfl⟩) hin
          · omega
        refine ⟨m + 1, by omega, ?_⟩
        rw [if_neg hin, show List.range (m + 1) = List.range m ++ [m] from List.range_succ, List.map_append,
          List.map_singleton, hjm]

theorem MembersHeld.place (H : MembersHeld lines trees log records tree)
    (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log) {pre : List Release} {r : Release}
    {post : List Release} (he : log = pre ++ r :: post) {j s n : ℕ} (hj : j < records.size)
    (hr : r.frame = .opening (tree j) s n) : (treesOpened pre).idxOf (tree j) = j := by
  obtain ⟨m, hm, hpre⟩ := H.prefix hT hR pre (r :: post) he
  have hle := H.le he hj hr hpre
  rw [hpre]
  rcases Nat.lt_or_ge j m with h | h
  · exact idxOf_map_range (fun a b ha hb => H.inj ha hb) hm h
  · have hjm : j = m := by omega
    have hn : tree j ∉ (List.range m).map tree := by
      intro hmem
      obtain ⟨a, ha, hfa⟩ := List.mem_map.1 hmem
      have ha' := List.mem_range.1 ha
      have := H.inj (by omega) hj hfa
      omega
    rw [List.idxOf_eq_length hn, List.length_map, List.length_range, hjm]

theorem MembersHeld.find (H : MembersHeld lines trees log records tree)
    (hrecs : records.toList.map (·.2.2.1) = recs) {t : ℕ} {x : String × Array (String × Option ℕ) × Json × String}
    (hx : records[t]? = some x) {ρ : String} (hρ : (Verity.Sampled.MemberForm.usedOf records)[t]? = some ρ) :
    recs[t]? = some x.2.2.1 ∧ recs.find? (Verity.Sampled.recordTree · == ρ) = some x.2.2.1 := by
  have hrec : ∀ (u : ℕ) (b : Json), recs[u]? = some b →
      ∃ y : String × Array (String × Option ℕ) × Json × String, records[u]? = some y ∧ y.2.2.1 = b := by
    intro u b h
    rw [← hrecs, List.getElem?_map, Array.getElem?_toList] at h
    cases hy : records[u]? with
    | none => rw [hy] at h; simp at h
    | some y => rw [hy] at h; exact ⟨y, rfl, Option.some.inj h⟩
  have hroot : ∀ (u : ℕ) (y : String × Array (String × Option ℕ) × Json × String), records[u]? = some y →
      (Verity.Sampled.MemberForm.usedOf records)[u]? = some (Verity.Sampled.recordTree y.2.2.1) := by
    intro u y hy
    obtain ⟨ρu, hr, hu⟩ := H.roots u y hy
    rw [hu]
    unfold Verity.Sampled.recordTree
    rw [hr]
  have hget : recs[t]? = some x.2.2.1 := by simp [← hrecs, hx]
  refine ⟨hget, find?_of_first hget ?_ fun u b hu hub => ?_⟩
  · have := hroot t x hx
    rw [hρ] at this
    simp [← Option.some.inj this]
  · obtain ⟨y, hy, rfl⟩ := hrec u b hub
    have h1 := hroot u y hy
    simp only [beq_eq_false_iff_ne, ne_eq]
    intro hEq
    rw [hEq] at h1
    have := Verity.Sampled.MemberForm.nodup_idx H.nodup (by rw [Array.getElem?_toList]; exact h1)
      (by rw [Array.getElem?_toList]; exact hρ)
    omega

theorem MembersHeld.own (H : MembersHeld lines trees log records tree)
    (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log)
    (hrecs : records.toList.map (·.2.2.1) = recs) {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String}
    {pre : List Release} {x : Release} {post : List Release} (he : log = pre ++ x :: post) {n₀ : ℕ}
    (hloc : sessLoc off K M tab pre x.frame = some n₀) :
    ∃ t j i s c rec, t < M ∧ j < K t ∧ n₀ = off t + j ∧ x.frame = .opening i s c ∧ recs[t]? = some rec ∧
      x.coins = ((Verity.Sampled.recordRound rec (slotStream (tab t) (repsOf (K t)) j).1
        (slotStream (tab t) (repsOf (K t)) j).2).map Verity.bitsOfBytes).getD [] := by
  cases hx : x.frame with
  | fresh _ => rw [hx] at hloc; simp [sessLoc] at hloc
  | hello _ => rw [hx] at hloc; simp [sessLoc] at hloc
  | opening i s c =>
    rw [hx] at hloc
    simp only [sessLoc] at hloc
    obtain ⟨j, hj, hif⟩ := Option.bind_eq_some_iff.1 hloc
    have hc : (treesOpened pre).idxOf i < M ∧ j < K ((treesOpened pre).idxOf i) := by
      by_contra hc
      rw [if_neg hc] at hif
      cases hif
    rw [if_pos hc] at hif
    obtain ⟨htM, hjK⟩ := hc
    obtain ⟨t, ht, rfl⟩ := H.opening hT hR he hx
    have hplace := H.place hT hR he ht hx
    rw [hplace] at htM hjK hif hj
    obtain ⟨l, ρ, name, k, -, hro, hkey, hname, hk⟩ := Verity.Sampled.readLog_round hT hR he hx
    rw [hname] at hj
    obtain ⟨hn, hkk⟩ := roundOf_slot hj
    obtain ⟨y, hy⟩ : ∃ y, records[t]? = some y := ⟨_, Array.getElem?_eq_getElem ht⟩
    obtain ⟨ρt, -, hut⟩ := H.roots t y hy
    obtain ⟨sp, hsp⟩ := Verity.Sampled.MemberForm.keyAt_root hkey
    have eρ : ρ = ρt := Verity.Sampled.MemberForm.findIdx_fst (H.idx t ρt hut) hsp
    subst eρ
    obtain ⟨hget, hfind⟩ := H.find hrecs hy hut
    refine ⟨t, j, tree t, s, c, y.2.2.1, htM, hjK, (Option.some.inj hif).symm, rfl, hget, ?_⟩
    rw [hro.2.1, hk, hkk, ← Option.some.inj hn]
    unfold Verity.Sampled.Lines.openCoins
    simp only [hfind]

theorem MembersHeld.key (H : MembersHeld lines trees log records tree)
    (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log) {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String}
    {pre : List Release} {x : Release} {post : List Release} (he : log = pre ++ x :: post) {n₀ : ℕ}
    (hloc : sessLoc off K M tab pre x.frame = some n₀) :
    ∃ t j s c, t < records.size ∧ x.frame = .opening (tree t) s c ∧ j < K t ∧ n₀ = off t + j ∧
      ((helloSpecs pre)[tree t]?.bind fun sp => sp.streams[s]?) = some (slotStream (tab t) (repsOf (K t)) j).1 ∧
      openedOf pre (tree t) s = (slotStream (tab t) (repsOf (K t)) j).2 := by
  cases hx : x.frame with
  | fresh _ => rw [hx] at hloc; simp [sessLoc] at hloc
  | hello _ => rw [hx] at hloc; simp [sessLoc] at hloc
  | opening i s c =>
    rw [hx] at hloc
    simp only [sessLoc] at hloc
    obtain ⟨j, hj, hif⟩ := Option.bind_eq_some_iff.1 hloc
    have hc : (treesOpened pre).idxOf i < M ∧ j < K ((treesOpened pre).idxOf i) := by
      by_contra hc
      rw [if_neg hc] at hif
      cases hif
    rw [if_pos hc] at hif
    obtain ⟨-, hjK⟩ := hc
    obtain ⟨t, ht, rfl⟩ := H.opening hT hR he hx
    have hplace := H.place hT hR he ht hx
    rw [hplace] at hjK hif hj
    obtain ⟨hn, hkk⟩ := roundOf_slot hj
    exact ⟨t, j, s, c, ht, rfl, hjK, (Option.some.inj hif).symm, hn, hkk⟩

theorem MembersHeld.one (H : MembersHeld lines trees log records tree)
    (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log) {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String}
    (hs : ∀ t, off (t + 1) = off t + K t) (n : ℕ) : (located (sessLoc off K M tab) n [] log).length ≤ 1 := by
  by_contra hlen
  obtain ⟨a, r, b, r', c, he, h1, h2⟩ := located_two (show 2 ≤ (located (sessLoc off K M tab) n [] log).length by omega)
  rw [List.nil_append] at h1 h2
  have he2 : log = (a ++ r :: b) ++ r' :: c := by rw [he]; simp
  obtain ⟨t₁, j₁, s₁, c₁, -, hf₁, hj₁, hn₁, hname₁, hk₁⟩ := H.key hT hR he h1
  obtain ⟨t₂, j₂, s₂, c₂, -, -, hj₂, hn₂, hname₂, hk₂⟩ := H.key hT hR he2 h2
  obtain ⟨e1, e2⟩ := off_sep hs hj₁ hj₂ (hn₁.symm.trans hn₂)
  rw [← e1, ← e2] at hname₂ hk₂
  obtain ⟨sp, hsp, hs₁⟩ := Option.bind_eq_some_iff.1 hname₁
  have hsp₂ : (helloSpecs (a ++ r :: b))[tree t₁]? = some sp := by
    rw [Verity.Sampled.helloSpecs_append, List.getElem?_append_left (List.getElem?_eq_some_iff.1 hsp).1, hsp]
  rw [hsp₂, Option.bind_some] at hname₂
  have hmem : sp ∈ helloSpecs log := by
    rw [he, Verity.Sampled.helloSpecs_append]
    exact List.mem_append_left _ (List.mem_of_getElem? hsp)
  have hss : s₁ = s₂ := Verity.Sampled.MemberForm.nodup_idx (Verity.Sampled.readLog_nodup hT hR sp hmem)
    (by rw [Array.getElem?_toList]; exact hs₁) (by rw [Array.getElem?_toList]; exact hname₂)
  rw [← hss] at hk₂
  have hcount : openedOf (a ++ r :: b) (tree t₁) s₁ = openedOf a (tree t₁) s₁ + (openedOf b (tree t₁) s₁ + 1) := by
    unfold openedOf
    rw [List.countP_append, List.countP_cons, if_pos (by simp [hf₁, Verity.Frame.opensAt])]
  omega

end Held

theorem located_own (A : Live₂) (C : CaseOf) (tab : ℕ → String) (hM : MembersAccepted C) : LocatedOwn A C tab := by
  intro o hacc l hl t j _ hj x hx
  obtain ⟨label, after, ξ, lines, trees, records, hT, hR, hMR, hS⟩ := hM o hacc l hl
  obtain ⟨tree, H⟩ := membersHeld_of hT hR hMR
  obtain ⟨a, r, b, he, hloc, rfl⟩ := mem_located hx
  rw [List.nil_append] at hloc
  unfold locOf at hloc
  obtain ⟨t', j', i, s, c, rec, -, hj', hn, hrf, hrec, hcoins⟩ := H.own hT hR rfl he hloc
  obtain ⟨rfl, rfl⟩ := off_sep (off := SpotFlock.off A.h) (K := SpotFlock.k₀ A.h) (SpotFlock.off_succ A.h) hj hj' hn
  exact ⟨_, rec, hS, hrec, ⟨i, s, c, hrf⟩, hcoins⟩

theorem located_one (A : Live₂) (C : CaseOf) (tab : ℕ → String) (hM : MembersAccepted C) :
    ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l → ∀ n, (located (locOf A tab) n [] l).length ≤ 1 := by
  intro o hacc l hl n
  obtain ⟨label, after, ξ, lines, trees, records, hT, hR, hMR, -⟩ := hM o hacc l hl
  obtain ⟨tree, H⟩ := membersHeld_of hT hR hMR
  unfold locOf
  exact H.one hT hR (off := SpotFlock.off A.h) (K := SpotFlock.k₀ A.h) (SpotFlock.off_succ A.h) n

theorem coinsAgree_of_some {A : Live₂} {Rd : Reading A} {Acc Good : Exchange → Prop} {g : Group}
    (h : CoinsReadSome A Rd Acc Good g) : CoinsAgree A Rd Acc Good g :=
  fun o hA hG i hr hg c hc _ hd => hd c (h o hA hG i hr hg c hc)

theorem coinsAgree_sessions (A : Live₂) (C : CaseOf) (Ms : SessionReading) (tab : ℕ → String)
    (dev : List Release → ℕ → Option FlockDeploy.Coin) (Good : Exchange → Prop)
    (hdev : DevLocal A (locOf A tab) dev) (hlog : InteriorLogged C Good) :
    CoinsAgree A (formReading A C (withCoins A C tab dev Ms) dev) (FormAccepts C) Good .sessions :=
  coinsAgree_of_some (coinsRead_sessions A C Ms tab dev Good hdev hlog)

end Verity.SampledTwo.SessionCoins
