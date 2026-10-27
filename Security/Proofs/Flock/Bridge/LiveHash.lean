import Proofs.Flock.Bridge.SampledSound
import Proofs.Flock.Soundness.StrictCR

/-!
# The live session with its roots under their hash

v2 releases a session's link points after the SHA-512 of its tables' `root_B` digests (`OneStageZk.pointsAfter`), and
the record names the roots only at the end. So the live audit a play drives is the hash-first one: the prover sends
the roots' hash (`pointsHash`), the challenger draws the link points, the prover runs its rounds, then sends the roots
with the proof files (`liveSessionH`, `liveOfZJH`). Its writer `canonWrH` writes the raw session's record (`canonWr`)
where the roots it is sent hash to what it was sent first, and a refused transcript elsewhere (`canonWrH_spec`).

The raw session (`Tables.liveSession`, roots first) is unchanged, and so is everything proved over it. `adaptH τ` is
the raw strategy of a hash-first one `τ`: at each draw it sends first the roots `rootAt` that some play of `τ` with
positive probability ends with under the hash it sent (a lookahead over `τ`'s own plays), then plays as `τ` does.
Wherever `τ` ends with other roots under the same hash, they and `rootAt` give a SHA-512 collision
(`coll_of_points`), which `rootFinder τ`, one run of `τ`, outputs.

* `hash_le`: any event the writers carry over (`canonWrH_spec`) is at most as likely in the hash-first audit at `τ`
  as in the raw audit at `adaptH τ`, plus the probability that `rootFinder τ` outputs a SHA-512 collision.
* `LiveAt.GH`, `LiveAt.adapt`, `LiveAt.RootCR`, `LiveAt.CRH`, `LiveAt.boundH`, `LiveAt.soundFH`: `LiveAt.soundF` at
  the hash-first audit, its bound `deltaPub` at `adapt τ` plus `qS²/2^513` for the root finder (cr/sha-512 at the
  self-clash finder's budget).
-/

namespace Flock.Guarantees.LiveHash

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Composed
open FlockSoundness.Discharge.Tables (LiveOutJ liveRoundsJ liveSession KLJ)
open FlockSoundness.Discharge.ZkSession (TabZK)
open scoped ENNReal
open Classical

local notation "𝔽" => FlockLevel3.GF128

/-! ## The roots' hash -/

/-- The SHA-512 digests of the roots, in turn. -/
def rootDigests (rb : Array ByteArray) : List UInt8 :=
  (rb.toList.map fun b => (Flock.Sha512.hash b).data.toList).flatten

/-- **What the link points are released after**: the SHA-512 of the roots' digests (`OneStageZk.pointsAfter`). -/
def pointsHash (rb : Array ByteArray) : ByteArray := Flock.Sha512.hash ⟨(rootDigests rb).toArray⟩

theorem flatten_inj_of_length {k : ℕ} : ∀ (L L' : List (List UInt8)), (∀ l ∈ L, l.length = k) →
    (∀ l ∈ L', l.length = k) → 0 < k → L.flatten = L'.flatten → L = L'
  | [], [], _, _, _, _ => rfl
  | [], l :: L', _, h', hk, e => by
    have := congrArg List.length e
    simp only [List.flatten_nil, List.flatten_cons, List.length_nil, List.length_append, h' l List.mem_cons_self]
      at this
    omega
  | l :: L, [], h, _, hk, e => by
    have := congrArg List.length e
    simp only [List.flatten_nil, List.flatten_cons, List.length_nil, List.length_append, h l List.mem_cons_self]
      at this
    omega
  | l :: L, l' :: L', h, h', hk, e => by
    rw [List.flatten_cons, List.flatten_cons] at e
    obtain ⟨e1, e2⟩ := List.append_inj e (by rw [h l List.mem_cons_self, h' l' List.mem_cons_self])
    rw [e1, flatten_inj_of_length L L' (fun x hx => h x (List.mem_cons_of_mem _ hx))
      (fun x hx => h' x (List.mem_cons_of_mem _ hx)) hk e2]

theorem exists_ne_of_map_eq {β γ : Type} (g : β → γ) : ∀ u v : List β, u.map g = v.map g → u ≠ v →
    ∃ a b, a ≠ b ∧ g a = g b
  | [], [], _, h => (h rfl).elim
  | [], _ :: _, e, _ => by simp at e
  | _ :: _, [], e, _ => by simp at e
  | a :: u, b :: v, e, h => by
    simp only [List.map_cons, List.cons.injEq] at e
    by_cases hab : a = b
    · subst hab
      exact exists_ne_of_map_eq g u v e.2 fun huv => h (by rw [huv])
    · exact ⟨a, b, hab, e.1⟩

theorem bytes_toList_toArray (a : ByteArray) : (⟨a.data.toList.toArray⟩ : ByteArray) = a := by
  cases a
  simp

/-- **Other roots under the same hash are a SHA-512 collision**: of the digests' concatenations where they differ,
else of two roots at one place. -/
theorem coll_of_points {rb r0 : Array ByteArray} (hne : rb ≠ r0) (he : pointsHash rb = pointsHash r0) :
    Merkle.Collision H512 := by
  by_cases hd : rootDigests rb = rootDigests r0
  · have hlen : ∀ (r : Array ByteArray), ∀ l ∈ r.toList.map (fun b => (Flock.Sha512.hash b).data.toList),
        l.length = 64 := by
      intro r l hl
      obtain ⟨b, _, rfl⟩ := List.mem_map.1 hl
      rw [Array.length_toList]
      exact sha512_size b
    have hL := flatten_inj_of_length _ _ (hlen rb) (hlen r0) (by decide) hd
    have hne' : rb.toList ≠ r0.toList := fun h => hne (Array.toList_inj.mp h)
    obtain ⟨a, b, hab, hg⟩ := exists_ne_of_map_eq _ _ _ hL hne'
    refine ⟨a.data.toList, b.data.toList, fun h => hab ?_, Subtype.ext ?_⟩
    · rw [← bytes_toList_toArray a, ← bytes_toList_toArray b, h]
    · show Flock.Sha512.hash ⟨a.data.toList.toArray⟩ = Flock.Sha512.hash ⟨b.data.toList.toArray⟩
      rw [bytes_toList_toArray, bytes_toList_toArray]
      have hd' : (Flock.Sha512.hash a).data = (Flock.Sha512.hash b).data := Array.toList_inj.mp hg
      rw [← bytes_toList_toArray (Flock.Sha512.hash a), ← bytes_toList_toArray (Flock.Sha512.hash b)]
      simp only [hd']
  · exact ⟨rootDigests rb, rootDigests r0, hd, Subtype.ext he⟩

/-! ## The hash-first session -/

/-- What the prover sends last: the roots, then the proof files. -/
abbrev Last : Type := Array ByteArray × Array ByteArray

/-- The session's outcome after the hash: the link points, every stream's rounds, the roots and the proof files. -/
abbrev OutC (J mPts : ℕ) : Type := (Fin 2 → Fin mPts → 𝔽) × (Fin J → Fin 2 → List Flock.Round) × Last

/-- The hash-first session's outcome: the hash, then the rest. -/
abbrev OutH (J mPts : ℕ) : Type := ByteArray × OutC J mPts

/-- The continuation after the rounds: the roots and the proof files. -/
def KH {J mPts : ℕ} (pts : Fin 2 → Fin mPts → 𝔽) (rs : Fin J → Fin 2 → List Flock.Round) : Game (OutC J mPts) :=
  .send Last fun m => .ret (pts, rs, m)

/-- The session after the hash: the link points, the rounds, then the roots and the proof files. -/
def contH (dg : ByteArray → ByteArray) (J mPts N : ℕ) : Game (OutC J mPts) :=
  .coin (Fin 2 → Fin mPts → 𝔽) fun pts => (liveRoundsJ dg J N fun _ _ => []).bind (KH pts)

/-- **The hash-first live session of `J` tables**: the prover sends the roots' hash, the server draws the link
points, the prover runs its rounds, then sends the roots and the proof files. -/
def liveSessionH (dg : ByteArray → ByteArray) (J mPts N : ℕ) : Game (OutH J mPts) :=
  .send ByteArray fun h => (contH dg J mPts N).map (Prod.mk h)

/-- The raw outcome with the roots `r` in place of the ones sent last. -/
def rawAt {J mPts : ℕ} (r : Array ByteArray) (x : OutC J mPts) : LiveOutJ J mPts := ⟨r, x.1, x.2.1, x.2.2.2⟩

/-- The raw outcome at the roots sent last. -/
def OutH.raw {J mPts : ℕ} (o : OutH J mPts) : LiveOutJ J mPts := rawAt o.2.2.2.1 o.2

/-- **The hash-first audit's writer**: the raw writer's record (`canonWr`) of the raw outcome where the roots sent last
hash to what was sent first, else a transcript flock-verify refuses (`exists_refused`). -/
noncomputable def canonWrH {n : ℕ} {Reg : Type} (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (Jf : Finset (Fin n) → Reg → ℕ) (mPts : ℕ) (S : Finset (Fin n)) (R : Reg) (o : OutH (Jf S R) mPts) :
    Discharge.Exec.Transcript :=
  if pointsHash o.2.2.2.1 = o.1 then canonWr I dj Jf mPts S R o.raw else Classical.choose (exists_refused I)

/-- **What flock-verify accepts of `canonWrH`'s is the raw writer's at roots under the hash sent.** -/
theorem canonWrH_spec {n : ℕ} {Reg : Type} {I : Inputs} {dj : Finset (Fin n) → Lean.Json}
    {Jf : Finset (Fin n) → Reg → ℕ} {mPts : ℕ} {S : Finset (Fin n)} {R : Reg} {o : OutH (Jf S R) mPts} {C : Prop}
    (h : AcceptsZK I (canonWrH I dj Jf mPts S R o) ∧ C) :
    pointsHash o.2.2.2.1 = o.1 ∧ (AcceptsZK I (canonWr I dj Jf mPts S R o.raw) ∧ C) := by
  by_cases hp : pointsHash o.2.2.2.1 = o.1
  · have e : canonWrH I dj Jf mPts S R o = canonWr I dj Jf mPts S R o.raw := if_pos hp
    rw [e] at h
    exact ⟨hp, h⟩
  · have e : canonWrH I dj Jf mPts S R o = Classical.choose (exists_refused I) := if_neg hp
    rw [e] at h
    exact absurd h.1 (Classical.choose_spec (exists_refused I))

/-- **Where a record of the raw outcome is accepted, and its roots hash to what was sent first, the hash-first writer's
is accepted** (`accepts_canonWr`). -/
theorem accepts_canonWrH {n : ℕ} {Reg : Type} {I : Inputs} {dj : Finset (Fin n) → Lean.Json}
    {Jf : Finset (Fin n) → Reg → ℕ} {mPts : ℕ} {S : Finset (Fin n)} {R : Reg} {h : ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {rs : Fin (Jf S R) → Fin 2 → List Flock.Round} {rb pf : Array ByteArray}
    {t : Discharge.Exec.Transcript} (hr : RecordOfZKJ I dj (Jf S R) S ⟨rb, pts, rs, pf⟩ t)
    (hh : pointsHash rb = h) (ha : AcceptsZK I t) :
    AcceptsZK I (canonWrH I dj Jf mPts S R (h, pts, rs, (rb, pf))) := by
  have e : canonWrH I dj Jf mPts S R (h, pts, rs, (rb, pf)) = canonWr I dj Jf mPts S R ⟨rb, pts, rs, pf⟩ :=
    if_pos hh
  rw [e]
  exact accepts_canonWr hr ha

/-- **The hash-first live audit's session at a draw**, written out by `wrH`. -/
def liveOfZJH {n mPts : ℕ} {Reg : Type} (dg : ByteArray → ByteArray)
    (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (N : ℕ)
    (wrH : (S : Finset (Fin n)) → (R : Reg) → OutH (planZ S R).length mPts → Discharge.Exec.Transcript)
    (S : Finset (Fin n)) (R : Reg) : Game Discharge.Exec.Transcript :=
  (liveSessionH dg (planZ S R).length mPts N).map (wrH S R)

/-! ## From a hash-first strategy to a raw one -/

/-- A strategy of `g.bind f` as one of `g.bind f'`, leaf by leaf. -/
def transB {α β β' : Type} {f : α → Game β} {f' : α → Game β'} (T : ∀ a, Strategy (f a) → Strategy (f' a)) :
    (g : Game α) → Strategy (g.bind f) → Strategy (g.bind f')
  | .ret a, s => T a s
  | .send _ k, s => ⟨s.1, transB T (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => transB T (k c) (s c)

theorem prob_transB {α β β' : Type} {f : α → Game β} {f' : α → Game β'} (T : ∀ a, Strategy (f a) → Strategy (f' a))
    (p : β' → Prop) (p' : β → Prop) (hT : ∀ a s, prob p (f' a) (T a s) = prob p' (f a) s) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), prob p (g.bind f') (transB T g s) = prob p' (g.bind f) s
  | .ret a, s => hT a s
  | .send _ k, s => prob_transB T p p' hT (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => congrArg avg (funext fun c => prob_transB T p p' hT (k c) (s c))

theorem prob_mapS {α β : Type} (p : β → Prop) (f : α → β) : ∀ (g : Game α) (s : Strategy g),
    prob p (g.map f) (Strategy.map f g s) = prob (fun a => p (f a)) g s
  | .ret _, _ => rfl
  | .send _ k, s => prob_mapS p f (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => congrArg avg (funext fun c => prob_mapS p f (k c) (s c))

/-- **An event of positive probability has an outcome of positive probability in it.** -/
theorem exists_atom {α : Type} (P : α → Prop) : ∀ (g : Game α) (s : Strategy g), prob P g s ≠ 0 →
    ∃ a, P a ∧ prob (· = a) g s ≠ 0
  | .ret a, _, h => by
    refine ⟨a, ?_, ?_⟩
    · by_contra hP
      exact h (by simp [prob, hP])
    · simp [prob]
  | .send _ k, s, h => exists_atom P (k s.1) s.2 h
  | @Game.coin _ C _ _ k, s, h => by
    rw [prob_coin] at h
    unfold avg at h
    obtain ⟨c, -, hc⟩ := Finset.exists_ne_zero_of_sum_ne_zero h
    obtain ⟨a, hPa, ha⟩ := exists_atom P (k c) (s c) (right_ne_zero_of_mul hc)
    refine ⟨a, hPa, ?_⟩
    rw [prob_coin]
    unfold avg
    exact fun h0 => ha ((mul_eq_zero.1 ((Finset.sum_eq_zero_iff.1 h0) c (Finset.mem_univ c))).resolve_left
      (left_ne_zero_of_mul hc))

/-- The raw session's strategy that sends the roots `r` first, then plays `s` (the roots `s` sends last dropped). -/
def adaptS (dg : ByteArray → ByteArray) (J mPts N : ℕ) (r : Array ByteArray) (s : Strategy (contH dg J mPts N)) :
    Strategy (liveSession dg J mPts N) :=
  ⟨r, fun pts => transB (f := KH pts) (f' := KLJ r pts) (fun _ t => ⟨t.1.2, PUnit.unit⟩) _ (s pts)⟩

theorem prob_adaptS (dg : ByteArray → ByteArray) (J mPts N : ℕ) (p : LiveOutJ J mPts → Prop) (r : Array ByteArray)
    (s : Strategy (contH dg J mPts N)) :
    prob p (liveSession dg J mPts N) (adaptS dg J mPts N r s) =
      prob (fun x => p (rawAt r x)) (contH dg J mPts N) s :=
  congrArg avg (funext fun pts => prob_transB (f := KH pts) (f' := KLJ r pts) (fun _ t => ⟨t.1.2, PUnit.unit⟩) p
    (fun x => p (rawAt r x)) (fun _ _ => rfl) _ (s pts))

/-- **The roots a play of `s` ends with under the hash `h`**, with positive probability, where there are any. -/
noncomputable def rootOf {J mPts : ℕ} (g : Game (OutC J mPts)) (s : Strategy g) (h : ByteArray) : Array ByteArray :=
  if hx : ∃ r, prob (fun x : OutC J mPts => x.2.2.1 = r ∧ pointsHash r = h) g s ≠ 0 then Classical.choose hx else #[]

/-- The play ends with roots other than `r0` under `r0`'s hash. -/
def collAt {J mPts : ℕ} (r0 : Array ByteArray) (x : OutC J mPts) : Prop :=
  x.2.2.1 ≠ r0 ∧ pointsHash x.2.2.1 = pointsHash r0

/-- **The hash-first event, at the roots `rootOf` reads, up to a collision**: an event that holds only where the roots
sent last hash to `h` is at most as likely as its raw version at `rootOf g s h`, plus other roots under that hash. -/
theorem step_le {J mPts : ℕ} (g : Game (OutC J mPts)) (s : Strategy g) (h : ByteArray) (EH : OutC J mPts → Prop)
    (E : LiveOutJ J mPts → Prop) (hw : ∀ x, EH x → pointsHash x.2.2.1 = h ∧ E (rawAt x.2.2.1 x)) :
    prob EH g s ≤ prob (fun x => E (rawAt (rootOf g s h) x)) g s + prob (collAt (rootOf g s h)) g s := by
  by_cases hx : ∃ r, prob (fun x : OutC J mPts => x.2.2.1 = r ∧ pointsHash r = h) g s ≠ 0
  · have e : rootOf g s h = Classical.choose hx := dif_pos hx
    rw [e]
    have hr : pointsHash (Classical.choose hx) = h := by
      by_contra hne
      exact Classical.choose_spec hx (le_antisymm ((prob_mono (q := fun _ => False) (fun x hx' => hne hx'.2) g s).trans
        (prob_false g s).le) bot_le)
    refine (prob_mono (q := fun x => E (rawAt (Classical.choose hx) x) ∨ collAt (Classical.choose hx) x)
      (fun x hEx => ?_) g s).trans (prob_or_le _ _ g s)
    obtain ⟨hp, hE⟩ := hw x hEx
    by_cases hxr : x.2.2.1 = Classical.choose hx
    · left
      rw [← hxr]
      exact hE
    · exact Or.inr ⟨hxr, hp.trans hr.symm⟩
  · have h0 : prob EH g s = 0 := by
      by_contra hne
      obtain ⟨a, hEa, ha⟩ := exists_atom EH g s hne
      refine hx ⟨a.2.2.1, fun h0 => ha (le_antisymm ((prob_mono (q := fun x : OutC J mPts =>
        x.2.2.1 = a.2.2.1 ∧ pointsHash a.2.2.1 = h) (fun x hxa => ?_) g s).trans h0.le) bot_le)⟩
      subst hxa
      exact ⟨rfl, (hw _ hEa).1⟩
    rw [h0]
    exact bot_le

section Audit

variable {n mPts : ℕ} {Reg : Type} (L : Law n) (dg : ByteArray → ByteArray)
  (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (N : ℕ)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Discharge.Exec.Transcript)
  (wrH : (S : Finset (Fin n)) → (R : Reg) → OutH (planZ S R).length mPts → Discharge.Exec.Transcript)

/-- A hash-first audit strategy's session strategy at the draw `ω`. -/
noncomputable def sH (τ : Strategy (auditLive L Reg (liveOfZJH dg planZ N wrH))) (ω : L.Ω) :
    Strategy (liveSessionH dg (planZ (L.draw ω) τ.1).length mPts N) :=
  Strategy.ofMap (wrH (L.draw ω) τ.1) _ (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω))

/-- The hash it sends at the draw `ω`. -/
noncomputable def hOf (τ : Strategy (auditLive L Reg (liveOfZJH dg planZ N wrH))) (ω : L.Ω) : ByteArray :=
  (sH L dg planZ N wrH τ ω).1

/-- Its strategy after the hash at the draw `ω`. -/
noncomputable def s0 (τ : Strategy (auditLive L Reg (liveOfZJH dg planZ N wrH))) (ω : L.Ω) :
    Strategy (contH dg (planZ (L.draw ω) τ.1).length mPts N) :=
  Strategy.ofMap (Prod.mk (hOf L dg planZ N wrH τ ω)) _ (sH L dg planZ N wrH τ ω).2

/-- The roots the raw strategy sends first at the draw `ω` (`rootOf`). -/
noncomputable def rootAt (τ : Strategy (auditLive L Reg (liveOfZJH dg planZ N wrH))) (ω : L.Ω) : Array ByteArray :=
  rootOf _ (s0 L dg planZ N wrH τ ω) (hOf L dg planZ N wrH τ ω)

/-- **The raw audit's strategy of a hash-first one**: the same registration, and at each draw the roots `rootAt` sent
first, then the hash-first strategy's play. -/
noncomputable def adaptH (τ : Strategy (auditLive L Reg (liveOfZJH dg planZ N wrH))) :
    Strategy (auditLive L Reg (liveOfZJ dg planZ N wr)) :=
  ⟨τ.1, fun ω => Strategy.map (fun t => (τ.1, L.draw ω, t)) _ (Strategy.map (wr (L.draw ω) τ.1) _
    (adaptS dg _ mPts N (rootAt L dg planZ N wrH τ ω) (s0 L dg planZ N wrH τ ω)))⟩

/-- **The root finder**: one run of the hash-first strategy, its roots and `rootAt`'s as a SHA-512 collision where they
differ under one hash (`coll_of_points`). -/
noncomputable def rootFinder (τ : Strategy (auditLive L Reg (liveOfZJH dg planZ N wrH))) :
    StrictCR.Finder StrictCR.Pair :=
  ⟨.coin L.Ω fun ω => (contH dg (planZ (L.draw ω) τ.1).length mPts N).map
      (StrictCR.onePick H512 (collAt (rootAt L dg planZ N wrH τ ω)) fun _ hx => coll_of_points hx.1 hx.2),
    fun ω => Strategy.map _ _ (s0 L dg planZ N wrH τ ω)⟩

theorem prob_GH (E : Discharge.Exec.Transcript → Prop) (τ : Strategy (auditLive L Reg (liveOfZJH dg planZ N wrH))) :
    prob (fun o => E o.2.2) (auditLive L Reg (liveOfZJH dg planZ N wrH)) τ =
      avg fun ω => prob (fun x => E (wrH (L.draw ω) τ.1 (hOf L dg planZ N wrH τ ω, x)))
        (contH dg (planZ (L.draw ω) τ.1).length mPts N) (s0 L dg planZ N wrH τ ω) := by
  show avg (fun ω => prob (fun o => E o.2.2)
    ((liveOfZJH dg planZ N wrH (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω)) = _
  exact congrArg avg (funext fun ω =>
    Eq.trans (prob_map _ _ _ _) (Eq.trans (prob_map _ _ _ _) (Eq.trans (prob_send _ _ _ _) (prob_map _ _ _ _))))

theorem prob_adaptH (E : Discharge.Exec.Transcript → Prop)
    (τ : Strategy (auditLive L Reg (liveOfZJH dg planZ N wrH))) :
    prob (fun o => E o.2.2) (auditLive L Reg (liveOfZJ dg planZ N wr)) (adaptH L dg planZ N wr wrH τ) =
      avg fun ω => prob (fun x => E (wr (L.draw ω) τ.1 (rawAt (rootAt L dg planZ N wrH τ ω) x)))
        (contH dg (planZ (L.draw ω) τ.1).length mPts N) (s0 L dg planZ N wrH τ ω) := by
  show avg (fun ω => prob (fun o => E o.2.2)
    ((liveOfZJ dg planZ N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t))
      (Strategy.map (fun t => (τ.1, L.draw ω, t)) _ (Strategy.map (wr (L.draw ω) τ.1) _
        (adaptS dg _ mPts N (rootAt L dg planZ N wrH τ ω) (s0 L dg planZ N wrH τ ω))))) = _
  exact congrArg avg (funext fun ω =>
    Eq.trans (prob_mapS _ _ _ _) (Eq.trans (prob_mapS _ _ _ _) (prob_adaptS _ _ _ _ _ _ _)))

theorem collAt_le_root (τ : Strategy (auditLive L Reg (liveOfZJH dg planZ N wrH))) :
    (avg fun ω => prob (collAt (rootAt L dg planZ N wrH τ ω)) (contH dg (planZ (L.draw ω) τ.1).length mPts N)
      (s0 L dg planZ N wrH τ ω)) ≤
      prob (StrictCR.IsColl H512) (rootFinder L dg planZ N wrH τ).g (rootFinder L dg planZ N wrH τ).s := by
  show _ ≤ avg fun ω => prob (StrictCR.IsColl H512) ((contH dg (planZ (L.draw ω) τ.1).length mPts N).map
      (StrictCR.onePick H512 (collAt (rootAt L dg planZ N wrH τ ω)) fun _ hx => coll_of_points hx.1 hx.2))
    (Strategy.map _ _ (s0 L dg planZ N wrH τ ω))
  refine avg_mono fun ω => ?_
  rw [prob_mapS]
  exact prob_mono (fun x hx => StrictCR.collOf_isColl H512 _ hx) _ _

/-- **The hash-first audit, up to a root collision**: an event of the transcript that the writers carry over (where
`wrH`'s is in it, the roots sent last hash to what was sent first and `wr`'s raw one is in it) is at most as likely
in the hash-first audit at `τ` as in the raw audit at `adaptH τ`, plus the root finder's collision. -/
theorem hash_le (E : Discharge.Exec.Transcript → Prop)
    (hw : ∀ S R (o : OutH (planZ S R).length mPts), E (wrH S R o) → pointsHash o.2.2.2.1 = o.1 ∧ E (wr S R o.raw))
    (τ : Strategy (auditLive L Reg (liveOfZJH dg planZ N wrH))) :
    prob (fun o => E o.2.2) (auditLive L Reg (liveOfZJH dg planZ N wrH)) τ ≤
      prob (fun o => E o.2.2) (auditLive L Reg (liveOfZJ dg planZ N wr)) (adaptH L dg planZ N wr wrH τ) +
        prob (StrictCR.IsColl H512) (rootFinder L dg planZ N wrH τ).g (rootFinder L dg planZ N wrH τ).s := by
  rw [prob_GH, prob_adaptH]
  refine (avg_mono fun ω => step_le (contH dg (planZ (L.draw ω) τ.1).length mPts N) (s0 L dg planZ N wrH τ ω)
    (hOf L dg planZ N wrH τ ω) (fun x => E (wrH (L.draw ω) τ.1 (hOf L dg planZ N wrH τ ω, x)))
    (fun o => E (wr (L.draw ω) τ.1 o)) fun x hx => hw (L.draw ω) τ.1 (hOf L dg planZ N wrH τ ω, x) hx).trans ?_
  rw [avg_add]
  exact add_le_add le_rfl (collAt_le_root L dg planZ N wrH τ)

end Audit

/-- **cr/sha-512 at a finder bounds its collision's probability.** -/
theorem coll_le_of_CR {Fd : StrictCR.Finder StrictCR.Pair} {q : ℝ} (h : Fd.CR H512 q) :
    prob (StrictCR.IsColl H512) Fd.g Fd.s ≤ ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  have h' : (prob (StrictCR.IsColl H512) Fd.g Fd.s).toReal ≤ q ^ 2 / 2 ^ (513 : ℝ) := h fun _ => le_rfl
  rw [← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top
    (prob_le_one (StrictCR.IsColl H512) Fd.g Fd.s))]
  exact ENNReal.ofReal_le_ofReal h'

end Flock.Guarantees.LiveHash

namespace Flock.Guarantees.LiveAt

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden Flock.Guarantees.LiveHash
open scoped ENNReal
open Classical

variable {n : ℕ} {D : Flock.Desc} {hU : Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}
  (A : LiveAt n D hU kd K₀ desc)

/-- The hash-first audit's writer (`canonWrH`). -/
noncomputable abbrev wrH : (S : Finset (Fin n)) → (R : A.Reg) →
    OutH (planZHJ (Reg := A.Reg) A.dj A.y₀ S R).length (mPtsOf D.unroll.g (kd / nTab A.I)) →
      Discharge.Exec.Transcript :=
  canonWrH A.I A.dj (fun S R => (planZHJ (Reg := A.Reg) A.dj A.y₀ S R).length) (mPtsOf D.unroll.g (kd / nTab A.I))

/-- **The hash-first live audit**: the one a play drives, the roots under their hash until the proof files. -/
noncomputable abbrev GH := auditLive A.L A.Reg (liveOfZJH A.dg (planZHJ A.dj A.y₀) A.N A.wrH)

/-- The raw audit's strategy of a hash-first one (`adaptH`). -/
noncomputable def adapt (τ : Strategy A.GH) : Strategy A.G :=
  adaptH A.L A.dg (planZHJ A.dj A.y₀) A.N A.wr A.wrH τ

/-- The root finder of a hash-first strategy (`rootFinder`). -/
noncomputable def rootF (τ : Strategy A.GH) : StrictCR.Finder StrictCR.Pair :=
  rootFinder A.L A.dg (planZHJ A.dj A.y₀) A.N A.wrH τ

/-- **cr/sha-512 at the root finder**, at the self-clash finder's budget `qS`: one run of the strategy and one further
verification. -/
def RootCR (τ : Strategy A.GH) : Prop := (A.rootF τ).CR H512 (A.qS (A.adapt τ))

/-- **The collision-resistance premises at a hash-first strategy**: C-Flock's at its raw strategy, and the root
finder's. -/
def CRH (τ : Strategy A.GH) : Prop := A.CR (A.adapt τ) ∧ A.RootCR τ

/-- **The bound at a hash-first strategy**: `deltaPub` at its raw strategy plus the root finder's `qS²/2^513`, where
the premises hold, `⊤` elsewhere. -/
noncomputable def boundH (τ : Strategy A.GH) : ℝ≥0∞ :=
  if A.CRH τ then
    ENNReal.ofReal (Bridge.deltaPub n kd K₀ A.k (A.qF (A.adapt τ)) (A.qS (A.adapt τ)) (A.t' (A.adapt τ)) A.M A.Rw
      (Bridge.ioSum A.P) A.ρ) + ENNReal.ofReal (A.qS (A.adapt τ) ^ 2 / 2 ^ (513 : ℝ))
  else ⊤

/-- **`soundF` at the hash-first audit**: at every hash-first prover `τ`, flock-verify accepts while more than an `ε`
share of the units are faulty under the extractor's witness at `adapt τ` with probability at most `boundH τ`, where
`K₀ − 1 ≤ ε n`. -/
theorem soundFH (q : Claims.Sampling.Query) (ε : ℝ) (hε : (K₀ : ℝ) - 1 ≤ ε * n) (τ : Strategy A.GH) :
    prob (fun o => LiveAcceptsZK A.I o ∧ ¬ A.HoldsF q ε (A.adapt τ)) A.GH τ ≤ A.boundH τ := by
  unfold boundH
  split_ifs with hCR
  · refine (hash_le A.L A.dg (planZHJ A.dj A.y₀) A.N A.wr A.wrH
      (fun t => AcceptsZK A.I t ∧ ¬ A.HoldsF q ε (A.adapt τ)) (fun _ _ _ h => canonWrH_spec h) τ).trans
      (add_le_add ?_ (coll_le_of_CR hCR.2))
    have h := A.soundF q ε hε (A.adapt τ)
    rw [LiveAt.bound, if_pos hCR.1] at h
    exact h
  · exact le_top

end Flock.Guarantees.LiveAt
