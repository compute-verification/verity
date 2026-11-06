import Proofs.Flock.VBridge.Family.Assembly
import Proofs.Flock.VBridge.Family.Lig
import Proofs.Flock.VBridge.Runs.ZcLc
import Proofs.Flock.Soundness.Refine.Seam

/-!
# VBridge G at V*'s family: every level's openings pass

`OpenedAll` (`hopen` of `vbridgeRead_assembled`) asks, at each rep's proof file, that every level's queries open to the
level's cap (`RecAlgebra.Opened`, Ligerito.lean:148-158) under the setup's scheme `st.merkle`, which `SetupRealizes`
pins to `plain` (`flock-leaf/sha512-unsalted`, the scheme the inner statements name). V*'s openings are the recursion's
open circuit `RecOpen` (rec_open.py:37, rec_vstar.py:15-19): it climbs from the plain leaf `SHA-512(row)` and has no salt
port. At hm96 `OpenedAll` would be false at V*: with no salts, hm96's check fails at its shape (`not_opened_hm96`: one
salt per query); with salts, hm96's leaf is not V*'s. This file proves the check at the scheme V* climbs, `plain`:

* `Conss`: the verifier's Ligerito run (`RecAlgebra.ligerito`) pushes one consistency context per level, in level order,
  each with the level's `q` positions (`ligerito_conss`).
* `climb_of_layer`: an opening unit's `out` columns, in the layer, are the climb (`Flock.climb`) from the plain leaf of
  its `row` columns through its `aux` columns at the position its `dirs` columns spell (`keyProg_parse`).
* `openedAllAt_plain`: `OpenedAll` at `plain` at every accepted outer proof, from the openings `lig` holds being the
  layers' (`LigOpens`), the `dirs` columns being the verifier's positions (`HDirCols`), the `out` columns being the
  cap's entries (`HTopsL`), the caps' shape (`CapsAt`) and the openings' constants (`HOneOpen`).
* `hopen_of_plain`: `vbridgeRead_assembled`'s `hopen`, at a setup whose scheme is `plain`.
* `ligOpens_unsalted`, `capsAt_capOf`: `LigOpens` and `CapsAt` at the lig lane's `capOf` and `ligOfRead` (Lig.lean) at
  `plain`'s layout, no salts (`sl` `none` or `some 0`; `ligOf`'s `some 192` is hm96's), and `hopen_of_plain_lig`:
  `hopen` there, from `HOneOpen`, `HDirCols` and `HTopsL`.
-/

namespace FlockVBridge.Family

namespace Conss

open Flock Flock.RecAlgebra

/-! ## The consistency contexts of the verifier's Ligerito run -/

/-- The level and query count of each of a Ligerito state's consistency contexts. -/
def startsOf (st : LSt Lin) : List (Nat × Nat) := st.conss.toList.map fun c => (c.start, c.pos.size)

/-- `x` appends `l` to the state's `startsOf` and returns a value with `P`, wherever it succeeds. -/
def CS {A : Type} (l : List (Nat × Nat)) (P : A → Prop) (x : LM Lin A) : Prop :=
  ∀ st k src a st' k', x st k src = .ok ((a, st'), k') → startsOf st' = startsOf st ++ l ∧ P a

namespace CS

variable {A B : Type} {l l₁ l₂ : List (Nat × Nat)} {P : A → Prop} {Q : B → Prop}

theorem bindL {x : LM Lin A} {f : A → LM Lin B} (hx : CS l₁ P x) (hf : ∀ a, P a → CS l₂ Q (f a))
    (hl : l₁ ++ l₂ = l) : CS l Q (x >>= f) := by
  intro st k src b st' k' h
  obtain ⟨⟨a, st₁⟩, k₁, h₁, h₂⟩ := cm_bind (x := x st) (f := fun p : A × LSt Lin => f p.1 p.2) h
  obtain ⟨e₁, ha⟩ := hx st k src a st₁ k₁ h₁
  obtain ⟨e₂, hb⟩ := hf a ha st₁ k₁ src b st' k' h₂
  exact ⟨by rw [e₂, e₁, List.append_assoc, hl], hb⟩

theorem bind0 {x : LM Lin A} {f : A → LM Lin B} (hx : CS [] P x) (hf : ∀ a, P a → CS l Q (f a)) :
    CS l Q (x >>= f) :=
  bindL hx hf (List.nil_append _)

theorem lift {x : CM A} : CS [] (fun _ => True) (StateT.lift x : LM Lin A) := by
  intro st k src a st' k' h
  obtain ⟨a', k₁, -, h₂⟩ := cm_bind (x := x) (f := fun a => (Pure.pure (a, st) : CM (A × LSt Lin))) h
  cases cm_pure h₂
  exact ⟨(List.append_nil _).symm, trivial⟩

theorem pure' {a : A} (h : P a) : CS [] P (Pure.pure a : LM Lin A) := by
  intro st k src a' st' k' hx
  cases hx
  exact ⟨(List.append_nil _).symm, h⟩

theorem throw' (e : String) : CS l P (throw e : LM Lin A) := by
  intro st k src a st' k' hx
  cases hx

theorem ite {c : Prop} [Decidable c] {x₁ x₂ : LM Lin A} (h₁ : c → CS l P x₁) (h₂ : ¬c → CS l P x₂) :
    CS l P (if c then x₁ else x₂) := by
  split
  · rename_i hc; exact h₁ hc
  · rename_i hc; exact h₂ hc

theorem get_bind {g : LSt Lin → LM Lin A}
    (h : ∀ s k src a st' k', g s s k src = .ok ((a, st'), k') → startsOf st' = startsOf s ++ l ∧ P a) :
    CS l P (get >>= g) :=
  fun st k src a st' k' hx => h st k src a st' k' hx

theorem set_bind {s₀ s' : LSt Lin} {f : PUnit → LM Lin A} (he : startsOf s' = startsOf s₀) (hf : CS l P (f ⟨⟩)) :
    ∀ k src a st' k', (set s' >>= f) s₀ k src = .ok ((a, st'), k') → startsOf st' = startsOf s₀ ++ l ∧ P a := by
  intro k src a st' k' hx
  obtain ⟨e, hp⟩ := hf s' k src a st' k' hx
  exact ⟨by rw [e, he], hp⟩

theorem modify {g : LSt Lin → LSt Lin} (h : ∀ s, startsOf (g s) = startsOf s ++ l) :
    CS l (fun _ => True) (modify g : LM Lin PUnit) := by
  intro st k src a st' k' hx
  cases hx
  exact ⟨h st, trivial⟩

theorem keep {g : LSt Lin → LSt Lin} (h : ∀ s, (g s).conss = s.conss) :
    CS [] (fun _ => True) (StateT.modifyGet (fun s => (PUnit.unit, g s)) : LM Lin PUnit) :=
  modify fun s => by rw [List.append_nil, startsOf, startsOf, h]

end CS

section Run

variable (M : Messages Lin)

theorem nextMsg_cs : CS [] (fun _ => True) (Flock.RecAlgebra.nextMsg M) := by
  unfold Flock.RecAlgebra.nextMsg
  refine CS.get_bind fun s => ?_
  cases M.msgs[s.msg]? with
  | none => exact CS.throw' _ s
  | some mm => exact CS.set_bind rfl (CS.pure' trivial)

theorem nextOod_cs : CS [] (fun _ => True) (Flock.RecAlgebra.nextOod M) := by
  unfold Flock.RecAlgebra.nextOod
  refine CS.get_bind fun s => ?_
  cases M.ood[s.ood]? with
  | none => exact CS.throw' _ s
  | some y => exact CS.set_bind rfl (CS.pure' trivial)

theorem challenge_cs : CS [] (fun _ => True) (Flock.RecAlgebra.challenge M) := by
  unfold Flock.RecAlgebra.challenge
  refine CS.bind0 CS.lift fun rk _ => CS.get_bind fun s => ?_
  obtain ⟨s, ⟨a0, a1, a2⟩, msg, ood, oods, conss⟩ := s
  exact CS.set_bind rfl (CS.bind0 (nextMsg_cs M) fun _ _ =>
    CS.bind0 (CS.keep fun _ => rfl) fun _ _ => CS.pure' trivial)

theorem challenges_cs : ∀ n, CS [] (fun _ => True) (challenges M n)
  | 0 => CS.pure' trivial
  | n + 1 => by
    simp only [challenges]
    exact CS.bind0 (challenge_cs M) fun _ _ => CS.bind0 (challenges_cs n) fun _ _ => CS.pure' trivial

theorem oodIntro_cs (dim split : Nat) : CS [] (fun _ => True) (Flock.RecAlgebra.oodIntro M dim split) := by
  unfold Flock.RecAlgebra.oodIntro
  exact CS.bind0 CS.lift fun _ _ => CS.bind0 (nextOod_cs M) fun _ _ => CS.bind0 (nextMsg_cs M) fun _ _ =>
    CS.bind0 CS.lift fun _ _ => CS.bind0 (x := addScaled _ _) (CS.keep fun _ => rfl) fun _ _ =>
      CS.keep fun _ => rfl

variable {sch : Schedule}

theorem query_cs (i : Nat) (fold : Array F256)
    (hq : ∀ c, (Flock.positions (sch.levels[i]!) c).size = (sch.levels[i]!).q) :
    CS [] (fun r => r.1.size = (sch.levels[i]!).q) (Flock.RecAlgebra.query sch M i fold) := by
  unfold Flock.RecAlgebra.query
  exact CS.bind0 CS.lift fun c _ => CS.bind0 CS.lift fun _ _ => CS.pure' (hq c)

theorem queryFold_cs (i cols : Nat) (fold : Array F256)
    (hq : ∀ c, (Flock.positions (sch.levels[i]!) c).size = (sch.levels[i]!).q) :
    CS [(i, (sch.levels[i]!).q)] (fun _ => True) (queryFold sch M i cols fold) := by
  unfold queryFold
  exact CS.bind0 (query_cs M i fold hq) fun ⟨pos, al, e⟩ hpos =>
    CS.bind0 (nextMsg_cs M) fun _ _ => CS.bind0 CS.lift fun _ _ =>
      CS.bind0 (x := addScaled _ _) (CS.keep fun _ => rfl) fun _ _ =>
        CS.modify fun s => by simp only at hpos; simp [startsOf, hpos]

theorem levelsLoop_cs (R : Nat)
    (hq : ∀ i, i ≤ R → ∀ c, (Flock.positions (sch.levels[i]!) c).size = (sch.levels[i]!).q) :
    ∀ fuel i split (levels : Array (Array F256)) ext, i + fuel ≤ R →
      CS ((List.range' (i + 1) fuel).map fun k => (k, (sch.levels[k]!).q)) (fun _ => True)
        (levelsLoop sch M R fuel i split levels ext)
  | 0, _, _, _, _, _ => CS.pure' trivial
  | fuel + 1, i, split, levels, ext, hf => by
    simp only [levelsLoop]
    refine CS.ite (fun _ => CS.throw' _) fun _ => ?_
    refine CS.bind0 (challenges_cs M 4) fun chs _ => ?_
    by_cases hc : i + 1 < R
    · rw [ite_eq_left hc]
      exact CS.bind0 (oodIntro_cs M _ _) fun _ _ => CS.bind0 (oodIntro_cs M _ _) fun _ _ =>
        CS.bindL (queryFold_cs M _ _ _ (hq _ (by omega)))
          (fun _ _ => levelsLoop_cs R hq fuel _ _ _ _ (by omega)) rfl
    · rw [ite_eq_right hc]
      have hR : R = i + 1 := by omega
      have e2 : fuel = 0 := by omega
      exact CS.bind0 (query_cs M R chs (hq R (Nat.le_refl R))) fun ⟨pos, al, e⟩ hpos =>
        CS.bind0 CS.lift fun _ _ =>
          CS.bindL (CS.modify (l := [(R, (sch.levels[R]!).q)]) fun s => by
              simp only at hpos; simp [startsOf, hpos])
            (fun _ _ => levelsLoop_cs R hq fuel _ _ _ _ (by omega)) (by subst e2; rw [hR]; rfl)

theorem range_eq_cons {n : Nat} (h : 0 < n) : List.range n = 0 :: List.range' 1 (n - 1) := by
  cases n with
  | zero => omega
  | succ m => rw [List.range_eq_range', List.range'_succ, Nat.add_sub_cancel]

theorem ligeritoRun_cs (h0 : 0 < sch.levels.size)
    (hq : ∀ i < sch.levels.size, ∀ c, (Flock.positions (sch.levels[i]!) c).size = (sch.levels[i]!).q) :
    CS ((List.range sch.levels.size).map fun k => (k, (sch.levels[k]!).q)) (fun _ => True) (ligeritoRun sch M) := by
  unfold ligeritoRun
  refine CS.bind0 CS.lift fun _ _ => CS.bind0 (nextOod_cs M) fun _ _ => CS.bind0 CS.lift fun _ _ => ?_
  refine CS.bind0 (CS.keep fun _ => rfl) fun _ _ => CS.bind0 (nextMsg_cs M) fun _ _ =>
    CS.bind0 (CS.keep fun _ => rfl) fun _ _ => CS.bind0 (challenges_cs M _) fun _ _ => ?_
  exact CS.bind0 (oodIntro_cs M _ _) fun _ _ => CS.bind0 (oodIntro_cs M _ _) fun _ _ =>
    CS.bindL (queryFold_cs M 0 _ _ (hq 0 h0))
      (fun _ _ => CS.bindL (levelsLoop_cs M _ (fun i hi => hq i (by omega)) _ 0 _ _ _ (by omega))
        (fun ⟨_, _⟩ _ => CS.pure' trivial) (List.append_nil _))
      (by rw [range_eq_cons h0]; rfl)

/-- **The consistency contexts of a Ligerito run are its levels, in order** (RecAlgebra.lean: `ligeritoRun` pushes
level 0's at `queryFold 0`, `levelsLoop` level `i + 1`'s at `queryFold (i + 1)` and the last level's after `query`):
context `i` is level `i`'s, with the level's `q` positions. -/
theorem ligerito_conss {tIn : Lin} {sws : Array Switched} {k k' : Nat} {src : Flock.RecAlgebra.CoinSource} {lgr : Array Lin}
    {lg : LgPts} (h : Flock.RecAlgebra.ligerito sch M tIn sws k src = .ok ((lgr, lg), k')) (h0 : 0 < sch.levels.size)
    (hq : ∀ i < sch.levels.size, ∀ c, (Flock.positions (sch.levels[i]!) c).size = (sch.levels[i]!).q) :
    lg.conss.toList.map (fun c => (c.start, c.pos.size)) =
      (List.range sch.levels.size).map fun k => (k, (sch.levels[k]!).q) := by
  unfold Flock.RecAlgebra.ligerito at h
  obtain ⟨⟨⟨initCh, levels, ext⟩, st⟩, k₁, h1, h⟩ := cm_bind h
  have e := (ligeritoRun_cs M h0 hq _ k src _ st k₁ h1).1
  obtain ⟨a, k₂, ha, h⟩ := cm_bind h
  cases ha
  obtain ⟨b, k₃, hb, h⟩ := cm_bind h
  cases hb
  by_cases cd : src.drawnAll k₁ = true
  rotate_left
  · rw [ite_eq_left (by simpa using cd)] at h
    obtain ⟨_, _, h0, _⟩ := cm_bind h
    cases h0
  rw [ite_eq_right (by simpa using cd)] at h
  dsimp only at h
  by_cases cm : st.msg = M.msgs.size ∧ st.ood = M.ood.size
  rotate_left
  · rw [ite_eq_left (by omega)] at h
    obtain ⟨_, _, h0, _⟩ := cm_bind h
    cases h0
  rw [ite_eq_right (by omega)] at h
  obtain ⟨res, k₆, h6, h⟩ := cm_bind h
  obtain ⟨res', hfw, h6⟩ := liftE_ok h6
  cases h6
  by_cases cy : M.yr.size = 2 <<< ext
  rotate_left
  · rw [ite_eq_left cy] at h
    obtain ⟨_, _, h0, _⟩ := cm_bind h
    cases h0
  rw [ite_eq_right (fun hn => hn cy)] at h
  cases cm_pure h
  exact e

/-- A list of contexts whose levels are `0, 1, …` holds level `i`'s at `i`. -/
theorem conss_mem {conss : Array ConsCtx} {n : Nat} {q : Nat → Nat}
    (E : conss.toList.map (fun c => (c.start, c.pos.size)) = (List.range n).map fun k => (k, q k)) {cs : ConsCtx}
    (hcs : cs ∈ conss.toList) : cs.start < n ∧ conss[cs.start]! = cs ∧ cs.pos.size = q cs.start := by
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hcs
  have hl : conss.toList.length = n := by simpa using congrArg List.length E
  have e := List.getElem_of_eq E (by simpa using hi)
  simp only [List.getElem_map, List.getElem_range] at e
  obtain ⟨e1, e2⟩ := Prod.mk.inj e
  rw [e1]
  refine ⟨by omega, ?_, e2⟩
  rw [getElem!_pos conss i (by simpa using hi)]
  simp

end Run

/-! ## Merkle checks -/

/-- A climb reads its index's first `|ss|` bits only. -/
theorem climb_congr (ms : MerkleScheme) : ∀ (ss : List ByteArray) (acc : ByteArray) (a b : Nat),
    (∀ h < ss.length, a.testBit h = b.testBit h) → climb ms acc a ss = climb ms acc b ss
  | [], _, _, _, _ => rfl
  | s :: rest, acc, a, b, h => by
    have h0 : (a % 2 = 0) = (b % 2 = 0) := by
      have := h 0 (by simp)
      simp only [Nat.testBit_zero, decide_eq_decide] at this
      apply propext
      omega
    simp only [climb, h0]
    exact climb_congr ms rest _ (a / 2) (b / 2) fun i hi => by
      rw [Nat.testBit_div_two, Nat.testBit_div_two]
      exact h (i + 1) (by simp; omega)

/-- **`merkleCheck` passes** (Merkle.lean:72-87) when the shapes and digest widths are right and every query opens. -/
theorem merkleCheck_of {ms : MerkleScheme} {cap : Array ByteArray} {c d : Nat} {pos : Array Nat}
    {rows : Array (Array F128)} {sibs salts : Array ByteArray} {lanes : Nat} (h1 : cap.size = 2 ^ c)
    (h2 : rows.size = pos.size) (h3 : sibs.size = pos.size * (d - c)) (h4 : salts.size = saltCount ms pos.size)
    (h5 : ∀ x ∈ cap, x.size = ms.digestLen) (h6 : ∀ x ∈ sibs, x.size = ms.digestLen)
    (h7 : ∀ j < pos.size,
      opens ms cap c d lanes pos[j]! rows[j]! (salts.getD j ByteArray.empty) (sibsOf sibs (d - c) j) = true) :
    merkleCheck ms cap c d pos rows sibs salts lanes = .ok () := by
  unfold merkleCheck
  rw [ite_eq_right (by simp [h1, h2, h3, h4]), ite_eq_right (by
    simp only [Bool.or_eq_true, not_or, Array.any_eq_true, not_exists, bne_iff_ne, ne_eq, Decidable.not_not]
    exact ⟨fun i hi => h5 _ (by simp), fun i hi => h6 _ (by simp)⟩)]
  rw [List.find?_eq_none.mpr]
  · rfl
  · intro j hj
    simpa using h7 j (List.mem_range.mp hj)

/-- **hm96's check refuses unsalted openings** (Merkle.lean:74-76): it asks one salt per query. -/
theorem merkleCheck_hm96_unsalted {cap : Array ByteArray} {c d : Nat} {pos : Array Nat} {rows : Array (Array F128)}
    {sibs : Array ByteArray} {lanes : Nat} (hq : 0 < pos.size) :
    merkleCheck MerkleScheme.hm96Sha512 cap c d pos rows sibs #[] lanes ≠ .ok () := by
  unfold merkleCheck
  rw [ite_eq_left (by simp [saltCount, MerkleScheme.hm96Sha512]; omega)]
  intro h
  cases h

theorem sibsOf_ofFn {β : Type} (g : Nat → Nat → β) (q H j : Nat) (hj : j < q) :
    ((Array.ofFn (n := q * H) fun t => g (t.val / H) (t.val % H)).extract (j * H) (j * H + H)).toList =
      (List.range H).map (g j) := by
  have hle : j * H + H ≤ q * H := by
    have := Nat.mul_le_mul_right H hj
    rwa [Nat.succ_mul] at this
  apply List.ext_getElem
  · simp; omega
  · intro i h1 h2
    have hi : i < H := by simpa using h2
    simp only [Array.getElem_toList, Array.getElem_extract, Array.getElem_ofFn, List.getElem_map,
      List.getElem_range]
    have e1 : (j * H + i) / H = j := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi, Nat.zero_add]
    have e2 : (j * H + i) % H = i := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
    rw [e1, e2]

theorem getD_default {α : Type} [Inhabited α] (a : Array α) (i : Nat) : a.getD i default = a[i]! := by
  rw [Array.getD_eq_getD_getElem?]
  by_cases h : i < a.size
  · rw [getElem!_pos a i h, Array.getElem?_eq_getElem h, Option.getD_some]
  · rw [getElem!_neg a i h, Array.getElem?_eq_none (by omega), Option.getD_none]

end Conss

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecAlgebra (verifierRows decoded valOf symbolic Lin Key Messages run_stages)
open FlockVBridge.AccChain (levelLanes)
open FlockSoundness.Discharge.Hm (bitL)
open Conss (climb_congr merkleCheck_of sibsOf_ofFn getD_default)

/-- The scheme V*'s openings climb (rec_open.py:37; `Tags.withPlainLeaves`): SHA-512 of the row, no salt. -/
abbrev plain : MerkleScheme := MerkleScheme.sha512 "flock-leaf/sha512-unsalted"

/-- **`verifierRows`' consistency contexts are the schedule's levels, in order**, each with the level's `q` positions
(`run_of_rows`, `run_stages`, `ligerito_conss`). -/
theorem rows_conss {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb v : Array F128}
    {info : Flock.RecAlgebra.Info Lin} {kappa : Array (Array (Array F128))}
    (h : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (h0 : 0 < sh.sched.levels.size)
    (hq : ∀ i < sh.sched.levels.size, ∀ c,
      (Flock.positions (sh.sched.levels[i]!) c).size = (sh.sched.levels[i]!).q) :
    info.lg.conss.toList.map (fun c => (c.start, c.pos.size)) =
      (List.range sh.sched.levels.size).map fun k => (k, (sh.sched.levels[k]!).q) := by
  obtain ⟨lins, k, hr⟩ := FlockVBridge.Runs.run_of_rows h
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, k3, -, -, -, h4, -⟩ := run_stages hr
  exact Conss.ligerito_conss _ h4 h0 hq

/-- `decoded`'s consistency sums are `cons`' (`symbolic`'s `cons`, `Lin.eval_var`). -/
theorem decoded_cons (sh : Shape) (w comb : Array F128) (cons : ℕ → ℕ → F128) (ℓ : ℕ) (ea : Array F128) :
    (decoded sh w comb cons).cons ℓ ea = (Array.range (2 ^ (sh.sched.levels[ℓ]!).lanes)).map (cons ℓ) := by
  show ((Array.range (2 ^ (sh.sched.levels[ℓ]!).lanes)).map (Lin.var ∘ Key.cons ℓ)).map
    (Lin.eval (valOf sh w comb cons)) = _
  rw [Array.map_map]
  congr 1
  funext i
  exact Flock.RecAlgebra.Lin.eval_var _ _

/-! ## One opening's climb, in the layer -/

theorem colsHold_left {f : ℕ → Bool} {base : ℕ} {a b : ByteArray} (h : ColsHold f base (a ++ b)) :
    ColsHold f base a := by
  intro t ht
  have hl : a.data.toList.length = a.size := Array.length_toList
  rw [← bitL_append_left b.data.toList (a := a.data.toList) (t := t) (by omega), ← ByteArray.toList_data_append]
  exact h t (by rw [ByteArray.size_append]; omega)

/-- The bytes the columns from `base` hold are `bytesAt`'s. -/
theorem bytesAt_of_cols {f : ℕ → Bool} {base : ℕ} {B : ByteArray} (h : ColsHold f base B) :
    bytesAt f base B.size = B := by
  have hc := colsHold_bytesAt f base B.size
  have hs := size_bytesAt f base B.size
  have hl : (bytesAt f base B.size).data.toList.length = B.size := by
    rw [Array.length_toList]
    exact hs
  refine ByteArray.ext (Array.toList_inj.1 (Hm.eq_of_bitL (by rw [hl]; rfl) fun t ht => ?_))
  rw [hl] at ht
  exact (hc t (by rw [hs]; exact ht)).symm.trans (h t ht)

/-- An opening unit's `H` siblings: the 64 bytes of each 512-column block of `aux`, bottom up (`RecOpen.ports`). -/
noncomputable def sibsAt (f : ℕ → Bool) (LANES H : ℕ) : List ByteArray :=
  (List.range H).map fun i => bytesAt f (128 * LANES + 512 * i) 64

/-- **An opening's climb, from its unit's columns in the layer** (`keyProg_parse`): copy 0 of unit `u`, a unit the
verifier parsed as `rec-open/l<LANES>-h<H>` carrying 1 on its constant, holds on its first 512 output columns the climb
under `plain` from the plain leaf of the `LANES` elements of its `row` columns, through the siblings of its `aux` columns,
at the index its `dirs` columns spell. -/
theorem climb_of_layer {Q : VStmt} {Reg' : Type} (zo : Q.Outer Reg') {bits : Q.Bits} {u : Fin Q.nv}
    (hj : 0 < Q.c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg Q.c Q.hU Q.nv) u) = true)
    (hu : Q.c.unit.unitLog ≤ 32) {LANES H : ℕ} (hp : Flock.RecOpen.params Q.c.unit.name = some (LANES, H)) :
    climb plain (Sha512.hash (MerkleScheme.rowBytes (elems (colOf Q bits u) 0 LANES)))
        (natOfBits (fun h => colOf Q bits u (accCol LANES H + 1024 + 256 * LANES + h)) H)
        (sibsAt (colOf Q bits u) LANES H) =
      bytesAt (colOf Q bits u) (outCol LANES H) 64 := by
  rw [colOf_eq Q bits u hj]
  set f := copyVal (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u) bits hj
  have hO := keyProg_parse zo.hc Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) bits u hj hone plain (fun _ _ => rfl) hp hu
    (elems f 0 LANES) (elems f (accCol LANES H) 2) (elems f (accCol LANES H + 1024) (2 * LANES))
    (bytesAt f (accCol LANES H + 256) 96) (natOfBits (fun h => f (accCol LANES H + 1024 + 256 * LANES + h)) H)
    (sibsAt f LANES H) (size_elems _ _ _) (size_elems _ _ _) (size_elems _ _ _) (size_bytesAt _ _ _)
    (by simp [sibsAt]) (colsHold_elems _ _ _)
    (fun i hi => by
      simp only [sibsAt, List.getElem_map, List.getElem_range]
      exact ⟨size_bytesAt _ _ _, colsHold_bytesAt _ _ _⟩)
    (colsHold_append (colsHold_elems _ _ _) (by
      rw [size_rowBytes, size_elems]
      exact colsHold_bytesAt _ _ _))
    (colsHold_elems _ _ _)
    (fun h hh => by
      rw [testBit_natOfBits]
      simp [hh]
      try rfl)
  have hX := bytesAt_of_cols (colsHold_left (colsHold_left hO))
  rw [AccChain.climb_size_of plain (fun _ _ => rfl) _ _ _ (sha512_size _)] at hX
  exact hX.symm

/-! ## The family's openings -/

/-- **Every level's openings pass at the scheme `ms`**: `OpenedAll` with `ms` in place of the setup's. -/
def OpenedAllAt (ms : MerkleScheme) (sch : Flock.Schedule) (sh : Shape) (p : Proof) (d : RepData) : Prop :=
  ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
    ∀ cs ∈ info.lg.conss.toList, RecAlgebra.Opened ms sch p.cap p.ligerito (decoded sh d.w d.comb d.cons) cs

theorem openedAll_iff {st : Setup} {sch : Flock.Schedule} {sh : Shape} {p : Proof} {d : RepData} :
    OpenedAll st sch sh p d ↔ OpenedAllAt st.merkle sch sh p d :=
  Iff.rfl

/-- **`OpenedAll` is false at V*'s unsalted openings** (`merkleCheck_hm96_unsalted`): at hm96, a level whose openings
carry no salts and that queries a position fails its Merkle check. -/
theorem not_opened_hm96 {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof} {M : Messages F128}
    {cs : Flock.ConsCtx} (hs : (openAt p (sch.levels.size - 1) cs.start).2.2 = #[]) (hq : 0 < cs.pos.size) :
    ¬ RecAlgebra.Opened MerkleScheme.hm96Sha512 sch cap p M cs := by
  intro h
  have h1 := h.1
  rw [hs] at h1
  exact Conss.merkleCheck_hm96_unsalted hq h1

/-- **Each level's caps have the schedule's shape**: `2^cap` digests of 64 bytes (`CapShaped`, and `LigShaped`'s
recursive caps, `capsAt_of`). -/
def CapsAt (sch : Flock.Schedule) (cap : Array ByteArray) (lig : LigeritoProof) : Prop :=
  ∀ ℓ < sch.levels.size, (capAt cap lig ℓ).size = 2 ^ (sch.levels[ℓ]!).cap ∧ ∀ x ∈ capAt cap lig ℓ, x.size = 64

/-- **`CapsAt` from the decoder's shapes** (`CapShaped`, `LigShaped`'s recursive caps), at a 64-byte digest. -/
theorem capsAt_of {st : Setup} {sch : Flock.Schedule} {cap : Array ByteArray} {lig : LigeritoProof}
    (hd : st.merkle.digestLen = 64) (hc : CapShaped st sch cap) (hs : lig.recursiveCaps.size = sch.levels.size - 1)
    (hr : ∀ i (h : i < lig.recursiveCaps.size),
      ZkView.CapOk (shapeOf st sch 0) (sch.levels.getD (i + 1) default) lig.recursiveCaps[i]) :
    CapsAt sch cap lig := by
  intro ℓ hℓ
  unfold capAt
  split
  · rename_i h0
    subst h0
    obtain ⟨h1, h2⟩ := hc
    rw [getD_default] at h1
    exact ⟨h1, fun x hx => (h2 x hx).trans hd⟩
  · rename_i h0
    have hi : ℓ - 1 < lig.recursiveCaps.size := by omega
    obtain ⟨h1, h2⟩ := hr (ℓ - 1) hi
    rw [getD_default, show ℓ - 1 + 1 = ℓ by omega] at h1
    rw [getElem!_pos lig.recursiveCaps (ℓ - 1) hi]
    exact ⟨h1, fun x hx => (h2 x hx).trans hd⟩

/-- Level `ℓ`'s openings in `ligOfRead 64 sl sch x`, at rows of `w` elements (Lig.lean's `lev`). -/
def levAt (sl : Option ℕ) (sch : Flock.Schedule) (x : LigRead) (ℓ w : ℕ) : LevelProof :=
  levelOf 64 sl (sch.levels.getD ℓ default).q w ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap)
    (x.row ℓ) (x.sib ℓ) (x.salt ℓ)

/-- **`openAt` of `ligOfRead`**: level `ℓ`'s openings are `levAt`'s, at rows of `2^lanes` elements or, past level 0,
16. -/
theorem openAt_ligOfRead {sch : Flock.Schedule} (sl : Option ℕ) (x : LigRead) {ℓ : ℕ} (hℓ : ℓ < sch.levels.size) :
    ∃ w, (w = 2 ^ (sch.levels.getD ℓ default).lanes ∨ (0 < ℓ ∧ w = 16)) ∧
      openAt (ligOfRead 64 sl sch x) (sch.levels.size - 1) ℓ =
        ((levAt sl sch x ℓ w).rows, (levAt sl sch x ℓ w).siblings, (levAt sl sch x ℓ w).salts) := by
  unfold openAt
  rcases Nat.eq_zero_or_pos ℓ with rfl | h0
  · exact ⟨_, .inl rfl, by rw [ite_eq_left (show (0 : ℕ) = 0 from rfl)]; rfl⟩
  · rw [ite_eq_right (show ¬ℓ = 0 by omega)]
    obtain ⟨i, rfl⟩ : ∃ i, ℓ = i + 1 := ⟨ℓ - 1, by omega⟩
    refine ⟨16, .inr ⟨h0, rfl⟩, ?_⟩
    by_cases h1 : i + 1 < sch.levels.size - 1
    · have hi : i < (ligOfRead 64 sl sch x).recursiveProofs.size := by
        simp only [ligOfRead, Array.size_ofFn]; omega
      rw [ite_eq_left h1, Nat.add_sub_cancel, getElem!_pos _ i hi]
      simp only [ligOfRead, Array.getElem_ofFn, levAt]
    · have hR : sch.levels.size - 1 = i + 1 := by omega
      rw [ite_eq_right h1]
      simp only [ligOfRead, levAt, hR]

/-- **`CapsAt` at `ligOfRead`'s caps** (`capArr`), the level-0 one any `capArr` of the level's `2^cap` entries. -/
theorem capsAt_ligOfRead {sch : Flock.Schedule} (sl : Option ℕ) (x : LigRead) (e : ℕ → ByteArray) :
    CapsAt sch (capArr 64 (sch.levels.getD 0 default).cap e) (ligOfRead 64 sl sch x) := by
  intro ℓ hℓ
  unfold capAt
  split
  · rename_i h0
    subst h0
    refine ⟨by rw [capArr, Array.size_ofFn, getD_default], fun y hy => ?_⟩
    obtain ⟨i, rfl⟩ := Array.mem_ofFn.1 hy
    exact size_fitB _ _
  · rename_i h0
    obtain ⟨i, rfl⟩ : ∃ i, ℓ = i + 1 := ⟨ℓ - 1, by omega⟩
    have hi : i < (ligOfRead 64 sl sch x).recursiveCaps.size := by simp only [ligOfRead, Array.size_ofFn]; omega
    rw [Nat.add_sub_cancel, getElem!_pos _ i hi]
    simp only [ligOfRead, Array.getElem_ofFn]
    refine ⟨by rw [capArr, Array.size_ofFn, getD_default], fun y hy => ?_⟩
    obtain ⟨k, rfl⟩ := Array.mem_ofFn.1 hy
    exact size_fitB _ _

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)

/-- **Each rep's positions of each level**: the ones the verifier's Ligerito run draws for the level's queries
(`RecAlgebra.query`, `ConsCtx.pos`), at the rep's coins. -/
def posOf (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2)
    (ℓ : ℕ) : Array ℕ :=
  match verifierRows F.sh (F.rep₀ R ω t l r).rounds (F.rep₀ R ω t l r).extras (F.rep₀ R ω t l r).comb with
  | .ok (_, info, _) => (info.lg.conss[ℓ]!).pos
  | .error _ => #[]

theorem posOf_ok (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits)
    (r : Fin 2) {v info kappa}
    (h : verifierRows F.sh (F.rep₀ R ω t l r).rounds (F.rep₀ R ω t l r).extras (F.rep₀ R ω t l r).comb =
      .ok (v, info, kappa)) (ℓ : ℕ) : F.posOf R ω t l r ℓ = (info.lg.conss[ℓ]!).pos := by
  unfold posOf
  rw [h]

/-- **Each rep's siblings of each level**, query-major: query `j`'s `depth` siblings are its unit's `aux` blocks. -/
noncomputable def sibsFlat (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) (ℓ : ℕ) : Array ByteArray :=
  Array.ofFn (n := F.q ℓ * F.depth ℓ) fun t =>
    bytesAt (F.openCol l r ℓ (t.val / F.depth ℓ)) (128 * levelLanes F.sh ℓ + 512 * (t.val % F.depth ℓ)) 64

theorem sibsOf_sibsFlat (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) {ℓ j : ℕ} (hj : j < F.q ℓ) :
    sibsOf (F.sibsFlat l r ℓ) (F.depth ℓ) j = sibsAt (F.openCol l r ℓ j) (levelLanes F.sh ℓ) (F.depth ℓ) :=
  sibsOf_ofFn (fun j i => bytesAt (F.openCol l r ℓ j) (128 * levelLanes F.sh ℓ + 512 * i) 64) _ _ j hj

/-- **The openings' units' parameters** (`hOpenUnits_of`): level `ℓ`'s unit is `rec-open/l<2^lanes>-h<d - cap>`. -/
theorem hOpenParams_of (hne : Nonempty (Reg × L.Ω × Reg₂)) (hname : F.HOpenName) :
    ∀ ℓ < F.sh.sched.levels.size, Flock.RecOpen.params (F.Vs (F.sessO ℓ)).c.unit.name =
      some (levelLanes F.sh ℓ, (F.sh.sched.levels[ℓ]!).d - (F.sh.sched.levels[ℓ]!).cap) := by
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  intro ℓ hℓ
  obtain ⟨s, -, -, hs, -, -, -⟩ := statements_eq F.hstmts
  have hss := F.sched_eq hs
  subst hss
  obtain ⟨hj, hst⟩ := statements_recOpen F.hstmts hs hℓ
  rw [F.hsessO ℓ ⟨ℓ, hj⟩ ⟨_, _, _, hst⟩]
  exact params_of_check (parse_facts (F.zs R ω [] R₂ ⟨ℓ, hj⟩).hc).recNet (hname ⟨ℓ, hj⟩ _ _ _ _ hst)

/-- Obligation: discharged by check 5's `RecCoins.checkOpen` (in the event as `ChecksC3`'s `RecCoins.check`), which
holds the `dirs` port's registration to its rep's query positions (`dirsRow`, `checkValue` at `rec-dirs`), as it holds
`coef` for `HCoefCols`.
**The `dirs` the openings read are the verifier's positions** (`rec_outer.stage`: `dirs` registered, rec_outer.py:
239-242): at every accepted outer proof, the `dirs` columns of rep `r`'s opening `j` of level `ℓ` spell, bit `h` at
column `h`, the position the verifier's Ligerito run draws for the query (`posOf`). -/
def HDirCols : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ, ∀ h < F.depth ℓ,
      F.openCol (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ j
          (accCol (levelLanes F.sh ℓ) (F.depth ℓ) + 1024 + 256 * levelLanes F.sh ℓ + h) =
        ((F.posOf R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ)[j]!).testBit h

/-- **The openings `lig` holds are the layers'** (the lig lane's definition of `lig`): at every accepted outer proof,
each level's opened rows are `rowsOf`, its siblings the openings' `aux` blocks (`sibsFlat`), and it carries no salts
(`RecOpen` has no salt port, rec_open.py:37). -/
def LigOpens : Prop :=
  F.AtAccepted dg N k Rw fun _ _ _ l => ∀ r, ∀ ℓ < F.sh.sched.levels.size,
    openAt (lig l r) (F.sh.sched.levels.size - 1) ℓ = (F.rowsOf l r ℓ, F.sibsFlat l r ℓ, #[])

/-- Obligation: discharged by each opening's `out` registered as a read of its level's cap row, whose bytes are the cap
message's (route not built).
**The openings climb to the caps** (rec_outer.py: each opening's `out` registered as a read of its level's cap row,
at the query's cap index): at every accepted outer proof, the first 512 output columns of rep `r`'s opening `j` of
level `ℓ`, in the layer, are entry `pos >>> depth` of the level's cap (`capAt`). The registration lane's (`out` an
output column, as `HZeroL`'s) with the lig lane's `cap` and `lig`. -/
def HTopsL : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ,
      bytesAt (F.openCol (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ j) (outCol (levelLanes F.sh ℓ) (F.depth ℓ)) 64 =
        (capAt (cap (layers F.zs dg N k Rw R ω t R₂ τ)) (lig (layers F.zs dg N k Rw R ω t R₂ τ) r) ℓ).getD
          ((F.posOf R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ)[j]! >>> F.depth ℓ) ByteArray.empty

/-- **Every level's openings pass at V*'s scheme `plain`**, at every accepted outer proof: level `ℓ`'s context is the
level's (`rows_conss`), its rows are the openings' `row` columns and its siblings their `aux` (`LigOpens`), each climbs
from its plain leaf at the verifier's position (`HDirCols`) to its `out` columns (`climb_of_layer`), which are the cap's
entry (`HTopsL`); the consistency sums are `rep`'s, `rowsOf`'s at `alphaOf`'s `α`. -/
theorem openedAllAt_plain (hne : Nonempty (Reg × L.Ω × Reg₂)) (hname : F.HOpenName) (hsch : F.sh.sched = sch)
    (hrows : F.rows = F.rowsOf) (halpha : F.alpha = F.alphaOf) (hone : F.HOneOpen dg N k Rw)
    (hdir : F.HDirCols dg N k Rw) (hlig : F.LigOpens dg N k Rw lig) (htop : F.HTopsL dg N k Rw cap lig)
    (hcaps : F.AtAccepted dg N k Rw fun _ _ _ l => ∀ r, CapsAt sch (cap l) (lig l r)) :
    F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAllAt plain sch F.sh ((F.tableSrc spec st cap lig R ω t l).proofAt r) (F.rep R ω t l r) := by
  intro R ω t R₂ τ o hacc r v info kappa hv cs hcs
  have hU := F.hOpenUnits_of hne hname
  have hP := F.hOpenParams_of hne hname
  have hf := (shape?_eq F.hsh).1
  have hsz := (fast100_q0 hf).1
  have hseam := fast100_level _ _ hf
  have hq : ∀ i < F.sh.sched.levels.size, ∀ c,
      (Flock.positions (F.sh.sched.levels[i]!) c).size = (F.sh.sched.levels[i]!).q := fun i hi c =>
    positions_size _ (hseam _ (by rw [getElem!_pos _ i hi]; exact Array.getElem_mem hi)) c
  obtain ⟨hℓ, hcs0, hpsz⟩ := Conss.conss_mem (rows_conss hv hsz hq) hcs
  have hposE : F.posOf R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r cs.start = cs.pos := by
    rw [F.posOf_ok R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r hv, hcs0]
  have halE : F.alphaOf R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r cs.start = cs.alpha := by
    rw [F.alphaOf_ok R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r hv, hcs0]
  have hlv : sch.levels.getD cs.start default = F.sh.sched.levels[cs.start]! := by
    rw [getD_default, hsch]
  have hsize : sch.levels.size = F.sh.sched.levels.size := by rw [hsch]
  have hdep : F.depth cs.start = (F.sh.sched.levels[cs.start]!).d - (F.sh.sched.levels[cs.start]!).cap := by
    simp only [VStarFamily.depth, hP cs.start hℓ, Option.getD_some]
  have hO := hlig R ω t R₂ τ o hacc r cs.start hℓ
  have hcapP : capAt ((F.tableSrc spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).proofAt r).cap
      ((F.tableSrc spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).proofAt r).ligerito cs.start =
        capAt (cap (layers F.zs dg N k Rw R ω t R₂ τ)) (lig (layers F.zs dg N k Rw R ω t R₂ τ) r) cs.start := by
    unfold capAt
    rfl
  have hopenP : openAt ((F.tableSrc spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).proofAt r).ligerito
      (F.sh.sched.levels.size - 1) cs.start =
        openAt (lig (layers F.zs dg N k Rw R ω t R₂ τ) r) (F.sh.sched.levels.size - 1) cs.start := by
    unfold openAt
    rfl
  obtain ⟨hc1, hc2⟩ := hcaps R ω t R₂ τ o hacc r cs.start (by rw [hsize]; exact hℓ)
  have hlv' : sch.levels[cs.start]! = F.sh.sched.levels[cs.start]! := by rw [hsch]
  rw [hlv'] at hc1
  unfold RecAlgebra.Opened
  rw [hlv, hsize, hcapP, hopenP, hO]
  refine ⟨merkleCheck_of (rows := F.rowsOf _ r cs.start) (sibs := F.sibsFlat _ r cs.start) (salts := #[]) hc1
    ?_ ?_ rfl hc2 ?_ ?_, ?_⟩
  · simp [rowsOf, hpsz, VStarFamily.q]
  · rw [← hdep, hpsz]
    simp [sibsFlat, VStarFamily.q]
  · intro x hx
    obtain ⟨i, rfl⟩ := Array.mem_ofFn.1 hx
    exact size_bytesAt _ _ _
  · intro j hj
    have hjq : j < F.q cs.start := by rw [VStarFamily.q, ← hpsz]; exact hj
    obtain ⟨hju, -, -⟩ := hU cs.start hℓ
    have hp' := hP cs.start hℓ
    unfold opens
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    rw [← hdep] at hp' ⊢
    rw [F.rowsOf_get _ r hjq, F.sibsOf_sibsFlat _ r hjq]
    refine ⟨⟨⟨size_elems _ _ _, by simp [sibsAt]⟩, by simp; rfl⟩, ?_⟩
    have hcl := climb_congr plain (sibsAt (F.openCol (layers F.zs dg N k Rw R ω t R₂ τ) r cs.start j)
        (levelLanes F.sh cs.start) (F.depth cs.start))
      (Sha512.hash (MerkleScheme.rowBytes
        (elems (F.openCol (layers F.zs dg N k Rw R ω t R₂ τ) r cs.start j) 0 (levelLanes F.sh cs.start))))
      cs.pos[j]!
      (natOfBits (fun h => F.openCol (layers F.zs dg N k Rw R ω t R₂ τ) r cs.start j
        (accCol (levelLanes F.sh cs.start) (F.depth cs.start) + 1024 + 256 * levelLanes F.sh cs.start + h))
        (F.depth cs.start))
      (fun h hh => by
        have hh' : h < F.depth cs.start := by simpa [sibsAt] using hh
        rw [testBit_natOfBits, ← hposE, ← hdir R ω t R₂ τ o hacc r cs.start hℓ j hjq h hh']
        simp [hh'])
    refine hcl.trans ((climb_of_layer (F.zs R ω t R₂ (F.sessO cs.start)) hju
      (hone R ω t R₂ τ o hacc r cs.start hℓ j hjq) (hU cs.start hℓ).2.1 hp').trans ?_)
    rw [← hposE]
    exact htop R ω t R₂ τ o hacc r cs.start hℓ j hjq
  · rw [decoded_cons]
    congr 1
    funext i
    show Flock.RecAlgebra.consAt (F.rows _ r cs.start) (F.q cs.start) (F.alpha R ω t _ r cs.start) i = _
    rw [hrows, halpha, halE]
    rfl

/-- **`vbridgeRead_assembled`'s `hopen` at a setup whose scheme is V*'s** (`openedAllAt_plain` at `assembled`, whose
`rows` and `alpha` are `rowsOf` and `alphaOf` by `rfl`). -/
theorem hopen_of_plain (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin F.stmts.size) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8)
    (hne : Nonempty (Reg × L.Ω × Reg₂)) (hname : F.HOpenName) (hsch : F.sh.sched = sch)
    (hm : ∀ R ω, (st R ω).merkle = plain) (hone : F.HOneOpen dg N k Rw)
    (hdir : (F.assembled dg N k Rw spec st cap lig own sel bc).HDirCols dg N k Rw) (hlig : F.LigOpens dg N k Rw lig)
    (htop : (F.assembled dg N k Rw spec st cap lig own sel bc).HTopsL dg N k Rw cap lig)
    (hcaps : F.AtAccepted dg N k Rw fun _ _ _ l => ∀ r, CapsAt sch (cap l) (lig l r)) :
    F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAll (st R ω) sch F.sh
        (((F.assembled dg N k Rw spec st cap lig own sel bc).tableSrc spec st cap lig R ω t l).proofAt r)
        ((F.assembled dg N k Rw spec st cap lig own sel bc).rep R ω t l r) := by
  intro R ω t R₂ τ o hacc r
  rw [openedAll_iff, hm R ω]
  exact (F.assembled dg N k Rw spec st cap lig own sel bc).openedAllAt_plain dg N k Rw spec st cap lig hne hname hsch rfl
    rfl hone hdir hlig htop hcaps R ω t R₂ τ o hacc r

/-- **The lig lane's openings at no salts are the layers'**: `ligOfRead 64 sl` at `ligRead`'s reads, at a layout that
salts nothing (`plain`'s `shapeOf` saltLen, `none` or `some 0` by `openedSalts`), opens at each level `rowsOf`'s rows
and `sibsFlat`'s siblings (past level 0 its rows are 16 elements, `DimSeam`'s 4 lanes). -/
theorem openAt_ligRead (hne : Nonempty (Reg × L.Ω × Reg₂)) (hname : F.HOpenName) (hsch : F.sh.sched = sch)
    {sl : Option ℕ} (hsl : sl = none ∨ sl = some 0) (ms : List (List (SMsg T)))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) {ℓ : ℕ} (hℓ : ℓ < F.sh.sched.levels.size) :
    openAt (ligOfRead 64 sl sch (F.ligRead ms l r)) (F.sh.sched.levels.size - 1) ℓ =
      (F.rowsOf l r ℓ, F.sibsFlat l r ℓ, #[]) := by
  have hsize : sch.levels.size = F.sh.sched.levels.size := by rw [hsch]
  have hlv : sch.levels.getD ℓ default = F.sh.sched.levels[ℓ]! := by rw [getD_default, hsch]
  have hdep : F.depth ℓ = (F.sh.sched.levels[ℓ]!).d - (F.sh.sched.levels[ℓ]!).cap := by
    simp only [VStarFamily.depth, F.hOpenParams_of hne hname ℓ hℓ, Option.getD_some]
  have hdims : ∀ i < F.sh.sched.levels.size, 1 ≤ i → (F.sh.sched.levels.getD i default).lanes = 4 := by
    have h := (fast100_dims _ _ (shape?_eq F.hsh).1).1
    unfold DimSeam at h
    exact h.2.2.2.2.1
  obtain ⟨w, hw, e⟩ := openAt_ligOfRead (sch := sch) sl (F.ligRead ms l r) (by rw [hsize]; exact hℓ)
  have hlw : levelLanes F.sh ℓ = w := by
    rcases hw with rfl | ⟨h0, rfl⟩
    · rw [hlv]; rfl
    · rw [levelLanes, ← getD_default, hdims ℓ hℓ h0]; rfl
  have h1 : (levAt sl sch (F.ligRead ms l r) ℓ w).rows = F.rowsOf l r ℓ := by
    apply Array.ext
    · simp only [levAt, levelOf, rowsOf, VStarFamily.q, hlv, Array.size_ofFn, Array.size_map, Array.size_range]
    · intro j _ hj2
      have hj : j < F.q ℓ := by simpa only [rowsOf, Array.size_map, Array.size_range] using hj2
      rw [← getElem!_pos (F.rowsOf l r ℓ) j hj2, F.rowsOf_get l r hj]
      simp only [levAt, levelOf, Array.getElem_ofFn, VStarFamily.ligRead, hlv]
      exact fitRow_of_size ((size_elems _ _ _).trans hlw)
  have h2 : (levAt sl sch (F.ligRead ms l r) ℓ w).siblings = F.sibsFlat l r ℓ := by
    apply Array.ext
    · simp only [levAt, levelOf, sibsFlat, Array.size_ofFn, hlv, ← hdep, VStarFamily.q]
    · intro t _ _
      simp only [levAt, levelOf, sibsFlat, Array.getElem_ofFn, VStarFamily.ligRead, hlv, ← hdep, levelLanes,
        fitB_of_size, size_bytesAt]
  rw [← hsize, e, h1, h2]
  rcases hsl with rfl | rfl <;> rfl

/-- **`LigOpens` at the lig lane's openings at no salts** (`ligOfRead 64 sl` at `ligRead`, any inner messages). -/
theorem ligOpens_unsalted (hne : Nonempty (Reg × L.Ω × Reg₂)) (hname : F.HOpenName) (hsch : F.sh.sched = sch)
    {sl : Option ℕ} (hsl : sl = none ∨ sl = some 0)
    (ms : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → List (List (SMsg T))) :
    F.LigOpens dg N k Rw (fun l r => ligOfRead 64 sl sch (F.ligRead (ms l) l r)) := by
  intro R ω t R₂ τ o _ r ℓ hℓ
  exact F.openAt_ligRead hne hname hsch hsl (ms _) _ r hℓ

/-- **`CapsAt` at the lig lane's caps** (`capOf`, `ligOfRead`'s recursive caps). -/
theorem capsAt_capOf (sl : Option ℕ) (ms : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → List (List (SMsg T))) :
    F.AtAccepted dg N k Rw fun _ _ _ l => ∀ r,
      CapsAt sch (capOf T sch (ms l)) (ligOfRead 64 sl sch (F.ligRead (ms l) l r)) := by
  intro R ω t R₂ τ o _ r
  exact capsAt_ligOfRead sl _ _

/-- **`hopen` at the lig lane's cap and openings**, at a setup whose scheme is `plain` and the openings at its layout
(no salts): `hopen_of_plain` with `LigOpens` and `CapsAt` proved, from `HOneOpen`, `HDirCols` and `HTopsL`. -/
theorem hopen_of_plain_lig (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin F.stmts.size) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8)
    (ms : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → List (List (SMsg T))) {sl : Option ℕ}
    (hsl : sl = none ∨ sl = some 0) (hcap : cap = fun l => capOf T sch (ms l))
    (hlig : lig = fun l r => ligOfRead 64 sl sch (F.ligRead (ms l) l r))
    (hne : Nonempty (Reg × L.Ω × Reg₂)) (hname : F.HOpenName) (hsch : F.sh.sched = sch)
    (hm : ∀ R ω, (st R ω).merkle = plain) (hone : F.HOneOpen dg N k Rw)
    (hdir : (F.assembled dg N k Rw spec st cap lig own sel bc).HDirCols dg N k Rw)
    (htop : (F.assembled dg N k Rw spec st cap lig own sel bc).HTopsL dg N k Rw cap lig) :
    F.AtAccepted dg N k Rw fun R ω t l => ∀ r,
      OpenedAll (st R ω) sch F.sh
        (((F.assembled dg N k Rw spec st cap lig own sel bc).tableSrc spec st cap lig R ω t l).proofAt r)
        ((F.assembled dg N k Rw spec st cap lig own sel bc).rep R ω t l r) := by
  subst hcap hlig
  exact F.hopen_of_plain dg N k Rw spec st _ _ own sel bc hne hname hsch hm hone hdir
    (F.ligOpens_unsalted dg N k Rw hne hname hsch hsl ms) htop (F.capsAt_capOf dg N k Rw sl ms)

end VStarFamily

end family

end FlockVBridge.Family
