import Proofs.Flock.Recursive.Target
import Proofs.Flock.Soundness.Discharge.Exec.Tables
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins

/-!
# VBridge G: the compiled inner verifier accepts from both reps' accepting runs

`inner_accepts_of_reps`, the last step of `StagesAccept` (`Compiled.lean`): when both reps' checks accept with one cap
(`verifyRep`, `Tables.lean`), the compiled inner protocol `flockInnerCL` accepts the inner transcript `t` whose steps on
the schedule are the level-0 cap, the link points, both reps' tapes in `Plays.interleave_first`'s order, and the
openings with their salts (`tableTape`).

* `playsT_bind`, `playsT_interleave_first`: `Plays.bind` and `Plays.interleave_first` on one tape. The tape says where
  the prover's choices fall: before each of rep 0's steps, `true` (rep 0 moves); none for rep 1's.
* `RepTape`: what the transcript holds for one rep: steps of the kinds of the rep's spine (`repShape`), the proof file's
  messages (`repMsgs`) and the stream's coin arrays (`coinArr`). A rep's messages and coins alone do not fix its tape;
  the spine places the coins among the messages (`Spine.tape`).
* `playsT_of_tape`: a rep's run on its messages and coins (`Plays`) is a run on any such tape (`PlaysT`).
* `flockInnerCL_rejects_short`: `flockInnerCL` accepts no transcript of at most `r` rounds, the rounds `Inner.game`
  plays: each of them ends with a coin, and the table ends with the openings' message.
-/

namespace FlockVBridge.InnerAccepts

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Discharge.PlaysCoin FlockSoundness.Refine FlockSoundness.Model
open Flock FlockLevel3

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Runs on one tape compose -/

section game

variable {α β : Type}

/-- **Runs on one tape compose.** A run of `g` to `a`, then a run of `f a`, is a run of `g.bind f`. -/
theorem playsT_bind {g : Game α} {f : α → Game β} {ts₁ ts₂ : List Game.Step} {a : α} {b : β}
    (h₁ : PlaysT g ts₁ a) (h₂ : PlaysT (f a) ts₂ b) : PlaysT (g.bind f) (ts₁ ++ ts₂) b := by
  induction h₁ with
  | ret a => exact h₂
  | send m _ ih => exact .send m (ih h₂)
  | coin x _ ih => exact .coin x (ih h₂)

theorem playsT_map {g : Game α} {ts : List Game.Step} {a : α} (f : α → β) (h : PlaysT g ts a) :
    PlaysT (g.map f) ts (f a) := by
  have h' := playsT_bind (f := fun a => Game.ret (f a)) h (.ret (f a))
  rw [List.append_nil] at h'
  exact h'

/-- The steps `ts`, each after the prover's choice `true`: the first of two interleaved interactions moves. -/
def firstSteps (ts : List Game.Step) : List Game.Step := ts.flatMap fun s => [.inl ⟨Bool, true⟩, s]

/-- **The first interaction first, on one tape** (`Plays.interleave_first`): while the second interaction has not
returned, the prover chooses the first at each of its steps; then the second runs alone. -/
theorem playsT_interleave_first {g₁ : Game α} {g₂ : Game β} {ts₁ ts₂ : List Game.Step} {a : α} {b : β}
    (h₁ : PlaysT g₁ ts₁ a) (h₂ : PlaysT g₂ ts₂ b) (hg₂ : ∀ b', g₂ ≠ .ret b') :
    PlaysT (interleave g₁ g₂) (firstSteps ts₁ ++ ts₂) (a, b) := by
  induction h₁ with
  | ret a => exact playsT_map (fun b => (a, b)) h₂
  | send m _ ih =>
      cases g₂ with
      | ret b' => exact absurd rfl (hg₂ b')
      | send M k => exact .send true (.send m ih)
      | coin C k => exact .send true (.send m ih)
  | coin x _ ih =>
      cases g₂ with
      | ret b' => exact absurd rfl (hg₂ b')
      | send M k => exact .send true (.coin x ih)
      | coin C k => exact .send true (.coin x ih)

/-- A game of a nonempty exact shape, then anything, does not return at once. -/
theorem exact_bind_ne_ret {g : Game α} {σ : List Exec.Kind} (h : Exact g σ) (hσ : σ ≠ []) (f : α → Game β) (b : β) :
    g >>= f ≠ .ret b := by
  cases h with
  | ret a => exact absurd rfl hσ
  | send _ => intro h; cases h
  | coin _ => intro h; cases h

theorem length_eq_lefts_rights (l : List Game.Step) :
    l.length = (l.filterMap Sum.getLeft?).length + (l.filterMap Sum.getRight?).length := by
  induction l with
  | nil => rfl
  | cons s l ih =>
      cases s with
      | inl x =>
          simp only [List.filterMap_cons, Sum.getLeft?_inl, Sum.getRight?_inl, List.length_cons, ih]
          omega
      | inr x =>
          simp only [List.filterMap_cons, Sum.getLeft?_inr, Sum.getRight?_inr, List.length_cons, ih]
          omega

end game

/-! ## A rep's tape -/

/-- A compiled rep starts with the zerocheck's first coin, so it never returns at once. -/
theorem repC_ne_ret (r : ℕ) (S : Model.Statement) (sch : Accounting.Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim 𝔽 S.m)) (p : PendL 𝔽 Dig) : repC (D := Dig) execArith r S sch hm extra ≠ .ret p := by
  unfold repC
  exact exact_bind_ne_ret (zerocheck_exact execArith S.m) (by simp [zcShape]) _ _

/-- **Rep `r`'s tape in the inner transcript**: steps whose kinds, in order, are a prefix of the rep's spine
(`repShape`, at the statement's dimensions, `n` extra claims and the schedule), whose messages are the proof file's
(`repMsgs`) and whose coin steps are the stream's coin arrays, as `coinArr` reads them. -/
def RepTape (st : Setup) (S : Model.Statement) (sch : Flock.Schedule) (n : ℕ) (p : Proof) (s : StreamRec)
    (ts : List Game.Step) : Prop :=
  ts.map kindOf <+: repShape 𝔽 𝕂 Dig S.m S.kLog n (schedOf sch) ∧
    ts.filterMap Sum.getLeft? = repMsgs st sch (2 + n) p capDig ∧
    ts.filterMap Sum.getRight? = (s.rounds.toList.map (·.coins)).map (coinArr fOf)

/-- **A rep's run on its messages and coins is a run on one tape** that takes steps of its spine's kinds, with those
messages and coin steps: `repShape` places the coins among the messages. -/
theorem playsT_of_tape (r : ℕ) (S : Model.Statement) (sch : Accounting.Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim 𝔽 S.m)) {ms : List Game.Msg} {cs : List (Array F128)} {pend : PendL 𝔽 Dig}
    {ts : List Game.Step} (h : Plays fOf (repC execArith r S sch hm extra) ms cs pend)
    (hk : ts.map kindOf <+: repShape 𝔽 𝕂 Dig S.m S.kLog extra.length sch)
    (hl : ts.filterMap Sum.getLeft? = ms) (hr : ts.filterMap Sum.getRight? = cs.map (coinArr fOf)) :
    PlaysT (repC execArith r S sch hm extra) ts pend := by
  obtain ⟨ts', hp, h1, h2⟩ := toT_exact fOf h
  have hlen : ts.length = ts'.length := by
    rw [length_eq_lefts_rights ts, length_eq_lefts_rights ts', hl, hr, h1, h2]
  have hw : ts = weave ((repShape 𝔽 𝕂 Dig S.m S.kLog extra.length sch).take ts.length)
      (ts.filterMap Sum.getLeft?) (ts.filterMap Sum.getRight?) := by
    have k := List.prefix_iff_eq_take.mp hk
    rw [List.length_map] at k
    calc ts = weave (ts.map kindOf) (ts.filterMap Sum.getLeft?) (ts.filterMap Sum.getRight?) := (weave_self ts).symm
      _ = _ := by rw [k]
  have e : ts = ts' := by
    rw [hw, Spine.tape hp (repC_spine execArith r S sch hm extra), hlen, hl, hr, h1, h2]
  rw [e]
  exact hp

/-! ## The inner transcript -/

/-! ## The inner statement's leaf -/

/-- **The inner statement's leaf**, `Refine.leafPlain`: SHA-512 of the row's encoding, its salt ignored (a recursive
session's inner statement has `flock-leaf/sha512-unsalted` leaves, `Tags.withPlainLeaves`; `MerkleScheme.sha512`'s
`leaf`, `Refine.leafPlain_leaf`). -/
abbrev leafPlain : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512 := Refine.leafPlain

/-- **One table's accepted reps refine the compiled reps, for `flock-leaf/sha512-unsalted`**: `table_refines_hm96`
(Tables.lean) with the unsalted path checks (`opensOK_of`), which at `leafPlain` are its salted ones. -/
theorem table_refines_plain (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S)
    (hms : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") (extra : List (Model.Claim 𝔽 S.m))
    (sess : Session) (hext : ExtraRealizes st S sess.points extra) (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (sch : Flock.Schedule) (hf : Flock.fast100 st.m = some sch) (i : ℕ) (b₀ b₁ c : ByteArray)
    (h0 : verifyRep spec st sch sess i 0 b₀ = .ok c) (h1 : verifyRep spec st sch sess i 1 b₁ = .ok c) :
    ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      decodeProof (shapeOf st sch extra.length) b₀ = .ok p₀ ∧
      decodeProof (shapeOf st sch extra.length) b₁ = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 S (schedOf sch) hm extra)
        (repMsgs st sch (2 + extra.length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 S (schedOf sch) hm extra)
        (repMsgs st sch (2 + extra.length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      pend₀.verdict (opensOf p₀ p₁ sch) = true ∧ pend₁.verdict (opensOf p₀ p₁ sch) = true ∧
      OpensOKL H512 enc512 leafPlain (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) ∧
      OpensOKL H512 enc512 leafPlain (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) := by
  have h15 : 15 ≤ S.m := by have := exec_fast100_ge hf; omega
  obtain ⟨hdim, hsm⟩ := fast100_dims st.m sch hf
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hschm : sch.m = S.m := hsm.trans hmS
  obtain ⟨s₀, p₀, hs₀, hp₀, hc₀, -, hsz₀, hw₀, hrest₀⟩ := rep_refines spec st S hm h15 sch hmS hkS hfold extra
    sess hext hdim hlv hα1 hschm hwf i 0 _ _ h0
  obtain ⟨s₁, p₁, hs₁, hp₁, hc₁, -, hsz₁, hw₁, hrest₁⟩ := rep_refines spec st S hm h15 sch hmS hkS hfold extra
    sess hext hdim hlv hα1 hschm hwf i 1 _ _ h1
  obtain ⟨pend₀, hpl₀, hcaps₀, hreqs₀, hver₀⟩ := hrest₀ capDig (opensOf p₀ p₁ sch) fun ℓ _ j _ => rfl
  obtain ⟨pend₁, hpl₁, hcaps₁, hreqs₁, hver₁⟩ := hrest₁ capDig (opensOf p₀ p₁ sch) fun ℓ _ j _ => rfl
  rw [hms] at hreqs₀ hreqs₁ hw₀ hw₁
  have hok₀ : OpensOKL H512 enc512 leafPlain (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) :=
    opensOK_of _ sch hlv p₀.cap p₀.ligerito 0 pend₀ (opensOf p₀ p₁ sch) hcaps₀ hreqs₀ fun ℓ j => rfl
  have hok₁ : OpensOKL H512 enc512 leafPlain (capDig p₁.cap) 1 pend₁ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) :=
    opensOK_of _ sch hlv p₁.cap p₁.ligerito 1 pend₁ (opensOf p₀ p₁ sch) hcaps₁ hreqs₁ fun ℓ j => rfl
  have hcapeq : p₁.cap = p₀.cap := by
    apply Array.ext'
    refine entries_eq_of_flat _ _ (fun x hx => hw₁ x hx) (fun x hx => hw₀ x hx) (by simp [hsz₀, hsz₁]) ?_
    have e := congrArg (fun b : ByteArray => b.data.toList) (hc₁.symm.trans hc₀)
    simp only [← Array.foldl_toList, flat_data] at e
    simpa using e
  rw [hcapeq] at hok₁
  exact ⟨p₀, p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hver₀, hver₁, hok₀, hok₁⟩

/-- **The table's steps after the link points**, as `Plays.interleave_first` schedules them: rep 0's steps, each after
the prover's choice `true`; then rep 1's; then the openings with their salts. -/
def tableTape (ts₀ ts₁ : List Game.Step) (os : Opens 𝔽 Dig × Salts (List UInt8)) : List Game.Step :=
  firstSteps ts₀ ++ ts₁ ++ [.inl ⟨Opens 𝔽 Dig × Salts (List UInt8), os⟩]

section target

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

open Classical in
/-- **The compiled inner verifier accepts from both reps' accepting checks** (the last step of `StagesAccept`).
When both reps of table `i` accept with one cap (`verifyRep`, with `table_tableAfterL`'s side conditions) on the
class's hidden statement `S`, and the inner transcript `t`'s steps on the schedule are the level-0 cap of rep 0's proof
file, the link points `pts`, the reps' tapes (`RepTape`, at the proof files `p₀ p₁` and the streams `s₀ s₁` the checks
read) in `tableTape`'s order, and the proof files' openings and salts, then `flockInnerCL` at the verifier's own
definitions accepts `t`. -/
theorem inner_accepts_of_reps (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls)
    (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hS : M.toStatement regs = S)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S)
    (hms : st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") (pts : Fin 2 → Fin cls.mPts → 𝔽)
    (sess : Session)
    (hext : ExtraRealizes st S sess.points (extraClaims S cls.ptLocal cls.mPts pts))
    (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (sch : Flock.Schedule) (hf : Flock.fast100 st.m = some sch) (i : ℕ) (b₀ b₁ c : ByteArray)
    (h0 : verifyRep spec st sch sess i 0 b₀ = .ok c) (h1 : verifyRep spec st sch sess i 1 b₁ = .ok c)
    (p₀ p₁ : Proof) (s₀ s₁ : StreamRec)
    (hp₀ : decodeProof (shapeOf st sch (extraClaims S cls.ptLocal cls.mPts pts).length) b₀ = .ok p₀)
    (hp₁ : decodeProof (shapeOf st sch (extraClaims S cls.ptLocal cls.mPts pts).length) b₁ = .ok p₁)
    (hs₀ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) = some s₀)
    (hs₁ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) = some s₁)
    (ts₀ ts₁ : List Game.Step)
    (ht₀ : RepTape st S sch (extraClaims S cls.ptLocal cls.mPts pts).length p₀ s₀ ts₀)
    (ht₁ : RepTape st S sch (extraClaims S cls.ptLocal cls.mPts pts).length p₁ s₁ ts₁)
    (t : List (List (SMsg T) × ((k : Fin r) → CT k)))
    (ht : schedSteps T CT 0 t = .inl ⟨Cap Dig, capDig p₀.cap⟩ :: .inr ⟨Fin 2 → Fin cls.mPts → 𝔽, pts⟩ ::
      tableTape ts₀ ts₁ (opensOf p₀ p₁ sch, saltsOf p₀ p₁ sch)) :
    (flockInnerCL T CT execArith H512 enc512 leafPlain cls regs (schedOf sch)).accepts M t := by
  subst hS
  obtain ⟨p₀', p₁', s₀', s₁', pend₀, pend₁, hp₀', hp₁', hs₀', hs₁', hpl₀, hpl₁, hv₀, hv₁, hok₀, hok₁⟩ :=
    table_refines_plain spec st (M.toStatement regs) cls.hm hmS hkS hfold hms
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts pts) sess hext hwf sch hf i b₀ b₁ c h0 h1
  obtain rfl : p₀ = p₀' := Except.ok.inj (hp₀.symm.trans hp₀')
  obtain rfl : p₁ = p₁' := Except.ok.inj (hp₁.symm.trans hp₁')
  obtain rfl : s₀ = s₀' := Option.some.inj (hs₀.symm.trans hs₀')
  obtain rfl : s₁ = s₁' := Option.some.inj (hs₁.symm.trans hs₁')
  have hT₀ := playsT_of_tape 0 _ _ cls.hm _ hpl₀ ht₀.1 ht₀.2.1 ht₀.2.2
  have hT₁ := playsT_of_tape 1 _ _ cls.hm _ hpl₁ ht₁.1 ht₁.2.1 ht₁.2.2
  have hint := playsT_interleave_first hT₀ hT₁ (repC_ne_ret 1 _ _ cls.hm _)
  refine (flockInnerCL_accepts T CT execArith H512 enc512 leafPlain cls regs (schedOf sch) M t).mpr
    ⟨capDig p₀.cap, pts, _,
      ⟨pend₀.verdict (opensOf p₀ p₁ sch) && pend₁.verdict (opensOf p₀ p₁ sch) &&
        decide (OpensOKL H512 enc512 leafPlain (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) ∧
          OpensOKL H512 enc512 leafPlain (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch)),
        capDig p₀.cap, (pend₀, pend₁), opensOf p₀ p₁ sch, saltsOf p₀ p₁ sch⟩, ht, ?_, ?_⟩
  · unfold tableAfterL
    exact playsT_bind hint (.send (opensOf p₀ p₁ sch, saltsOf p₀ p₁ sch) (.ret _))
  · simp only [hv₀, hv₁, Bool.true_and, decide_eq_true_eq]
    exact ⟨hok₀, hok₁⟩

end target

/-! ## The round count -/

section rounds

variable {α β : Type}

/-- A run of `g.bind f` is a run of `g`, then a run of `f` at its output. -/
theorem playsT_bind_inv {g : Game α} {f : α → Game β} {ts : List Game.Step} {b : β} (h : PlaysT (g.bind f) ts b) :
    ∃ ts₁ ts₂ a, ts = ts₁ ++ ts₂ ∧ PlaysT g ts₁ a ∧ PlaysT (f a) ts₂ b := by
  induction g generalizing ts with
  | ret a => exact ⟨[], ts, a, rfl, .ret a, h⟩
  | send M k ih =>
      cases ts with
      | nil => cases h.nil_inv
      | cons s ts =>
          obtain ⟨m, rfl, hk⟩ := PlaysT.send_inv h
          obtain ⟨ts₁, ts₂, a, rfl, h₁, h₂⟩ := ih m hk
          exact ⟨_ :: ts₁, ts₂, a, rfl, .send m h₁, h₂⟩
  | @coin C iF iN k ih =>
      cases ts with
      | nil => cases h.nil_inv
      | cons s ts =>
          obtain ⟨x, rfl, hk⟩ := PlaysT.coin_inv h
          obtain ⟨ts₁, ts₂, a, rfl, h₁, h₂⟩ := ih x hk
          exact ⟨_ :: ts₁, ts₂, a, rfl, .coin x h₁, h₂⟩

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

omit [Fintype K] in
/-- Every run of `tableAfterL` ends with the openings' message. -/
theorem tableAfterL_last (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Model.Statement) (sch : Accounting.Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) {ts : List Game.Step} {out : OutCL F D Sl}
    (h : PlaysT (tableAfterL A H E Lf S sch hm ptLocal mPts cap₀ pts) ts out) :
    ∃ l os, ts = l ++ [.inl ⟨Opens F D × Salts Sl, os⟩] := by
  unfold tableAfterL at h
  obtain ⟨ts₁, ts₂, a, rfl, -, h₂⟩ := playsT_bind_inv h
  cases ts₂ with
  | nil => cases h₂.nil_inv
  | cons s ts₂ =>
      obtain ⟨os, rfl, h₃⟩ := h₂.send_inv
      obtain ⟨rfl, -⟩ := h₃.ret_inv
      exact ⟨ts₁, os, rfl⟩

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

omit [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)] in
/-- On a nonempty transcript of rounds `j, …` that all draw coins, the schedule's last step is a coin. -/
theorem schedSteps_last : ∀ (j : ℕ) (t : List (List (SMsg T) × ((k : Fin r) → CT k))), t ≠ [] →
    j + t.length ≤ r → ∃ l x, schedSteps T CT j t = l ++ [.inr x]
  | _, [], h, _ => absurd rfl h
  | j, [(ms, ρ)], _, hl => by
      have hj : j < r := by simp at hl; omega
      exact ⟨ms.map (fun e => Sum.inl ⟨T e.1, e.2⟩), ⟨CT ⟨j, hj⟩, ρ ⟨j, hj⟩⟩, by simp [schedSteps, hj]⟩
  | j, (ms, ρ) :: e :: t, _, hl => by
      have hj : j < r := by simp at hl; omega
      obtain ⟨l, x, hx⟩ := schedSteps_last (j + 1) (e :: t) (by simp) (by simp at hl ⊢; omega)
      refine ⟨ms.map (fun e => Sum.inl ⟨T e.1, e.2⟩) ++ (Sum.inr ⟨CT ⟨j, hj⟩, ρ ⟨j, hj⟩⟩ :: l), x, ?_⟩
      rw [schedSteps, hx]
      simp [hj]

/-- **`flockInnerCL` accepts no transcript of at most `r` rounds.** On such a transcript the schedule's last step is a
coin, and every run of the table after the link points ends with the openings' message. -/
theorem flockInnerCL_rejects_short (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Accounting.Schedule) (M : Hidden cls) (t : List (List (SMsg T) × ((k : Fin r) → CT k)))
    (ht : t.length ≤ r) : ¬ (flockInnerCL T CT A H E Lf cls regs sch).accepts M t := by
  rw [flockInnerCL_accepts]
  rintro ⟨cap₀, pts, ts, out, hs, hp, -⟩
  obtain ⟨l, os, rfl⟩ := tableAfterL_last A H E Lf _ sch cls.hm cls.ptLocal cls.mPts cap₀ pts hp
  have hne : t ≠ [] := by
    rintro rfl
    simp [schedSteps] at hs
  obtain ⟨l', x, hx⟩ := schedSteps_last T CT 0 t hne (by omega)
  rw [hx] at hs
  have h := congrArg List.reverse hs
  simp at h

end rounds

end FlockVBridge.InnerAccepts
