import Proofs.Flock.Soundness.Refine.Basis
import Flock.Verify

/-!
# One rep refines the model's (PROTOCOL.md §9–§14)

`rep_refines`: when the executable's `verifyRep` accepts rep `r`, the record carries the rep's stream and the proof
decodes. Then, at the proof's openings, the model's compiled rep `repC`, played with the proof's messages (`repMsgs`) and
the coins of every round of the stream, ends in a result:
- whose verdict is true;
- whose caps are the proof's;
- whose requests passed the executable's Merkle checks.

It composes the four phases: `zerocheck_refines`, `lincheck_refines`, `opening_refines` and `ligerito_accepts_opening`.
The statement's own facts enter as `Realizes`-style hypotheses: `FoldRealizes` for the circuit fold, `ExtraRealizes`
for the extra claims.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- The shape `verifyRep` decodes a proof with (§8.3), `nExtra` the statement's extra claims. -/
def shapeOf (st : Setup) (sch : Flock.Schedule) (nExtra : ℕ) : Shape :=
  { m := st.m, kLog := st.kLog, claims := 2 + nExtra, sched := sch, hashLen := st.merkle.digestLen,
    saltLen := if st.merkle.saltBytes = 0 && !st.openedSalts then none else some st.merkle.saltBytes }

/-- The statement's extra claims at the link points `points` are the model's `extra`, claim by claim. -/
def ExtraRealizes (st : Setup) (S : Model.Statement) (points : Array (Array F128))
    (extra : List (Model.Claim 𝔽 S.m)) : Prop :=
  ∀ extras, st.extra points = .ok extras →
    extras.size = extra.length ∧ ∀ (j : ℕ) (hj : j < extra.length), ClaimRel S.m extras[j]! extra[j]

/-- **The messages of one rep** (§9–§13): the zerocheck's, the lincheck's, one ring switch per claim, then
Ligerito's. -/
def repMsgs {D : Type} (st : Setup) (sch : Flock.Schedule) (nClaims : ℕ) (p : Proof)
    (capD : Array ByteArray → Cap D) : List Msg :=
  zcMsgs st.m p.zerocheck ++ lcMsgs st.kLog p.lincheck ++ opMsgs nClaims p.ringSwitches ++ ligMsgs p.ligerito capD sch

theorem ebind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases hx : x with
  | error e => rw [hx] at h; cases h
  | ok a => rw [hx] at h; exact ⟨a, rfl, h⟩

theorem consumed_all (rs : Array Round) : consumed rs 0 rs.size = rs.toList.map (·.coins) := by
  unfold consumed
  refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
  simp only [List.getElem_map, List.getElem_range', Nat.sub_zero, Nat.zero_add, one_mul, Array.getElem_toList]
  rw [getElem!_pos rs i (by simpa using h₂)]

/-- A passed Merkle check had a cap of `2^c` entries, each of the scheme's digest width. -/
theorem merkleCheck_cap {ms : MerkleScheme} {cap : Array ByteArray} {c d : ℕ} {pos : Array ℕ}
    {rows : Array (Array F128)} {sibs salts : Array ByteArray} {lanes : ℕ}
    (h : merkleCheck ms cap c d pos rows sibs salts lanes = .ok ()) :
    cap.size = 2 ^ c ∧ ∀ b ∈ cap.toList, b.size = ms.digestLen := by
  unfold merkleCheck at h
  split at h
  · cases h
  rename_i h₁
  split at h
  · cases h
  rename_i h₂
  simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not] at h₁
  simp only [Bool.or_eq_true, not_or, Array.any_eq_true, not_exists, bne_iff_ne, ne_eq, not_not] at h₂
  refine ⟨h₁.1.1.1, fun b hb => ?_⟩
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hb)
  exact h₂.1 i hi

/-- **The table's cap** (§14): an accepting Ligerito run's level-0 cap has `2^c` entries of the digest width, since its
level-0 queries passed their Merkle check. -/
theorem ligerito_cap (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (basisAt : Array F256 → ℕ → Array F256) (hq : (sch.levels.getD 0 default).q ≠ 0) {t t' : Transcript}
    (h : Flock.ligerito ms sch T cap p basisAt t = .ok ((), t')) :
    cap.size = 2 ^ (sch.levels.getD 0 default).cap ∧ ∀ b ∈ cap.toList, b.size = ms.digestLen := by
  unfold Flock.ligerito at h
  obtain ⟨_, h⟩ := check_bind h
  obtain ⟨_, _, h⟩ := label_bind h
  obtain ⟨_, _, h⟩ := absorbF_bind h
  obtain ⟨_, _, h⟩ := absorbBytes_bind h
  obtain ⟨_, _, h₄, -⟩ := bind_ok h
  rw [StateT.run] at h₄
  obtain ⟨_, _, _, h₄⟩ := lift_bind h₄
  obtain ⟨_, _, h₄⟩ := nextOod_bind h₄
  obtain ⟨_, _, _, h₄⟩ := lift_bind h₄
  obtain ⟨_, _, h₄⟩ := claimPow_bind h₄
  obtain ⟨_, _, _, h₄⟩ := lift_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨_, _, _, _, h₄⟩ := nextMsg_bind h₄
  have h₄ := lmodify_bind h₄
  obtain ⟨_, _, _, _, h₄⟩ := lbind_ok h₄
  obtain ⟨_, _, _, h₄⟩ := lift_bind h₄
  obtain ⟨-, -, _, _, _, _, _, _, _, h₄⟩ := oodIntro_bind (by omega) h₄
  obtain ⟨-, -, _, _, _, _, _, _, _, h₄⟩ := oodIntro_bind (by omega) h₄
  obtain ⟨-, -, hmc, -⟩ := queryLevel_bind hq h₄
  exact merkleCheck_cap hmc

/-- **One rep refines the model's** (§9–§14). When the executable's `verifyRep` accepts table `i`'s rep `r`, on a statement whose
`m`, `k_log`, circuit fold and extra claims are the model's `S` and `extra`, and on a schedule whose dimensions and
levels are the model's and whose levels' consistency coins are not one word, the record carries the rep's stream and
the proof decodes. Then at the proof's openings `o`, the model's compiled rep, played with the proof's messages and the
coins of every round of the stream, accepts. Its caps are the proof's, and its requests passed the executable's Merkle
checks. -/
theorem rep_refines {D : Type} (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m) (h15 : 15 ≤ S.m)
    (sch : Flock.Schedule) (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S)
    (extra : List (Model.Claim 𝔽 S.m)) (sess : Session) (hext : ExtraRealizes st S sess.points extra)
    (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1) (hschm : sch.m = S.m)
    (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds) (i r : ℕ) (bytes cap : ByteArray)
    (h : verifyRep spec st sch sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (p : Proof),
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream ∧
      decodeProof (shapeOf st sch extra.length) bytes = .ok p ∧
      cap = p.cap.foldl (· ++ ·) ByteArray.empty ∧ (cap == sess.rootB[i]!) = true ∧
      p.cap.size = 2 ^ (sch.levels.getD 0 default).cap ∧ (∀ b ∈ p.cap.toList, b.size = st.merkle.digestLen) ∧
      ∀ (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D),
        (∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
          (o r ℓ j).1 = ((openAt p.ligerito (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf) →
        ∃ pend : PendL 𝔽 D,
          Plays fOf (Model.repC execArith r S (schedOf sch) hm extra) (repMsgs st sch (2 + extra.length) p capD)
            (stream.rounds.toList.map (·.coins)) pend ∧
          pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capD p.ligerito.recursiveCaps[j]!) ∧
          (∀ req ∈ pend.reqs, ReqOK st.merkle sch p.cap p.ligerito req) ∧ pend.verdict o = true := by
  unfold verifyRep at h
  dsimp only at h
  cases hs : Array.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r)
      sess.streams with
  | none => rw [hs] at h; cases h
  | some stream =>
      rw [hs] at h
      obtain ⟨extras, hex, h⟩ := ebind_ok h
      obtain ⟨p, hp, h⟩ := ebind_ok h
      by_cases c1 : (p.params != pcsParams st.merkle sch) = true
      · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
      rw [ite_eq_right c1] at h
      obtain ⟨⟨u, t⟩, hrun, h⟩ := ebind_ok h
      change (if (t.k != stream.rounds.size) = true then _ else _) = _ at h
      by_cases c2 : (t.k != stream.rounds.size) = true
      · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
      rw [ite_eq_right c2] at h
      by_cases c3 : (Array.foldl (fun x1 x2 => x1 ++ x2) ByteArray.empty p.cap != sess.rootB[i]!) = true
      · rw [ite_eq_left c3] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
      rw [ite_eq_right c3] at h
      cases h
      simp only [bne_iff_ne, ne_eq, not_not] at c2
      have c3' : (Array.foldl (fun x1 x2 => x1 ++ x2) ByteArray.empty p.cap == sess.rootB[i]!) = true := by
        cases hb : (Array.foldl (fun x1 x2 => x1 ++ x2) ByteArray.empty p.cap == sess.rootB[i]!)
        · exact absurd (by simp [bne, hb]) c3
        · rfl
      obtain ⟨hlenx, hcx⟩ := hext extras hex
      have hcapE : p.cap.size = 2 ^ (sch.levels.getD 0 default).cap ∧
          ∀ b ∈ p.cap.toList, b.size = st.merkle.digestLen := by
        rw [StateT.run] at hrun
        obtain ⟨_, _, _, hrun⟩ := bind_ok hrun
        obtain ⟨_, _, _, hrun⟩ := bind_ok hrun
        obtain ⟨⟨_, _⟩, _, _, hlig⟩ := bind_ok hrun
        exact ligerito_cap _ _ _ _ _ _ (hlv 0 (by have := hdim.2.1; omega)).q_ne_zero hlig
      refine ⟨stream, p, rfl, by rw [← hlenx]; exact hp, rfl, c3', hcapE.1, hcapE.2, fun capD o ho => ?_⟩
      have hwf₀ : RoundsWF stream.rounds := hwf stream (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs))
      rw [StateT.run] at hrun
      obtain ⟨zc, t₁, h₁, hrun⟩ := bind_ok hrun
      obtain ⟨lc, t₂, h₂, hrun⟩ := bind_ok hrun
      obtain ⟨⟨T, sws⟩, t₃, h₃, h₄⟩ := bind_ok hrun
      rw [hmS] at h₁
      obtain ⟨hr₁, hk₁, zc', hpl₁, hzc⟩ := zerocheck_refines S.m h15 _ _ _ _ _ zc hwf₀ h₁
      obtain ⟨hr₂, hk₂, ab, hpl₂, hab⟩ := lincheck_refines S st.fold hfold zc zc' hzc p.lincheck t₁ t₂ lc
        (hr₁ ▸ hwf₀) h₂
      rw [hmS, hkS] at h₃
      have hcsz : (abcClaims S.m S.kLog zc lc ++ extras).size = (ab :: Model.cClaim execArith zc' :: extra).length := by
        simp [abcClaims, hlenx]; omega
      obtain ⟨hr₃, hk₃, op, hpl₃, hop⟩ := opening_refines S.m (by omega) _ (ab :: Model.cClaim execArith zc' :: extra)
        hcsz (fun k hk => by
          rcases k with _ | _ | k
          · rw [getElem!_pos _ 0 (by simp [abcClaims]), Array.getElem_append_left (by simp [abcClaims])]
            rw [getElem!_pos _ 0 (by simp [abcClaims])] at hab
            exact hab
          · rw [getElem!_pos _ 1 (by simp [abcClaims]; omega), Array.getElem_append_left (by simp [abcClaims])]
            exact cClaim_rel zc zc' hzc
          · have hk' : k < extra.length := by simp at hk; omega
            have hsz2 : (abcClaims S.m S.kLog zc lc).size = 2 := rfl
            rw [getElem!_pos _ (k + 2) (by rw [Array.size_append, hsz2]; omega),
              Array.getElem_append_right (by rw [hsz2]; omega)]
            simp only [hsz2, Nat.add_sub_cancel]
            rw [← getElem!_pos extras k (by omega)]
            exact hcx k hk')
        (by simp [abcClaims]) (by rw [Array.size_append, show (abcClaims S.m S.kLog zc lc).size = 2 from rfl]; omega)
        p.ringSwitches t₂ t₃ T sws (hr₂ ▸ hr₁ ▸ hwf₀) h₃
      rw [← hschm] at hop
      obtain ⟨pend, hr₄, hk₄, hpl₄, hcaps, hreqs, hver⟩ := ligerito_accepts_opening st.merkle sch T sws _ op hop p.cap
        p.ligerito r capD o hdim hlv hα1 ho (hr₃ ▸ hr₂ ▸ hr₁ ▸ hwf₀) h₄
      refine ⟨pend, ?_, hcaps, hreqs, hver⟩
      have hcs : stream.rounds.toList.map (·.coins) =
          consumed stream.rounds 0 t₁.k ++ (consumed t₁.rounds t₁.k t₂.k ++ (consumed t₂.rounds t₂.k t₃.k ++
            consumed t₃.rounds t₃.k t.k)) := by
        have e1 : t₁.rounds = stream.rounds := hr₁
        have e2 : t₂.rounds = stream.rounds := hr₂.trans hr₁
        have e3 : t₃.rounds = stream.rounds := hr₃.trans e2
        rw [e1, e2, e3, ← consumed_append _ hk₃ hk₄, ← consumed_append _ hk₂ (hk₃.trans hk₄),
          ← consumed_append _ (Nat.zero_le t₁.k) (hk₂.trans (hk₃.trans hk₄)), c2]
        exact (consumed_all _).symm
      have hsz : (abcClaims S.m S.kLog zc lc ++ extras).size = 2 + extra.length := by
        simp [abcClaims, hlenx]
      rw [hsz] at hpl₃
      rw [hcs]
      unfold Model.repC
      refine Plays.bind' (ms₂ := lcMsgs S.kLog p.lincheck ++ opMsgs (2 + extra.length) p.ringSwitches ++
        ligMsgs p.ligerito capD sch) hpl₁ ?_ (by simp [repMsgs, hmS, hkS]) rfl
      refine Plays.bind' (ms₂ := opMsgs (2 + extra.length) p.ringSwitches ++ ligMsgs p.ligerito capD sch) hpl₂ ?_
        (by simp) rfl
      exact Plays.bind' hpl₃ hpl₄ rfl rfl

end FlockSoundness.Refine
