import Proofs.Flock.Claims.Recursion.OpenedPlain
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes

/-!
# A decoded proof's openings have each level's shape

`OpenedPlain.opened_plain_of` takes each level's openings' shape (`LevelShaped`): the level's `q` rows, `q` blocks of
`d − c` siblings of 64 bytes, and no salts. `decodeProof` checks each of these lengths as it reads the openings
(`vecExact`, Proof.lean:201-221), so a decoded proof has them at a shape of 64-byte digests whose openings carry no
salts: none (`saltLen = none`) or none of their 0 bytes (`some 0`), as at a `Refine.shapeOf` of an unsalted scheme
(`saltLen_plain`, whether or not the statement's openings carry `opened_salts`): `levelShaped_of_decode`.

* `yields_forIn_all`: a loop whose step `i` pushes an element with `P i` returns its start followed by one element per
  index, element `k` with `P k` (the recursive levels' loop).
* `saltLen_plain`: an unsalted scheme's shape reads no salts, or empty ones.
* `levelShaped_of_decode`: at every level of the shape's schedule, `LevelShaped` of the decoded Ligerito proof.
-/

namespace RecDeploy.OpenedPlain

open Flock Reader
open FlockSoundness.Discharge.DecodeZL (Yields Post yields_bind yields_pure post_bind post_pure post_ite
  post_bind_throw post_bind_vec post_bind_vecBytes post_bind_switches post_bind_caps forall₂_range' yields_vecExact)
open FlockSoundness.Refine (openAt shapeOf)

/-- A loop whose step `i` pushes an element with `P i` returns its start followed by one element per index, in order,
each with its index's `P`. -/
theorem yields_forIn_all {α : Type} {P : ℕ → α → Prop} (body : ℕ → Array α → Reader.M (ForInStep (Array α)))
    (hb : ∀ i s, Post (body i s) fun st => ∃ x, P i x ∧ st = .yield (s.push x)) :
    ∀ (l : List ℕ) (acc out : Array α), Yields (forIn l acc body) out →
      ∃ xs : List α, out.toList = acc.toList ++ xs ∧ List.Forall₂ P l xs
  | [], acc, out, h => ⟨[], by rw [List.forIn_nil] at h; rw [yields_pure h, List.append_nil], .nil⟩
  | i :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := yields_bind h
    obtain ⟨x, hx, rfl⟩ := hb i acc st hst
    obtain ⟨xs, hout, hxs⟩ := yields_forIn_all body hb l _ out h
    refine ⟨x :: xs, ?_, .cons hx hxs⟩
    rw [hout, Array.toList_push, List.append_assoc, List.singleton_append]

/-- Openings of level `lv`: its `q` rows, `q` blocks of `d − c` siblings of `n` bytes, no salts. -/
abbrev LvShape (lv : Flock.Level) (n : ℕ) (rows : Array (Array F128)) (sibs salts : Array ByteArray) : Prop :=
  rows.size = lv.q ∧ sibs.size = lv.q * (lv.d - lv.cap) ∧ (∀ x ∈ sibs, x.size = n) ∧ salts = #[]

/-- A salts read at no salts or at 0-byte salts returns none. -/
theorem salts_nil {sl : Array ByteArray}
    (h : Yields (pure #[] : Reader.M (Array ByteArray)) sl ∨ ∃ w, Yields (vecExact 0 (Reader.bytes 0) w) sl) :
    sl = #[] := by
  rcases h with h | ⟨_, h⟩
  · exact yields_pure h
  · exact Array.eq_empty_of_size_eq_zero (yields_vecExact h).1

/-- **An unsalted scheme's shape reads no salts, or empty ones**, whether or not the statement's openings carry
`opened_salts`. -/
theorem saltLen_plain {st : Setup} {sch : Flock.Schedule} {n : ℕ} (h : st.merkle.saltBytes = 0) :
    (shapeOf st sch n).saltLen = none ∨ (shapeOf st sch n).saltLen = some 0 := by
  unfold shapeOf
  rw [h]
  cases st.openedSalts <;> simp

/-- **The openings a decoded proof carries have their levels' shapes**: level 0's, each recursive level's and the
last level's, at a shape whose openings carry no salts, or empty ones. -/
theorem decodeProof_openings {sh : Shape} {bytes : ByteArray} {p : Proof}
    (hsl : sh.saltLen = none ∨ sh.saltLen = some 0)
    (h : decodeProof sh bytes = .ok p) :
    LvShape (sh.sched.levels.getD 0 default) sh.hashLen p.ligerito.initialProof.rows
        p.ligerito.initialProof.siblings p.ligerito.initialProof.salts ∧
      p.ligerito.recursiveProofs.size = sh.sched.levels.size - 1 - 1 ∧
      (∀ k (hk : k < p.ligerito.recursiveProofs.size),
        LvShape (sh.sched.levels.getD (k + 1) default) sh.hashLen p.ligerito.recursiveProofs[k].rows
          p.ligerito.recursiveProofs[k].siblings p.ligerito.recursiveProofs[k].salts) ∧
      LvShape (sh.sched.levels.getD (sh.sched.levels.size - 1) default) sh.hashLen p.ligerito.finalProof.rows
        p.ligerito.finalProof.siblings p.ligerito.finalProof.salts := by
  replace h := (Flock.decodeProof_read h).1
  unfold decodeRead at h
  obtain ⟨r', hrun, -⟩ := Flock.Reader.runPlain_ok h
  refine (?_ : Post _ fun q : Proof =>
    LvShape (sh.sched.levels.getD 0 default) sh.hashLen q.ligerito.initialProof.rows
        q.ligerito.initialProof.siblings q.ligerito.initialProof.salts ∧
      q.ligerito.recursiveProofs.size = sh.sched.levels.size - 1 - 1 ∧
      (∀ k (hk : k < q.ligerito.recursiveProofs.size),
        LvShape (sh.sched.levels.getD (k + 1) default) sh.hashLen q.ligerito.recursiveProofs[k].rows
          q.ligerito.recursiveProofs[k].siblings q.ligerito.recursiveProofs[k].salts) ∧
      LvShape (sh.sched.levels.getD (sh.sched.levels.size - 1) default) sh.hashLen q.ligerito.finalProof.rows
        q.ligerito.finalProof.siblings q.ligerito.finalProof.salts) p ⟨_, _, hrun⟩
  dsimp only
  rcases hsl with hsl | hsl <;> simp only [hsl] <;>
    refine post_bind_vecBytes fun _ _ _ => ?_ <;>
    refine post_bind fun m _ => post_bind fun lir _ => post_bind fun lbs _ => post_bind fun profile _ => ?_ <;>
    refine post_bind fun tag _ => ?_ <;>
    split
  rotate_left 2
  · exact post_bind_throw
  rotate_left 2
  · exact post_bind_throw
  all_goals
    refine post_bind fun numLanes _ => post_bind fun mh _ => ?_
    refine post_bind_vec fun _ _ => post_bind_vec fun _ _ => post_bind_vec fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind_vec fun _ _ => post_bind_vec fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind_switches fun _ _ _ => post_bind fun _ _ => ?_
    refine post_bind_vecBytes fun icap _ _ => ?_
    refine post_bind_vec fun rows0 hrows0 => post_bind_vecBytes fun sib0 hsib0 hsib0E => ?_
    refine post_bind fun sl0 hsl0 => ?_
    refine post_bind fun n _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind_caps fun _ _ _ => ?_
    refine post_bind fun np _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind fun rproofs hrp => post_bind_vec fun _ _ => ?_
    refine post_bind_vec fun rowsF hrowsF => post_bind_vecBytes fun sibF hsibF hsibFE => ?_
    refine post_bind fun slF hslF => post_bind fun _ _ => ?_
    refine post_bind_vec fun _ _ => post_bind_vec fun _ _ => post_bind_vec fun _ _ => ?_
    refine post_bind fun _ _ => post_bind_vec fun _ _ => post_bind_vec fun _ _ => ?_
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hrp
    obtain ⟨xs, hout, hxs⟩ := yields_forIn_all
      (P := fun i (x : LevelProof) => LvShape (sh.sched.levels.getD (i + 1) default) sh.hashLen x.rows x.siblings
        x.salts) _ (by
          intro i s
          exact post_bind_vec fun a ha => post_bind_vecBytes fun b hb hbE => post_bind fun c hc =>
            post_pure ⟨⟨a, b, c⟩, ⟨ha, hb, hbE, salts_nil (by first | exact .inl hc | exact .inr ⟨_, hc⟩)⟩,
              rfl⟩) _ _ _ hrp
    have hsz : List.range' ([:sh.sched.levels.size - 1 - 1] : Std.Legacy.Range).start
        ([:sh.sched.levels.size - 1 - 1] : Std.Legacy.Range).size
        ([:sh.sched.levels.size - 1 - 1] : Std.Legacy.Range).step = List.range' 0 (sh.sched.levels.size - 1 - 1) := by
      simp [Std.Legacy.Range.size]
    rw [hsz] at hxs
    obtain ⟨hl, hk⟩ := forall₂_range' 0 _ hxs
    have hout' : rproofs.toList = xs := hout.trans (by simp)
    subst hout'
    have hrs : rproofs.size = sh.sched.levels.size - 1 - 1 := by rw [← Array.length_toList]; exact hl
    refine post_pure ⟨⟨hrows0, hsib0, hsib0E, salts_nil (by first | exact .inl hsl0 | exact .inr ⟨_, hsl0⟩)⟩,
      hrs, fun k hk' => ?_, ⟨hrowsF, hsibF, hsibFE, salts_nil (by first | exact .inl hslF | exact .inr ⟨_, hslF⟩)⟩⟩
    have hx := hk k (by simpa using hk')
    rw [Array.getElem_toList, Nat.zero_add] at hx
    exact hx

/-- **A decoded proof's openings have each level's shape** (`LevelShaped`), at a shape of 64-byte digests whose
openings carry no salts, or empty ones: `decodeProof_openings` read at `openAt`'s level `ℓ` (level 0, a recursive
level, or the last). -/
theorem levelShaped_of_decode {sh : Shape} {bytes : ByteArray} {p : Proof}
    (hsl : sh.saltLen = none ∨ sh.saltLen = some 0)
    (hh : sh.hashLen = 64) (h : decodeProof sh bytes = .ok p) :
    ∀ ℓ < sh.sched.levels.size, LevelShaped sh.sched p.ligerito ℓ (sh.sched.levels[ℓ]!).q := by
  obtain ⟨h0, hrs, hk, hF⟩ := decodeProof_openings hsl h
  rw [hh] at h0 hF hk
  intro ℓ hℓ
  have hof : ∀ {rows : Array (Array F128)} {sibs salts : Array ByteArray},
      LvShape (sh.sched.levels.getD ℓ default) 64 rows sibs salts →
      openAt p.ligerito (sh.sched.levels.size - 1) ℓ = (rows, sibs, salts) →
      LevelShaped sh.sched p.ligerito ℓ (sh.sched.levels[ℓ]!).q := by
    intro rows sibs salts hs ho
    rw [FlockVBridge.Family.Conss.getD_default] at hs
    obtain ⟨h1, h2, h3, h4⟩ := hs
    unfold LevelShaped
    rw [ho]
    exact ⟨h1, h2, h3, h4⟩
  by_cases hz : ℓ = 0
  · subst hz
    exact hof h0 (by simp [openAt])
  · by_cases hr : ℓ < sh.sched.levels.size - 1
    · have hi : ℓ - 1 < p.ligerito.recursiveProofs.size := by omega
      have hs := hk (ℓ - 1) hi
      rw [show ℓ - 1 + 1 = ℓ by omega] at hs
      refine hof hs ?_
      simp [openAt, hz, hr, getElem!_pos p.ligerito.recursiveProofs (ℓ - 1) hi]
    · have hℓR : ℓ = sh.sched.levels.size - 1 := by omega
      subst hℓR
      exact hof hF (by simp [openAt, hz])

end RecDeploy.OpenedPlain
