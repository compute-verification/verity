import Proofs.Flock.VBridge.Family.Opened
import Proofs.Flock.VBridge.Algebra.Holo
import Proofs.Flock.Claims.Recursion.VStarParts

/-!
# A rep's Ligerito openings at V*'s scheme, query by query

`RunsFrom.opened` (`FlockVBridge.Family.runs_of_files`) asks, at each rep's decoded proof, that every level the
verifier's Ligerito run queries opens to the level's cap (`RecAlgebra.Opened`) under V*'s scheme `plain`
(`flock-leaf/sha512-unsalted`). V*'s rec-open parts give the openings one query at a time: a copy climbs from
`SHA-512(row)` through its siblings at the index its `dirs` bits spell (`Readable.climb_of_check`). This file states
what one query must give and assembles a level and a rep from it; nothing in it reads the verdict's pieces.

* `QueryOpens`: query `j` of level `ℓ` opens at position `pos`: the opened row has the level's lanes and climbs under
  `plain` through the query's siblings to the cap's entry above them, `pos >>> (d − c)`. `LevelShaped`: the level's
  openings are `n` rows, `n` blocks of `d − c` 64-byte siblings, and no salts.
* `opened_plain_of`: `Opened plain` at a context, from its level's shape, every query's `QueryOpens` at the context's
  positions, and the consistency sums being the opened rows' (`consAt`).
* `openedAll_of_levels`: every context `verifierRows` gives, from one fact per level at the level's context
  (`rows_conss`). `verifierRows_info_comb`: the contexts and `κ` don't read the comb, so the coins' run at
  `RecCoins.ZERO_COMB` (`staged_rep_drawn`) gives the rep's.
* `queryOpens_of_climb`: `QueryOpens` from a climb at any index whose low `d − c` bits are the position's.
  `climb_of_outs`, `queryOpens_of_outs`: the climb a rec-open copy computes is on its first 64 output bytes
  (`Readable.OpenOuts`), so `QueryOpens` holds where those bytes are the cap's entry. `dirsRow_testBit`: the `dirs` row
  check 5 registers (`RecCoins.checkOpen`) spells the position's low `H` bits, 16 a word.
-/

namespace RecDeploy.OpenedPlain

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockVBridge.Family

/-- **Query `j` of level `ℓ` opens at `pos`** under `plain`: the row `p` opens there has the level's lanes, and the
climb from its plain leaf through the query's `d − c` siblings at `pos` is the cap's entry `pos >>> (d − c)`. -/
def QueryOpens (sch : Flock.Schedule) (cap : Array ByteArray) (p : LigeritoProof) (ℓ j pos : ℕ) : Prop :=
  ((openAt p (sch.levels.size - 1) ℓ).1[j]!).size = 2 ^ (sch.levels[ℓ]!).lanes ∧
    climb plain (Sha512.hash (MerkleScheme.rowBytes (openAt p (sch.levels.size - 1) ℓ).1[j]!)) pos
        (sibsOf (openAt p (sch.levels.size - 1) ℓ).2.1 ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) j) =
      (capAt cap p ℓ).getD (pos >>> ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap)) ByteArray.empty

/-- **Level `ℓ`'s openings have `n` queries' shape**: `n` rows, `n` blocks of `d − c` siblings of 64 bytes, no salts
(V*'s openings have no salt port, rec_open.py:37). -/
def LevelShaped (sch : Flock.Schedule) (p : LigeritoProof) (ℓ n : ℕ) : Prop :=
  (openAt p (sch.levels.size - 1) ℓ).1.size = n ∧
    (openAt p (sch.levels.size - 1) ℓ).2.1.size = n * ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) ∧
    (∀ x ∈ (openAt p (sch.levels.size - 1) ℓ).2.1, x.size = 64) ∧ (openAt p (sch.levels.size - 1) ℓ).2.2 = #[]

/-- **A context's openings pass at `plain`**: where its level's caps have the schedule's shape (`CapsAt`), its openings
have the context's number of queries (`LevelShaped`), every query opens at the context's position, and the decoded
consistency sums are the opened rows' (`consAt` at the context's `α`). -/
theorem opened_plain_of {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof} {M : Messages F128}
    {cs : Flock.ConsCtx} (hℓ : cs.start < sch.levels.size) (hcaps : CapsAt sch cap p)
    (hsh : LevelShaped sch p cs.start cs.pos.size)
    (hq : ∀ j < cs.pos.size, QueryOpens sch cap p cs.start j cs.pos[j]!)
    (hcons : M.cons cs.start ((eqTable cs.alpha).extract 0 (sch.levels[cs.start]!).q) =
      (Array.range (2 ^ (sch.levels[cs.start]!).lanes)).map
        (consAt (openAt p (sch.levels.size - 1) cs.start).1 (sch.levels[cs.start]!).q cs.alpha)) :
    Opened plain sch cap p M cs := by
  obtain ⟨hc1, hc2⟩ := hcaps cs.start hℓ
  obtain ⟨h2, h3, h6, h4⟩ := hsh
  unfold Opened
  rw [Conss.getD_default]
  refine ⟨?_, hcons⟩
  rw [h4]
  refine Conss.merkleCheck_of hc1 h2 h3 rfl hc2 h6 fun j hj => ?_
  obtain ⟨hs, hc⟩ := hq j hj
  unfold opens
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨⟨⟨hs, ?_⟩, by simp; rfl⟩, hc⟩
  have hle : j * ((sch.levels[cs.start]!).d - (sch.levels[cs.start]!).cap) +
      ((sch.levels[cs.start]!).d - (sch.levels[cs.start]!).cap) ≤
        cs.pos.size * ((sch.levels[cs.start]!).d - (sch.levels[cs.start]!).cap) := by
    have := Nat.mul_le_mul_right ((sch.levels[cs.start]!).d - (sch.levels[cs.start]!).cap) hj
    rwa [Nat.succ_mul] at this
  simp only [sibsOf, Array.length_toList, Array.size_extract]
  omega

/-- **Every context of the verifier's Ligerito run**, from one fact per level: `verifierRows`' contexts are the
schedule's levels in order, level `ℓ`'s at index `ℓ` with the level's `q` positions (`rows_conss`). -/
theorem openedAll_of_levels {ms : MerkleScheme} {sch : Flock.Schedule} {sh : Shape} {cap : Array ByteArray}
    {p : LigeritoProof} {M : Messages F128} {rounds : Array (Array F128)} {extras : Array Claim}
    {comb v : Array F128} {info : Info Lin} {kappa : Array (Array (Array F128))}
    (hv : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (h0 : 0 < sh.sched.levels.size)
    (hq : ∀ i < sh.sched.levels.size, ∀ c,
      (Flock.positions (sh.sched.levels[i]!) c).size = (sh.sched.levels[i]!).q)
    (hlev : ∀ ℓ < sh.sched.levels.size, (info.lg.conss[ℓ]!).start = ℓ →
      (info.lg.conss[ℓ]!).pos.size = (sh.sched.levels[ℓ]!).q → Opened ms sch cap p M (info.lg.conss[ℓ]!)) :
    ∀ cs ∈ info.lg.conss.toList, Opened ms sch cap p M cs := by
  intro cs hcs
  obtain ⟨hℓ, he, hpsz⟩ := Conss.conss_mem (rows_conss hv h0 hq) hcs
  have h := hlev cs.start hℓ (by rw [he]) (by rw [he]; exact hpsz)
  rwa [he] at h

/-- **`verifierRows`' contexts and `κ` don't read the comb** (`verifierRows_comb`): two runs at the same coins and
claims, at any two combs, give one `Info` and one `κ`. -/
theorem verifierRows_info_comb {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim}
    {comb comb' v v' : Array F128} {info info' : Info Lin} {kappa kappa' : Array (Array (Array F128))}
    (h : verifierRows sh rounds extras comb = .ok (v, info, kappa))
    (h' : verifierRows sh rounds extras comb' = .ok (v', info', kappa')) : info' = info ∧ kappa' = kappa := by
  obtain ⟨V, -, hall⟩ := verifierRows_comb h
  rw [hall comb'] at h'
  simp only [Except.ok.injEq, Prod.mk.injEq] at h'
  exact ⟨h'.2.1.symm, h'.2.2.symm⟩

/-- **`QueryOpens` from a copy's climb**: where the row and siblings a climb starts from are query `j`'s, the row has
the level's lanes, and the climb's index agrees with `pos` on its low `d − c` bits, the climb reaching the cap's entry
`pos >>> (d − c)` is query `j`'s opening at `pos` (`climb_congr`: a climb reads its index's first `|ss|` bits). -/
theorem queryOpens_of_climb {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof} {ℓ j pos idx : ℕ}
    {r : Array F128} {ss : List ByteArray} (hr : (openAt p (sch.levels.size - 1) ℓ).1[j]! = r)
    (hss : sibsOf (openAt p (sch.levels.size - 1) ℓ).2.1 ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) j = ss)
    (hsize : r.size = 2 ^ (sch.levels[ℓ]!).lanes) (hlen : ss.length = (sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap)
    (hbits : ∀ h < (sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap, idx.testBit h = pos.testBit h)
    (hclimb : climb plain (Sha512.hash (MerkleScheme.rowBytes r)) idx ss =
      (capAt cap p ℓ).getD (pos >>> ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap)) ByteArray.empty) :
    QueryOpens sch cap p ℓ j pos := by
  unfold QueryOpens
  rw [hr, hss]
  exact ⟨hsize, (Conss.climb_congr plain ss _ pos idx fun h hh => (hbits h (hlen ▸ hh)).symm).trans hclimb⟩

/-- **A rec-open copy's first 64 output bytes are its climb** (`Readable.OpenOuts`, what `Readable.climb_of_check`
gives at a satisfied table): the bytes from `outCol` are the climb under `plain` from the row's plain leaf. -/
theorem climb_of_outs {v : ℕ → Bool} {LANES H : ℕ} {r a cf : Array F128} {rest : ByteArray} {idx : ℕ}
    {ss : List ByteArray} (h : Readable.OpenOuts v plain LANES H r a cf rest idx ss) :
    bytesAt v (outCol LANES H) 64 = climb plain (Sha512.hash (MerkleScheme.rowBytes r)) idx ss := by
  unfold Readable.OpenOuts at h
  have hX := bytesAt_of_cols (colsHold_left (colsHold_left h))
  rw [FlockVBridge.AccChain.climb_size_of plain (fun _ _ => rfl) _ _ _ (sha512_size _)] at hX
  exact hX

/-- **`QueryOpens` from a rec-open copy's outputs**: `queryOpens_of_climb` at the climb on the copy's output columns
(`climb_of_outs`), where those columns' first 64 bytes are the cap's entry `pos >>> (d − c)` (`htop`, the copy's `out`
registered as a read of the level's top, `RecCoins.checkOpenTops`). -/
theorem queryOpens_of_outs {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof} {ℓ j pos : ℕ}
    {v : ℕ → Bool} {LANES H : ℕ} {r a cf : Array F128} {rest : ByteArray} {idx : ℕ} {ss : List ByteArray}
    (h : Readable.OpenOuts v plain LANES H r a cf rest idx ss) (hr : (openAt p (sch.levels.size - 1) ℓ).1[j]! = r)
    (hss : sibsOf (openAt p (sch.levels.size - 1) ℓ).2.1 ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) j = ss)
    (hsize : r.size = 2 ^ (sch.levels[ℓ]!).lanes) (hlen : ss.length = (sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap)
    (hbits : ∀ h < (sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap, idx.testBit h = pos.testBit h)
    (htop : bytesAt v (outCol LANES H) 64 =
      (capAt cap p ℓ).getD (pos >>> ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap)) ByteArray.empty) :
    QueryOpens sch cap p ℓ j pos :=
  queryOpens_of_climb hr hss hsize hlen hbits ((climb_of_outs h).symm.trans htop)

/-- **The `dirs` row spells the position's low bits**: bit `h % 16` of word `h / 16` of `RecCoins.dirsRow pos H` is
bit `h` of `pos`, for `h < H ≤ 32` (`rec_open.dirs_words`: the low `H` bits, 16 a word). -/
theorem dirsRow_testBit {pos H h : ℕ} (hH : H ≤ 32) (hh : h < H) :
    ((RecCoins.dirsRow pos H)[h / 16]!).testBit (h % 16) = pos.testBit h := by
  by_cases h16 : h < 16
  · rw [Nat.div_eq_of_lt h16, Nat.mod_eq_of_lt h16]
    have e : (RecCoins.dirsRow pos H)[0]! = pos % 2 ^ H % 2 ^ 16 := by
      rw [getElem!_pos _ 0 (by simp [RecCoins.dirsRow, RecCoins.DIRS_WORDS])]
      simp [RecCoins.dirsRow]
    rw [e, Nat.testBit_mod_two_pow, Nat.testBit_mod_two_pow]
    simp [hh, h16]
  · rw [show h / 16 = 1 by omega]
    have e : (RecCoins.dirsRow pos H)[1]! = pos % 2 ^ H / 2 ^ 16 := by
      rw [getElem!_pos _ 1 (by simp [RecCoins.dirsRow, RecCoins.DIRS_WORDS])]
      simp [RecCoins.dirsRow]
    rw [e, Nat.testBit_div_two_pow, Nat.testBit_mod_two_pow, show h % 16 + 16 = h by omega]
    simp [hh]

end RecDeploy.OpenedPlain
