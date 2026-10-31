import Proofs.Flock.Claims.DeployRoundsZk
import Proofs.Flock.Claims.KS.ConflictMerkle
import Proofs.Flock.Claims.KS.ConflictWithin

/-!
# The pads openings an accepted masked rep logs

`Zk.verifyInnerWith` opens each of its `Q` queried pads columns as an `hm96-sha512/v1` opening of the row `[a, b]`
against the inner proof's root: the pair hashed, its leaf under the query's salt, then the climb at the query's
position `w.lo mod innerN`, `w` the query's coin.

* `verifyInner_logs`: run with `hashesIn L`, the inner proof reads four rounds, and at each query its path has the
  tree's depth, its climb reaches the root, and every SHA-512 input of the opening is in `L` (`opensIns` at the row
  `[a, b]`, whose encoding is the pair's, `rowBytes_pair`).
* `tiedColl_of_climbs`: two `hm96-sha512/v1` climbs from different rows to one top, along paths of one length of
  digest-width siblings, collide among their inputs; `tiedColl_of_opens` without the cap and the salt's width.
* `tiedColl_of_verifyInner`: so two queries of one accepted inner proof at one position, with different pairs, collide
  in `L`.
* `zkVerifyRepWith_inner`: an accepted masked rep ran its inner proof on its stream's rounds, ending at the last one.
-/

namespace FlockDeploy.KS

open Flock Transcript FlockSoundness.Refine
open FlockSoundness.Discharge.ZkExec (innerN layZ shapeZK)

attribute [local irreducible] Sha512.hash

/-- A loop whose every step yields and keeps the transcript ran each step from the transcript it started at. -/
theorem forIn_each {γ : Type} (body : γ → PUnit → TM (ForInStep PUnit))
    (hbody : ∀ x s t b t', body x s t = .ok (b, t') → b = ForInStep.yield PUnit.unit ∧ t' = t) :
    ∀ (l : List γ) (t t' : Transcript) (out : PUnit), forIn l PUnit.unit body t = .ok (out, t') →
      t' = t ∧ ∀ x ∈ l, ∃ b, body x PUnit.unit t = .ok (b, t) := by
  intro l
  induction l with
  | nil =>
    intro t t' out h
    obtain ⟨-, rfl⟩ := pure_ok h
    exact ⟨rfl, fun x hx => by cases hx⟩
  | cons y l ih =>
    intro t t' out h
    rw [List.forIn_cons] at h
    obtain ⟨b, t₁, h₁, h₂⟩ := bind_ok h
    obtain ⟨rfl, rfl⟩ := hbody y _ t b t₁ h₁
    obtain ⟨he, hl⟩ := ih _ t' out h₂
    refine ⟨he, fun x hx => ?_⟩
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨_, h₁⟩
    · exact hl x hx

theorem rowBytes_pair (a b : F128) : MerkleScheme.rowBytes #[a, b] = a.toBytes ++ b.toBytes := by
  show (ByteArray.empty ++ a.toBytes) ++ b.toBytes = _
  rw [ByteArray.empty_append]

theorem qs_get (a : Array F128) (n k : ℕ) : (a.map fun w => w.lo.toNat % n)[k]! = (a[k]!).lo.toNat % n := by
  by_cases hk : k < a.size
  · rw [getElem!_pos _ k (by simpa using hk), getElem!_pos a k hk, Array.getElem_map]
  · rw [getElem!_neg _ k (by simpa using hk), getElem!_neg a k hk]
    show (0 : ℕ) = (0 : UInt64).toNat % n
    simp

/-- **The inner proof's openings, logged**: when `Zk.verifyInnerWith` accepts with SHA-512 answered only within `L`,
it read four rounds, and at each query `k` (its position the coin `w` of the fourth round, at `w.lo mod innerN`) the
path has the tree's depth, the climb from the leaf of `[a, b]` under the query's salt reaches the root, and every
SHA-512 input of that opening is in `L`. -/
theorem verifyInner_logs {L : List Claims.Msg} {lay : Zk.PadLayout} {pr : Zk.InnerProof} {finalC : F128}
    {cons : Array (String × Zk.Lin)} {t t' : Transcript}
    (h : Zk.verifyInnerWith (hashesIn L) lay pr finalC cons t = .ok ((), t')) :
    t'.rounds = t.rounds ∧ t'.k = t.k + 4 ∧ pr.paths.size = Zk.Q ∧
      ∀ k < Zk.Q, (pr.paths[k]!).size = Nat.log2 (innerN lay) ∧
        climb MerkleScheme.hm96Sha512 (MerkleScheme.hm96Sha512.leaf #[(pr.opened[k]!).1, (pr.opened[k]!).2]
            pr.salts[k]!) (((t.rounds[t.k + 3]!).coins[k]!).lo.toNat % innerN lay) (pr.paths[k]!).toList =
          pr.root ∧
        ∀ b ∈ opensIns #[(pr.opened[k]!).1, (pr.opened[k]!).2] pr.salts[k]!
          (((t.rounds[t.k + 3]!).coins[k]!).lo.toNat % innerN lay) (pr.paths[k]!).toList, Claims.toMsg b ∈ L := by
  unfold Zk.verifyInnerWith at h
  simp only [liftTM_V] at h
  obtain ⟨hshape, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨-, -, t₃, r₃, kk₃, h⟩ := sampleWith_bind _ h
  obtain ⟨t₄, k₄, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₅, r₅, kk₅, h⟩ := sampleVecWith_bind _ (by simp [Zk.triple]) h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨-, -, t₇, r₇, kk₇, h⟩ := sampleWith_bind _ h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₈, k₈, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₉, r₉, kk₉, h⟩ := sampleVecWith_bind _ (by decide) h
  obtain ⟨tc, -, h⟩ := vlift_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨s, t₁₀, hloop, h⟩ := bind_ok h
  obtain ⟨-, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  have hr₈ : t₈.rounds = t.rounds := by rw [k₈.2, r₇, k₆.2, r₅, k₄.2, r₃, k₂.2, k₁.2]
  have hk₈ : t₈.k = t.k + 3 := by rw [k₈.1, kk₇, k₆.1, kk₅, k₄.1, kk₃, k₂.1, k₁.1]
  have hps : pr.paths.size = Zk.Q := by
    simp only [Bool.and_eq_true, beq_iff_eq] at hshape
    exact hshape.1.1.1.2
  have hall := forIn_each _ (by
      intro i s tt b tt' hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨_, -, hb⟩ := vlift_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      exact ⟨rfl, rfl⟩) _ _ _ _ hloop
  obtain ⟨rfl, hall⟩ := hall
  refine ⟨by rw [r₉, hr₈], by rw [kk₉, hk₈], hps, fun k hk => ?_⟩
  obtain ⟨b, hb⟩ := hall k (List.mem_range'.2 ⟨k, hk, by omega⟩)
  obtain ⟨-, hb⟩ := check_bind hb
  obtain ⟨hdep, hb⟩ := check_bind hb
  obtain ⟨x, hx, hb⟩ := vlift_bind hb
  obtain ⟨leaf, hleaf, hb⟩ := vlift_bind hb
  obtain ⟨top, htop, hb⟩ := vlift_bind hb
  obtain ⟨hroot, -⟩ := check_bind hb
  obtain ⟨rfl, hxm⟩ := hashesIn_sha512 hx
  obtain ⟨rfl, hsm, hlm⟩ :=
    Runs.leafW ((With.leafWith_sim (back L) (back_hashes L) _ _) leaf hleaf)
  have hops := With.opsWith_sim (back L) (back_hashes L) MerkleScheme.hm96Sha512
  change (back L).Ops (MerkleOps.hm96With Claims.Hashes.oc) (MerkleOps.hm96With (hashesIn L)) at hops
  obtain ⟨rfl, hcm⟩ := Runs.climbW ((With.climbWith_sim (back L) hops _ _ _) top htop)
  rw [qs_get, hr₈, hk₈] at hcm hroot
  have hlf : Hm96.Default512.leaf (Sha512.hash ((pr.opened[k]!).1.toBytes ++ (pr.opened[k]!).2.toBytes))
      pr.salts[k]! = MerkleScheme.hm96Sha512.leaf #[(pr.opened[k]!).1, (pr.opened[k]!).2] pr.salts[k]! := by
    show _ = Hm96.Default512.leaf (Sha512.hash (MerkleScheme.rowBytes _)) _
    rw [rowBytes_pair]
  rw [hlf] at hcm hroot
  refine ⟨beq_iff_eq.1 hdep, bytes_eq_of_beq hroot, fun y hy => ?_⟩
  rcases List.mem_append.1 hy with hy | hy
  · simp only [leafIns, rowBytes_pair, List.mem_cons, List.not_mem_nil, or_false] at hy
    rcases hy with rfl | rfl | rfl
    · exact hxm
    · exact hsm
    · exact hlm
  · exact hcm y hy

/-- `opens_binding` from the climbs' equality alone: two climbs from different rows to one top, along paths of one
length of digest-width siblings, give a colliding pair. -/
theorem climbs_binding (ms : MerkleScheme) (hs : ms.Sized) (pos : ℕ) (r1 r2 : Array F128) (y1 y2 : ByteArray)
    (s1 s2 : List ByteArray) (hw1 : ∀ x ∈ s1, x.size = ms.digestLen) (hw2 : ∀ x ∈ s2, x.size = ms.digestLen)
    (hlen : s1.length = s2.length) (heq : climb ms (ms.leaf r1 y1) pos s1 = climb ms (ms.leaf r2 y2) pos s2)
    (hr : r1 ≠ r2) : (opensPair ms pos r1 y1 s1 r2 y2 s2).Collides ms := by
  by_cases hl : ms.leaf r1 y1 = ms.leaf r2 y2
  · simp only [opensPair, hl, ↓reduceIte]
    exact ⟨hr, hl⟩
  · simp only [opensPair, hl, ↓reduceIte]
    exact climb_binding ms hs s1 s2 _ _ pos hlen hw1 hw2 (hs.2 _ _) (hs.2 _ _) hl heq

/-- **Two climbs from different rows to one top collide in the joined logs**: `tiedColl_of_opens` from the climbs'
equality alone, along paths of one length of digest-width siblings. -/
theorem tiedColl_of_climbs {pos : ℕ} {r1 r2 : Array F128} {y1 y2 : ByteArray} {s1 s2 : List ByteArray}
    {l : List (Claims.Msg × Claims.Msg)} (hw1 : ∀ x ∈ s1, x.size = MerkleScheme.hm96Sha512.digestLen)
    (hw2 : ∀ x ∈ s2, x.size = MerkleScheme.hm96Sha512.digestLen) (hlen : s1.length = s2.length)
    (heq : climb MerkleScheme.hm96Sha512 (MerkleScheme.hm96Sha512.leaf r1 y1) pos s1 =
      climb MerkleScheme.hm96Sha512 (MerkleScheme.hm96Sha512.leaf r2 y2) pos s2) (hr : r1 ≠ r2)
    (hl1 : ∀ b ∈ opensIns r1 y1 pos s1, Claims.toMsg b ∈ l.map Prod.fst)
    (hl2 : ∀ b ∈ opensIns r2 y2 pos s2, Claims.toMsg b ∈ l.map Prod.fst) : Claims.TiedColl sha512 l := by
  have hc := climbs_binding MerkleScheme.hm96Sha512 MerkleScheme.hm96Sha512_sized pos r1 r2 y1 y2 s1 s2 hw1 hw2 hlen
    heq hr
  obtain ⟨hne, heq'⟩ := MerklePair.hm96_inputs _ hc
  obtain ⟨m1, m2⟩ := opensPair_mem pos r1 y1 s1 r2 y2 s2 hc
  refine ⟨_, _, ⟨fun e => hne (toMsg_injective e), ?_⟩, hl1 _ m1, hl2 _ m2⟩
  show Claims.toMsg (Sha512.hash (Claims.ofMsg (Claims.toMsg _))) =
    Claims.toMsg (Sha512.hash (Claims.ofMsg (Claims.toMsg _)))
  rw [Claims.ofMsg_toMsg, Claims.ofMsg_toMsg, heq']

/-- **Two queries of an accepted inner proof at one position, with different pairs, collide in its log**: both
openings climb to the inner proof's root along paths of the tree's depth (`verifyInner_logs`). -/
theorem tiedColl_of_verifyInner {L : List (Claims.Msg × Claims.Msg)} {lay : Zk.PadLayout} {pr : Zk.InnerProof}
    {finalC : F128} {cons : Array (String × Zk.Lin)} {t t' : Transcript}
    (h : Zk.verifyInnerWith (hashesIn (L.map Prod.fst)) lay pr finalC cons t = .ok ((), t'))
    (hpaths : ∀ k < pr.paths.size, ∀ b ∈ (pr.paths[k]!).toList, b.size = 64) {k k' : ℕ} (hk : k < Zk.Q)
    (hk' : k' < Zk.Q)
    (hpos : ((t.rounds[t.k + 3]!).coins[k]!).lo.toNat % innerN lay =
      ((t.rounds[t.k + 3]!).coins[k']!).lo.toNat % innerN lay)
    (hne : pr.opened[k]! ≠ pr.opened[k']!) : Claims.TiedColl sha512 L := by
  obtain ⟨-, -, hps, hall⟩ := verifyInner_logs h
  obtain ⟨hd, hc, hl⟩ := hall k hk
  obtain ⟨hd', hc', hl'⟩ := hall k' hk'
  rw [hpos] at hc hl
  refine tiedColl_of_climbs (hpaths k (by omega)) (hpaths k' (by omega)) (by simp [hd, hd']) (hc.trans hc'.symm)
    ?_ hl hl'
  intro e
  apply hne
  exact Prod.ext (by simpa using congrArg (fun r : Array F128 => r[0]!) e)
    (by simpa using congrArg (fun r : Array F128 => r[1]!) e)

/-- **An accepted masked rep ran its inner proof on its stream's rounds, ending at the last one**: when
`Zk.verifyRepWith hs` accepts rep `r` of table `i`, its stream's decoded proof's inner proof passed
`Zk.verifyInnerWith hs` at the pads layout `layZ`, from a transcript on the stream's rounds four rounds before their
end. -/
theorem zkVerifyRepWith_inner (hs : Hashes Flock.V) {spec : SessionSpec} {st : Setup} {sch : Schedule} {tPad : ℕ}
    {sess : RunRecord} {i r : ℕ} {bytes c : ByteArray} (hsch : Zk.schedule st.m = .ok (sch, tPad))
    (h : Zk.verifyRepWith hs spec st sch tPad sess i r bytes = .ok c) :
    ∃ (stream : StreamRec) (extras : Array Claim) (zp : Zk.ZkProof) (finalC : F128)
      (cons : Array (String × Zk.Lin)) (t t' : Transcript),
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) = some stream ∧
      st.extra sess.points = .ok extras ∧ Zk.decode (shapeZK st sch extras) bytes = .ok zp ∧
      t.rounds = stream.rounds ∧ t.k + 4 = stream.rounds.size ∧
      Zk.verifyInnerWith hs (layZ st sch) zp.inner finalC cons t = .ok ((), t') := by
  obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, -, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ hsch
  obtain ⟨hn7, h2, -, -, -, -, -⟩ := hdim
  unfold Zk.verifyRepWith at h
  dsimp only at h
  cases hs' : Array.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r)
      sess.streams with
  | none => rw [hs'] at h; cases h
  | some stream =>
  rw [hs'] at h
  obtain ⟨extras, hex, h⟩ := ebind_ok h
  obtain ⟨zp, hzp, h⟩ := ebind_ok h
  by_cases c1 : (zp.proof.params != pcsParams st.merkle sch) = true
  · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1] at h
  by_cases c1' : (zp.inner.root.size != 64) = true
  · rw [ite_eq_left c1'] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c1'] at h
  obtain ⟨⟨u, tr⟩, hrun, h⟩ := ebind_ok h
  change (if (tr.k != stream.rounds.size) = true then _ else _) = _ at h
  by_cases c2 : (tr.k != stream.rounds.size) = true
  · rw [ite_eq_left c2] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  simp only [bne_iff_ne, ne_eq, not_not] at c2
  rw [StateT.run] at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  obtain ⟨hr₁, -⟩ := replayWith_rounds hs (by omega) h₁
  try dsimp only at hrun
  obtain ⟨⟨T, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨hr₂, -⟩ := ringSwitchWith_rounds hs (by simp) h₂
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  obtain ⟨hr₃, -⟩ := zkLigeritoWith_rounds hs _ sch _ _ _ _ _ (by omega) hk0 h2
    (fun i hi => (hlv i hi).q_ne_zero) h₃
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄
  | ok v =>
  rw [hv] at hrun
  try dsimp only at hrun
  obtain ⟨lg, t₄, h₄, hrun'⟩ := bind_ok hrun
  clear hrun
  obtain ⟨rfl, rfl⟩ := pure_ok h₄
  obtain ⟨-, hk₅⟩ := verifyInnerWith_rounds hs hrun'
  refine ⟨stream, extras, zp, _, _, _, tr, rfl, hex, hzp, ?_, ?_, hrun'⟩
  · rw [hr₃, hr₂, hr₁]
    rfl
  · rw [← c2, hk₅]

end FlockDeploy.KS
