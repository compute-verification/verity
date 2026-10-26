import Proofs.Flock.Claims.Complete.Zk.Link
import Proofs.Flock.Claims.Complete.ZkTables.Faithful
import Proofs.Flock.Claims.Complete.Replay.LigeritoWeight
import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLigPad

/-!
# The masked prover's first message, from its level-0 leaves

Round `0` of the masked session on the executable's side: `ZkTables.honestPR`'s first message is the link message of
the tables' level-0 caps (`honestPR_first`), and each cap is a function of its level's leaves alone.

* `layerOf`: layer `h` of a Merkle tree over `2^d` leaves, as `commitRowsP` builds it.
* `commitRowsP_cap_eq`: through `Hashes` whose node and leaf are functions, `commitRowsP` returns, and its cap is layer
  `d − cap` of the tree over the rows' leaves under their salts.
* `commitsPZk_caps`: at `SHA-512` and `hm96-sha512/v1`, each table's level-0 cap is layer `d − cap` of the tree of
  `MerkleScheme.hm96Sha512.leaf` over `encodeRowsZk`'s rows under the tape's level-0 salts.
* `encodeRowsZk_entry`: a row's entry at position `q` and lane `l`, as the model writes a level-0 column: the lane's
  message encoded at `ω_q` (`X̂_c(ω_q)`), plus its `t_pad` padding coefficients on C-Flock ZK's padding columns
  (`fOf_padF`, `ZkExec.padAt_eq` at `c₀ = PAD_C`).
* `sessionPZk_go_head`: once the schedules and the level-0 commitments return, the pure masked session's first body
  is the link message of the caps, whatever the session does after its coin.
* `honestPR_first`: so `ZkTables.honestPR`'s message before any reply is that link message, tagged `0`.
-/

open Flock FlockDeploy

namespace FlockComplete

open With

/-! ## A level's cap from its leaves -/

/-- **Layer `h` of a Merkle tree** over the leaves `lf q` (`q < 2^d`) with the node `nd`, as `commitRowsP` builds it:
layer `h + 1` holds the nodes of layer `h`'s pairs. -/
def layerOf (nd : ByteArray → ByteArray → ByteArray) (d : ℕ) (lf : ℕ → ByteArray) : ℕ → Array ByteArray
  | 0 => (Array.range (2 ^ d)).map lf
  | h + 1 => (Array.range ((layerOf nd d lf h).size / 2)).map fun i =>
      nd (layerOf nd d lf h)[2 * i]! (layerOf nd d lf h)[2 * i + 1]!

theorem mapM_ok {α β : Type} {f : α → V β} (g : α → β) (hf : ∀ a, f a = .ok (g a)) (xs : Array α) :
    xs.mapM f = .ok (xs.map g) := by
  have e : f = fun a => pure (g a) := funext hf
  subst e
  exact Array.mapM_pure

theorem list_mapM_ok_mem {α β : Type} {f : α → V β} (g : α → β) :
    ∀ l : List α, (∀ a ∈ l, f a = .ok (g a)) → l.mapM f = .ok (l.map g)
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons, h a List.mem_cons_self, list_mapM_ok_mem g l fun b hb => h b (List.mem_cons_of_mem a hb)]
    rfl

theorem mapM_ok_mem {α β : Type} {f : α → V β} (g : α → β) (xs : Array α) (hf : ∀ a ∈ xs, f a = .ok (g a)) :
    xs.mapM f = .ok (xs.map g) := by
  rw [Array.mapM_eq_mapM_toList, list_mapM_ok_mem g xs.toList fun a ha => hf a (Array.mem_toList_iff.1 ha)]
  exact congrArg Except.ok (Array.ext' (by simp))

/-- A loop over `[0:n]` whose every step yields returns, and keeps an invariant that each step carries. -/
theorem forIn_ok_inv {β : Type} (n : ℕ) (b : β) {f : ℕ → β → V (ForInStep β)} (I : ℕ → β → Prop) (h0 : I 0 b)
    (h : ∀ i < n, ∀ b, I i b → ∃ b', f i b = .ok (.yield b') ∧ I (i + 1) b') :
    ∃ r, forIn [0:n] b f = .ok r ∧ I n r := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices ∀ j s (b : β), s + j = n → I s b → ∃ r, forIn (List.range' s j 1) b f = .ok r ∧ I n r from
    this n 0 b (by omega) h0
  intro j
  induction j with
  | zero =>
    intro s b hs hb
    refine ⟨b, by rw [List.range'_zero, List.forIn_nil]; rfl, ?_⟩
    rw [← hs]
    exact hb
  | succ j ih =>
    intro s b hs hb
    obtain ⟨b', hf, hb'⟩ := h s (by omega) b hb
    obtain ⟨r, hr, hI⟩ := ih (s + 1) b' (by omega) hb'
    refine ⟨r, ?_, hI⟩
    rw [List.range'_succ, List.forIn_cons, hf]
    exact hr

theorem forIn_ok_inv' {β : Type} (n : ℕ) (b : β) {f : ℕ → β → V (ForInStep β)} (I : ℕ → β → Prop) (h0 : I 0 b)
    (h : ∀ i < n, ∀ b, I i b → ∃ b', f i b = .ok (.yield b') ∧ I (i + 1) b') {e : V β} (he : forIn [0:n] b f = e) :
    ∃ r, e = .ok r ∧ I n r := by
  subst he
  exact forIn_ok_inv n b I h0 h

/-- **A level's cap is layer `d − cap` of its leaves' tree**: through `Hashes` whose node is `nd` and whose leaf is
`lf`, `commitRowsP` returns, and its cap is that layer of the tree over the rows' leaves under their fitted salts. -/
theorem commitRowsP_cap_eq {hs : Hashes V} {ms : MerkleScheme} {nd : ByteArray → ByteArray → ByteArray}
    {lf : Array F128 → ByteArray → ByteArray} (hn : ∀ l r, (ms.opsWith hs).node l r = .ok (nd l r))
    (hl : ∀ row s, (ms.opsWith hs).leaf row s = .ok (lf row s)) (lv : Level) (rows : Array (Array F128))
    (salt : ℕ → ByteArray) :
    ∃ cm, (commitRowsP hs ms lv rows salt : V Commit) = .ok cm ∧
      cm.cap = layerOf nd lv.d (fun q => lf rows[q]! (fitBytes ms.saltBytes (salt q))) (lv.d - lv.cap) := by
  set L := layerOf nd lv.d (fun q => lf rows[q]! (fitBytes ms.saltBytes (salt q))) with hL
  have h0 : ((Array.range (2 ^ lv.d)).mapM fun q => (ms.opsWith hs).leaf rows[q]!
      ((Array.range (2 ^ lv.d)).map fun q => fitBytes ms.saltBytes (salt q))[q]! : V _) = .ok (L 0) := by
    rw [mapM_ok _ fun q => hl _ _]
    congr 1
    refine Array.ext (by simp [hL, layerOf]) fun q h1 _ => ?_
    have hq : q < 2 ^ lv.d := by simpa using h1
    simp only [hL, layerOf, Array.getElem_map, Array.getElem_range]
    rw [getElem!_pos ((Array.range (2 ^ lv.d)).map fun q => fitBytes ms.saltBytes (salt q)) q (by simpa using hq)]
    simp
  unfold commitRowsP
  simp only [bind, Except.bind, h0]
  generalize hF : forIn (m := V) (β := Array (Array ByteArray) × Array ByteArray) [0:lv.d - lv.cap] (#[L 0], L 0) _ = e
  obtain ⟨r, hr, hI⟩ := forIn_ok_inv' (lv.d - lv.cap) (#[L 0], L 0)
    (fun i (p : Array (Array ByteArray) × Array ByteArray) => p.2 = L i) rfl (by
      intro i _ p hp
      refine ⟨(p.1.push (L (i + 1)), L (i + 1)), ?_, rfl⟩
      rw [hp, mapM_ok _ fun i => hn _ _]
      rfl) hF
  subst hr
  exact ⟨_, rfl, hI⟩

/-- `hm96-sha512/v1` through `SHA-512` is its own leaf. -/
theorem hm96_leaf_std (row : Array F128) (s : ByteArray) :
    (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 Sha512.hash : Hashes V)).leaf row s =
      .ok (MerkleScheme.hm96Sha512.leaf row s) := rfl

/-- `hm96-sha512/v1` through `SHA-512` is its own node. -/
theorem hm96_node_std (l r : ByteArray) :
    (MerkleScheme.hm96Sha512.opsWith (Hashes.ofSha512 Sha512.hash : Hashes V)).node l r =
      .ok (Sha512.hash (l ++ r)) := rfl

/-! ## The masked level-0 caps -/

/-- **Each table's masked level-0 cap from its leaves**: at `SHA-512`, where every table commits under
`hm96-sha512/v1`, the level-0 commitments return, one per table, table `i`'s cap layer `d − cap` of the tree of
`MerkleScheme.hm96Sha512.leaf` over its `encodeRowsZk` rows under the tape's level-0 salts, fitted to 192 bytes. -/
theorem commitsPZk_caps (tabs : Array Setup) (schs : Array (Schedule × ℕ)) (tbls : Array (Array F128))
    (tape : TapeZk) (hms : ∀ i < tabs.size, tabs[i]!.merkle = MerkleScheme.hm96Sha512) :
    ∃ cms, (commitsPZk (Hashes.ofSha512 Sha512.hash) tabs schs tbls tape : V _) = .ok cms ∧ cms.size = tabs.size ∧
      ∀ i < tabs.size, cms[i]!.cap = layerOf (fun l r => Sha512.hash (l ++ r))
        (schs[i]!.1.levels.getD 0 default).d
        (fun q => MerkleScheme.hm96Sha512.leaf
          (encodeRowsZk (schs[i]!.1.levels.getD 0 default) schs[i]!.2 tbls[i]! (tape.extra i) (tape.lanePad i))[q]!
          (fitBytes 192 (tape.salt i 0 0 q)))
        ((schs[i]!.1.levels.getD 0 default).d - (schs[i]!.1.levels.getD 0 default).cap) := by
  have hc : ∀ i ∈ Array.range tabs.size, ∃ cm, (commitPZk (Hashes.ofSha512 Sha512.hash) tabs[i]!.merkle
      (schs[i]!.1.levels.getD 0 default) schs[i]!.2 tbls[i]! (tape.extra i) (tape.lanePad i) (tape.salt i 0 0) :
        V _) = .ok cm := by
    intro i hi
    rw [hms i (by simpa using hi)]
    obtain ⟨cm, h, -⟩ := commitRowsP_cap_eq (hm96_node_std) (hm96_leaf_std) (schs[i]!.1.levels.getD 0 default)
      (encodeRowsZk (schs[i]!.1.levels.getD 0 default) schs[i]!.2 tbls[i]! (tape.extra i) (tape.lanePad i))
      (tape.salt i 0 0)
    exact ⟨cm, h⟩
  classical
  let cmOf : ℕ → Commit := fun i => if h : i < tabs.size then (hc i (by simpa using h)).choose else default
  have hcm : ∀ i ∈ Array.range tabs.size, (commitPZk (Hashes.ofSha512 Sha512.hash) tabs[i]!.merkle
      (schs[i]!.1.levels.getD 0 default) schs[i]!.2 tbls[i]! (tape.extra i) (tape.lanePad i) (tape.salt i 0 0) :
        V _) = .ok (cmOf i) := by
    intro i hi
    have h : i < tabs.size := by simpa using hi
    simp only [cmOf, h, ↓reduceDIte]
    exact (hc i hi).choose_spec
  refine ⟨(Array.range tabs.size).map cmOf, mapM_ok_mem cmOf _ hcm, by simp, fun i hi => ?_⟩
  have h := hcm i (by simpa using hi)
  rw [hms i hi] at h
  obtain ⟨cm, h', hcap⟩ := commitRowsP_cap_eq hm96_node_std hm96_leaf_std (schs[i]!.1.levels.getD 0 default)
    (encodeRowsZk (schs[i]!.1.levels.getD 0 default) schs[i]!.2 tbls[i]! (tape.extra i) (tape.lanePad i))
    (tape.salt i 0 0)
  unfold commitPZk at h
  rw [h'] at h
  injection h with h
  subst h
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_map, Array.getElem_range]
  exact hcap

/-! ## A masked level-0 row -/

section Rows

open FlockSoundness FlockSoundness.Refine FlockLevel3

theorem fOf_sumF (n : ℕ) (f : ℕ → F128) : fOf (sumF n f) = ∑ x ∈ Finset.range n, fOf (f x) :=
  Replay.Zc.of_sumF n f

/-- **A lane's padding at a position is the model's** (`ZkExec.padAt_eq` at `c₀ = PAD_C`): for `t_pad ≤ 2^cols`
coefficients `μ`, `Σ_j μ_j·(c₀·X̂_j(ω_p) + X̂_{2^cols+j}(ω_p))`. -/
theorem fOf_padF (p cols tPad : ℕ) (mu : Array F128) (hy : mu.size = tPad) (ht : tPad ≤ 2 ^ cols) :
    fOf (padF (whats (svTable (cols + 1)) p (cols + 1)) cols mu) =
      ∑ j : Fin tPad, fOf mu[j.val]! * (fOf Zk.PAD_C * execArith.Xhat j.val (cols + 1) (execArith.omega p) +
        execArith.Xhat (2 ^ cols + j.val) (cols + 1) (execArith.omega p)) := by
  have hg : gOf (Zk.padAt (whats (svTable (cols + 1)) p (cols + 1)) cols (mu.map F256.ofF)) =
      embed (∑ j : Fin tPad, fOf mu[j.val]! * (fOf Zk.PAD_C * execArith.Xhat j.val (cols + 1)
        (execArith.omega p) + execArith.Xhat (2 ^ cols + j.val) (cols + 1) (execArith.omega p))) := by
    rw [FlockSoundness.Discharge.ZkExec.padAt_eq p cols tPad _ (by simp [hy]) ht]
    unfold Model.padCorr
    rw [map_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    have hj : j.val < mu.size := by omega
    simp only [getElem!_pos (mu.map F256.ofF) j.val (by simpa using hj), Array.getElem_map, getElem!_pos mu j.val hj,
      gOf_ofF, map_mul]
    rfl
  have hx : Zk.padAt (whats (svTable (cols + 1)) p (cols + 1)) cols (mu.map F256.ofF) =
      F256.ofF ((∑ j : Fin tPad, fOf mu[j.val]! * (fOf Zk.PAD_C * execArith.Xhat j.val (cols + 1)
        (execArith.omega p) + execArith.Xhat (2 ^ cols + j.val) (cols + 1) (execArith.omega p)) : GF128)) := hg
  unfold padF
  rw [hx]
  rfl

theorem encodeRowsZk_size (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) {q : ℕ}
    (hq : q < 2 ^ lv.d) : ((encodeRowsZk lv tPad g extra lanePad)[q]!).size = 2 ^ lv.lanes + Zk.EXTRA_LANES := by
  unfold encodeRowsZk
  rw [getElem!_pos _ q (by simpa using hq)]
  simp

/-- **A masked level-0 row's entry** at position `q < 2^d`, lane `l`: the lane's message (the table's witness lane `l`,
or extra lane `l − 2^lanes`) encoded at `ω_q`, plus its padding by its `t_pad` coefficients. -/
theorem encodeRowsZk_entry (lv : Level) (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) {q l : ℕ}
    (hq : q < 2 ^ lv.d) (hl : l < 2 ^ lv.lanes + Zk.EXTRA_LANES) (ht : tPad ≤ 2 ^ lv.cols) :
    fOf ((encodeRowsZk lv tPad g extra lanePad)[q]!)[l]! =
      ∑ c ∈ Finset.range (2 ^ lv.cols), fOf ((if l < 2 ^ lv.lanes then fun c => g[c * 2 ^ lv.lanes + l]!
          else extra (l - 2 ^ lv.lanes)) c) * execArith.Xhat c (lv.cols + 1) (execArith.omega q) +
        ∑ j : Fin tPad, fOf (lanePad l j.val) * (fOf Zk.PAD_C * execArith.Xhat j.val (lv.cols + 1)
          (execArith.omega q) + execArith.Xhat (2 ^ lv.cols + j.val) (lv.cols + 1) (execArith.omega q)) := by
  unfold encodeRowsZk
  rw [getElem!_pos _ q (by simpa using hq), Array.getElem_map, Array.getElem_range]
  dsimp only
  rw [getElem!_pos _ l (by simpa using hl), Array.getElem_map, Array.getElem_range]
  rw [fOf_add, fOf_sumF]
  congr 1
  · refine Finset.sum_congr rfl fun c hc => ?_
    rw [fOf_mul, Replay.Lg.fOf_xhats q lv.cols c (Finset.mem_range.1 hc),
      Xhat_succ_low execArith lv.cols c (Finset.mem_range.1 hc)]
  · rw [getElem!_pos _ l (by simpa using hl), Array.getElem_map, Array.getElem_range,
      fOf_padF q lv.cols tPad _ (by simp) ht]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [getElem!_pos _ j.val (by simp), Array.getElem_map, Array.getElem_range]

end Rows

/-! ## The first send -/

/-- **The first body is the link message**: once the schedules and the level-0 commitments return, the pure masked
session's first body is the link message of the caps, with no public words, whatever the session does after its coin. -/
theorem sessionPZk_go_head (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec)
    (tabs : Array Setup) (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk) {schs : Array (Schedule × ℕ)}
    {cms : Array Commit} (hs : (schedsPZk tabs : V _) = .ok schs)
    (hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) tabs schs (tablesP tabs trs) tape : V _) = .ok cms) :
    ((sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[]).2.toList.head? =
      some (linkBody (cms.map fun c => flat c.cap).toList (tabs.map fun _ => ByteArray.empty).toList) := by
  rw [sessionPZk_eq, go_bind, schedsPZk_quiet, hs]
  dsimp only
  rw [go_bind, commitsPZk_quiet, hc]
  dsimp only
  rw [go_bind, asksS_link_go]
  dsimp only
  obtain ⟨bs, -, hl⟩ := afterLinkPZk_sim H resp spec tabs words trs tape schs (tablesP tabs trs) cms
    (coinOf (resp (#[] : Array Claims.Msg).size)) (#[].push (linkBody (cms.map fun c => flat c.cap).toList
      (tabs.map fun _ => ByteArray.empty).toList))
  rw [hl]
  simp

/-- **`ZkTables.honestPR`'s first message is the link message of the level-0 caps**: before any reply, at a statement
that sets up, once the schedules and the level-0 commitments of the tables at the tape's mask words return. -/
theorem honestPR_first (H : Claims.Msg → Claims.Msg) (w : Claims.Flock.Wit) (x : Claims.Msg) (ω : TapeZk)
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} (hset : setupOf true x = some (s, spec, sts))
    {schs : Array (Schedule × ℕ)} {cms : Array Commit} (hs : (schedsPZk (tabsOf sts) : V _) = .ok schs)
    (hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) (tabsOf sts) schs
      (tablesP (tabsOf sts) (ZkTables.traces sts (ZkTables.maskWOf x w ω.mask))) ω : V _) = .ok cms) :
    (((ZkTables.honestPR w).next x [] ω).eval H).1 =
      0 :: linkBody (cms.map fun c => flat c.cap).toList ((tabsOf sts).map fun _ => ByteArray.empty).toList := by
  rw [ZkTables.honestPR_next (by simp [ZkTables.outerHit])]
  change ((feed (ZkTables.prog x (ZkTables.maskWOf x w ω.mask) ω) []).eval H).1 = _
  have hf := feed_sim H (fun _ => []) (ZkTables.prog x (ZkTables.maskWOf x w ω.mask) ω) 0 0 (Nat.zero_le _)
  simp only [List.range_zero, List.map_nil] at hf
  rw [hf]
  obtain ⟨bs, h1, h2⟩ := sessionPZk_sim H (fun _ => []) spec (tabsOf sts) (MaskRank.masksOf sts)
    (ZkTables.traces sts (ZkTables.maskWOf x w ω.mask)) ω #[]
  have hd := sessionPZk_go_head H (fun _ => []) spec (tabsOf sts) (MaskRank.masksOf sts)
    (ZkTables.traces sts (ZkTables.maskWOf x w ω.mask)) ω hs hc
  rw [h2, Array.empty_append, List.toList_toArray] at hd
  obtain ⟨b, bs', rfl⟩ : ∃ b bs', bs = b :: bs' := List.exists_cons_of_ne_nil (fun h => by simp [h] at hd)
  simp only [List.head?_cons, Option.some.injEq] at hd
  subst hd
  unfold ZkTables.prog
  rw [hset]
  dsimp only
  change (((PI.bind (sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts)
    (ZkTables.traces sts (ZkTables.maskWOf x w ω.mask)) ω).run _).sim H (fun _ => []) 0).1 ++ _).getD 0 [] = _
  rw [PI.sim_bind]
  simp only [Array.size_empty] at h1
  rw [h1]
  simp

end FlockComplete
